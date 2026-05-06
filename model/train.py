import os
import json
import torch
import torch.nn as nn
import pandas as pd
import torchvision.models as models
from torchvision.models import ResNet18_Weights
from torchvision import transforms
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# -----------------------------
# Settings
# -----------------------------
DATASET_FILE = "dataset.csv"
MODEL_SAVE_PATH = "model/sea_model.pth"
REPORT_SAVE_PATH = "model/evaluation_report.txt"
LABEL_MAP_SAVE_PATH = "model/label_map.json"

SAMPLE_SIZE = 7200
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 0.001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv(DATASET_FILE)

print("Original dataset size:", len(df))

df["image_exists"] = df["image_path"].apply(os.path.exists)
missing_df = df[df["image_exists"] == False]

print("Missing images:", len(missing_df))

if len(missing_df) > 0:
    print("Example missing images:")
    print(missing_df.head())

df = df[df["image_exists"] == True].drop(columns=["image_exists"])

print("Dataset after removing missing images:", len(df))

if len(df) > SAMPLE_SIZE:
    df = df.sample(n=SAMPLE_SIZE, random_state=42)

print("Final dataset size:", len(df))

# -----------------------------
# Encode labels
# -----------------------------
labels = sorted(df["label"].unique())
label_map = {label: i for i, label in enumerate(labels)}
reverse_label_map = {i: label for label, i in label_map.items()}

df["label_encoded"] = df["label"].map(label_map)

print("Label map:", label_map)

os.makedirs("model", exist_ok=True)

with open(LABEL_MAP_SAVE_PATH, "w") as f:
    json.dump(label_map, f)

# -----------------------------
# Train / Validation / Test Split
# -----------------------------
train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    random_state=42,
    stratify=df["label_encoded"]
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=42,
    stratify=temp_df["label_encoded"]
)

print("Train size:", len(train_df))
print("Validation size:", len(val_df))
print("Test size:", len(test_df))

# -----------------------------
# Image Transform
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# -----------------------------
# Dataset Class
# -----------------------------
class SeaDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe):
        self.df = dataframe.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["image_path"]

        try:
            image = Image.open(img_path).convert("RGB")
            image = transform(image)

        except Exception as e:
            print("\n❌ Image loading error")
            print("Image path:", img_path)
            print("Error:", e)
            next_idx = (idx + 1) % len(self.df)
            return self.__getitem__(next_idx)

        sensor = torch.tensor(
            [row["wave_height"], row["wind_speed"]],
            dtype=torch.float32
        )

        label = torch.tensor(row["label_encoded"], dtype=torch.long)

        return image, sensor, label

# -----------------------------
# DataLoaders
# -----------------------------
train_loader = torch.utils.data.DataLoader(
    SeaDataset(train_df),
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = torch.utils.data.DataLoader(
    SeaDataset(val_df),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

test_loader = torch.utils.data.DataLoader(
    SeaDataset(test_df),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

# -----------------------------
# ResNet Multimodal Model
# -----------------------------
class MultiModalModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.cnn = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.cnn.fc = nn.Linear(512, 128)

        self.sensor_net = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(128 + 16, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, image, sensor):
        image_features = self.cnn(image)
        sensor_features = self.sensor_net(sensor)

        combined = torch.cat((image_features, sensor_features), dim=1)
        output = self.classifier(combined)

        return output

model = MultiModalModel(num_classes=len(labels)).to(device)

# -----------------------------
# Loss and Optimizer
# -----------------------------
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# -----------------------------
# Evaluation Function
# -----------------------------
def evaluate_model(loader):
    model.eval()

    total_loss = 0
    correct = 0
    total = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, sensors, labels_batch in loader:
            images = images.to(device)
            sensors = sensors.to(device)
            labels_batch = labels_batch.to(device)

            outputs = model(images, sensors)
            loss = criterion(outputs, labels_batch)

            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)

            correct += (predicted == labels_batch).sum().item()
            total += labels_batch.size(0)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels_batch.cpu().numpy())

    accuracy = correct / total
    avg_loss = total_loss / len(loader)

    return avg_loss, accuracy, all_labels, all_preds

# -----------------------------
# Training Loop
# -----------------------------
try:
    best_val_accuracy = 0.0

    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0

        print(f"\n🚀 Starting Epoch {epoch + 1}/{EPOCHS}")

        for batch_idx, (images, sensors, labels_batch) in enumerate(train_loader):
            images = images.to(device)
            sensors = sensors.to(device)
            labels_batch = labels_batch.to(device)

            optimizer.zero_grad()

            outputs = model(images, sensors)
            loss = criterion(outputs, labels_batch)

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

            if batch_idx % 10 == 0:
                print(
                    f"Epoch {epoch + 1}, "
                    f"Batch {batch_idx}/{len(train_loader)}, "
                    f"Loss: {loss.item():.4f}"
                )

        train_loss = total_train_loss / len(train_loader)
        val_loss, val_accuracy, _, _ = evaluate_model(val_loader)

        print(f"\n✅ Epoch {epoch + 1} completed")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print("💾 Best ResNet model saved!")

except Exception as e:
    print("\n🔥 TRAINING CRASHED")
    print("Error:", e)
    exit()

# -----------------------------
# Final Test Evaluation
# -----------------------------
try:
    print("\n📊 Testing best ResNet model...")

    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))

    test_loss, test_accuracy, y_true, y_pred = evaluate_model(test_loader)

    target_names = [reverse_label_map[i] for i in range(len(labels))]

    report = classification_report(
        y_true,
        y_pred,
        target_names=target_names
    )

    cm = confusion_matrix(y_true, y_pred)

    print("\n✅ Final ResNet Test Accuracy:", round(test_accuracy * 100, 2), "%")

    print("\nLabel Mapping:")
    for idx, label in reverse_label_map.items():
        print(f"{idx} = {label}")

    print("\nClassification Report:\n", report)
    print("\nConfusion Matrix:\n", cm)

    with open(REPORT_SAVE_PATH, "w", encoding="utf-8") as f:
        f.write("RESNET MULTIMODAL SEA STATE CLASSIFICATION REPORT\n")
        f.write("=================================================\n\n")

        f.write(f"Final Dataset Size: {len(df)}\n")
        f.write(f"Train Size: {len(train_df)}\n")
        f.write(f"Validation Size: {len(val_df)}\n")
        f.write(f"Test Size: {len(test_df)}\n\n")

        f.write(f"Best Validation Accuracy: {best_val_accuracy * 100:.2f}%\n")
        f.write(f"Final Test Accuracy: {test_accuracy * 100:.2f}%\n\n")

        f.write("Label Mapping:\n")
        for idx, label in reverse_label_map.items():
            f.write(f"{idx} = {label}\n")

        f.write("\nClassification Report:\n")
        f.write(report)

        f.write("\nConfusion Matrix:\n")
        f.write(str(cm))

    print("\n✅ ResNet report saved to:", REPORT_SAVE_PATH)
    print("✅ ResNet model saved to:", MODEL_SAVE_PATH)
    print("✅ Label map saved to:", LABEL_MAP_SAVE_PATH)

except Exception as e:
    print("\n🔥 TESTING / REPORT GENERATION CRASHED")
    print("Error:", e)