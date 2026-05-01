import torch
import torch.nn as nn
import pandas as pd
import torchvision.models as models
from torchvision.models import ResNet18_Weights, MobileNet_V2_Weights
from torchvision import transforms
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# -----------------------------
# SELECT MODEL HERE
# -----------------------------
MODEL_TYPE = "mobilenet"  # "resnet" or "mobilenet"

if MODEL_TYPE == "resnet":
    MODEL_PATH = "model/sea_model.pth"
    REPORT_PATH = "model/resnet_report_updated.txt"
else:
    MODEL_PATH = "model/mobilenet_model.pth"
    REPORT_PATH = "model/mobilenet_report_updated.txt"

DATASET_FILE = "dataset.csv"
device = torch.device("cpu")

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv(DATASET_FILE)

labels = sorted(df["label"].unique())
label_map = {label: i for i, label in enumerate(labels)}
reverse_label_map = {i: label for label, i in label_map.items()}

df["label_encoded"] = df["label"].map(label_map)

# Split (same as before)
train_df, temp_df = train_test_split(
    df, test_size=0.30, random_state=42, stratify=df["label_encoded"]
)

val_df, test_df = train_test_split(
    temp_df, test_size=0.50, random_state=42, stratify=temp_df["label_encoded"]
)

# -----------------------------
# Transform
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# -----------------------------
# Dataset
# -----------------------------
class SeaDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image = Image.open(row["image_path"]).convert("RGB")
        image = transform(image)

        sensor = torch.tensor(
            [row["wave_height"], row["wind_speed"]],
            dtype=torch.float32
        )

        label = torch.tensor(row["label_encoded"], dtype=torch.long)

        return image, sensor, label

test_loader = torch.utils.data.DataLoader(
    SeaDataset(test_df), batch_size=32
)

# -----------------------------
# MODEL DEFINITION
# -----------------------------
class MultiModalModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        if MODEL_TYPE == "resnet":
            self.cnn = models.resnet18(weights=ResNet18_Weights.DEFAULT)
            self.cnn.fc = nn.Linear(512, 128)
        else:
            self.cnn = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
            self.cnn.classifier[1] = nn.Linear(1280, 128)

        self.sensor_net = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 16)
        )

        self.fc = nn.Sequential(
            nn.Linear(128 + 16, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, img, sensor):
        img_feat = self.cnn(img)
        sensor_feat = self.sensor_net(sensor)
        combined = torch.cat((img_feat, sensor_feat), dim=1)
        return self.fc(combined)

model = MultiModalModel(len(labels))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# -----------------------------
# Evaluation
# -----------------------------
all_preds = []
all_labels = []

with torch.no_grad():
    for img, sensor, label in test_loader:
        output = model(img, sensor)
        _, pred = torch.max(output, 1)

        all_preds.extend(pred.numpy())
        all_labels.extend(label.numpy())

# -----------------------------
# Report
# -----------------------------
target_names = [reverse_label_map[i] for i in range(len(labels))]

report = classification_report(
    all_labels,
    all_preds,
    target_names=target_names
)

cm = confusion_matrix(all_labels, all_preds)

print("\nLabel Mapping:")
for idx, label in reverse_label_map.items():
    print(f"{idx} = {label}")

print("\nClassification Report:\n", report)
print("\nConfusion Matrix:\n", cm)

# Save
with open(REPORT_PATH, "w") as f:
    f.write("UPDATED EVALUATION REPORT\n\n")

    f.write("Label Mapping:\n")
    for idx, label in reverse_label_map.items():
        f.write(f"{idx} = {label}\n")

    f.write("\nClassification Report:\n")
    f.write(report)

    f.write("\nConfusion Matrix:\n")
    f.write(str(cm))

print("\nSaved updated report:", REPORT_PATH)