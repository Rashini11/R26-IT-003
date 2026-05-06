from pathlib import Path
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# =========================
# CONFIG
# =========================
DATA_DIR = Path("ml/dataset_v2_balanced")
MODEL_DIR = Path("ml/models")
RESULTS_FILE = MODEL_DIR / "cnn_experiment_results.csv"

EPOCHS = 5
BATCH_SIZE = 32
IMAGE_SIZE = 128
LEARNING_RATE = 0.001

MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Device selection
if torch.backends.mps.is_available():
    device = torch.device("mps")   # Mac GPU if available
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("Using device:", device)

# =========================
# DATA LOADERS
# =========================
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
])

train_dataset = datasets.ImageFolder(DATA_DIR / "train", transform=transform)
val_dataset = datasets.ImageFolder(DATA_DIR / "val", transform=transform)
test_dataset = datasets.ImageFolder(DATA_DIR / "test", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print("Classes:", train_dataset.classes)
print("Train images:", len(train_dataset))
print("Val images:", len(val_dataset))
print("Test images:", len(test_dataset))


# =========================
# MODEL 1: SIMPLE CNN
# =========================
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(SimpleCNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)

        self.fc1 = nn.Linear(32 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 128 -> 64
        x = self.pool(F.relu(self.conv2(x)))  # 64 -> 32
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# =========================
# MODEL 2: DEEPER CNN
# =========================
class DeeperCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(DeeperCNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(64 * 16 * 16, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 128 -> 64
        x = self.pool(F.relu(self.conv2(x)))  # 64 -> 32
        x = self.pool(F.relu(self.conv3(x)))  # 32 -> 16
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# =========================
# MODEL 3: TINY VGG STYLE
# =========================
class TinyVGG(nn.Module):
    def __init__(self, num_classes=3):
        super(TinyVGG, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 128 -> 64

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 64 -> 32
        )

        self.classifier = nn.Sequential(
            nn.Linear(64 * 32 * 32, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


# =========================
# EVALUATION FUNCTION
# =========================
def evaluate(model, loader):
    model.eval()

    correct = 0
    total = 0
    total_loss = 0.0

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    avg_loss = total_loss / len(loader)

    return avg_loss, accuracy


# =========================
# TRAINING FUNCTION
# =========================
def train_model(model_name, model):
    print("\n==============================")
    print(f"Training {model_name}")
    print("==============================")

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_acc = 0.0
    best_model_path = MODEL_DIR / f"{model_name}_best.pth"

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        val_loss, val_acc = evaluate(model, val_loader)

        print(
            f"{model_name} | Epoch {epoch+1}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.2f}%"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)

    # Load best model and test
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    test_loss, test_acc = evaluate(model, test_loader)

    print(f"{model_name} Best Val Acc: {best_val_acc:.2f}%")
    print(f"{model_name} Test Acc: {test_acc:.2f}%")
    print(f"Saved best model: {best_model_path}")

    return {
        "model_name": model_name,
        "best_val_accuracy": round(best_val_acc, 2),
        "test_accuracy": round(test_acc, 2),
        "model_path": str(best_model_path)
    }


# =========================
# RUN EXPERIMENTS
# =========================
models = {
    "SimpleCNN": SimpleCNN(num_classes=3),
    "DeeperCNN": DeeperCNN(num_classes=3),
    "TinyVGG": TinyVGG(num_classes=3)
}

results = []

for model_name, model in models.items():
    result = train_model(model_name, model)
    results.append(result)

# Save results to CSV
with open(RESULTS_FILE, mode="w", newline="") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=["model_name", "best_val_accuracy", "test_accuracy", "model_path"]
    )
    writer.writeheader()
    writer.writerows(results)

print("\n==============================")
print("CNN Experiments Completed")
print("==============================")
print(f"Results saved to: {RESULTS_FILE}")

for r in results:
    print(r)