from pathlib import Path
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# =========================
# CONFIG
# =========================
DATA_DIR = Path("ml/dataset_v2_balanced")
MODEL_DIR = Path("ml/models")
RESULTS_FILE = MODEL_DIR / "rnn_experiment_results.csv"

EPOCHS = 5
BATCH_SIZE = 32
IMAGE_SIZE = 128
LEARNING_RATE = 0.001

MODEL_DIR.mkdir(parents=True, exist_ok=True)

# RNNs can sometimes be unstable on Mac MPS, but we will try MPS first
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("Using device:", device)

# =========================
# DATA LOADERS
# =========================
# Convert heatmap image to grayscale because RNN uses row sequences
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.Grayscale(num_output_channels=1),
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
# MODEL 1: SIMPLE RNN
# =========================
class SimpleRNNModel(nn.Module):
    def __init__(self, input_size=128, hidden_size=128, num_classes=3):
        super(SimpleRNNModel, self).__init__()

        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x shape: [batch, 1, 128, 128]
        x = x.squeeze(1)  # [batch, 128, 128]

        output, hidden = self.rnn(x)

        # use last timestep
        last_output = output[:, -1, :]

        return self.fc(last_output)


# =========================
# MODEL 2: LSTM
# =========================
class LSTMModel(nn.Module):
    def __init__(self, input_size=128, hidden_size=128, num_classes=3):
        super(LSTMModel, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.squeeze(1)  # [batch, 128, 128]

        output, (hidden, cell) = self.lstm(x)

        last_output = output[:, -1, :]

        return self.fc(last_output)


# =========================
# MODEL 3: GRU
# =========================
class GRUModel(nn.Module):
    def __init__(self, input_size=128, hidden_size=128, num_classes=3):
        super(GRUModel, self).__init__()

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.squeeze(1)  # [batch, 128, 128]

        output, hidden = self.gru(x)

        last_output = output[:, -1, :]

        return self.fc(last_output)


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
    "SimpleRNN": SimpleRNNModel(num_classes=3),
    "LSTM": LSTMModel(num_classes=3),
    "GRU": GRUModel(num_classes=3)
}

results = []

for model_name, model in models.items():
    result = train_model(model_name, model)
    results.append(result)

with open(RESULTS_FILE, mode="w", newline="") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=["model_name", "best_val_accuracy", "test_accuracy", "model_path"]
    )
    writer.writeheader()
    writer.writerows(results)

print("\n==============================")
print("RNN Experiments Completed")
print("==============================")
print(f"Results saved to: {RESULTS_FILE}")

for r in results:
    print(r)