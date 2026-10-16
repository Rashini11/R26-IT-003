from pathlib import Path
import json, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "ml" / "dataset_v3_grouped"
OUT = ROOT / "ml" / "models" / "v3_runs" / "simplecnn"
OUT.mkdir(parents=True, exist_ok=True)

MODEL_PATH = OUT / "simplecnn_v3_best.pth"

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = (
    torch.device("mps")
    if torch.backends.mps.is_available()
    else torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("cpu")
)

print("Using device:", device)

train_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomAffine(
        degrees=8,
        translate=(0.05, 0.05),
        scale=(0.90, 1.10),
    ),
    transforms.GaussianBlur(3, sigma=(0.1, 1.2)),
    transforms.ToTensor(),
    transforms.RandomErasing(
        p=0.30,
        scale=(0.02, 0.10),
        value="random",
    ),
])

eval_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

train_ds = datasets.ImageFolder(
    DATA / "train",
    transform=train_transform,
)

val_ds = datasets.ImageFolder(
    DATA / "val",
    transform=eval_transform,
)

train_loader = DataLoader(
    train_ds,
    batch_size=32,
    shuffle=True,
    num_workers=0,
)

val_loader = DataLoader(
    val_ds,
    batch_size=32,
    shuffle=False,
    num_workers=0,
)

print("Classes:", train_ds.classes)
print("Train:", len(train_ds))
print("Validation:", len(val_ds))
print("TEST SET: NOT LOADED")


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 12, 3, padding=1)
        self.conv2 = nn.Conv2d(12, 24, 3, padding=1)

        self.pool = nn.MaxPool2d(2)

        self.fc1 = nn.Linear(
            24 * 32 * 32,
            64,
        )

        self.dropout = nn.Dropout(0.55)
        self.fc2 = nn.Linear(64, 3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        x = x.flatten(1)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        return self.fc2(x)


model = SimpleCNN().to(device)

criterion = nn.CrossEntropyLoss(
    label_smoothing=0.10
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.0005,
    weight_decay=0.001,
)


def evaluate():
    model.eval()

    actual = []
    predicted = []
    total_loss = 0
    total = 0

    with torch.inference_mode():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            output = model(images)
            loss = criterion(output, labels)

            total_loss += loss.item() * labels.size(0)
            total += labels.size(0)

            pred = output.argmax(1)

            actual.extend(labels.cpu().tolist())
            predicted.extend(pred.cpu().tolist())

    actual = np.array(actual)
    predicted = np.array(predicted)

    return {
        "loss": total_loss / total,
        "accuracy": float((actual == predicted).mean()),
        "f1": float(
            f1_score(
                actual,
                predicted,
                average="macro",
                zero_division=0,
            )
        ),
    }


best_loss = float("inf")
best = None
patience = 3
no_improve = 0

for epoch in range(1, 11):

    model.train()

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        output = model(images)
        loss = criterion(output, labels)

        loss.backward()
        optimizer.step()

    metrics = evaluate()

    print(
        f"Epoch {epoch:02d} | "
        f"Val loss {metrics['loss']:.4f} | "
        f"Val acc {metrics['accuracy']*100:.2f}% | "
        f"F1 {metrics['f1']:.4f}"
    )

    if metrics["loss"] < best_loss:
        best_loss = metrics["loss"]
        best = dict(metrics)

        torch.save(
            model.state_dict(),
            MODEL_PATH,
        )

        no_improve = 0
        print("Saved best model")

    else:
        no_improve += 1

    if no_improve >= patience:
        print("Early stopping")
        break


summary = {
    "validation_accuracy_percent": round(
        best["accuracy"] * 100,
        2,
    ),
    "validation_macro_f1": round(
        best["f1"],
        4,
    ),
    "validation_loss": round(
        best["loss"],
        6,
    ),
    "test_set_evaluated": False,
}

with (OUT / "validation_summary.json").open("w") as f:
    json.dump(summary, f, indent=2)

print()
print("=" * 55)
print("SIMPLECNN V3 COMPLETE")
print("=" * 55)
print(
    "Validation accuracy:",
    f"{summary['validation_accuracy_percent']:.2f}%"
)
print(
    "Validation macro F1:",
    f"{summary['validation_macro_f1']:.4f}"
)
print("TEST SET HAS NOT BEEN EVALUATED.")
print("=" * 55)
