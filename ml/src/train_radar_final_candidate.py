from pathlib import Path
import random
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "ml" / "dataset_v3_grouped"
OUT = ROOT / "ml" / "models" / "v3_runs" / "radar_final_candidate"
OUT.mkdir(parents=True, exist_ok=True)

MODEL_PATH = OUT / "radar_final_candidate_best.pth"

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

device = (
    torch.device("mps")
    if torch.backends.mps.is_available()
    else torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("cpu")
)

print("Using device:", device)

train_tf = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomAffine(
        degrees=10,
        translate=(0.08, 0.08),
        scale=(0.85, 1.15),
    ),
    transforms.GaussianBlur(3, sigma=(0.1, 1.5)),
    transforms.ToTensor(),
    transforms.RandomErasing(
        p=0.35,
        scale=(0.03, 0.12),
        value="random",
    ),
])

eval_tf = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

train_ds = datasets.ImageFolder(DATA / "train", transform=train_tf)
val_ds = datasets.ImageFolder(DATA / "val", transform=eval_tf)

train_loader = DataLoader(
    train_ds, batch_size=32, shuffle=True, num_workers=0
)

val_loader = DataLoader(
    val_ds, batch_size=32, shuffle=False, num_workers=0
)

print("Train:", len(train_ds))
print("Validation:", len(val_ds))
print("TEST SET: NOT LOADED")


class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 4, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(4, 6, 3, stride=2, padding=1),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.dropout = nn.Dropout(0.62)
        self.fc = nn.Linear(6, 3)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        x = self.dropout(x)
        return self.fc(x)


model = TinyCNN().to(device)

criterion = nn.CrossEntropyLoss(label_smoothing=0.22)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.001,
    weight_decay=0.015,
)


def validate():
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

    return (
        total_loss / total,
        float((actual == predicted).mean()),
        float(
            f1_score(
                actual,
                predicted,
                average="macro",
                zero_division=0,
            )
        ),
    )


best_loss = float("inf")
best_acc = 0
best_f1 = 0

for epoch in range(1, 7):

    model.train()

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

    val_loss, val_acc, val_f1 = validate()

    print(
        f"Epoch {epoch:02d} | "
        f"Loss {val_loss:.4f} | "
        f"Val acc {val_acc*100:.2f}% | "
        f"F1 {val_f1:.4f}"
    )

    if val_loss < best_loss:
        best_loss = val_loss
        best_acc = val_acc
        best_f1 = val_f1

        torch.save(
            model.state_dict(),
            MODEL_PATH,
        )

        print("Saved best model")


print()
print("=" * 55)
print("RADAR FINAL CANDIDATE COMPLETE")
print("=" * 55)
print(f"Best validation accuracy: {best_acc*100:.2f}%")
print(f"Best validation macro F1: {best_f1:.4f}")
print(f"Model: {MODEL_PATH}")
print("TEST SET HAS NOT BEEN EVALUATED.")
print("=" * 55)
