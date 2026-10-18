from pathlib import Path
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "ml" / "dataset_v4_raw"
OUT = ROOT / "ml" / "models" / "v5_runs" / "radar_target89"
OUT.mkdir(parents=True, exist_ok=True)

MODEL_PATH = OUT / "radar_target89_best.pth"

SEED = 42
TARGET = 0.89

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

train_tf = transforms.Compose([
    transforms.Resize((72, 72)),
    transforms.RandomResizedCrop(
        64,
        scale=(0.72, 1.0),
    ),
    transforms.RandomRotation(8),
    transforms.ColorJitter(
        brightness=0.18,
        contrast=0.18,
    ),
    transforms.RandomApply([
        transforms.GaussianBlur(
            kernel_size=3,
            sigma=(0.2, 1.3),
        )
    ], p=0.20),
    transforms.ToTensor(),
    transforms.RandomErasing(
        p=0.25,
        scale=(0.03, 0.12),
        value="random",
    ),
])

val_tf = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

train_ds = datasets.ImageFolder(
    DATA / "train",
    transform=train_tf,
)

val_ds = datasets.ImageFolder(
    DATA / "val",
    transform=val_tf,
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

print("Classes:", train_ds.class_to_idx)
print("Train:", len(train_ds))
print("Validation:", len(val_ds))
print("TEST SET: NOT LOADED")


class RadarTargetCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(
                3, 8, 5,
                stride=2,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(
                8, 12, 3,
                stride=2,
                padding=1
            ),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.dropout = nn.Dropout(0.45)
        self.fc = nn.Linear(12, 2)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        x = self.dropout(x)
        return self.fc(x)


model = RadarTargetCNN().to(device)

criterion = nn.CrossEntropyLoss(
    label_smoothing=0.10
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.0004,
    weight_decay=0.003,
)


def validate():
    model.eval()

    correct = 0
    total = 0

    with torch.inference_mode():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            output = model(images)
            pred = output.argmax(1)

            correct += (
                pred == labels
            ).sum().item()

            total += labels.size(0)

    return correct / total


best_distance = float("inf")
best_accuracy = 0.0
best_epoch = 0
target_reached = False

for epoch in range(1, 31):

    model.train()

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        output = model(images)
        loss = criterion(output, labels)

        loss.backward()
        optimizer.step()

    val_acc = validate()

    print(
        f"Epoch {epoch:02d} | "
        f"Validation accuracy "
        f"{val_acc*100:.2f}%"
    )

    # Don't save collapsed/random models.
    if val_acc >= 0.75:

        distance = abs(
            val_acc - TARGET
        )

        if distance < best_distance:
            best_distance = distance
            best_accuracy = val_acc
            best_epoch = epoch

            torch.save(
                model.state_dict(),
                MODEL_PATH,
            )

            print(
                "Saved closest checkpoint: "
                f"{val_acc*100:.2f}%"
            )

    if 0.88 <= val_acc <= 0.90:
        target_reached = True

        print()
        print(
            "TARGET 88-90% RANGE REACHED"
        )
        break


print()
print("=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)

if best_epoch == 0:
    print(
        "No usable checkpoint reached "
        "at least 75% validation accuracy."
    )
else:
    print("Selected epoch:", best_epoch)

    print(
        f"Selected validation accuracy: "
        f"{best_accuracy*100:.2f}%"
    )

    print("Model:", MODEL_PATH)

print("Target range reached:", target_reached)
print("TEST SET WAS NOT USED.")
print("=" * 60)
