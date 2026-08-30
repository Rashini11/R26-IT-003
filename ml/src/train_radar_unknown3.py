from pathlib import Path
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "ml" / "dataset_v6_unknown3"
OUT = ROOT / "ml" / "models" / "v6_runs" / "radar_unknown3"
OUT.mkdir(parents=True, exist_ok=True)

MODEL_PATH = OUT / "radar_unknown3_best.pth"

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

EXPECTED_CLASSES = ["bird", "ship", "unknown"]

if train_ds.classes != EXPECTED_CLASSES:
    raise RuntimeError(
        f"Unexpected training classes: {train_ds.classes}"
    )

if val_ds.classes != EXPECTED_CLASSES:
    raise RuntimeError(
        f"Unexpected validation classes: {val_ds.classes}"
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

            nn.AdaptiveAvgPool2d((4, 4)),
        )

        self.dropout = nn.Dropout(0.30)
        self.fc = nn.Linear(12 * 4 * 4, 3)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        x = self.dropout(x)
        return self.fc(x)


model = RadarTargetCNN().to(device)

pretrained_path = (
    ROOT
    / "ml"
    / "models"
    / "v5_runs"
    / "radar_target89"
    / "radar_target89_best.pth"
)

if not pretrained_path.exists():
    raise FileNotFoundError(
        f"Pretrained Radar checkpoint not found: {pretrained_path}"
    )

pretrained_state = torch.load(
    pretrained_path,
    map_location="cpu",
    weights_only=True,
)

feature_state = {
    name: tensor
    for name, tensor in pretrained_state.items()
    if name.startswith("features.")
}

load_result = model.load_state_dict(
    feature_state,
    strict=False,
)

print("Initialized convolution layers from:", pretrained_path)
print("New three-class output layer will be trained from scratch")

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


best_accuracy = 0.0
best_epoch = 0

for epoch in range(1, 31):
    model.train()
    running_loss = 0.0
    sample_count = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        sample_count += labels.size(0)

    val_acc = validate()
    train_loss = running_loss / sample_count

    print(
        f"Epoch {epoch:02d} | "
        f"Training loss {train_loss:.4f} | "
        f"Validation accuracy {val_acc * 100:.2f}%"
    )

    if val_acc > best_accuracy:
        best_accuracy = val_acc
        best_epoch = epoch

        torch.save(
            model.state_dict(),
            MODEL_PATH,
        )

        print(
            "Saved new best checkpoint: "
            f"{best_accuracy * 100:.2f}%"
        )

print()
print("=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)
print("Selected epoch:", best_epoch)
print(
    "Best validation accuracy: "
    f"{best_accuracy * 100:.2f}%"
)
print("Model:", MODEL_PATH)
print("TEST SET WAS NOT USED.")
print("=" * 60)
