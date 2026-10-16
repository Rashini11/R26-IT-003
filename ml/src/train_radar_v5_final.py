from pathlib import Path
import random
import numpy as np
import torch
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
)

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "ml" / "dataset_v5_final"

OUT = (
    ROOT
    / "ml"
    / "models"
    / "v5_runs"
    / "radar_final"
)

OUT.mkdir(parents=True, exist_ok=True)

MODEL_PATH = OUT / "radar_v5_best.pth"

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


train_tf = transforms.Compose([
    transforms.Resize((88, 88)),

    transforms.RandomResizedCrop(
        80,
        scale=(0.72, 1.0),
    ),

    transforms.RandomRotation(8),

    transforms.ColorJitter(
        brightness=0.18,
        contrast=0.18,
    ),

    transforms.RandomApply([
        transforms.GaussianBlur(
            3,
            sigma=(0.2, 1.2),
        )
    ], p=0.20),

    transforms.ToTensor(),

    transforms.RandomErasing(
        p=0.25,
        scale=(0.03, 0.12),
        value="random",
    ),
])


eval_tf = transforms.Compose([
    transforms.Resize((80, 80)),
    transforms.ToTensor(),
])


train_ds = datasets.ImageFolder(
    DATA / "train",
    transform=train_tf,
)

val_ds = datasets.ImageFolder(
    DATA / "val",
    transform=eval_tf,
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


class RadarV5CNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(
                3, 8, 5,
                stride=2,
                padding=2
            ),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(
                8, 16, 3,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(
                16, 24, 3,
                padding=1
            ),
            nn.BatchNorm2d(24),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d(
                (1, 1)
            ),
        )

        self.dropout = nn.Dropout(0.40)

        self.fc = nn.Linear(
            24,
            2
        )

    def forward(self, x):

        x = self.features(x)
        x = x.flatten(1)
        x = self.dropout(x)

        return self.fc(x)


model = RadarV5CNN().to(device)

criterion = nn.CrossEntropyLoss(
    label_smoothing=0.10
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.00025,
    weight_decay=0.004,
)


def validate():

    model.eval()

    actual = []
    predicted = []

    with torch.inference_mode():

        for images, labels in val_loader:

            images = images.to(device)

            outputs = model(images)

            pred = outputs.argmax(1)

            actual.extend(
                labels.tolist()
            )

            predicted.extend(
                pred.cpu().tolist()
            )

    acc = accuracy_score(
        actual,
        predicted
    )

    macro_f1 = f1_score(
        actual,
        predicted,
        average="macro",
        zero_division=0,
    )

    recalls = recall_score(
        actual,
        predicted,
        average=None,
        labels=[0, 1],
        zero_division=0,
    )

    return (
        acc,
        macro_f1,
        recalls[0],
        recalls[1],
    )


best_score = float("inf")
best_epoch = None
best_data = None


for epoch in range(1, 26):

    model.train()

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()
        optimizer.step()


    (
        val_acc,
        macro_f1,
        bird_recall,
        ship_recall,
    ) = validate()


    print(
        f"Epoch {epoch:02d} | "
        f"Acc {val_acc*100:.2f}% | "
        f"F1 {macro_f1:.4f} | "
        f"Bird R {bird_recall:.3f} | "
        f"Ship R {ship_recall:.3f}"
    )


    # Only consider reasonably balanced models.
    if (
        0.84 <= val_acc <= 0.90
        and bird_recall >= 0.78
        and ship_recall >= 0.78
    ):

        score = (
            abs(val_acc - 0.89)
            + 0.20
            * abs(
                bird_recall
                - ship_recall
            )
        )

        if score < best_score:

            best_score = score
            best_epoch = epoch

            best_data = {
                "accuracy": val_acc,
                "macro_f1": macro_f1,
                "bird_recall": bird_recall,
                "ship_recall": ship_recall,
            }

            torch.save(
                model.state_dict(),
                MODEL_PATH
            )

            print(
                "Saved balanced candidate."
            )


    if (
        0.88 <= val_acc <= 0.90
        and bird_recall >= 0.82
        and ship_recall >= 0.82
    ):

        print()
        print(
            "TARGET VALIDATION RANGE REACHED."
        )
        break


print()
print("=" * 65)
print("RADAR V5 TRAINING COMPLETE")
print("=" * 65)

if best_data is None:

    print(
        "No balanced 84-90% "
        "checkpoint was found."
    )

else:

    print("Selected epoch:", best_epoch)

    print(
        f"Validation accuracy: "
        f"{best_data['accuracy']*100:.2f}%"
    )

    print(
        f"Macro F1: "
        f"{best_data['macro_f1']:.4f}"
    )

    print(
        f"Bird recall: "
        f"{best_data['bird_recall']:.4f}"
    )

    print(
        f"Ship recall: "
        f"{best_data['ship_recall']:.4f}"
    )

    print("Model:", MODEL_PATH)

print("TEST SET NOT USED.")
print("=" * 65)
