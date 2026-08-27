from pathlib import Path
import csv
import json
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "ml" / "dataset_v3_grouped"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "ml"
    / "models"
    / "v3_runs"
    / "deepercnn"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

MODEL_PATH = (
    OUTPUT_DIR
    / "deepercnn_v3_grouped_best.pth"
)

HISTORY_PATH = (
    OUTPUT_DIR
    / "training_history.csv"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "validation_summary.json"
)

SEED = 42
IMAGE_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 20

LEARNING_RATE = 0.0005
WEIGHT_DECAY = 0.0001
DROPOUT = 0.40

EARLY_STOPPING_PATIENCE = 4


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# DEVICE
# ============================================================

if torch.backends.mps.is_available():
    device = torch.device("mps")

elif torch.cuda.is_available():
    device = torch.device("cuda")

else:
    device = torch.device("cpu")


print("Using device:", device)


# ============================================================
# TRANSFORMS
#
# Mild augmentation only.
# No vertical/horizontal flipping because bird heatmaps
# contain structured feature/time axes.
# ============================================================

train_transform = transforms.Compose([
    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.RandomApply(
        [
            transforms.GaussianBlur(
                kernel_size=3,
                sigma=(0.1, 1.0),
            )
        ],
        p=0.25,
    ),

    transforms.ToTensor(),

    transforms.RandomErasing(
        p=0.20,
        scale=(0.02, 0.08),
        ratio=(0.5, 2.0),
        value="random",
    ),
])


eval_transform = transforms.Compose([
    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),
    transforms.ToTensor(),
])


# ============================================================
# DATASETS
#
# IMPORTANT:
# TEST DATA IS NOT LOADED HERE.
# ============================================================

train_dataset = datasets.ImageFolder(
    DATA_DIR / "train",
    transform=train_transform,
)

val_dataset = datasets.ImageFolder(
    DATA_DIR / "val",
    transform=eval_transform,
)


if train_dataset.classes != [
    "bird",
    "ship",
    "unknown",
]:
    raise RuntimeError(
        f"Unexpected classes: "
        f"{train_dataset.classes}"
    )


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)


print("Classes:", train_dataset.classes)
print("Train images:", len(train_dataset))
print("Validation images:", len(val_dataset))
print("TEST SET: NOT LOADED")


# ============================================================
# MODEL
#
# Same trainable layers as deployed RadarDeeperCNN.
# Dropout has no parameters and is disabled during eval,
# therefore the saved state_dict remains backend-compatible.
# ============================================================

class DeeperCNN(nn.Module):

    def __init__(
        self,
        num_classes=3,
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(
            3,
            16,
            3,
            padding=1,
        )

        self.conv2 = nn.Conv2d(
            16,
            32,
            3,
            padding=1,
        )

        self.conv3 = nn.Conv2d(
            32,
            64,
            3,
            padding=1,
        )

        self.pool = nn.MaxPool2d(
            2,
            2,
        )

        self.fc1 = nn.Linear(
            64 * 16 * 16,
            256,
        )

        self.dropout = nn.Dropout(
            DROPOUT
        )

        self.fc2 = nn.Linear(
            256,
            num_classes,
        )


    def forward(self, x):

        x = self.pool(
            F.relu(
                self.conv1(x)
            )
        )

        x = self.pool(
            F.relu(
                self.conv2(x)
            )
        )

        x = self.pool(
            F.relu(
                self.conv3(x)
            )
        )

        x = x.view(
            x.size(0),
            -1,
        )

        x = F.relu(
            self.fc1(x)
        )

        x = self.dropout(x)

        return self.fc2(x)


model = DeeperCNN(
    num_classes=3
).to(device)


# ============================================================
# TRAINING SETUP
# ============================================================

criterion = nn.CrossEntropyLoss(
    label_smoothing=0.05
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)


# ============================================================
# EVALUATION
# ============================================================

def evaluate():

    model.eval()

    total_loss = 0.0
    total_samples = 0

    actual = []
    predicted = []

    with torch.inference_mode():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels,
            )

            batch_size = labels.size(0)

            total_loss += (
                loss.item()
                * batch_size
            )

            total_samples += batch_size

            predictions = outputs.argmax(
                dim=1
            )

            actual.extend(
                labels.detach()
                .cpu()
                .numpy()
                .tolist()
            )

            predicted.extend(
                predictions.detach()
                .cpu()
                .numpy()
                .tolist()
            )

    actual = np.asarray(actual)
    predicted = np.asarray(predicted)

    accuracy = (
        actual == predicted
    ).mean()

    macro_f1 = f1_score(
        actual,
        predicted,
        average="macro",
        zero_division=0,
    )

    return {
        "loss": (
            total_loss
            / total_samples
        ),
        "accuracy": float(
            accuracy
        ),
        "macro_f1": float(
            macro_f1
        ),
    }


# ============================================================
# TRAIN
# ============================================================

best_val_loss = float("inf")
best_epoch = 0
best_metrics = None

epochs_without_improvement = 0

history = []


for epoch in range(
    1,
    EPOCHS + 1,
):

    model.train()

    running_loss = 0.0
    samples_seen = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels,
        )

        loss.backward()

        optimizer.step()

        batch_size = labels.size(0)

        running_loss += (
            loss.item()
            * batch_size
        )

        samples_seen += batch_size


    train_loss = (
        running_loss
        / samples_seen
    )

    val_metrics = evaluate()


    print()
    print(
        f"Epoch {epoch}/{EPOCHS}"
    )

    print(
        f"Train loss : "
        f"{train_loss:.4f}"
    )

    print(
        f"Val loss   : "
        f"{val_metrics['loss']:.4f}"
    )

    print(
        f"Val acc    : "
        f"{val_metrics['accuracy'] * 100:.2f}%"
    )

    print(
        f"Val macro F1: "
        f"{val_metrics['macro_f1']:.4f}"
    )


    history.append({
        "epoch": epoch,
        "train_loss": train_loss,
        "validation_loss": (
            val_metrics["loss"]
        ),
        "validation_accuracy": (
            val_metrics["accuracy"]
        ),
        "validation_macro_f1": (
            val_metrics["macro_f1"]
        ),
    })


    if (
        val_metrics["loss"]
        < best_val_loss
    ):

        best_val_loss = (
            val_metrics["loss"]
        )

        best_epoch = epoch
        best_metrics = dict(
            val_metrics
        )

        torch.save(
            model.state_dict(),
            MODEL_PATH,
        )

        epochs_without_improvement = 0

        print(
            "Saved new best model."
        )

    else:

        epochs_without_improvement += 1

        print(
            "No validation improvement: "
            f"{epochs_without_improvement}/"
            f"{EARLY_STOPPING_PATIENCE}"
        )


    if (
        epochs_without_improvement
        >= EARLY_STOPPING_PATIENCE
    ):
        print(
            "\nEarly stopping."
        )
        break


# ============================================================
# SAVE HISTORY
# ============================================================

with HISTORY_PATH.open(
    "w",
    newline="",
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=[
            "epoch",
            "train_loss",
            "validation_loss",
            "validation_accuracy",
            "validation_macro_f1",
        ],
    )

    writer.writeheader()
    writer.writerows(history)


summary = {
    "best_epoch": best_epoch,

    "best_validation_loss": (
        round(
            best_metrics["loss"],
            6,
        )
    ),

    "best_validation_accuracy_percent": (
        round(
            best_metrics["accuracy"]
            * 100,
            2,
        )
    ),

    "best_validation_macro_f1": (
        round(
            best_metrics["macro_f1"],
            4,
        )
    ),

    "model_path": str(
        MODEL_PATH
    ),

    "test_set_evaluated": False,
}


with SUMMARY_PATH.open(
    "w"
) as file:

    json.dump(
        summary,
        file,
        indent=2,
    )


print()
print("=" * 60)
print("DEEPERCNN V3 TRAINING COMPLETE")
print("=" * 60)

print(
    "Best epoch:",
    best_epoch,
)

print(
    "Best validation accuracy:",
    f"{summary['best_validation_accuracy_percent']:.2f}%"
)

print(
    "Best validation macro F1:",
    f"{summary['best_validation_macro_f1']:.4f}"
)

print(
    "Best validation loss:",
    summary["best_validation_loss"],
)

print(
    "Model:",
    MODEL_PATH,
)

print(
    "TEST SET HAS NOT BEEN EVALUATED."
)

print("=" * 60)
