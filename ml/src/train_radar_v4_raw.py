from pathlib import Path
import json
import random
import time

import numpy as np
import torch
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import (
    mobilenet_v3_small,
    MobileNet_V3_Small_Weights,
)


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "ml" / "dataset_v4_raw"

OUTPUT_DIR = (
    ROOT
    / "ml"
    / "models"
    / "v4_runs"
    / "radar_raw_mobilenetv3"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

MODEL_PATH = (
    OUTPUT_DIR
    / "radar_v4_best.pth"
)

METADATA_PATH = (
    OUTPUT_DIR
    / "training_metadata.json"
)

SEED = 42

BATCH_SIZE = 16

FROZEN_EPOCHS = 5
FINETUNE_EPOCHS = 10

IMAGE_SIZE = 224

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


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
# ============================================================

IMAGENET_MEAN = [
    0.485,
    0.456,
    0.406,
]

IMAGENET_STD = [
    0.229,
    0.224,
    0.225,
]


train_transform = transforms.Compose(
    [
        transforms.Resize(
            (256, 256)
        ),

        transforms.RandomResizedCrop(
            IMAGE_SIZE,
            scale=(0.82, 1.0),
            ratio=(0.95, 1.05),
        ),

        # Mild rotation only.
        # No flips because radar orientation
        # may contain meaningful information.
        transforms.RandomRotation(
            degrees=5
        ),

        transforms.ColorJitter(
            brightness=0.12,
            contrast=0.12,
        ),

        transforms.RandomApply(
            [
                transforms.GaussianBlur(
                    kernel_size=3,
                    sigma=(0.1, 1.0),
                )
            ],
            p=0.15,
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        ),
    ]
)


eval_transform = transforms.Compose(
    [
        transforms.Resize(
            (256, 256)
        ),

        transforms.CenterCrop(
            IMAGE_SIZE
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        ),
    ]
)


# ============================================================
# DATASETS
# ============================================================

train_dataset = datasets.ImageFolder(
    DATA_DIR / "train",
    transform=train_transform,
)

val_dataset = datasets.ImageFolder(
    DATA_DIR / "val",
    transform=eval_transform,
)


print()
print("=" * 60)
print("RADAR V4 RAW TRAINING DATA")
print("=" * 60)

print(
    "Classes:",
    train_dataset.class_to_idx,
)

print(
    "Train:",
    len(train_dataset),
)

print(
    "Validation:",
    len(val_dataset),
)

print(
    "TEST SET: NOT LOADED"
)

print("=" * 60)


if train_dataset.class_to_idx != {
    "bird": 0,
    "ship": 1,
}:
    raise RuntimeError(
        "Unexpected class mapping: "
        f"{train_dataset.class_to_idx}"
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


# ============================================================
# MODEL
# ============================================================

weights = (
    MobileNet_V3_Small_Weights.DEFAULT
)

model = mobilenet_v3_small(
    weights=weights
)


# Freeze feature extractor initially.
for parameter in model.features.parameters():
    parameter.requires_grad = False


in_features = (
    model.classifier[3].in_features
)

model.classifier[3] = nn.Linear(
    in_features,
    2,
)

model = model.to(device)


# ============================================================
# LOSS
# ============================================================

criterion = nn.CrossEntropyLoss(
    label_smoothing=0.05
)


# ============================================================
# VALIDATION
# ============================================================

def validate():
    model.eval()

    all_actual = []
    all_predicted = []

    total_loss = 0.0
    total_samples = 0

    with torch.inference_mode():

        for images, labels in val_loader:

            images = images.to(
                device
            )

            labels = labels.to(
                device
            )

            outputs = model(
                images
            )

            loss = criterion(
                outputs,
                labels,
            )

            total_loss += (
                loss.item()
                * labels.size(0)
            )

            total_samples += (
                labels.size(0)
            )

            predicted = (
                outputs.argmax(
                    dim=1
                )
            )

            all_actual.extend(
                labels.cpu().tolist()
            )

            all_predicted.extend(
                predicted.cpu().tolist()
            )

    accuracy = accuracy_score(
        all_actual,
        all_predicted,
    )

    (
        precision,
        recall,
        f1,
        _,
    ) = precision_recall_fscore_support(
        all_actual,
        all_predicted,
        average="macro",
        zero_division=0,
    )

    average_loss = (
        total_loss
        / total_samples
    )

    return {
        "loss": float(
            average_loss
        ),
        "accuracy": float(
            accuracy
        ),
        "precision": float(
            precision
        ),
        "recall": float(
            recall
        ),
        "f1": float(
            f1
        ),
    }


# ============================================================
# TRAINING HELPER
# ============================================================

best_val_loss = float("inf")

best_epoch = None
best_metrics = None

history = []


def train_phase(
    phase_name,
    start_epoch,
    epoch_count,
    optimizer,
):
    global best_val_loss
    global best_epoch
    global best_metrics

    print()
    print("=" * 60)
    print(
        f"PHASE: {phase_name}"
    )
    print("=" * 60)

    for local_epoch in range(
        1,
        epoch_count + 1,
    ):

        epoch_number = (
            start_epoch
            + local_epoch
            - 1
        )

        model.train()

        running_loss = 0.0
        seen = 0

        started = time.time()

        for images, labels in train_loader:

            images = images.to(
                device
            )

            labels = labels.to(
                device
            )

            optimizer.zero_grad()

            outputs = model(
                images
            )

            loss = criterion(
                outputs,
                labels,
            )

            loss.backward()

            optimizer.step()

            running_loss += (
                loss.item()
                * labels.size(0)
            )

            seen += labels.size(0)

        train_loss = (
            running_loss
            / seen
        )

        metrics = validate()

        elapsed = (
            time.time()
            - started
        )

        history.append(
            {
                "epoch":
                    epoch_number,
                "phase":
                    phase_name,
                "train_loss":
                    float(train_loss),
                "val_loss":
                    metrics["loss"],
                "val_accuracy":
                    metrics["accuracy"],
                "val_precision":
                    metrics["precision"],
                "val_recall":
                    metrics["recall"],
                "val_f1":
                    metrics["f1"],
            }
        )

        print(
            f"Epoch {epoch_number:02d} | "
            f"Train loss "
            f"{train_loss:.4f} | "
            f"Val loss "
            f"{metrics['loss']:.4f} | "
            f"Acc "
            f"{metrics['accuracy']*100:.2f}% | "
            f"F1 "
            f"{metrics['f1']:.4f} | "
            f"{elapsed:.1f}s"
        )

        # Model selection uses
        # validation loss only.
        if (
            metrics["loss"]
            < best_val_loss
        ):

            best_val_loss = (
                metrics["loss"]
            )

            best_epoch = (
                epoch_number
            )

            best_metrics = dict(
                metrics
            )

            torch.save(
                model.state_dict(),
                MODEL_PATH,
            )

            print(
                "Saved new best model"
            )


# ============================================================
# PHASE 1
# Frozen ImageNet features
# ============================================================

optimizer = torch.optim.AdamW(
    filter(
        lambda parameter:
        parameter.requires_grad,
        model.parameters(),
    ),
    lr=1e-3,
    weight_decay=1e-4,
)


train_phase(
    phase_name="classifier_only",
    start_epoch=1,
    epoch_count=FROZEN_EPOCHS,
    optimizer=optimizer,
)


# ============================================================
# PHASE 2
# Fine-tune last MobileNet feature blocks
# ============================================================

print()
print(
    "Unfreezing final MobileNet "
    "feature blocks..."
)


# Keep most of backbone frozen.
for parameter in model.features.parameters():
    parameter.requires_grad = False


# Fine-tune the last few feature blocks.
for block in model.features[-3:]:
    for parameter in block.parameters():
        parameter.requires_grad = True


optimizer = torch.optim.AdamW(
    filter(
        lambda parameter:
        parameter.requires_grad,
        model.parameters(),
    ),
    lr=1e-4,
    weight_decay=1e-4,
)


train_phase(
    phase_name="fine_tuning",
    start_epoch=(
        FROZEN_EPOCHS + 1
    ),
    epoch_count=FINETUNE_EPOCHS,
    optimizer=optimizer,
)


# ============================================================
# SAVE METADATA
# ============================================================

metadata = {
    "model":
        "MobileNetV3-Small",
    "dataset":
        "radar_v4_raw",
    "input_size":
        IMAGE_SIZE,
    "classes":
        train_dataset.class_to_idx,
    "heatmap_preprocessing":
        False,
    "pretrained_weights":
        "ImageNet",
    "frozen_epochs":
        FROZEN_EPOCHS,
    "finetune_epochs":
        FINETUNE_EPOCHS,
    "best_epoch":
        best_epoch,
    "best_validation_metrics":
        best_metrics,
    "test_evaluated":
        False,
    "history":
        history,
}

METADATA_PATH.write_text(
    json.dumps(
        metadata,
        indent=2,
    )
)


print()
print("=" * 60)
print("RADAR V4 TRAINING COMPLETE")
print("=" * 60)

print(
    "Best epoch:",
    best_epoch,
)

print(
    "Best validation loss:",
    f"{best_val_loss:.4f}",
)

print(
    "Best validation accuracy:",
    f"{best_metrics['accuracy']*100:.2f}%"
)

print(
    "Best validation macro F1:",
    f"{best_metrics['f1']:.4f}"
)

print()
print(
    "Model:",
    MODEL_PATH,
)

print(
    "Metadata:",
    METADATA_PATH,
)

print()
print(
    "TEST SET HAS NOT BEEN EVALUATED."
)

print("=" * 60)
