from pathlib import Path
import json

import numpy as np
import torch
import torch.nn as nn

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import mobilenet_v3_small


ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "ml" / "dataset_v4_raw" / "val"

MODEL_PATH = (
    ROOT
    / "ml"
    / "models"
    / "v4_runs"
    / "radar_raw_mobilenetv3"
    / "radar_v4_best.pth"
)

OUT_DIR = (
    ROOT
    / "ml"
    / "evaluation"
    / "radar_v4_validation"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


print("Using device:", device)


# ============================================================
# TRANSFORM
# Must match training validation transform exactly
# ============================================================

transform = transforms.Compose(
    [
        transforms.Resize(
            (256, 256)
        ),
        transforms.CenterCrop(
            224
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[
                0.485,
                0.456,
                0.406,
            ],
            std=[
                0.229,
                0.224,
                0.225,
            ],
        ),
    ]
)


dataset = datasets.ImageFolder(
    DATA_DIR,
    transform=transform,
)

loader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=False,
    num_workers=0,
)


print(
    "Classes:",
    dataset.class_to_idx,
)

print(
    "Validation samples:",
    len(dataset),
)


# ============================================================
# MODEL
# ============================================================

model = mobilenet_v3_small(
    weights=None
)

in_features = (
    model.classifier[3].in_features
)

model.classifier[3] = nn.Linear(
    in_features,
    2,
)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True,
    )
)

model.to(device)
model.eval()


# ============================================================
# INFERENCE
# ============================================================

actual = []
predicted = []
confidences = []
bird_probs = []
ship_probs = []


with torch.inference_mode():

    for images, labels in loader:

        images = images.to(device)

        outputs = model(images)

        probabilities = torch.softmax(
            outputs,
            dim=1,
        )

        confidence, prediction = (
            probabilities.max(
                dim=1
            )
        )

        actual.extend(
            labels.tolist()
        )

        predicted.extend(
            prediction.cpu().tolist()
        )

        confidences.extend(
            confidence.cpu().tolist()
        )

        bird_probs.extend(
            probabilities[:, 0]
            .cpu()
            .tolist()
        )

        ship_probs.extend(
            probabilities[:, 1]
            .cpu()
            .tolist()
        )


actual = np.array(actual)
predicted = np.array(predicted)
confidences = np.array(confidences)


# ============================================================
# RESULTS
# ============================================================

accuracy = float(
    (actual == predicted).mean()
)


print()
print("=" * 65)
print("RADAR V4 VALIDATION DIAGNOSTICS")
print("=" * 65)

print(
    f"Accuracy: "
    f"{accuracy * 100:.2f}%"
)

print()
print(
    classification_report(
        actual,
        predicted,
        target_names=[
            "bird",
            "ship",
        ],
        digits=4,
        zero_division=0,
    )
)

print(
    "Confusion matrix:"
)

cm = confusion_matrix(
    actual,
    predicted,
)

print(cm)


# ============================================================
# CONFIDENCE DISTRIBUTION
# ============================================================

print()
print("CONFIDENCE DISTRIBUTION")
print("-" * 65)

for percentile in [
    0,
    1,
    5,
    10,
    25,
    50,
    75,
    90,
    95,
    99,
    100,
]:
    value = np.percentile(
        confidences,
        percentile,
    )

    print(
        f"P{percentile:>3}: "
        f"{value:.4f}"
    )


# ============================================================
# UNKNOWN THRESHOLD ANALYSIS
# ============================================================

print()
print("UNKNOWN THRESHOLD ANALYSIS")
print("-" * 65)

threshold_results = {}

for threshold in [
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.95,
]:

    accepted = (
        confidences
        >= threshold
    )

    accepted_count = int(
        accepted.sum()
    )

    unknown_count = int(
        (~accepted).sum()
    )

    coverage = (
        accepted_count
        / len(confidences)
    )

    if accepted_count:
        accepted_accuracy = float(
            (
                actual[accepted]
                == predicted[accepted]
            ).mean()
        )
    else:
        accepted_accuracy = None

    threshold_results[
        str(threshold)
    ] = {
        "accepted":
            accepted_count,
        "unknown":
            unknown_count,
        "coverage":
            float(coverage),
        "accepted_accuracy":
            accepted_accuracy,
    }

    print(
        f"Threshold {threshold:.2f} | "
        f"Accepted {accepted_count:3d} | "
        f"Unknown {unknown_count:3d} | "
        f"Coverage {coverage*100:6.2f}% | "
        f"Accepted accuracy "
        f"{accepted_accuracy*100 if accepted_accuracy is not None else 0:.2f}%"
    )


results = {
    "samples":
        len(dataset),
    "accuracy":
        accuracy,
    "confusion_matrix":
        cm.tolist(),
    "confidence": {
        "minimum":
            float(
                confidences.min()
            ),
        "mean":
            float(
                confidences.mean()
            ),
        "median":
            float(
                np.median(
                    confidences
                )
            ),
        "maximum":
            float(
                confidences.max()
            ),
    },
    "threshold_analysis":
        threshold_results,
}


output_path = (
    OUT_DIR
    / "validation_diagnostics.json"
)

output_path.write_text(
    json.dumps(
        results,
        indent=2,
    )
)


print()
print(
    "Saved:",
    output_path,
)

print("=" * 65)
print(
    "TEST SET WAS NOT USED."
)
print("=" * 65)
