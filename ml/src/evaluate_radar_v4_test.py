from pathlib import Path
import json

import numpy as np
import torch
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import mobilenet_v3_small


ROOT = Path(__file__).resolve().parents[2]

TEST_DIR = (
    ROOT
    / "ml"
    / "dataset_v4_raw"
    / "test"
)

MODEL_PATH = (
    ROOT
    / "ml"
    / "models"
    / "v4_runs"
    / "radar_raw_mobilenetv3"
    / "radar_v4_best.pth"
)

CONFIG_PATH = (
    ROOT
    / "ml"
    / "models"
    / "v4_runs"
    / "radar_raw_mobilenetv3"
    / "deployment_config.json"
)

OUTPUT_DIR = (
    ROOT
    / "ml"
    / "evaluation"
    / "radar_v4_test"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# LOAD FROZEN DEPLOYMENT CONFIG
# ============================================================

config = json.loads(
    CONFIG_PATH.read_text()
)

UNKNOWN_THRESHOLD = float(
    config["unknown_threshold"]
)


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
print(
    "Frozen unknown threshold:",
    UNKNOWN_THRESHOLD,
)


# ============================================================
# TRANSFORM
# Must exactly match validation preprocessing.
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
    TEST_DIR,
    transform=transform,
)

loader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=False,
    num_workers=0,
)


print()
print("=" * 65)
print("RADAR V4 ONE-TIME HELD-OUT TEST")
print("=" * 65)

print(
    "Classes:",
    dataset.class_to_idx,
)

print(
    "Test samples:",
    len(dataset),
)


if dataset.class_to_idx != {
    "bird": 0,
    "ship": 1,
}:
    raise RuntimeError(
        "Unexpected test class mapping: "
        f"{dataset.class_to_idx}"
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
filenames = []


with torch.inference_mode():

    sample_index = 0

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

        batch_size = labels.size(0)

        actual.extend(
            labels.tolist()
        )

        predicted.extend(
            prediction.cpu().tolist()
        )

        confidences.extend(
            confidence.cpu().tolist()
        )

        for i in range(batch_size):
            path, _ = dataset.samples[
                sample_index + i
            ]

            filenames.append(
                Path(path).name
            )

        sample_index += batch_size


actual = np.asarray(actual)
predicted = np.asarray(predicted)
confidences = np.asarray(confidences)


# ============================================================
# STANDARD 2-CLASS TEST METRICS
# ============================================================

accuracy = accuracy_score(
    actual,
    predicted,
)

(
    macro_precision,
    macro_recall,
    macro_f1,
    _,
) = precision_recall_fscore_support(
    actual,
    predicted,
    average="macro",
    zero_division=0,
)

cm = confusion_matrix(
    actual,
    predicted,
)


print()
print("STANDARD BIRD / SHIP CLASSIFICATION")
print("-" * 65)

print(
    f"Accuracy       : "
    f"{accuracy * 100:.2f}%"
)

print(
    f"Macro precision: "
    f"{macro_precision:.4f}"
)

print(
    f"Macro recall   : "
    f"{macro_recall:.4f}"
)

print(
    f"Macro F1       : "
    f"{macro_f1:.4f}"
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

print(cm)


# ============================================================
# FROZEN CONFIDENCE-ABSTENTION ANALYSIS
# ============================================================

accepted_mask = (
    confidences
    >= UNKNOWN_THRESHOLD
)

unknown_mask = (
    ~accepted_mask
)

accepted_count = int(
    accepted_mask.sum()
)

unknown_count = int(
    unknown_mask.sum()
)

coverage = (
    accepted_count
    / len(dataset)
)

if accepted_count > 0:

    accepted_accuracy = float(
        (
            actual[accepted_mask]
            == predicted[accepted_mask]
        ).mean()
    )

else:
    accepted_accuracy = None


print()
print("FROZEN UNKNOWN / ABSTENTION POLICY")
print("-" * 65)

print(
    f"Threshold         : "
    f"{UNKNOWN_THRESHOLD:.2f}"
)

print(
    f"Accepted          : "
    f"{accepted_count}"
)

print(
    f"Marked unknown    : "
    f"{unknown_count}"
)

print(
    f"Coverage          : "
    f"{coverage * 100:.2f}%"
)

if accepted_accuracy is not None:

    print(
        f"Accepted accuracy : "
        f"{accepted_accuracy * 100:.2f}%"
    )


# ============================================================
# CONFIDENCE DISTRIBUTION
# ============================================================

print()
print("TEST CONFIDENCE DISTRIBUTION")
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
# SAVE PER-SAMPLE RESULTS
# ============================================================

index_to_class = {
    0: "bird",
    1: "ship",
}

samples = []

for filename, truth, pred, confidence in zip(
    filenames,
    actual,
    predicted,
    confidences,
):

    accepted = bool(
        confidence
        >= UNKNOWN_THRESHOLD
    )

    final_prediction = (
        index_to_class[
            int(pred)
        ]
        if accepted
        else "unknown"
    )

    samples.append(
        {
            "filename":
                filename,
            "ground_truth":
                index_to_class[
                    int(truth)
                ],
            "binary_prediction":
                index_to_class[
                    int(pred)
                ],
            "confidence":
                float(confidence),
            "accepted":
                accepted,
            "final_prediction":
                final_prediction,
        }
    )


results = {
    "dataset":
        "radar_v4_raw_test",
    "samples":
        len(dataset),
    "heatmap_preprocessing":
        False,
    "classes": [
        "bird",
        "ship",
    ],
    "standard_binary_metrics": {
        "accuracy":
            float(accuracy),
        "macro_precision":
            float(
                macro_precision
            ),
        "macro_recall":
            float(
                macro_recall
            ),
        "macro_f1":
            float(
                macro_f1
            ),
        "confusion_matrix":
            cm.tolist(),
    },
    "confidence_abstention": {
        "threshold":
            UNKNOWN_THRESHOLD,
        "accepted":
            accepted_count,
        "unknown":
            unknown_count,
        "coverage":
            float(coverage),
        "accepted_accuracy":
            accepted_accuracy,
    },
    "samples_detail":
        samples,
}


output_path = (
    OUTPUT_DIR
    / "test_metrics.json"
)

output_path.write_text(
    json.dumps(
        results,
        indent=2,
    )
)


print()
print("=" * 65)
print("FINAL TEST EVALUATION COMPLETE")
print("=" * 65)

print(
    "Saved:",
    output_path,
)

print()
print(
    "DO NOT RETUNE THE MODEL OR "
    "UNKNOWN THRESHOLD USING THIS TEST SET."
)

print("=" * 65)
