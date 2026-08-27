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
    / "v5_runs"
    / "radar_target89"
    / "radar_target89_best.pth"
)

OUT_DIR = (
    ROOT
    / "ml"
    / "evaluation"
    / "radar_target89_test"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


device = (
    torch.device("mps")
    if torch.backends.mps.is_available()
    else torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("cpu")
)

print("Using device:", device)


# ============================================================
# EXACT VALIDATION/TEST PREPROCESSING
# ============================================================

test_tf = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])


test_ds = datasets.ImageFolder(
    TEST_DIR,
    transform=test_tf,
)

test_loader = DataLoader(
    test_ds,
    batch_size=32,
    shuffle=False,
    num_workers=0,
)

print("Classes:", test_ds.class_to_idx)
print("Test samples:", len(test_ds))


# ============================================================
# EXACT SAME MODEL AS TRAINING
# ============================================================

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

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True,
    )
)

model.eval()


# ============================================================
# TEST
# ============================================================

actual = []
predicted = []
confidences = []


with torch.inference_mode():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        probabilities = torch.softmax(
            outputs,
            dim=1,
        )

        confidence, prediction = (
            probabilities.max(dim=1)
        )

        actual.extend(
            labels.cpu().tolist()
        )

        predicted.extend(
            prediction.cpu().tolist()
        )

        confidences.extend(
            confidence.cpu().tolist()
        )


actual = np.asarray(actual)
predicted = np.asarray(predicted)
confidences = np.asarray(confidences)


# ============================================================
# METRICS
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
print("=" * 65)
print("RADAR TARGET-89 TEST RESULTS")
print("=" * 65)

print(
    f"Test Accuracy     : "
    f"{accuracy*100:.2f}%"
)

print(
    f"Macro Precision   : "
    f"{macro_precision:.4f}"
)

print(
    f"Macro Recall      : "
    f"{macro_recall:.4f}"
)

print(
    f"Macro F1          : "
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

print("Confusion Matrix:")
print(cm)

print()
print(
    f"Mean Confidence   : "
    f"{confidences.mean()*100:.2f}%"
)

print(
    f"Minimum Confidence: "
    f"{confidences.min()*100:.2f}%"
)

print(
    f"Maximum Confidence: "
    f"{confidences.max()*100:.2f}%"
)


results = {
    "model":
        "RadarTargetCNN",

    "selected_validation_accuracy":
        0.8909,

    "test_samples":
        len(test_ds),

    "test_accuracy":
        float(accuracy),

    "macro_precision":
        float(macro_precision),

    "macro_recall":
        float(macro_recall),

    "macro_f1":
        float(macro_f1),

    "confusion_matrix":
        cm.tolist(),

    "mean_confidence":
        float(confidences.mean()),

    "minimum_confidence":
        float(confidences.min()),

    "maximum_confidence":
        float(confidences.max()),
}


output_path = (
    OUT_DIR
    / "test_metrics.json"
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
