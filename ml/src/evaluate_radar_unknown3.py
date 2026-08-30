from pathlib import Path
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

ROOT = Path(__file__).resolve().parents[2]

TEST_DATA = (
    ROOT
    / "ml"
    / "dataset_v6_unknown3"
    / "test"
)

MODEL_PATH = (
    ROOT
    / "ml"
    / "models"
    / "v6_runs"
    / "radar_unknown3"
    / "radar_unknown3_best.pth"
)

METRICS_PATH = (
    MODEL_PATH.parent
    / "radar_unknown3_test_metrics.json"
)

EXPECTED_CLASSES = [
    "bird",
    "ship",
    "unknown",
]

device = (
    torch.device("mps")
    if torch.backends.mps.is_available()
    else torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("cpu")
)

test_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

test_dataset = datasets.ImageFolder(
    TEST_DATA,
    transform=test_transform,
)

if test_dataset.classes != EXPECTED_CLASSES:
    raise RuntimeError(
        f"Unexpected class order: {test_dataset.classes}"
    )

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0,
)


class RadarTargetCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(
                3,
                8,
                kernel_size=5,
                stride=2,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(
                8,
                12,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )

        self.dropout = nn.Dropout(0.30)
        self.fc = nn.Linear(
            12 * 4 * 4,
            3,
        )

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        x = self.dropout(x)
        return self.fc(x)


if not MODEL_PATH.exists():
    raise FileNotFoundError(MODEL_PATH)

model = RadarTargetCNN()

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=True,
    )
)

model.to(device)
model.eval()

actual_labels = []
predicted_labels = []
prediction_confidences = []

with torch.inference_mode():
    for images, labels in test_loader:
        images = images.to(device)

        output = model(images)
        probabilities = torch.softmax(
            output,
            dim=1,
        )

        confidence, prediction = probabilities.max(
            dim=1
        )

        actual_labels.extend(labels.tolist())
        predicted_labels.extend(
            prediction.cpu().tolist()
        )
        prediction_confidences.extend(
            confidence.cpu().tolist()
        )

accuracy = accuracy_score(
    actual_labels,
    predicted_labels,
)

report = classification_report(
    actual_labels,
    predicted_labels,
    labels=[0, 1, 2],
    target_names=EXPECTED_CLASSES,
    output_dict=True,
    zero_division=0,
)

matrix = confusion_matrix(
    actual_labels,
    predicted_labels,
    labels=[0, 1, 2],
)

metrics = {
    "model": str(MODEL_PATH),
    "classes": EXPECTED_CLASSES,
    "test_samples": len(test_dataset),
    "test_accuracy": round(accuracy, 6),
    "macro_precision": round(
        report["macro avg"]["precision"],
        6,
    ),
    "macro_recall": round(
        report["macro avg"]["recall"],
        6,
    ),
    "macro_f1": round(
        report["macro avg"]["f1-score"],
        6,
    ),
    "average_confidence": round(
        sum(prediction_confidences)
        / len(prediction_confidences),
        6,
    ),
    "per_class": {
        name: {
            key: round(value, 6)
            if isinstance(value, float)
            else value
            for key, value in report[name].items()
        }
        for name in EXPECTED_CLASSES
    },
    "confusion_matrix": matrix.tolist(),
}

METRICS_PATH.write_text(
    json.dumps(
        metrics,
        indent=2,
    )
)

print("Using device:", device)
print("Model:", MODEL_PATH)
print("Test samples:", len(test_dataset))
print("Classes:", test_dataset.class_to_idx)

print()
print("CLASSIFICATION REPORT")
print(
    classification_report(
        actual_labels,
        predicted_labels,
        labels=[0, 1, 2],
        target_names=EXPECTED_CLASSES,
        zero_division=0,
        digits=4,
    )
)

print("CONFUSION MATRIX")
print("Rows = actual; columns = predicted")
print("Class order:", EXPECTED_CLASSES)
print(matrix)

print()
print(f"Test accuracy: {accuracy * 100:.2f}%")
print(
    "Macro precision: "
    f"{report['macro avg']['precision']:.4f}"
)
print(
    "Macro recall: "
    f"{report['macro avg']['recall']:.4f}"
)
print(
    "Macro F1: "
    f"{report['macro avg']['f1-score']:.4f}"
)
print(
    "Unknown recall: "
    f"{report['unknown']['recall'] * 100:.2f}%"
)
print("Metrics saved to:", METRICS_PATH)
