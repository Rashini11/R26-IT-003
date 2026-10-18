from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[2]
TEST_DIR = ROOT / "ml" / "dataset_v3_grouped" / "test"

MODEL_PATH = (
    ROOT / "ml" / "models" / "v3_runs"
    / "radar_final" / "radar_final_best.pth"
)

OUTPUT_DIR = ROOT / "ml" / "evaluation" / "radar_final"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["bird", "ship", "unknown"]

device = (
    torch.device("mps")
    if torch.backends.mps.is_available()
    else torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("cpu")
)

class RadarFinalCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 4, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(4, 8, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.dropout = nn.Dropout(0.60)
        self.fc = nn.Linear(8, 3)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        x = self.dropout(x)
        return self.fc(x)

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

dataset = datasets.ImageFolder(
    TEST_DIR,
    transform=transform,
)

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0,
)

model = RadarFinalCNN().to(device)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True,
    )
)

model.eval()

actual = []
predicted = []

with torch.inference_mode():
    for images, labels in loader:
        images = images.to(device)

        outputs = model(images)
        preds = outputs.argmax(dim=1)

        actual.extend(labels.tolist())
        predicted.extend(preds.cpu().tolist())

accuracy = accuracy_score(actual, predicted)

report = classification_report(
    actual,
    predicted,
    target_names=CLASS_NAMES,
    output_dict=True,
    zero_division=0,
)

matrix = confusion_matrix(
    actual,
    predicted,
)

summary = {
    "test_images": len(dataset),
    "test_accuracy_percent": round(accuracy * 100, 2),
    "test_macro_precision": round(report["macro avg"]["precision"], 4),
    "test_macro_recall": round(report["macro avg"]["recall"], 4),
    "test_macro_f1": round(report["macro avg"]["f1-score"], 4),
    "per_class": {
        name: {
            "precision": round(report[name]["precision"], 4),
            "recall": round(report[name]["recall"], 4),
            "f1": round(report[name]["f1-score"], 4),
            "support": int(report[name]["support"]),
        }
        for name in CLASS_NAMES
    },
    "confusion_matrix": matrix.tolist(),
}

with (OUTPUT_DIR / "test_metrics.json").open("w") as f:
    json.dump(summary, f, indent=2)

fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(matrix)

ax.set_title("Final Radar Test Confusion Matrix")
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")

ax.set_xticks(range(3))
ax.set_yticks(range(3))
ax.set_xticklabels(CLASS_NAMES)
ax.set_yticklabels(CLASS_NAMES)

for r in range(3):
    for c in range(3):
        ax.text(c, r, str(matrix[r, c]), ha="center", va="center")

fig.colorbar(im, ax=ax)
fig.tight_layout()
fig.savefig(
    OUTPUT_DIR / "confusion_matrix.png",
    dpi=250,
)
plt.close(fig)

print()
print("=" * 60)
print("FINAL RADAR TEST RESULTS")
print("=" * 60)

print("Test images     :", len(dataset))
print(f"Test accuracy   : {summary['test_accuracy_percent']:.2f}%")
print(f"Macro precision : {summary['test_macro_precision']:.4f}")
print(f"Macro recall    : {summary['test_macro_recall']:.4f}")
print(f"Macro F1        : {summary['test_macro_f1']:.4f}")

print("\nPer class:")
for name, values in summary["per_class"].items():
    print(
        f"{name:8} | "
        f"P={values['precision']:.4f} | "
        f"R={values['recall']:.4f} | "
        f"F1={values['f1']:.4f}"
    )

print("\nConfusion matrix:")
print(matrix)

print("\nSaved:", OUTPUT_DIR)
print("=" * 60)
