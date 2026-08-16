from pathlib import Path
import csv
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from torchvision import datasets, transforms
from ultralytics import YOLO


# ============================================================
# PATHS AND CONFIGURATION
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]

TEST_DIR = PROJECT_ROOT / "ml" / "dataset_v2_balanced" / "test"

CNN_MODEL_PATH = (
    PROJECT_ROOT
    / "ml"
    / "models"
    / "final"
    / "deepercnn_best.pth"
)

YOLO_MODEL_PATH = (
    PROJECT_ROOT
    / "ml"
    / "models"
    / "final"
    / "yolo11_medium_best.pt"
)

OUTPUT_DIR = PROJECT_ROOT / "ml" / "evaluation" / "final_system"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["bird", "ship", "unknown"]

IMAGE_SIZE = 128
BATCH_SIZE = 32


# ============================================================
# DEVICE CONFIGURATION
# ============================================================
if torch.backends.mps.is_available():
    TORCH_DEVICE = torch.device("mps")
    YOLO_DEVICE = "mps"
elif torch.cuda.is_available():
    TORCH_DEVICE = torch.device("cuda")
    YOLO_DEVICE = 0
else:
    TORCH_DEVICE = torch.device("cpu")
    YOLO_DEVICE = "cpu"


def synchronize_device():
    if TORCH_DEVICE.type == "cuda":
        torch.cuda.synchronize()
    elif TORCH_DEVICE.type == "mps":
        torch.mps.synchronize()


print("PyTorch device:", TORCH_DEVICE)
print("YOLO device:", YOLO_DEVICE)


# ============================================================
# CNN ARCHITECTURE
# Must match training and backend architecture exactly.
# ============================================================
class DeeperCNN(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(64 * 16 * 16, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))

        return self.fc2(x)


cnn_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])


# ============================================================
# LOAD TEST DATA
# ============================================================
test_dataset = datasets.ImageFolder(TEST_DIR)

if test_dataset.classes != CLASS_NAMES:
    raise ValueError(
        f"Unexpected class order: {test_dataset.classes}. "
        f"Expected: {CLASS_NAMES}"
    )

samples = test_dataset.samples

print("Classes:", test_dataset.classes)
print("Test images:", len(samples))


# ============================================================
# LOAD MODELS
# ============================================================
cnn_model = DeeperCNN(num_classes=len(CLASS_NAMES))

cnn_checkpoint = torch.load(
    CNN_MODEL_PATH,
    map_location=TORCH_DEVICE,
)

if (
    isinstance(cnn_checkpoint, dict)
    and "model_state_dict" in cnn_checkpoint
):
    cnn_checkpoint = cnn_checkpoint["model_state_dict"]

cnn_model.load_state_dict(cnn_checkpoint)
cnn_model.to(TORCH_DEVICE)
cnn_model.eval()

yolo_model = YOLO(str(YOLO_MODEL_PATH))

# Use the image size stored in the YOLO checkpoint when available.
yolo_image_size = yolo_model.overrides.get("imgsz", IMAGE_SIZE)

if isinstance(yolo_image_size, (list, tuple)):
    yolo_image_size = yolo_image_size[0]

yolo_image_size = int(yolo_image_size)

print("YOLO image size:", yolo_image_size)


# ============================================================
# UTILITY FUNCTIONS
# ============================================================
def batch_items(items, size):
    for index in range(0, len(items), size):
        yield items[index:index + size]


def reorder_yolo_probabilities(result):
    probabilities = np.zeros(
        len(CLASS_NAMES),
        dtype=np.float32,
    )

    raw_probabilities = (
        result.probs.data.detach().cpu().numpy()
    )

    for model_index, model_class_name in result.names.items():
        model_index = int(model_index)

        if model_class_name not in CLASS_NAMES:
            raise ValueError(
                f"Unexpected YOLO class: {model_class_name}"
            )

        expected_index = CLASS_NAMES.index(model_class_name)
        probabilities[expected_index] = raw_probabilities[model_index]

    return probabilities


def save_confusion_matrix(
    actual,
    predicted,
    labels,
    display_names,
    title,
    filename,
):
    matrix = confusion_matrix(
        actual,
        predicted,
        labels=labels,
    )

    figure, axis = plt.subplots(figsize=(8, 7))
    image = axis.imshow(matrix)

    axis.set_title(title)
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("Actual class")

    axis.set_xticks(range(len(display_names)))
    axis.set_yticks(range(len(display_names)))

    axis.set_xticklabels(display_names, rotation=30)
    axis.set_yticklabels(display_names)

    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
            )

    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(
        OUTPUT_DIR / filename,
        dpi=250,
    )
    plt.close(figure)


# ============================================================
# WARM-UP
# ============================================================
first_path = samples[0][0]

first_image = Image.open(first_path).convert("RGB")
first_tensor = (
    cnn_transform(first_image)
    .unsqueeze(0)
    .to(TORCH_DEVICE)
)

with torch.inference_mode():
    cnn_model(first_tensor)

yolo_model.predict(
    source=first_path,
    imgsz=yolo_image_size,
    device=YOLO_DEVICE,
    verbose=False,
)

synchronize_device()


# ============================================================
# EVALUATION
# ============================================================
actual_labels = []

cnn_predictions = []
cnn_confidences = []

yolo_predictions = []
yolo_confidences = []

image_paths = []

cnn_total_time = 0.0
yolo_total_time = 0.0


for batch_number, batch in enumerate(
    batch_items(samples, BATCH_SIZE),
    start=1,
):
    paths = [item[0] for item in batch]
    labels = [item[1] for item in batch]

    images = torch.stack([
        cnn_transform(
            Image.open(path).convert("RGB")
        )
        for path in paths
    ]).to(TORCH_DEVICE)

    synchronize_device()
    cnn_start = time.perf_counter()

    with torch.inference_mode():
        cnn_output = cnn_model(images)
        cnn_probabilities = torch.softmax(cnn_output, dim=1)

    synchronize_device()
    cnn_total_time += time.perf_counter() - cnn_start

    cnn_confidence, cnn_prediction = torch.max(
        cnn_probabilities,
        dim=1,
    )

    cnn_predictions.extend(
        cnn_prediction.detach().cpu().numpy().tolist()
    )

    cnn_confidences.extend(
        cnn_confidence.detach().cpu().numpy().tolist()
    )

    yolo_start = time.perf_counter()

    yolo_results = yolo_model.predict(
        source=paths,
        imgsz=yolo_image_size,
        batch=len(paths),
        device=YOLO_DEVICE,
        verbose=False,
    )

    yolo_total_time += time.perf_counter() - yolo_start

    for result in yolo_results:
        probabilities = reorder_yolo_probabilities(result)

        yolo_predictions.append(
            int(np.argmax(probabilities))
        )

        yolo_confidences.append(
            float(np.max(probabilities))
        )

    actual_labels.extend(labels)
    image_paths.extend(paths)

    processed = min(
        batch_number * BATCH_SIZE,
        len(samples),
    )

    print(
        f"Processed {processed}/{len(samples)} images"
    )


# ============================================================
# CONVERT TO NUMPY
# ============================================================
actual_labels = np.array(actual_labels)
cnn_predictions = np.array(cnn_predictions)
cnn_confidences = np.array(cnn_confidences)

yolo_predictions = np.array(yolo_predictions)
yolo_confidences = np.array(yolo_confidences)


# ============================================================
# AGREEMENT-BASED FINAL DECISION
# ============================================================
agreement_mask = cnn_predictions == yolo_predictions
disagreement_mask = ~agreement_mask

agreement_count = int(agreement_mask.sum())
disagreement_count = int(disagreement_mask.sum())

agreement_rate = agreement_mask.mean() * 100
disagreement_rate = disagreement_mask.mean() * 100

final_predictions = np.where(
    agreement_mask,
    yolo_predictions,
    len(CLASS_NAMES),  # uncertain class index
)

if agreement_count > 0:
    accepted_accuracy = (
        accuracy_score(
            actual_labels[agreement_mask],
            yolo_predictions[agreement_mask],
        )
        * 100
    )
else:
    accepted_accuracy = 0.0

accepted_wrong_count = int(
    (
        agreement_mask
        & (yolo_predictions != actual_labels)
    ).sum()
)


# ============================================================
# STANDARD MODEL METRICS
# ============================================================
cnn_accuracy = (
    accuracy_score(
        actual_labels,
        cnn_predictions,
    )
    * 100
)

yolo_accuracy = (
    accuracy_score(
        actual_labels,
        yolo_predictions,
    )
    * 100
)

cnn_report = classification_report(
    actual_labels,
    cnn_predictions,
    target_names=CLASS_NAMES,
    output_dict=True,
    zero_division=0,
)

yolo_report = classification_report(
    actual_labels,
    yolo_predictions,
    target_names=CLASS_NAMES,
    output_dict=True,
    zero_division=0,
)


# ============================================================
# PER-CLASS VERIFICATION RESULTS
# ============================================================
per_class_results = {}

for class_index, class_name in enumerate(CLASS_NAMES):
    class_mask = actual_labels == class_index
    class_agreement = agreement_mask[class_mask]

    class_coverage = (
        class_agreement.mean() * 100
        if class_mask.sum() > 0
        else 0.0
    )

    accepted_class_mask = class_mask & agreement_mask

    if accepted_class_mask.sum() > 0:
        class_accepted_accuracy = (
            accuracy_score(
                actual_labels[accepted_class_mask],
                yolo_predictions[accepted_class_mask],
            )
            * 100
        )
    else:
        class_accepted_accuracy = 0.0

    per_class_results[class_name] = {
        "test_images": int(class_mask.sum()),
        "accepted_images": int(accepted_class_mask.sum()),
        "manual_review_images": int(
            (class_mask & disagreement_mask).sum()
        ),
        "coverage_percent": round(class_coverage, 2),
        "accepted_accuracy_percent": round(
            class_accepted_accuracy,
            2,
        ),
    }


# ============================================================
# INFERENCE TIME
# ============================================================
total_images = len(actual_labels)

cnn_ms_per_image = (
    cnn_total_time / total_images
) * 1000

yolo_ms_per_image = (
    yolo_total_time / total_images
) * 1000

combined_ms_per_image = (
    cnn_ms_per_image + yolo_ms_per_image
)


# ============================================================
# SAVE CONFUSION MATRICES
# ============================================================
save_confusion_matrix(
    actual_labels,
    cnn_predictions,
    labels=[0, 1, 2],
    display_names=CLASS_NAMES,
    title="DeeperCNN Confusion Matrix",
    filename="cnn_confusion_matrix.png",
)

save_confusion_matrix(
    actual_labels,
    yolo_predictions,
    labels=[0, 1, 2],
    display_names=CLASS_NAMES,
    title="YOLO11-Medium Confusion Matrix",
    filename="yolo_confusion_matrix.png",
)

save_confusion_matrix(
    actual_labels,
    final_predictions,
    labels=[0, 1, 2, 3],
    display_names=[
        "bird",
        "ship",
        "unknown",
        "uncertain",
    ],
    title="Agreement-Based Final System Confusion Matrix",
    filename="final_system_confusion_matrix.png",
)


# ============================================================
# SAVE INDIVIDUAL PREDICTIONS
# ============================================================
predictions_file = OUTPUT_DIR / "all_predictions.csv"

with predictions_file.open("w", newline="") as file:
    writer = csv.writer(file)

    writer.writerow([
        "image_path",
        "actual_class",
        "cnn_prediction",
        "cnn_confidence_percent",
        "yolo_prediction",
        "yolo_confidence_percent",
        "final_prediction",
        "decision_status",
    ])

    for (
        path,
        actual,
        cnn_prediction,
        cnn_confidence,
        yolo_prediction,
        yolo_confidence,
        agreed,
    ) in zip(
        image_paths,
        actual_labels,
        cnn_predictions,
        cnn_confidences,
        yolo_predictions,
        yolo_confidences,
        agreement_mask,
    ):
        if agreed:
            final_prediction = CLASS_NAMES[yolo_prediction]
            status = "accepted_models_agree"
        else:
            final_prediction = "uncertain"
            status = "manual_review_models_disagree"

        writer.writerow([
            path,
            CLASS_NAMES[actual],
            CLASS_NAMES[cnn_prediction],
            round(cnn_confidence * 100, 2),
            CLASS_NAMES[yolo_prediction],
            round(yolo_confidence * 100, 2),
            final_prediction,
            status,
        ])


# ============================================================
# SAVE SUMMARY
# ============================================================
summary = {
    "dataset": {
        "test_images": total_images,
        "classes": CLASS_NAMES,
        "images_per_class": {
            name: int(
                (actual_labels == index).sum()
            )
            for index, name in enumerate(CLASS_NAMES)
        },
    },
    "model_accuracy_percent": {
        "deepercnn": round(cnn_accuracy, 2),
        "yolo11_medium": round(yolo_accuracy, 2),
    },
    "agreement_verification": {
        "accepted_predictions": agreement_count,
        "manual_review_predictions": disagreement_count,
        "coverage_percent": round(agreement_rate, 2),
        "manual_review_rate_percent": round(
            disagreement_rate,
            2,
        ),
        "accepted_prediction_accuracy_percent": round(
            accepted_accuracy,
            2,
        ),
        "accepted_but_incorrect_predictions": (
            accepted_wrong_count
        ),
    },
    "inference_time": {
        "cnn_ms_per_image": round(cnn_ms_per_image, 2),
        "yolo_ms_per_image": round(yolo_ms_per_image, 2),
        "combined_ms_per_image": round(
            combined_ms_per_image,
            2,
        ),
    },
    "per_class_verification": per_class_results,
    "cnn_classification_report": cnn_report,
    "yolo_classification_report": yolo_report,
}

with (
    OUTPUT_DIR / "metrics_summary.json"
).open("w") as file:
    json.dump(summary, file, indent=2)


# ============================================================
# DISPLAY RESULTS
# ============================================================
print("\n")
print("=" * 65)
print("FINAL RADAR CLASSIFICATION EVALUATION")
print("=" * 65)

print(f"Test images                 : {total_images}")
print(f"DeeperCNN accuracy          : {cnn_accuracy:.2f}%")
print(f"YOLO11-Medium accuracy      : {yolo_accuracy:.2f}%")

print("\nAgreement-based verification")
print(f"Models agreed               : {agreement_count}")
print(f"Models disagreed            : {disagreement_count}")
print(f"System coverage             : {agreement_rate:.2f}%")
print(f"Manual-review rate          : {disagreement_rate:.2f}%")
print(f"Accepted prediction accuracy: {accepted_accuracy:.2f}%")
print(f"Accepted but incorrect      : {accepted_wrong_count}")

print("\nAverage inference time")
print(f"DeeperCNN                   : {cnn_ms_per_image:.2f} ms/image")
print(f"YOLO11-Medium               : {yolo_ms_per_image:.2f} ms/image")
print(f"Combined models             : {combined_ms_per_image:.2f} ms/image")

print("\nPer-class verification")

for class_name, values in per_class_results.items():
    print(
        f"{class_name:8} | "
        f"coverage: {values['coverage_percent']:.2f}% | "
        f"accepted accuracy: "
        f"{values['accepted_accuracy_percent']:.2f}%"
    )

print("\nResults saved to:")
print(OUTPUT_DIR)
print("=" * 65)
