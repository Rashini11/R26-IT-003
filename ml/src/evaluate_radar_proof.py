from pathlib import Path
import csv
import json

import matplotlib.pyplot as plt
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


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = (
    ROOT
    / "ml"
    / "dataset_v4_raw"
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
    / "radar_target89_proof"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# DEVICE
# ============================================================

device = (
    torch.device("mps")
    if torch.backends.mps.is_available()
    else torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("cpu")
)

print("Using device:", device)


# ============================================================
# PREPROCESSING
# EXACTLY MATCHES THE MODEL EVALUATION PIPELINE
# ============================================================

eval_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])


# ============================================================
# MODEL
# MUST MATCH train_radar_target89.py
# ============================================================

class RadarTargetCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(
                3,
                8,
                5,
                stride=2,
                padding=2,
            ),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(
                8,
                12,
                3,
                stride=2,
                padding=1,
            ),

            nn.ReLU(),

            nn.AdaptiveAvgPool2d(
                (1, 1)
            ),
        )

        self.dropout = nn.Dropout(
            0.45
        )

        self.fc = nn.Linear(
            12,
            2
        )

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

print("Model loaded:", MODEL_PATH)


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate_split(split_name, directory):

    dataset = datasets.ImageFolder(
        directory,
        transform=eval_transform,
    )

    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
    )

    actual = []
    predicted = []
    confidences = []
    rows = []

    sample_index = 0

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

                image_path, _ = (
                    dataset.samples[
                        sample_index + i
                    ]
                )

                truth = int(
                    labels[i].item()
                )

                pred = int(
                    prediction[i]
                    .cpu()
                    .item()
                )

                rows.append({
                    "split":
                        split_name,

                    "filename":
                        Path(
                            image_path
                        ).name,

                    "ground_truth":
                        dataset.classes[
                            truth
                        ],

                    "prediction":
                        dataset.classes[
                            pred
                        ],

                    "confidence":
                        float(
                            confidence[i]
                            .cpu()
                            .item()
                        ),

                    "correct":
                        truth == pred,
                })

            sample_index += (
                batch_size
            )

    actual = np.asarray(actual)

    predicted = np.asarray(
        predicted
    )

    confidences = np.asarray(
        confidences
    )


    accuracy = accuracy_score(
        actual,
        predicted,
    )

    (
        precision,
        recall,
        f1,
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
    print("=" * 70)
    print(
        f"{split_name.upper()} RESULTS"
    )
    print("=" * 70)

    print(
        f"Samples         : "
        f"{len(dataset)}"
    )

    print(
        f"Accuracy        : "
        f"{accuracy * 100:.2f}%"
    )

    print(
        f"Macro Precision : "
        f"{precision:.4f}"
    )

    print(
        f"Macro Recall    : "
        f"{recall:.4f}"
    )

    print(
        f"Macro F1        : "
        f"{f1:.4f}"
    )

    print(
        f"Mean Confidence : "
        f"{confidences.mean() * 100:.2f}%"
    )

    print()

    print(
        classification_report(
            actual,
            predicted,
            target_names=
                dataset.classes,
            digits=4,
            zero_division=0,
        )
    )

    print(
        "Confusion Matrix:"
    )

    print(cm)


    # ========================================================
    # SAVE CONFUSION MATRIX IMAGE
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(5, 4)
    )

    image = ax.imshow(cm)

    ax.set_title(
        f"Radar {split_name.title()} "
        "Confusion Matrix"
    )

    ax.set_xlabel(
        "Predicted Class"
    )

    ax.set_ylabel(
        "Actual Class"
    )

    ax.set_xticks(
        range(
            len(dataset.classes)
        )
    )

    ax.set_yticks(
        range(
            len(dataset.classes)
        )
    )

    ax.set_xticklabels(
        dataset.classes
    )

    ax.set_yticklabels(
        dataset.classes
    )

    for i in range(
        cm.shape[0]
    ):

        for j in range(
            cm.shape[1]
        ):

            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
            )

    fig.colorbar(
        image,
        ax=ax
    )

    fig.tight_layout()

    figure_path = (
        OUT_DIR
        / f"{split_name}_confusion_matrix.png"
    )

    fig.savefig(
        figure_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


    return {
        "split":
            split_name,

        "samples":
            len(dataset),

        "accuracy":
            float(accuracy),

        "accuracy_percent":
            float(
                accuracy * 100
            ),

        "macro_precision":
            float(precision),

        "macro_recall":
            float(recall),

        "macro_f1":
            float(f1),

        "mean_confidence":
            float(
                confidences.mean()
            ),

        "confusion_matrix":
            cm.tolist(),

        "class_names":
            dataset.classes,

        "per_sample":
            rows,
    }


# ============================================================
# RUN VALIDATION + TEST
# ============================================================

validation = evaluate_split(
    "validation",
    DATA_DIR / "val",
)

test = evaluate_split(
    "test",
    DATA_DIR / "test",
)


# ============================================================
# SAVE JSON
# ============================================================

summary = {
    "model":
        "RadarTargetCNN",

    "model_path":
        str(MODEL_PATH),

    "dataset":
        "dataset_v4_raw",

    "classes": [
        "bird",
        "ship",
    ],

    "heatmap_preprocessing":
        False,

    "validation": {
        key: value
        for key, value
        in validation.items()
        if key != "per_sample"
    },

    "test": {
        key: value
        for key, value
        in test.items()
        if key != "per_sample"
    },
}


json_path = (
    OUT_DIR
    / "radar_evaluation_metrics.json"
)

json_path.write_text(
    json.dumps(
        summary,
        indent=2,
    )
)


# ============================================================
# SAVE PER-IMAGE CSV
# ============================================================

csv_path = (
    OUT_DIR
    / "radar_predictions.csv"
)

all_rows = (
    validation["per_sample"]
    + test["per_sample"]
)

with csv_path.open(
    "w",
    newline="",
    encoding="utf-8",
) as handle:

    writer = csv.DictWriter(
        handle,
        fieldnames=[
            "split",
            "filename",
            "ground_truth",
            "prediction",
            "confidence",
            "correct",
        ],
    )

    writer.writeheader()

    writer.writerows(
        all_rows
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("RADAR FINAL EVALUATION SUMMARY")
print("=" * 70)

print(
    f"Validation Accuracy : "
    f"{validation['accuracy_percent']:.2f}%"
)

print(
    f"Validation Macro F1 : "
    f"{validation['macro_f1']:.4f}"
)

print()

print(
    f"Test Accuracy       : "
    f"{test['accuracy_percent']:.2f}%"
)

print(
    f"Test Macro Precision: "
    f"{test['macro_precision']:.4f}"
)

print(
    f"Test Macro Recall   : "
    f"{test['macro_recall']:.4f}"
)

print(
    f"Test Macro F1       : "
    f"{test['macro_f1']:.4f}"
)

print()
print(
    "Metrics JSON:",
    json_path,
)

print(
    "Predictions CSV:",
    csv_path,
)

print(
    "Validation CM:",
    OUT_DIR
    / "validation_confusion_matrix.png"
)

print(
    "Test CM:",
    OUT_DIR
    / "test_confusion_matrix.png"
)

print("=" * 70)
