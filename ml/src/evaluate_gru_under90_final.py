from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    classification_report,
)

from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = (
    ROOT
    / "ml"
    / "ais_motion"
    / "sequences_5min"
)

MODEL_DIR = (
    ROOT
    / "ml"
    / "models"
    / "ais_motion_under90_final"
)

MODEL_PATH = (
    MODEL_DIR
    / "ais_motion_gru_under90_best.pth"
)

CONFIG_PATH = (
    MODEL_DIR
    / "config.json"
)

OUTPUT_PATH = (
    MODEL_DIR
    / "final_evaluation_metrics.json"
)

CM_PATH = (
    MODEL_DIR
    / "test_confusion_matrix.png"
)


CLASS_NAMES = [
    "Stopped",
    "Slow",
    "Moderate",
    "Fast",
]

BATCH_SIZE = 1024


# ============================================================
# DEVICE
# ============================================================

DEVICE = (
    torch.device("mps")
    if torch.backends.mps.is_available()
    else torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("cpu")
)

print("Using device:", DEVICE)


# ============================================================
# LOAD SELECTED CONFIG
# ============================================================

config = json.loads(
    CONFIG_PATH.read_text()
)

HIDDEN_SIZE = int(
    config["hidden"]
)

SHARED_SIZE = int(
    config["shared"]
)

DROPOUT = float(
    config["dropout"]
)

print()
print("Selected configuration:")
print("Hidden size :", HIDDEN_SIZE)
print("Shared size :", SHARED_SIZE)
print("Dropout     :", DROPOUT)
print(
    "Saved validation accuracy:",
    f"{config['accuracy']*100:.2f}%"
)


# ============================================================
# LOAD DATA
# ============================================================

def load_split(name):

    data = np.load(
        DATA_DIR
        / f"{name}_motion_sequences.npz"
    )

    return {
        "X":
            data["X"].astype(
                np.float32
            ),

        "position":
            data["y_position"].astype(
                np.float32
            ),

        "speed":
            data["y_speed"].astype(
                np.float32
            ),

        "class":
            data["y_class"].astype(
                np.int64
            ),

        "mmsi":
            data["mmsi"].astype(
                np.int64
            ),
    }


train = load_split("train")
val = load_split("val")
test = load_split("test")


print()
print("Train samples     :", len(train["X"]))
print("Validation samples:", len(val["X"]))
print("Test samples      :", len(test["X"]))


# ============================================================
# VERIFY MMSI SEPARATION
# ============================================================

train_mmsi = set(
    train["mmsi"].tolist()
)

val_mmsi = set(
    val["mmsi"].tolist()
)

test_mmsi = set(
    test["mmsi"].tolist()
)

assert not (
    train_mmsi & val_mmsi
)

assert not (
    train_mmsi & test_mmsi
)

assert not (
    val_mmsi & test_mmsi
)

print(
    "MMSI split verification: PASSED"
)


# ============================================================
# TARGET NORMALISATION
# TRAINING DATA ONLY
# ============================================================

position_mean = (
    train["position"].mean(
        axis=0
    )
)

position_std = (
    train["position"].std(
        axis=0
    )
)

position_std = np.where(
    position_std < 1e-6,
    1.0,
    position_std,
)


speed_mean = float(
    train["speed"].mean()
)

speed_std = float(
    train["speed"].std()
)

if speed_std < 1e-6:
    speed_std = 1.0


def create_loader(data):

    position_norm = (
        (
            data["position"]
            - position_mean
        )
        / position_std
    ).astype(np.float32)

    speed_norm = (
        (
            data["speed"]
            - speed_mean
        )
        / speed_std
    ).astype(np.float32)

    dataset = TensorDataset(
        torch.from_numpy(
            data["X"]
        ),
        torch.from_numpy(
            position_norm
        ),
        torch.from_numpy(
            speed_norm
        ),
        torch.from_numpy(
            data["class"]
        ),
    )

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )


val_loader = create_loader(
    val
)

test_loader = create_loader(
    test
)


# ============================================================
# EXACT SELECTED GRU ARCHITECTURE
# ============================================================

class SmallGRU(nn.Module):

    def __init__(
        self,
        hidden,
        shared,
        dropout,
    ):
        super().__init__()

        self.gru = nn.GRU(
            input_size=7,
            hidden_size=hidden,
            num_layers=1,
            batch_first=True,
        )

        self.shared = nn.Sequential(
            nn.Linear(
                hidden,
                shared
            ),
            nn.ReLU(),
            nn.Dropout(
                dropout
            ),
        )

        self.position_head = (
            nn.Linear(
                shared,
                2
            )
        )

        self.speed_head = (
            nn.Linear(
                shared,
                1
            )
        )

        self.class_head = (
            nn.Linear(
                shared,
                4
            )
        )

    def forward(self, x):

        x, _ = self.gru(x)

        features = self.shared(
            x[:, -1, :]
        )

        position = (
            self.position_head(
                features
            )
        )

        speed = (
            self.speed_head(
                features
            ).squeeze(1)
        )

        classes = (
            self.class_head(
                features
            )
        )

        return (
            position,
            speed,
            classes,
        )


model = SmallGRU(
    hidden=HIDDEN_SIZE,
    shared=SHARED_SIZE,
    dropout=DROPOUT,
).to(DEVICE)


model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True,
    )
)

model.eval()

print()
print(
    "Model loaded:",
    MODEL_PATH
)


# ============================================================
# EVALUATION
# ============================================================

def evaluate(
    loader,
    raw_data,
    split_name,
):

    actual_classes = []
    predicted_classes = []

    predicted_positions = []
    predicted_speeds = []

    with torch.inference_mode():

        for (
            inputs,
            _,
            _,
            labels,
        ) in loader:

            inputs = inputs.to(
                DEVICE
            )

            (
                position_output,
                speed_output,
                class_output,
            ) = model(inputs)

            predicted_classes.extend(
                class_output
                .argmax(1)
                .cpu()
                .numpy()
                .tolist()
            )

            actual_classes.extend(
                labels.numpy().tolist()
            )

            predicted_positions.append(
                position_output
                .cpu()
                .numpy()
            )

            predicted_speeds.append(
                speed_output
                .cpu()
                .numpy()
            )


    actual_classes = np.asarray(
        actual_classes
    )

    predicted_classes = np.asarray(
        predicted_classes
    )

    predicted_positions = np.concatenate(
        predicted_positions,
        axis=0,
    )

    predicted_speeds = np.concatenate(
        predicted_speeds,
        axis=0,
    )


    # --------------------------------------------------------
    # DENORMALISE REGRESSION OUTPUTS
    # --------------------------------------------------------

    predicted_positions = (
        predicted_positions
        * position_std
        + position_mean
    )

    predicted_speeds = (
        predicted_speeds
        * speed_std
        + speed_mean
    )


    # --------------------------------------------------------
    # POSITION ERROR
    # Euclidean error in metres
    # --------------------------------------------------------

    position_errors = np.linalg.norm(
        predicted_positions
        - raw_data["position"],
        axis=1,
    )

    position_mae = float(
        position_errors.mean()
    )

    position_median = float(
        np.median(
            position_errors
        )
    )


    # --------------------------------------------------------
    # SPEED ERROR
    # --------------------------------------------------------

    speed_mae = float(
        np.abs(
            predicted_speeds
            - raw_data["speed"]
        ).mean()
    )


    # --------------------------------------------------------
    # CLASSIFICATION
    # --------------------------------------------------------

    accuracy = float(
        accuracy_score(
            actual_classes,
            predicted_classes,
        )
    )

    (
        precision,
        recall,
        f1,
        support,
    ) = precision_recall_fscore_support(
        actual_classes,
        predicted_classes,
        labels=[
            0,
            1,
            2,
            3,
        ],
        zero_division=0,
    )

    macro_f1 = float(
        np.mean(f1)
    )

    matrix = confusion_matrix(
        actual_classes,
        predicted_classes,
        labels=[
            0,
            1,
            2,
            3,
        ],
    )


    print()
    print("=" * 70)
    print(
        f"{split_name.upper()} RESULTS"
    )
    print("=" * 70)

    print(
        f"Motion-state accuracy : "
        f"{accuracy*100:.2f}%"
    )

    print(
        f"Macro F1              : "
        f"{macro_f1:.4f}"
    )

    print(
        f"Position MAE          : "
        f"{position_mae:.2f} m"
    )

    print(
        f"Median position error : "
        f"{position_median:.2f} m"
    )

    print(
        f"Speed MAE             : "
        f"{speed_mae:.3f} knots"
    )

    print()

    print(
        classification_report(
            actual_classes,
            predicted_classes,
            target_names=
                CLASS_NAMES,
            digits=4,
            zero_division=0,
        )
    )

    print(
        "Confusion matrix:"
    )

    print(matrix)


    per_class = {}

    for i, name in enumerate(
        CLASS_NAMES
    ):

        per_class[name] = {
            "precision":
                float(
                    precision[i]
                ),

            "recall":
                float(
                    recall[i]
                ),

            "f1":
                float(
                    f1[i]
                ),

            "support":
                int(
                    support[i]
                ),
        }


    return {
        "samples":
            int(
                len(
                    actual_classes
                )
            ),

        "accuracy_percent":
            accuracy * 100,

        "macro_f1":
            macro_f1,

        "position_mae_metres":
            position_mae,

        "median_position_error_metres":
            position_median,

        "speed_mae_knots":
            speed_mae,

        "per_class":
            per_class,

        "confusion_matrix":
            matrix.tolist(),
    }


validation_results = evaluate(
    val_loader,
    val,
    "validation",
)

test_results = evaluate(
    test_loader,
    test,
    "test",
)


# ============================================================
# CONFUSION MATRIX FIGURE
# ============================================================

matrix = np.asarray(
    test_results[
        "confusion_matrix"
    ]
)

fig, ax = plt.subplots(
    figsize=(6, 5)
)

image = ax.imshow(
    matrix
)

ax.set_title(
    "GRU Motion-State Test Confusion Matrix"
)

ax.set_xlabel(
    "Predicted Class"
)

ax.set_ylabel(
    "Actual Class"
)

ax.set_xticks(
    range(4)
)

ax.set_yticks(
    range(4)
)

ax.set_xticklabels(
    CLASS_NAMES
)

ax.set_yticklabels(
    CLASS_NAMES
)

for i in range(4):

    for j in range(4):

        ax.text(
            j,
            i,
            str(
                matrix[i, j]
            ),
            ha="center",
            va="center",
        )

fig.colorbar(
    image,
    ax=ax
)

fig.tight_layout()

fig.savefig(
    CM_PATH,
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# SAVE RESULTS
# ============================================================

result = {
    "model":
        "SmallGRU-under90",

    "forecast_horizon_minutes":
        5,

    "selected_configuration":
        config,

    "validation":
        validation_results,

    "test":
        test_results,
}


OUTPUT_PATH.write_text(
    json.dumps(
        result,
        indent=2,
    )
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("FINAL GRU UNDER-90 SUMMARY")
print("=" * 70)

print(
    f"Validation Accuracy : "
    f"{validation_results['accuracy_percent']:.2f}%"
)

print(
    f"Validation Macro F1 : "
    f"{validation_results['macro_f1']:.4f}"
)

print()

print(
    f"Test Accuracy       : "
    f"{test_results['accuracy_percent']:.2f}%"
)

print(
    f"Test Macro F1       : "
    f"{test_results['macro_f1']:.4f}"
)

print(
    f"Position MAE        : "
    f"{test_results['position_mae_metres']:.2f} m"
)

print(
    f"Median Position Err : "
    f"{test_results['median_position_error_metres']:.2f} m"
)

print(
    f"Speed MAE           : "
    f"{test_results['speed_mae_knots']:.3f} knots"
)

print()
print(
    "Metrics:",
    OUTPUT_PATH
)

print(
    "Confusion matrix:",
    CM_PATH
)

print("=" * 70)
