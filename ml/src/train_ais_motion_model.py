from pathlib import Path
import argparse
import csv
import json
import random
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# CONFIGURATION
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = (
    PROJECT_ROOT
    / "ml"
    / "ais_motion"
    / "sequences_5min"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "ml"
    / "models"
    / "ais_motion"
)

MODEL_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = [
    "Stopped",
    "Slow",
    "Moderate",
    "Fast",
]

RANDOM_SEED = 42

BATCH_SIZE = 512
EPOCHS = 15
LEARNING_RATE = 0.001

HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.20

POSITION_LOSS_WEIGHT = 1.0
SPEED_LOSS_WEIGHT = 0.5
CLASS_LOSS_WEIGHT = 1.0

EARLY_STOPPING_PATIENCE = 4


# ============================================================
# REPRODUCIBILITY
# ============================================================
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


# ============================================================
# DEVICE
# ============================================================
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print("Using device:", DEVICE)


# ============================================================
# ARGUMENTS
# ============================================================
parser = argparse.ArgumentParser()

parser.add_argument(
    "--model",
    choices=["gru", "lstm"],
    default="gru",
)

parser.add_argument(
    "--epochs",
    type=int,
    default=EPOCHS,
)

parser.add_argument(
    "--batch-size",
    type=int,
    default=BATCH_SIZE,
)

parser.add_argument(
    "--evaluate-only",
    action="store_true",
    help="Skip training and evaluate the saved best checkpoint.",
)

args = parser.parse_args()

MODEL_TYPE = args.model
EPOCHS = args.epochs
BATCH_SIZE = args.batch_size

RUN_DIR = MODEL_DIR / MODEL_TYPE
RUN_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH = RUN_DIR / f"ais_motion_{MODEL_TYPE}_best.pth"
HISTORY_PATH = RUN_DIR / "training_history.csv"
METRICS_PATH = RUN_DIR / "test_metrics.json"
CONFUSION_MATRIX_PATH = RUN_DIR / "confusion_matrix.png"


# ============================================================
# LOAD DATA
# ============================================================
def load_split(split_name):
    path = DATA_DIR / f"{split_name}_motion_sequences.npz"

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    data = np.load(path)

    return {
        "X": data["X"].astype(np.float32, copy=True),
        "y_position": data["y_position"].astype(
            np.float32,
            copy=True,
        ),
        "y_speed": data["y_speed"].astype(
            np.float32,
            copy=True,
        ),
        "y_class": data["y_class"].astype(
            np.int64,
            copy=True,
        ),
        "mmsi": data["mmsi"].astype(
            np.int64,
            copy=True,
        ),
    }


print("Loading datasets...")

train_data = load_split("train")
val_data = load_split("val")
test_data = load_split("test")

print("Train shape:", train_data["X"].shape)
print("Validation shape:", val_data["X"].shape)
print("Test shape:", test_data["X"].shape)


# ============================================================
# VERIFY VESSEL SPLIT
# ============================================================
train_vessels = set(train_data["mmsi"].tolist())
val_vessels = set(val_data["mmsi"].tolist())
test_vessels = set(test_data["mmsi"].tolist())

if train_vessels & val_vessels:
    raise RuntimeError("Train and validation vessels overlap.")

if train_vessels & test_vessels:
    raise RuntimeError("Train and test vessels overlap.")

if val_vessels & test_vessels:
    raise RuntimeError("Validation and test vessels overlap.")

print("MMSI split verification: PASSED")


# ============================================================
# NORMALISE REGRESSION TARGETS
# Use training data only.
# ============================================================
position_mean = train_data["y_position"].mean(axis=0)
position_std = train_data["y_position"].std(axis=0)

position_std = np.where(
    position_std < 1e-6,
    1.0,
    position_std,
)

speed_mean = float(train_data["y_speed"].mean())
speed_std = float(train_data["y_speed"].std())

if speed_std < 1e-6:
    speed_std = 1.0


def normalise_targets(data):
    position = (
        data["y_position"] - position_mean
    ) / position_std

    speed = (
        data["y_speed"] - speed_mean
    ) / speed_std

    return (
        position.astype(np.float32),
        speed.astype(np.float32),
    )


train_position_norm, train_speed_norm = normalise_targets(
    train_data
)

val_position_norm, val_speed_norm = normalise_targets(
    val_data
)

test_position_norm, test_speed_norm = normalise_targets(
    test_data
)


# ============================================================
# CLASS WEIGHTS
# Uses square-root inverse frequency to avoid extreme weights.
# ============================================================
class_counts = np.bincount(
    train_data["y_class"],
    minlength=len(CLASS_NAMES),
)

class_weights = np.sqrt(
    class_counts.sum()
    / (
        len(CLASS_NAMES)
        * np.maximum(class_counts, 1)
    )
)

class_weights = class_weights / class_weights.mean()

class_weights_tensor = torch.tensor(
    class_weights,
    dtype=torch.float32,
    device=DEVICE,
)

print("\nTraining class counts:")

for index, class_name in enumerate(CLASS_NAMES):
    print(
        f"{class_name:8}: "
        f"{class_counts[index]:,} "
        f"| weight: {class_weights[index]:.4f}"
    )


# ============================================================
# DATA LOADERS
# ============================================================
def create_loader(
    data,
    position_targets,
    speed_targets,
    shuffle,
):
    dataset = TensorDataset(
        torch.from_numpy(data["X"]),
        torch.from_numpy(position_targets),
        torch.from_numpy(speed_targets),
        torch.from_numpy(data["y_class"]),
    )

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,
        drop_last=False,
    )


train_loader = create_loader(
    train_data,
    train_position_norm,
    train_speed_norm,
    shuffle=True,
)

val_loader = create_loader(
    val_data,
    val_position_norm,
    val_speed_norm,
    shuffle=False,
)

test_loader = create_loader(
    test_data,
    test_position_norm,
    test_speed_norm,
    shuffle=False,
)


# ============================================================
# MULTI-TASK GRU/LSTM
# ============================================================
class MultiTaskMotionModel(nn.Module):
    def __init__(
        self,
        input_size,
        model_type,
        hidden_size=128,
        num_layers=2,
        dropout=0.2,
    ):
        super().__init__()

        recurrent_class = (
            nn.GRU
            if model_type == "gru"
            else nn.LSTM
        )

        self.recurrent = recurrent_class(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=(
                dropout
                if num_layers > 1
                else 0.0
            ),
        )

        self.shared = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

        self.position_head = nn.Linear(64, 2)
        self.speed_head = nn.Linear(64, 1)
        self.class_head = nn.Linear(
            64,
            len(CLASS_NAMES),
        )

    def forward(self, inputs):
        recurrent_output, _ = self.recurrent(inputs)

        final_hidden = recurrent_output[:, -1, :]
        shared_features = self.shared(final_hidden)

        position = self.position_head(shared_features)

        speed = self.speed_head(
            shared_features
        ).squeeze(1)

        motion_class = self.class_head(
            shared_features
        )

        return position, speed, motion_class


input_size = train_data["X"].shape[2]

model = MultiTaskMotionModel(
    input_size=input_size,
    model_type=MODEL_TYPE,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT,
).to(DEVICE)

print("\nModel type:", MODEL_TYPE.upper())
print(model)


# ============================================================
# LOSSES AND OPTIMISER
# ============================================================
regression_loss = nn.SmoothL1Loss()

classification_loss = nn.CrossEntropyLoss(
    weight=class_weights_tensor
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-4,
)


def calculate_loss(
    predicted_position,
    predicted_speed,
    predicted_class,
    actual_position,
    actual_speed,
    actual_class,
):
    position_loss = regression_loss(
        predicted_position,
        actual_position,
    )

    speed_loss = regression_loss(
        predicted_speed,
        actual_speed,
    )

    class_loss = classification_loss(
        predicted_class,
        actual_class,
    )

    total_loss = (
        POSITION_LOSS_WEIGHT * position_loss
        + SPEED_LOSS_WEIGHT * speed_loss
        + CLASS_LOSS_WEIGHT * class_loss
    )

    return (
        total_loss,
        position_loss,
        speed_loss,
        class_loss,
    )


# ============================================================
# TRAINING
# ============================================================
def train_one_epoch():
    model.train()

    total_loss_sum = 0.0
    total_samples = 0

    for batch_number, batch in enumerate(
        train_loader,
        start=1,
    ):
        (
            inputs,
            actual_position,
            actual_speed,
            actual_class,
        ) = [
            value.to(DEVICE)
            for value in batch
        ]

        optimizer.zero_grad(set_to_none=True)

        (
            predicted_position,
            predicted_speed,
            predicted_class,
        ) = model(inputs)

        total_loss, _, _, _ = calculate_loss(
            predicted_position,
            predicted_speed,
            predicted_class,
            actual_position,
            actual_speed,
            actual_class,
        )

        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=5.0,
        )

        optimizer.step()

        batch_size = inputs.size(0)

        total_loss_sum += (
            total_loss.item() * batch_size
        )

        total_samples += batch_size

        if batch_number % 200 == 0:
            print(
                f"  Batch "
                f"{batch_number}/"
                f"{len(train_loader)}"
            )

    return total_loss_sum / total_samples


# ============================================================
# EVALUATION
# ============================================================
def evaluate(loader):
    model.eval()

    total_loss_sum = 0.0
    total_samples = 0

    all_actual_classes = []
    all_predicted_classes = []

    all_actual_positions = []
    all_predicted_positions = []

    all_actual_speeds = []
    all_predicted_speeds = []

    with torch.inference_mode():
        for batch in loader:
            (
                inputs,
                actual_position,
                actual_speed,
                actual_class,
            ) = [
                value.to(DEVICE)
                for value in batch
            ]

            (
                predicted_position,
                predicted_speed,
                predicted_class_logits,
            ) = model(inputs)

            total_loss, _, _, _ = calculate_loss(
                predicted_position,
                predicted_speed,
                predicted_class_logits,
                actual_position,
                actual_speed,
                actual_class,
            )

            batch_size = inputs.size(0)

            total_loss_sum += (
                total_loss.item() * batch_size
            )

            total_samples += batch_size

            predicted_class = torch.argmax(
                predicted_class_logits,
                dim=1,
            )

            actual_position_original = (
                actual_position.cpu().numpy()
                * position_std
                + position_mean
            )

            predicted_position_original = (
                predicted_position.cpu().numpy()
                * position_std
                + position_mean
            )

            actual_speed_original = (
                actual_speed.cpu().numpy()
                * speed_std
                + speed_mean
            )

            predicted_speed_original = (
                predicted_speed.cpu().numpy()
                * speed_std
                + speed_mean
            )

            all_actual_classes.extend(
                actual_class.cpu().numpy()
            )

            all_predicted_classes.extend(
                predicted_class.cpu().numpy()
            )

            all_actual_positions.append(
                actual_position_original
            )

            all_predicted_positions.append(
                predicted_position_original
            )

            all_actual_speeds.append(
                actual_speed_original
            )

            all_predicted_speeds.append(
                predicted_speed_original
            )

    actual_classes = np.asarray(
        all_actual_classes
    )

    predicted_classes = np.asarray(
        all_predicted_classes
    )

    actual_positions = np.concatenate(
        all_actual_positions,
        axis=0,
    )

    predicted_positions = np.concatenate(
        all_predicted_positions,
        axis=0,
    )

    actual_speeds = np.concatenate(
        all_actual_speeds,
        axis=0,
    )

    predicted_speeds = np.concatenate(
        all_predicted_speeds,
        axis=0,
    )

    position_errors = np.linalg.norm(
        predicted_positions - actual_positions,
        axis=1,
    )

    speed_errors = np.abs(
        predicted_speeds - actual_speeds
    )

    metrics = {
        "loss": total_loss_sum / total_samples,
        "accuracy": accuracy_score(
            actual_classes,
            predicted_classes,
        ),
        "macro_f1": f1_score(
            actual_classes,
            predicted_classes,
            average="macro",
            zero_division=0,
        ),
        "position_mae_metres": float(
            position_errors.mean()
        ),
        "position_median_error_metres": float(
            np.median(position_errors)
        ),
        "speed_mae_knots": float(
            speed_errors.mean()
        ),
        "actual_classes": actual_classes,
        "predicted_classes": predicted_classes,
    }

    return metrics


# ============================================================
# RUN TRAINING
# ============================================================
if not args.evaluate_only:
    history_rows = []

    best_validation_loss = float("inf")
    epochs_without_improvement = 0

    training_start = time.perf_counter()

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.perf_counter()

        print("\n" + "=" * 70)
        print(f"EPOCH {epoch}/{EPOCHS}")
        print("=" * 70)

        training_loss = train_one_epoch()
        validation_metrics = evaluate(val_loader)

        epoch_seconds = time.perf_counter() - epoch_start

        print(
            f"Training loss       : {training_loss:.6f}"
        )

        print(
            f"Validation loss     : "
            f"{validation_metrics['loss']:.6f}"
        )

        print(
            f"Validation accuracy : "
            f"{validation_metrics['accuracy'] * 100:.2f}%"
        )

        print(
            f"Validation macro F1 : "
            f"{validation_metrics['macro_f1']:.4f}"
        )

        print(
            f"Position MAE        : "
            f"{validation_metrics['position_mae_metres']:.2f} m"
        )

        print(
            f"Speed MAE           : "
            f"{validation_metrics['speed_mae_knots']:.2f} knots"
        )

        print(
            f"Epoch time          : "
            f"{epoch_seconds:.2f} seconds"
        )

        history_rows.append({
            "epoch": epoch,
            "training_loss": training_loss,
            "validation_loss": validation_metrics["loss"],
            "validation_accuracy": validation_metrics["accuracy"],
            "validation_macro_f1": validation_metrics["macro_f1"],
            "validation_position_mae_metres": (
                validation_metrics[
                    "position_mae_metres"
                ]
            ),
            "validation_speed_mae_knots": (
                validation_metrics[
                    "speed_mae_knots"
                ]
            ),
            "epoch_seconds": epoch_seconds,
        })

        if (
            validation_metrics["loss"]
            < best_validation_loss
        ):
            best_validation_loss = (
                validation_metrics["loss"]
            )

            epochs_without_improvement = 0

            torch.save(
                {
                    "model_type": MODEL_TYPE,
                    "model_state_dict": model.state_dict(),
                    "input_size": input_size,
                    "hidden_size": HIDDEN_SIZE,
                    "num_layers": NUM_LAYERS,
                    "dropout": DROPOUT,
                    "class_names": CLASS_NAMES,
                    "position_mean": torch.from_numpy(position_mean.astype(np.float32).copy()),
                    "position_std": torch.from_numpy(position_std.astype(np.float32).copy()),
                    "speed_mean": speed_mean,
                    "speed_std": speed_std,
                    "best_validation_loss": (
                        best_validation_loss
                    ),
                },
                BEST_MODEL_PATH,
            )

            print("Saved new best model:", BEST_MODEL_PATH)

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
                print("Early stopping activated.")
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
            fieldnames=history_rows[0].keys(),
        )

        writer.writeheader()
        writer.writerows(history_rows)



else:
    print("\nSkipping training and evaluating the saved best checkpoint.")

# ============================================================
# LOAD BEST MODEL AND TEST
# ============================================================
checkpoint = torch.load(
    BEST_MODEL_PATH,
    map_location=DEVICE,
    weights_only=True,
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

test_metrics = evaluate(test_loader)

precision, recall, f1, support = (
    precision_recall_fscore_support(
        test_metrics["actual_classes"],
        test_metrics["predicted_classes"],
        labels=list(range(len(CLASS_NAMES))),
        zero_division=0,
    )
)

matrix = confusion_matrix(
    test_metrics["actual_classes"],
    test_metrics["predicted_classes"],
    labels=list(range(len(CLASS_NAMES))),
)


# ============================================================
# SAVE CONFUSION MATRIX
# ============================================================
figure, axis = plt.subplots(
    figsize=(8, 7)
)

image = axis.imshow(matrix)

axis.set_title(
    f"{MODEL_TYPE.upper()} Motion-State Confusion Matrix"
)

axis.set_xlabel("Predicted motion state")
axis.set_ylabel("Actual motion state")

axis.set_xticks(range(len(CLASS_NAMES)))
axis.set_yticks(range(len(CLASS_NAMES)))

axis.set_xticklabels(
    CLASS_NAMES,
    rotation=30,
)

axis.set_yticklabels(CLASS_NAMES)

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
    CONFUSION_MATRIX_PATH,
    dpi=250,
)

plt.close(figure)


# ============================================================
# SAVE TEST METRICS
# ============================================================
total_training_seconds = (
    time.perf_counter() - training_start
    if not args.evaluate_only
    else 0.0
)

results = {
    "model_type": MODEL_TYPE,
    "forecast_horizon_minutes": 5,
    "test_samples": int(
        len(test_data["X"])
    ),
    "test_unique_vessels": int(
        len(test_vessels)
    ),
    "test_accuracy_percent": round(
        test_metrics["accuracy"] * 100,
        2,
    ),
    "test_macro_f1": round(
        test_metrics["macro_f1"],
        4,
    ),
    "test_position_mae_metres": round(
        test_metrics["position_mae_metres"],
        2,
    ),
    "test_position_median_error_metres": round(
        test_metrics[
            "position_median_error_metres"
        ],
        2,
    ),
    "test_speed_mae_knots": round(
        test_metrics["speed_mae_knots"],
        3,
    ),
    "training_seconds": round(
        total_training_seconds,
        2,
    ),
    "per_class": {
        CLASS_NAMES[index]: {
            "precision": round(
                float(precision[index]),
                4,
            ),
            "recall": round(
                float(recall[index]),
                4,
            ),
            "f1_score": round(
                float(f1[index]),
                4,
            ),
            "support": int(
                support[index]
            ),
        }
        for index in range(len(CLASS_NAMES))
    },
    "confusion_matrix": matrix.tolist(),
}

with METRICS_PATH.open("w") as file:
    json.dump(
        results,
        file,
        indent=2,
    )


# ============================================================
# FINAL OUTPUT
# ============================================================
print("\n" + "=" * 70)
print(f"FINAL {MODEL_TYPE.upper()} TEST RESULTS")
print("=" * 70)

print(
    f"Test accuracy       : "
    f"{results['test_accuracy_percent']:.2f}%"
)

print(
    f"Macro F1-score      : "
    f"{results['test_macro_f1']:.4f}"
)

print(
    f"Position MAE        : "
    f"{results['test_position_mae_metres']:.2f} metres"
)

print(
    f"Median position err : "
    f"{results['test_position_median_error_metres']:.2f} metres"
)

print(
    f"Speed MAE           : "
    f"{results['test_speed_mae_knots']:.3f} knots"
)

print("\nPer-class results:")

for class_name, values in results["per_class"].items():
    print(
        f"{class_name:8} | "
        f"precision={values['precision']:.4f} | "
        f"recall={values['recall']:.4f} | "
        f"F1={values['f1_score']:.4f} | "
        f"support={values['support']:,}"
    )

print("\nBest model:")
print(BEST_MODEL_PATH)

print("\nResults:")
print(METRICS_PATH)

print("\nConfusion matrix:")
print(CONFUSION_MATRIX_PATH)
