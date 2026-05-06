from pathlib import Path
import csv
import torch
from ultralytics import YOLO

# =========================
# CONFIG
# =========================
DATA_DIR = Path("ml/dataset_v2_balanced")
MODEL_DIR = Path("ml/models")
RESULTS_FILE = MODEL_DIR / "yolo_experiment_results.csv"

EPOCHS = 5
IMG_SIZE = 128
BATCH_SIZE = 32

MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Device
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

print("Using device:", DEVICE)

# =========================
# YOLO CLASSIFICATION MODELS
# =========================
models = {
    "YOLO11_Nano": "yolo11n-cls.pt",
    "YOLO11_Small": "yolo11s-cls.pt",
    "YOLO11_Medium": "yolo11m-cls.pt"
}

results = []


def get_top1_accuracy(metrics):
    """
    Extract top-1 accuracy from YOLO classification metrics.
    """
    if hasattr(metrics, "top1"):
        return round(metrics.top1 * 100, 2)

    if hasattr(metrics, "results_dict"):
        for key, value in metrics.results_dict.items():
            if "top1" in key.lower():
                return round(value * 100, 2)

    return None


# =========================
# TRAIN + TEST EACH MODEL
# =========================
for model_name, model_file in models.items():
    print("\n==============================")
    print(f"Training {model_name}")
    print("==============================")

    model = YOLO(model_file)

    model.train(
        data=str(DATA_DIR),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        project="ml/models/yolo_runs",
        name=model_name,
        exist_ok=True
    )

    best_model_path = Path(model.trainer.save_dir) / "weights" / "best.pt"

    print(f"Best model saved at: {best_model_path}")

    best_model = YOLO(str(best_model_path))

    test_metrics = best_model.val(
        data=str(DATA_DIR),
        split="test",
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE
    )

    test_acc = get_top1_accuracy(test_metrics)

    print(f"{model_name} Test Accuracy: {test_acc}%")

    results.append({
        "model_name": model_name,
        "model_file": model_file,
        "test_accuracy": test_acc,
        "model_path": str(best_model_path)
    })


# =========================
# SAVE RESULTS
# =========================
with open(RESULTS_FILE, mode="w", newline="") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=["model_name", "model_file", "test_accuracy", "model_path"]
    )
    writer.writeheader()
    writer.writerows(results)

print("\n==============================")
print("YOLO Experiments Completed")
print("==============================")
print(f"Results saved to: {RESULTS_FILE}")

for r in results:
    print(r)