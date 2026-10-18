from pathlib import Path
from collections import defaultdict
import csv
import hashlib
import json

import torch
from PIL import Image, ImageOps
from torch import nn
from torchvision.models import (
    mobilenet_v3_small,
    MobileNet_V3_Small_Weights,
)
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

ROOT = Path(__file__).resolve().parents[2]
FOLDER = ROOT / "ml/models/image_router_v1"
MANIFEST = ROOT / "ml/chatbot/_router_v1/manifest.csv"

def main():
    metrics_path = FOLDER / "test_metrics.json"
    if metrics_path.exists():
        raise SystemExit(
            f"Evaluation already exists: {metrics_path}\n"
            "Share the saved results; do not overwrite them."
        )

    checkpoint = torch.load(
        FOLDER / "image_router_best.pth",
        map_location="cpu",
        weights_only=True,
    )
    metadata = checkpoint["metadata"]
    classes = metadata["classes"]

    manifest_hash = hashlib.sha256(MANIFEST.read_bytes()).hexdigest()
    if manifest_hash != metadata["manifest_sha256"]:
        raise SystemExit("Manifest changed since training. Evaluation stopped.")

    with MANIFEST.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    for field in ("group", "pixel_hash"):
        assignments = defaultdict(set)
        for row in rows:
            assignments[row[field]].add(row["split"])
        if any(len(splits) > 1 for splits in assignments.values()):
            raise SystemExit(f"Cross-split overlap detected: {field}")

    test_rows = [row for row in rows if row["split"] == "test"]
    if not test_rows:
        raise SystemExit("No test images found.")

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )

    # Reconstruct architecture without downloading pretrained weights.
    model = mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(
        model.classifier[3].in_features, len(classes)
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()

    preprocess = MobileNet_V3_Small_Weights.IMAGENET1K_V1.transforms()

    actual = []
    predicted = []
    records = []

    with torch.inference_mode():
        for start in range(0, len(test_rows), 16):
            batch = test_rows[start:start + 16]
            tensors = []

            for row in batch:
                with Image.open(ROOT / row["path"]) as source:
                    image = ImageOps.exif_transpose(source).convert("RGB")
                    image.load()

                    digest = hashlib.sha256(
                        str(image.size).encode() + image.tobytes()
                    ).hexdigest()
                    if digest != row["pixel_hash"]:
                        raise RuntimeError(
                            f"Test image changed: {row['path']}"
                        )

                    tensors.append(preprocess(image))

            probabilities = model(
                torch.stack(tensors).to(device)
            ).softmax(dim=1).cpu()

            for row, scores in zip(batch, probabilities):
                target = classes.index(row["label"])
                prediction = int(scores.argmax().item())

                actual.append(target)
                predicted.append(prediction)

                records.append({
                    "path": row["path"],
                    "group": row["group"],
                    "actual": row["label"],
                    "predicted": classes[prediction],
                    "correct": target == prediction,
                    "top_score": float(scores[prediction]),
                    **{
                        f"score_{name}": float(scores[index])
                        for index, name in enumerate(classes)
                    },
                })

    indexes = list(range(len(classes)))
    report = classification_report(
        actual, predicted,
        labels=indexes,
        target_names=classes,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(actual, predicted, labels=indexes)
    accuracy = accuracy_score(actual, predicted)

    result = {
        "model_version": metadata["model_version"],
        "classes": classes,
        "test_samples": len(test_rows),
        "accuracy": accuracy,
        "macro_f1": report["macro avg"]["f1-score"],
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
        "manifest_sha256": manifest_hash,
        "decision_rule": "five_class_argmax",
        "limitations": metadata["limitations"],
    }

    metrics_path.write_text(json.dumps(result, indent=2))

    with (FOLDER / "test_predictions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    print("\nIMAGE ROUTER — HELD-OUT TEST")
    print("Device:", device)
    print("Test samples:", len(test_rows))
    print("Class order:", classes)
    print()
    print(classification_report(
        actual, predicted,
        labels=indexes,
        target_names=classes,
        digits=4,
        zero_division=0,
    ))
    print("Confusion matrix: actual rows / predicted columns")
    print(matrix)
    print(f"\nTest accuracy: {accuracy * 100:.2f}%")
    print(f"Macro F1: {report['macro avg']['f1-score']:.4f}")

    print("\nMISCLASSIFIED IMAGES")
    for record in records:
        if not record["correct"]:
            print(
                f"{record['actual']} -> {record['predicted']} | "
                f"{record['top_score'] * 100:.2f}% | {record['path']}"
            )

    print("\nSaved:", metrics_path)
    print("Checkpoint and application routing were not changed.")

if __name__ == "__main__":
    main()
