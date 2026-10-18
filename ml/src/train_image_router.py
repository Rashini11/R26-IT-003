from pathlib import Path
import csv
import hashlib
import json
import random

import torch
from PIL import Image, ImageOps
from torch import nn
from torchvision.models import (
    mobilenet_v3_small,
    MobileNet_V3_Small_Weights,
)

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "ml/chatbot/_router_v1/manifest.csv"
OUT = ROOT / "ml/models/image_router_v1"
CLASSES = ["radar", "vessel", "sea", "hull", "unsupported"]
SEED = 42

def macro_f1(target, predicted):
    scores = []
    for index in range(len(CLASSES)):
        tp = ((target == index) & (predicted == index)).sum().item()
        fp = ((target != index) & (predicted == index)).sum().item()
        fn = ((target == index) & (predicted != index)).sum().item()
        denominator = 2 * tp + fp + fn
        scores.append(2 * tp / denominator if denominator else 0.0)
    return sum(scores) / len(scores)

def main():
    random.seed(SEED)
    torch.manual_seed(SEED)

    if OUT.exists() and any(OUT.iterdir()):
        raise SystemExit(
            f"Output folder is not empty: {OUT}\n"
            "Existing experiment will not be overwritten."
        )

    with MANIFEST.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    # Test rows are excluded before image loading.
    train_rows = [r for r in rows if r["split"] == "train"]
    val_rows = [r for r in rows if r["split"] == "val"]

    for split_name, selected in (
        ("train", train_rows), ("val", val_rows)
    ):
        if {r["label"] for r in selected} != set(CLASSES):
            raise SystemExit(f"A category is missing from {split_name}")

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print("Device:", device)
    print("Classes:", CLASSES)
    print("Train:", len(train_rows), "| Validation:", len(val_rows))
    print("TEST IMAGES: NOT LOADED")

    weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
    preprocess = weights.transforms()

    # First run may download pretrained ImageNet weights.
    model = mobilenet_v3_small(weights=weights)

    for parameter in model.parameters():
        parameter.requires_grad = False

    model.classifier[3] = nn.Linear(
        model.classifier[3].in_features, len(CLASSES)
    )
    model.to(device)
    model.eval()

    # Cache frozen features; no test images enter this operation.
    def extract(selected):
        feature_batches = []
        label_batches = []

        with torch.inference_mode():
            for start in range(0, len(selected), 16):
                batch = selected[start:start + 16]
                images = []

                for row in batch:
                    with Image.open(ROOT / row["path"]) as source:
                        image = ImageOps.exif_transpose(source).convert("RGB")
                        images.append(preprocess(image))

                inputs = torch.stack(images).to(device)
                features = model.features(inputs)
                features = model.avgpool(features).flatten(1)

                # Frozen classifier projection, excluding dropout/output.
                features = model.classifier[0](features)
                features = model.classifier[1](features)

                feature_batches.append(features.cpu())
                label_batches.extend(
                    CLASSES.index(row["label"]) for row in batch
                )

        # Produce ordinary tensors for output-layer training.
        features = torch.cat(feature_batches).clone()
        labels = torch.tensor(label_batches, dtype=torch.long)
        return features, labels

    print("Extracting frozen training features...")
    train_x, train_y = extract(train_rows)

    print("Extracting frozen validation features...")
    val_x, val_y = extract(val_rows)

    # Train the lightweight output layer on CPU.
    head = model.classifier[3].to("cpu")
    counts = torch.bincount(train_y, minlength=len(CLASSES)).float()
    class_weights = counts.sum() / (len(CLASSES) * counts)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=0.001, weight_decay=0.01
    )

    best_f1 = -1.0
    best_loss = float("inf")
    best_state = None
    best_epoch = 0
    best_accuracy = 0.0
    history = []

    for epoch in range(1, 61):
        head.train()
        order = torch.randperm(len(train_y))
        total_loss = 0.0

        for indexes in order.split(32):
            optimizer.zero_grad()
            logits = head(train_x[indexes])
            loss = criterion(logits, train_y[indexes])
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(indexes)

        head.eval()
        with torch.no_grad():
            logits = head(val_x)
            val_loss = criterion(logits, val_y).item()
            predicted = logits.argmax(1)
            accuracy = (predicted == val_y).float().mean().item()
            f1 = macro_f1(val_y, predicted)

        history.append({
            "epoch": epoch,
            "train_loss": total_loss / len(train_y),
            "validation_loss": val_loss,
            "validation_accuracy": accuracy,
            "validation_macro_f1": f1,
        })

        better = (
            f1 > best_f1 + 1e-9
            or (abs(f1 - best_f1) <= 1e-9 and val_loss < best_loss)
        )
        if better:
            best_f1 = f1
            best_loss = val_loss
            best_accuracy = accuracy
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in head.state_dict().items()
            }

        print(
            f"Epoch {epoch:02d} | "
            f"Loss {total_loss / len(train_y):.4f} | "
            f"Val accuracy {accuracy * 100:.2f}% | "
            f"Val macro F1 {f1:.4f}"
            + (" | BEST" if better else "")
        )

    head.load_state_dict(best_state)
    model.to("cpu")
    model.eval()

    metadata = {
        "architecture": "mobilenet_v3_small",
        "model_version": "image_router_v1",
        "classes": CLASSES,
        "class_to_index": {name: i for i, name in enumerate(CLASSES)},
        "pretrained_weights": "IMAGENET1K_V1",
        "preprocessing": {
            "exif_transpose": True,
            "color_mode": "RGB",
            "resize_shorter_edge": 256,
            "center_crop": 224,
            "interpolation": "bilinear",
            "antialias": True,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
        "seed": SEED,
        "selected_epoch": best_epoch,
        "validation_accuracy": best_accuracy,
        "validation_macro_f1": best_f1,
        "manifest_sha256": hashlib.sha256(
            MANIFEST.read_bytes()
        ).hexdigest(),
        "training_method": "frozen_features_linear_output_layer",
        "test_evaluated": False,
        "routing_threshold": None,
        "requires_confirmation": True,
        "limitations": [
            "Small development dataset.",
            "Filename-based source groups are provisional.",
            "Very limited unsupported examples and validation support.",
            "Vessel examples may not represent aerial imagery.",
            "No general open-set rejection guarantee.",
        ],
    }

    OUT.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "metadata": metadata,
        },
        OUT / "image_router_best.pth",
    )
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2))
    (OUT / "training_history.json").write_text(
        json.dumps(history, indent=2)
    )

    print("\nTRAINING COMPLETE")
    print("Selected epoch:", best_epoch)
    print(f"Validation accuracy: {best_accuracy * 100:.2f}%")
    print(f"Validation macro F1: {best_f1:.4f}")
    print("Saved:", OUT / "image_router_best.pth")
    print("TEST IMAGES WERE NOT LOADED.")
    print("Router is not yet connected to the application.")

if __name__ == "__main__":
    main()
