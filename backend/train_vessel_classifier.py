#!/usr/bin/env python3
"""
Train a Local/Foreign vessel classifier for boat/ship crops.

Dataset structure expected:
    data/classifier/train/
        local/
            *.jpg
        foreign/
            *.jpg
    data/classifier/val/
        local/
            *.jpg
        foreign/
            *.jpg

Then run:
    python train_vessel_classifier.py --data_dir data/classifier --epochs 20 --batch_size 32 --lr 1e-4 --output_dir experiments/vessel_classifier
"""

import argparse
import os
from pathlib import Path

import cv2
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.datasets import ImageFolder


class VesselDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.dataset = ImageFolder(root=root_dir, transform=transform)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


def build_model(num_classes=2, arch="resnet18"):
    if arch == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif arch == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif arch == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Unsupported arch: {arch}")
    return model


def build_transform(input_size=224):
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ColorJitter(0.1, 0.1, 0.1, 0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def validate_model(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    acc = correct / total if total else 0.0
    return acc


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dir = Path(args.data_dir) / "train"
    val_dir = Path(args.data_dir) / "val"

    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError(
            "Expected dataset folders: data_dir/train and data_dir/val with local/foreign subfolders."
        )

    train_tf = build_transform(args.input_size)
    val_tf = transforms.Compose([
        transforms.Resize((args.input_size, args.input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_ds = VesselDataset(str(train_dir), transform=train_tf)
    val_ds = VesselDataset(str(val_dir), transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    model = build_model(num_classes=2, arch=args.arch).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    os.makedirs(args.output_dir, exist_ok=True)
    best_val_acc = 0.0
    best_path = os.path.join(args.output_dir, "classifier_best.pth")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)

        train_loss = total_loss / len(train_ds)
        val_acc = validate_model(model, val_loader, device)
        scheduler.step()

        print(f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_path)
            print(f"Saved best checkpoint: {best_path}")

    print(f"Training finished. Best val accuracy: {best_val_acc:.4f}")
    print(f"Best model saved to: {best_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Local/Foreign vessel classifier")
    parser.add_argument("--data_dir", type=str, default="data/classifier", help="Parent folder containing train/val with local/foreign folders")
    parser.add_argument("--arch", type=str, default="resnet18", choices=["resnet18", "resnet50", "mobilenet_v3_small"], help="Backbone architecture")
    parser.add_argument("--input_size", type=int, default=224, help="Input image size")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--workers", type=int, default=2, help="DataLoader workers")
    parser.add_argument("--output_dir", type=str, default="experiments/vessel_classifier", help="Output checkpoint directory")
    args = parser.parse_args()

    train(args)
