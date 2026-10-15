from pathlib import Path
import copy
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from torchvision.models import MobileNet_V2_Weights


# ============================================================
# SETTINGS
# ============================================================

SEED = 42

DATA_DIR = Path("data/demo_finetune")

CURRENT_MODEL = Path(
    "model/image_only_model.pth"
)

OUTPUT_MODEL = Path(
    "model/image_only_model_demo.pth"
)

EXPECTED_CLASSES = [
    "calm",
    "moderate",
    "rough",
    "very_rough",
]

BATCH_SIZE = 16
EPOCHS = 15
LEARNING_RATE = 0.001
VAL_RATIO = 0.20


# ============================================================
# RANDOM SEED
# ============================================================

random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print("Using device:", device)


# ============================================================
# MODEL
# ============================================================

class ImageOnlyMobileNet(nn.Module):

    def __init__(self, num_classes=4):
        super().__init__()

        self.cnn = models.mobilenet_v2(
            weights=MobileNet_V2_Weights.DEFAULT
        )

        self.cnn.classifier[1] = nn.Linear(
            1280,
            num_classes,
        )

    def forward(self, image):
        return self.cnn(image)


model = ImageOnlyMobileNet(
    num_classes=4
).to(device)

print("\nUsing fresh ImageNet-pretrained MobileNetV2")
print("Research model remains untouched.")


# ============================================================
# FREEZE COMPLETE FEATURE EXTRACTOR
# ============================================================

# Freeze the entire existing model.
for parameter in model.parameters():
    parameter.requires_grad = False

# Replace only the final 4-class classifier.
# This removes the old external-image class bias while
# preserving the MobileNet features learned previously.
model.cnn.classifier[1] = nn.Linear(
    1280,
    4,
).to(device)

# Train only the classifier.
for parameter in model.cnn.classifier.parameters():
    parameter.requires_grad = True


# ============================================================
# PARAMETER CHECK
# ============================================================

trainable_parameters = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

total_parameters = sum(
    p.numel()
    for p in model.parameters()
)

print(
    f"Trainable parameters: "
    f"{trainable_parameters:,} / {total_parameters:,}"
)

# We expect only the classifier to be trainable.
if trainable_parameters > 10000:
    raise RuntimeError(
        "Too many trainable parameters. "
        "The MobileNet backbone may not be fully frozen."
    )


# ============================================================
# TRAINING AUGMENTATION
# ============================================================

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(
        224,
        scale=(0.70, 1.0),
    ),

    transforms.RandomHorizontalFlip(),

    transforms.ColorJitter(
        brightness=0.25,
        contrast=0.25,
        saturation=0.20,
        hue=0.04,
    ),

    transforms.RandomRotation(5),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406,
        ],
        std=[
            0.229,
            0.224,
            0.225,
        ],
    ),
])


# ============================================================
# VALIDATION TRANSFORM
# ============================================================

validation_transform = transforms.Compose([
    transforms.Resize(
        (224, 224)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406,
        ],
        std=[
            0.229,
            0.224,
            0.225,
        ],
    ),
])


# ============================================================
# DATASET
# ============================================================

raw_dataset = datasets.ImageFolder(
    DATA_DIR
)

print("\nDetected classes:")
print(raw_dataset.classes)


if raw_dataset.classes != EXPECTED_CLASSES:
    raise RuntimeError(
        "\nClass folder order does not match the model.\n"
        f"Expected: {EXPECTED_CLASSES}\n"
        f"Found:    {raw_dataset.classes}"
    )


train_full = datasets.ImageFolder(
    DATA_DIR,
    transform=train_transform,
)

validation_full = datasets.ImageFolder(
    DATA_DIR,
    transform=validation_transform,
)


# ============================================================
# CLASS-BALANCED TRAIN / VALIDATION SPLIT
# ============================================================

train_indices = []
validation_indices = []


for class_index in range(
    len(EXPECTED_CLASSES)
):

    indices = [
        index
        for index, (_, target)
        in enumerate(raw_dataset.samples)
        if target == class_index
    ]

    random.shuffle(indices)

    validation_count = max(
        1,
        round(
            len(indices)
            * VAL_RATIO
        )
    )

    validation_indices.extend(
        indices[:validation_count]
    )

    train_indices.extend(
        indices[validation_count:]
    )


train_dataset = Subset(
    train_full,
    train_indices,
)

validation_dataset = Subset(
    validation_full,
    validation_indices,
)


print(
    "\nTraining images:",
    len(train_dataset)
)

print(
    "Validation images:",
    len(validation_dataset)
)


# ============================================================
# DATA LOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)


# ============================================================
# OPTIMIZER
# ============================================================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    filter(
        lambda p: p.requires_grad,
        model.parameters(),
    ),
    lr=LEARNING_RATE,
)


best_model_state = copy.deepcopy(
    model.state_dict()
)

best_validation_accuracy = 0.0


# ============================================================
# TRAINING
# ============================================================

for epoch in range(EPOCHS):

    model.train()

    # Important:
    # Keep the frozen MobileNet backbone in eval mode so its
    # BatchNorm statistics do not change on this tiny dataset.
    model.cnn.features.eval()

    training_correct = 0
    training_total = 0
    training_loss_sum = 0.0


    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels,
        )

        loss.backward()

        optimizer.step()


        training_loss_sum += (
            loss.item()
            * labels.size(0)
        )

        predictions = outputs.argmax(
            dim=1
        )

        training_correct += (
            predictions == labels
        ).sum().item()

        training_total += labels.size(0)


    training_loss = (
        training_loss_sum
        / training_total
    )

    training_accuracy = (
        100
        * training_correct
        / training_total
    )


    # ========================================================
    # VALIDATION
    # ========================================================

    model.eval()

    validation_correct = 0
    validation_total = 0
    validation_loss_sum = 0.0


    with torch.no_grad():

        for images, labels in validation_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels,
            )

            validation_loss_sum += (
                loss.item()
                * labels.size(0)
            )

            predictions = outputs.argmax(
                dim=1
            )

            validation_correct += (
                predictions == labels
            ).sum().item()

            validation_total += labels.size(0)


    validation_loss = (
        validation_loss_sum
        / validation_total
    )

    validation_accuracy = (
        100
        * validation_correct
        / validation_total
    )


    print("\n======================================")

    print(
        f"Epoch {epoch + 1}/{EPOCHS}"
    )

    print(
        f"Train Loss: "
        f"{training_loss:.4f}"
    )

    print(
        f"Train Accuracy: "
        f"{training_accuracy:.2f}%"
    )

    print(
        f"Validation Loss: "
        f"{validation_loss:.4f}"
    )

    print(
        f"Validation Accuracy: "
        f"{validation_accuracy:.2f}%"
    )


    if (
        validation_accuracy
        > best_validation_accuracy
    ):

        best_validation_accuracy = (
            validation_accuracy
        )

        best_model_state = copy.deepcopy(
            model.state_dict()
        )

        print(
            "New best demo model checkpoint."
        )


# ============================================================
# SAVE SEPARATE DEMO MODEL
# ============================================================

torch.save(
    best_model_state,
    OUTPUT_MODEL,
)


print("\n======================================")
print("DEMO FINE-TUNING COMPLETE")
print("======================================")

print(
    "Best external validation accuracy:",
    f"{best_validation_accuracy:.2f}%"
)

print(
    "\nDemo model saved:"
)

print(
    OUTPUT_MODEL
)

print(
    "\nOriginal model remains unchanged:"
)

print(
    CURRENT_MODEL
)

print("======================================")