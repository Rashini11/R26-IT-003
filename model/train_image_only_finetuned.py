import os
import json
import random
import copy

import torch
import torch.nn as nn
import torchvision.models as models

from torchvision.models import MobileNet_V2_Weights
from torchvision import transforms
from PIL import Image

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)


# ============================================================
# SETTINGS
# ============================================================

TRAIN_DIR = "data/raw/images/train"
VAL_DIR = "data/raw/images/validation"
TEST_DIR = "data/raw/images/test"

STAGE1_MODEL_PATH = (
    "model/image_only_model_finetuned_stage1.pth"
)

FINAL_MODEL_PATH = (
    "model/image_only_model_finetuned.pth"
)

REPORT_PATH = (
    "model/image_only_finetuned_report.txt"
)

TRAIN_SIZE = 5040
VAL_SIZE = 1080
TEST_SIZE = 1080

BATCH_SIZE = 32

# Keep total maximum training comparable to baseline:
# 3 classifier-training epochs + 5 fine-tuning epochs = 8.
STAGE1_EPOCHS = 3
STAGE2_EPOCHS = 5

# Stage 1:
# Train only new classifier.
STAGE1_LR = 0.001

# Stage 2:
# Smaller LR for pretrained feature layers.
STAGE2_BACKBONE_LR = 0.00001
STAGE2_CLASSIFIER_LR = 0.0001

PATIENCE = 2
SEED = 42

CLASSES = [
    "calm",
    "moderate",
    "rough",
    "very_rough",
]

label_map = {
    "calm": 0,
    "moderate": 1,
    "rough": 2,
    "very_rough": 3,
}

VALID_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".JPG",
    ".JPEG",
    ".PNG",
    ".BMP",
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Using device:", device)


# ============================================================
# DATA COLLECTION
# ============================================================

def collect_images(root_dir):

    records = []

    print("\nScanning:", root_dir)

    for class_name in CLASSES:

        class_dir = os.path.join(
            root_dir,
            class_name,
        )

        if not os.path.isdir(class_dir):
            raise FileNotFoundError(
                f"Missing class folder: {class_dir}"
            )

        class_images = []

        for root, _, files in os.walk(
            class_dir
        ):

            for filename in files:

                if filename.endswith(
                    VALID_EXTENSIONS
                ):

                    class_images.append(
                        os.path.join(
                            root,
                            filename,
                        )
                    )

        print(
            f"{class_name}: "
            f"{len(class_images)} images"
        )

        for image_path in class_images:

            records.append(
                (
                    image_path,
                    label_map[class_name],
                    class_name,
                )
            )

    return records


# ============================================================
# BALANCED SAMPLING
# ============================================================

def balanced_sample(
    records,
    total_size,
    seed,
):

    rng = random.Random(seed)

    samples_per_class = (
        total_size
        // len(CLASSES)
    )

    selected = []

    for class_name in CLASSES:

        class_records = [
            record
            for record in records
            if record[2] == class_name
        ]

        print(
            f"{class_name}: "
            f"available={len(class_records)}, "
            f"required={samples_per_class}"
        )

        if (
            len(class_records)
            < samples_per_class
        ):
            raise ValueError(
                f"Not enough images for "
                f"{class_name}"
            )

        selected.extend(
            rng.sample(
                class_records,
                samples_per_class,
            )
        )

    rng.shuffle(selected)

    return selected


train_pool = collect_images(
    TRAIN_DIR
)

val_pool = collect_images(
    VAL_DIR
)

test_pool = collect_images(
    TEST_DIR
)


print(
    "\nCreating balanced samples..."
)

train_records = balanced_sample(
    train_pool,
    TRAIN_SIZE,
    SEED,
)

val_records = balanced_sample(
    val_pool,
    VAL_SIZE,
    SEED + 1,
)

test_records = balanced_sample(
    test_pool,
    TEST_SIZE,
    SEED + 2,
)


print("\n========================================")
print("EXPERIMENT DATASET")
print("========================================")
print("Train:", len(train_records))
print("Validation:", len(val_records))
print("Test:", len(test_records))
print(
    "Total:",
    len(train_records)
    + len(val_records)
    + len(test_records),
)


# ============================================================
# CHECK FOR DATA LEAKAGE
# ============================================================

train_paths = {
    os.path.abspath(x[0])
    for x in train_records
}

val_paths = {
    os.path.abspath(x[0])
    for x in val_records
}

test_paths = {
    os.path.abspath(x[0])
    for x in test_records
}


train_val_overlap = (
    train_paths & val_paths
)

train_test_overlap = (
    train_paths & test_paths
)

val_test_overlap = (
    val_paths & test_paths
)


print(
    "\nTrain / Validation overlap:",
    len(train_val_overlap),
)

print(
    "Train / Test overlap:",
    len(train_test_overlap),
)

print(
    "Validation / Test overlap:",
    len(val_test_overlap),
)


if (
    train_val_overlap
    or train_test_overlap
    or val_test_overlap
):
    raise RuntimeError(
        "Dataset leakage detected."
    )


# ============================================================
# IMAGE PREPROCESSING
#
# Same preprocessing as baseline.
# No new augmentation is added so that we isolate the effect
# of the fine-tuning strategy itself.
# ============================================================

train_transform = transforms.Compose([
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


eval_transform = transforms.Compose([
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

class SeaImageDataset(
    torch.utils.data.Dataset
):

    def __init__(
        self,
        records,
        transform,
    ):

        self.records = records
        self.transform = transform


    def __len__(self):
        return len(
            self.records
        )


    def __getitem__(
        self,
        idx,
    ):

        image_path, label, _ = (
            self.records[idx]
        )

        try:

            image = Image.open(
                image_path
            ).convert(
                "RGB"
            )

            image = self.transform(
                image
            )

        except Exception as e:

            print(
                "Image loading error:",
                image_path,
                e,
            )

            new_idx = (
                idx + 1
            ) % len(
                self.records
            )

            return self.__getitem__(
                new_idx
            )

        label_tensor = torch.tensor(
            label,
            dtype=torch.long,
        )

        return (
            image,
            label_tensor,
        )


train_dataset = SeaImageDataset(
    train_records,
    train_transform,
)

val_dataset = SeaImageDataset(
    val_records,
    eval_transform,
)

test_dataset = SeaImageDataset(
    test_records,
    eval_transform,
)


train_loader = (
    torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )
)

val_loader = (
    torch.utils.data.DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )
)

test_loader = (
    torch.utils.data.DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )
)


# ============================================================
# MODEL
# ============================================================

class ImageOnlyMobileNet(
    nn.Module
):

    def __init__(
        self,
        num_classes,
    ):

        super().__init__()

        self.cnn = models.mobilenet_v2(
            weights=(
                MobileNet_V2_Weights.DEFAULT
            )
        )

        self.cnn.classifier[1] = (
            nn.Linear(
                1280,
                num_classes,
            )
        )


    def forward(
        self,
        image,
    ):

        return self.cnn(
            image
        )


model = ImageOnlyMobileNet(
    num_classes=len(
        CLASSES
    )
).to(
    device
)


criterion = (
    nn.CrossEntropyLoss()
)


# ============================================================
# EVALUATION
# ============================================================

def evaluate(loader):

    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():

        for (
            images,
            labels_batch,
        ) in loader:

            images = images.to(
                device
            )

            labels_batch = (
                labels_batch.to(
                    device
                )
            )

            outputs = model(
                images
            )

            loss = criterion(
                outputs,
                labels_batch,
            )

            total_loss += (
                loss.item()
                * labels_batch.size(0)
            )

            predicted = torch.argmax(
                outputs,
                dim=1,
            )

            correct += (
                predicted
                == labels_batch
            ).sum().item()

            total += (
                labels_batch.size(0)
            )

            all_preds.extend(
                predicted
                .cpu()
                .numpy()
                .tolist()
            )

            all_labels.extend(
                labels_batch
                .cpu()
                .numpy()
                .tolist()
            )

    return (
        total_loss / total,
        correct / total,
        all_labels,
        all_preds,
    )


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(
    optimizer,
    stage,
):

    model.train()

    # During Stage 1 the complete feature extractor is frozen.
    # Keep BatchNorm layers in evaluation mode as well.
    if stage == 1:
        model.cnn.features.eval()

    # During Stage 2 only the last three MobileNetV2
    # feature blocks are trainable.
    if stage == 2:

        for block in (
            model.cnn.features[:-3]
        ):
            block.eval()

        for block in (
            model.cnn.features[-3:]
        ):
            block.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for (
        images,
        labels_batch,
    ) in train_loader:

        images = images.to(
            device
        )

        labels_batch = (
            labels_batch.to(
                device
            )
        )

        optimizer.zero_grad()

        outputs = model(
            images
        )

        loss = criterion(
            outputs,
            labels_batch,
        )

        loss.backward()

        optimizer.step()

        total_loss += (
            loss.item()
            * labels_batch.size(0)
        )

        predicted = torch.argmax(
            outputs,
            dim=1,
        )

        correct += (
            predicted
            == labels_batch
        ).sum().item()

        total += (
            labels_batch.size(0)
        )

    return (
        total_loss / total,
        correct / total,
    )


# ============================================================
# STAGE 1
# FEATURE EXTRACTION:
# Freeze complete MobileNet backbone and train classifier only.
# ============================================================

print("\n")
print("========================================")
print("STAGE 1 - CLASSIFIER TRAINING")
print("========================================")

for parameter in model.parameters():
    parameter.requires_grad = False

for parameter in (
    model.cnn.classifier.parameters()
):
    parameter.requires_grad = True


stage1_trainable = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print(
    "Stage 1 trainable parameters:",
    f"{stage1_trainable:,}",
)


stage1_optimizer = torch.optim.Adam(
    model.cnn.classifier.parameters(),
    lr=STAGE1_LR,
)


stage1_best_loss = float(
    "inf"
)

stage1_best_accuracy = 0.0

stage1_history = []


for epoch in range(
    STAGE1_EPOCHS
):

    (
        train_loss,
        train_accuracy,
    ) = train_one_epoch(
        stage1_optimizer,
        stage=1,
    )

    (
        val_loss,
        val_accuracy,
        _,
        _,
    ) = evaluate(
        val_loader
    )

    print(
        f"\nStage 1 Epoch "
        f"{epoch + 1}/{STAGE1_EPOCHS}"
    )

    print(
        f"Train Loss: "
        f"{train_loss:.4f}"
    )

    print(
        f"Train Accuracy: "
        f"{train_accuracy * 100:.2f}%"
    )

    print(
        f"Validation Loss: "
        f"{val_loss:.4f}"
    )

    print(
        f"Validation Accuracy: "
        f"{val_accuracy * 100:.2f}%"
    )

    stage1_history.append({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "val_loss": val_loss,
        "val_accuracy": val_accuracy,
    })

    if val_loss < stage1_best_loss:

        stage1_best_loss = val_loss
        stage1_best_accuracy = val_accuracy

        torch.save(
            model.state_dict(),
            STAGE1_MODEL_PATH,
        )

        print(
            "Best Stage 1 model saved."
        )


# Load best Stage 1 model before selective fine-tuning.
model.load_state_dict(
    torch.load(
        STAGE1_MODEL_PATH,
        map_location=device,
        weights_only=True,
    )
)


# ============================================================
# STAGE 2
# SELECTIVE FINE-TUNING:
# Unfreeze last three MobileNetV2 feature blocks.
# ============================================================

print("\n")
print("========================================")
print("STAGE 2 - SELECTIVE FINE-TUNING")
print("========================================")


# Freeze everything again first.
for parameter in model.parameters():
    parameter.requires_grad = False


# Unfreeze final 3 MobileNetV2 feature blocks.
for block in (
    model.cnn.features[-3:]
):
    for parameter in block.parameters():
        parameter.requires_grad = True


# Classifier stays trainable.
for parameter in (
    model.cnn.classifier.parameters()
):
    parameter.requires_grad = True


stage2_trainable = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

total_parameters = sum(
    p.numel()
    for p in model.parameters()
)


print(
    "Stage 2 trainable parameters:",
    f"{stage2_trainable:,}",
)

print(
    "Total model parameters:",
    f"{total_parameters:,}",
)


backbone_parameters = []

for block in (
    model.cnn.features[-3:]
):

    backbone_parameters.extend(
        list(
            block.parameters()
        )
    )


stage2_optimizer = torch.optim.Adam(
    [
        {
            "params":
                backbone_parameters,
            "lr":
                STAGE2_BACKBONE_LR,
        },
        {
            "params":
                model.cnn.classifier.parameters(),
            "lr":
                STAGE2_CLASSIFIER_LR,
        },
    ]
)


stage2_best_loss = float(
    "inf"
)

stage2_best_accuracy = 0.0

stage2_history = []

epochs_without_improvement = 0


for epoch in range(
    STAGE2_EPOCHS
):

    (
        train_loss,
        train_accuracy,
    ) = train_one_epoch(
        stage2_optimizer,
        stage=2,
    )

    (
        val_loss,
        val_accuracy,
        _,
        _,
    ) = evaluate(
        val_loader
    )

    print(
        f"\nStage 2 Epoch "
        f"{epoch + 1}/{STAGE2_EPOCHS}"
    )

    print(
        f"Train Loss: "
        f"{train_loss:.4f}"
    )

    print(
        f"Train Accuracy: "
        f"{train_accuracy * 100:.2f}%"
    )

    print(
        f"Validation Loss: "
        f"{val_loss:.4f}"
    )

    print(
        f"Validation Accuracy: "
        f"{val_accuracy * 100:.2f}%"
    )

    stage2_history.append({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "val_loss": val_loss,
        "val_accuracy": val_accuracy,
    })

    if val_loss < stage2_best_loss:

        stage2_best_loss = val_loss
        stage2_best_accuracy = val_accuracy

        epochs_without_improvement = 0

        torch.save(
            model.state_dict(),
            FINAL_MODEL_PATH,
        )

        print(
            "Best fine-tuned model saved."
        )

    else:

        epochs_without_improvement += 1

        print(
            "Validation loss did not improve."
        )

        if (
            epochs_without_improvement
            >= PATIENCE
        ):

            print(
                "Early stopping activated."
            )

            break


# ============================================================
# FINAL TEST
# ============================================================

print("\n")
print("========================================")
print("FINAL TEST - FINE-TUNED MODEL")
print("========================================")


model.load_state_dict(
    torch.load(
        FINAL_MODEL_PATH,
        map_location=device,
        weights_only=True,
    )
)


(
    test_loss,
    test_accuracy,
    y_true,
    y_pred,
) = evaluate(
    test_loader
)


report = classification_report(
    y_true,
    y_pred,
    target_names=CLASSES,
    digits=4,
)


cm = confusion_matrix(
    y_true,
    y_pred,
)


report_dict = classification_report(
    y_true,
    y_pred,
    target_names=CLASSES,
    output_dict=True,
)


macro_precision = (
    report_dict[
        "macro avg"
    ][
        "precision"
    ]
)

macro_recall = (
    report_dict[
        "macro avg"
    ][
        "recall"
    ]
)

macro_f1 = (
    report_dict[
        "macro avg"
    ][
        "f1-score"
    ]
)


print(
    "\nFine-Tuned Test Loss:",
    round(
        test_loss,
        4,
    )
)

print(
    "Fine-Tuned Test Accuracy:",
    round(
        test_accuracy * 100,
        2,
    ),
    "%",
)

print(
    "Macro Precision:",
    round(
        macro_precision * 100,
        2,
    ),
    "%",
)

print(
    "Macro Recall:",
    round(
        macro_recall * 100,
        2,
    ),
    "%",
)

print(
    "Macro F1:",
    round(
        macro_f1 * 100,
        2,
    ),
    "%",
)


print(
    "\nClassification Report:\n"
)

print(
    report
)

print(
    "\nConfusion Matrix:\n"
)

print(
    cm
)


# ============================================================
# SAVE EXPERIMENT REPORT
# ============================================================

with open(
    REPORT_PATH,
    "w",
    encoding="utf-8",
) as f:

    f.write(
        "SEA STATE STAGED FINE-TUNING REPORT\n"
    )

    f.write(
        "========================================\n\n"
    )

    f.write(
        "Baseline Test Accuracy: 83.89%\n"
    )

    f.write(
        "Baseline Validation Accuracy: 86.39%\n"
    )

    f.write(
        "Baseline Test Loss: 0.5979\n\n"
    )

    f.write(
        "Fine-Tuning Strategy:\n"
    )

    f.write(
        "Stage 1: Freeze complete MobileNetV2 "
        "feature extractor and train classifier only.\n"
    )

    f.write(
        "Stage 2: Unfreeze final 3 MobileNetV2 "
        "feature blocks and classifier.\n"
    )

    f.write(
        f"Stage 1 LR: {STAGE1_LR}\n"
    )

    f.write(
        f"Stage 2 backbone LR: "
        f"{STAGE2_BACKBONE_LR}\n"
    )

    f.write(
        f"Stage 2 classifier LR: "
        f"{STAGE2_CLASSIFIER_LR}\n\n"
    )

    f.write(
        f"Train Size: {len(train_records)}\n"
    )

    f.write(
        f"Validation Size: {len(val_records)}\n"
    )

    f.write(
        f"Test Size: {len(test_records)}\n\n"
    )

    f.write(
        "Image overlap between splits: 0\n\n"
    )

    f.write(
        f"Stage 1 Best Validation Accuracy: "
        f"{stage1_best_accuracy * 100:.2f}%\n"
    )

    f.write(
        f"Stage 2 Best Validation Accuracy: "
        f"{stage2_best_accuracy * 100:.2f}%\n"
    )

    f.write(
        f"Fine-Tuned Test Accuracy: "
        f"{test_accuracy * 100:.2f}%\n"
    )

    f.write(
        f"Fine-Tuned Test Loss: "
        f"{test_loss:.4f}\n"
    )

    f.write(
        f"Macro Precision: "
        f"{macro_precision * 100:.2f}%\n"
    )

    f.write(
        f"Macro Recall: "
        f"{macro_recall * 100:.2f}%\n"
    )

    f.write(
        f"Macro F1: "
        f"{macro_f1 * 100:.2f}%\n\n"
    )

    f.write(
        "Classification Report:\n"
    )

    f.write(
        report
    )

    f.write(
        "\nConfusion Matrix:\n"
    )

    f.write(
        str(cm)
    )


print("\n========================================")
print("EXPERIMENT COMPLETE")
print("========================================")
print(
    "Fine-tuned model:",
    FINAL_MODEL_PATH,
)

print(
    "Report:",
    REPORT_PATH,
)

print(
    "Original 83.89% model was NOT modified."
)
print("========================================")
