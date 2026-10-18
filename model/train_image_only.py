import os
import json
import random

import torch
import torch.nn as nn
import torchvision.models as models

from torchvision.models import MobileNet_V2_Weights
from torchvision import transforms
from PIL import Image

from sklearn.metrics import classification_report, confusion_matrix


# =====================================================
# SETTINGS
# =====================================================

TRAIN_DIR = "data/raw/images/train"
VAL_DIR = "data/raw/images/validation"
TEST_DIR = "data/raw/images/test"

MODEL_SAVE_PATH = "model/image_only_model.pth"
REPORT_SAVE_PATH = "model/image_only_report.txt"
LABEL_MAP_SAVE_PATH = "model/label_map.json"

# Keep the same experimental sizes as before:
# 5040 train + 1080 validation + 1080 test = 7200
TRAIN_SIZE = 5040
VAL_SIZE = 1080
TEST_SIZE = 1080

BATCH_SIZE = 32

EPOCHS = 8

LEARNING_RATE = 0.0001

SEED = 42

CLASSES = [
    "calm",
    "moderate",
    "rough",
    "very_rough"
]

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

random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Using device:", device)


# =====================================================
# LABEL MAPPING
# =====================================================

label_map = {
    "calm": 0,
    "moderate": 1,
    "rough": 2,
    "very_rough": 3,
}

reverse_label_map = {
    value: key
    for key, value in label_map.items()
}

os.makedirs(
    "model",
    exist_ok=True
)

with open(
    LABEL_MAP_SAVE_PATH,
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        label_map,
        f,
        indent=4
    )

print(
    "Label map:",
    label_map
)


# =====================================================
# LOAD IMAGE PATHS FROM REAL DATASET SPLITS
# =====================================================

def collect_images(root_dir):

    records = []

    print(
        f"\nScanning: {root_dir}"
    )

    for class_name in CLASSES:

        class_dir = os.path.join(
            root_dir,
            class_name
        )

        if not os.path.isdir(
            class_dir
        ):
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

                    image_path = os.path.join(
                        root,
                        filename
                    )

                    class_images.append(
                        image_path
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
                    class_name
                )
            )

    return records


train_pool = collect_images(
    TRAIN_DIR
)

val_pool = collect_images(
    VAL_DIR
)

test_pool = collect_images(
    TEST_DIR
)


# =====================================================
# BALANCED SAMPLING
# =====================================================

def balanced_sample(
    records,
    total_size,
    seed
):

    rng = random.Random(
        seed
    )

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
                f"{class_name}. "
                f"Need {samples_per_class}, "
                f"found {len(class_records)}."
            )

        selected.extend(
            rng.sample(
                class_records,
                samples_per_class
            )
        )

    rng.shuffle(
        selected
    )

    return selected


print(
    "\nCreating balanced training sample..."
)

train_records = balanced_sample(
    train_pool,
    TRAIN_SIZE,
    SEED
)


print(
    "\nCreating balanced validation sample..."
)

val_records = balanced_sample(
    val_pool,
    VAL_SIZE,
    SEED + 1
)


print(
    "\nCreating balanced testing sample..."
)

test_records = balanced_sample(
    test_pool,
    TEST_SIZE,
    SEED + 2
)


print(
    "\n========================================"
)

print(
    "FINAL EXPERIMENT SIZES"
)

print(
    "Train:",
    len(train_records)
)

print(
    "Validation:",
    len(val_records)
)

print(
    "Test:",
    len(test_records)
)

print(
    "Total:",
    (
        len(train_records)
        + len(val_records)
        + len(test_records)
    )
)

print(
    "========================================"
)


# =====================================================
# VERIFY SPLITS ARE COMPLETELY SEPARATE
# =====================================================

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
    train_paths
    & val_paths
)

train_test_overlap = (
    train_paths
    & test_paths
)

val_test_overlap = (
    val_paths
    & test_paths
)


print(
    "\nTrain / Validation image overlap:",
    len(train_val_overlap)
)

print(
    "Train / Test image overlap:",
    len(train_test_overlap)
)

print(
    "Validation / Test image overlap:",
    len(val_test_overlap)
)


if (
    train_val_overlap
    or train_test_overlap
    or val_test_overlap
):
    raise RuntimeError(
        "Dataset leakage detected."
    )


# =====================================================
# IMAGE TRANSFORMS
# =====================================================

# MobileNetV2 was pretrained using ImageNet.
# Therefore use ImageNet normalization.

train_transform = transforms.Compose(
    [
        transforms.Resize(
            (224, 224)
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[
                0.485,
                0.456,
                0.406
            ],
            std=[
                0.229,
                0.224,
                0.225
            ]
        ),
    ]
)


eval_transform = transforms.Compose(
    [
        transforms.Resize(
            (224, 224)
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[
                0.485,
                0.456,
                0.406
            ],
            std=[
                0.229,
                0.224,
                0.225
            ]
        ),
    ]
)


# =====================================================
# DATASET CLASS
# =====================================================

class SeaImageDataset(
    torch.utils.data.Dataset
):

    def __init__(
        self,
        records,
        transform
    ):

        self.records = records

        self.transform = transform


    def __len__(
        self
    ):

        return len(
            self.records
        )


    def __getitem__(
        self,
        idx
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
                image_path
            )

            print(
                e
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
            dtype=torch.long
        )

        return (
            image,
            label_tensor
        )


# =====================================================
# DATA LOADERS
# =====================================================

train_dataset = SeaImageDataset(
    train_records,
    train_transform
)

val_dataset = SeaImageDataset(
    val_records,
    eval_transform
)

test_dataset = SeaImageDataset(
    test_records,
    eval_transform
)


train_loader = (
    torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )
)


val_loader = (
    torch.utils.data.DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )
)


test_loader = (
    torch.utils.data.DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )
)


# =====================================================
# MOBILENETV2 MODEL
# SAME ARCHITECTURE AS OCEANIQ MAIN.PY
# =====================================================

class ImageOnlyMobileNet(
    nn.Module
):

    def __init__(
        self,
        num_classes
    ):

        super().__init__()

        self.cnn = (
            models.mobilenet_v2(
                weights=(
                    MobileNet_V2_Weights.DEFAULT
                )
            )
        )

        self.cnn.classifier[1] = (
            nn.Linear(
                1280,
                num_classes
            )
        )


    def forward(
        self,
        image
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


# =====================================================
# LOSS + OPTIMIZER
# =====================================================

criterion = (
    nn.CrossEntropyLoss()
)


optimizer = (
    torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )
)


# =====================================================
# EVALUATION
# =====================================================

def evaluate(
    loader
):

    model.eval()

    total_loss = 0.0

    correct = 0

    total = 0

    all_preds = []

    all_labels = []


    with torch.no_grad():

        for (
            images,
            labels_batch
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
                labels_batch
            )

            total_loss += (
                loss.item()
                * labels_batch.size(0)
            )

            predicted = (
                torch.argmax(
                    outputs,
                    dim=1
                )
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


    average_loss = (
        total_loss
        / total
    )

    accuracy = (
        correct
        / total
    )


    return (
        average_loss,
        accuracy,
        all_labels,
        all_preds
    )


# =====================================================
# TRAINING LOOP
# =====================================================

best_val_loss = float(
    "inf"
)

best_val_accuracy = 0.0

training_history = []

PATIENCE = 2

epochs_without_improvement = 0


for epoch in range(
    EPOCHS
):

    model.train()

    total_train_loss = 0.0

    train_correct = 0

    train_total = 0


    print(
        "\n========================================"
    )

    print(
        f"Epoch {epoch + 1}/{EPOCHS}"
    )

    print(
        "========================================"
    )


    for (
        images,
        labels_batch
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
            labels_batch
        )

        loss.backward()

        optimizer.step()


        total_train_loss += (
            loss.item()
            * labels_batch.size(0)
        )


        predicted = (
            torch.argmax(
                outputs,
                dim=1
            )
        )


        train_correct += (
            predicted
            == labels_batch
        ).sum().item()


        train_total += (
            labels_batch.size(0)
        )


    train_loss = (
        total_train_loss
        / train_total
    )


    train_accuracy = (
        train_correct
        / train_total
    )


    (
        val_loss,
        val_accuracy,
        _,
        _
    ) = evaluate(
        val_loader
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


    training_history.append(
        {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
        }
    )


    # Save according to validation loss
    if (
        val_loss
        < best_val_loss
    ):

        best_val_loss = (
            val_loss
        )

        best_val_accuracy = (
            val_accuracy
        )

        epochs_without_improvement = 0


        torch.save(
            model.state_dict(),
            MODEL_SAVE_PATH
        )


        print(
            "Best model saved."
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


# =====================================================
# FINAL TEST
# =====================================================

print(
    "\n========================================"
)

print(
    "FINAL TEST ON OFFICIAL TEST FOLDER"
)

print(
    "========================================"
)


model.load_state_dict(
    torch.load(
        MODEL_SAVE_PATH,
        map_location=device,
        weights_only=True
    )
)


(
    test_loss,
    test_accuracy,
    y_true,
    y_pred
) = evaluate(
    test_loader
)


target_names = CLASSES


report = (
    classification_report(
        y_true,
        y_pred,
        target_names=target_names,
        digits=4
    )
)


cm = confusion_matrix(
    y_true,
    y_pred
)


print(
    "\nFinal Test Loss:",
    round(
        test_loss,
        4
    )
)


print(
    "Final Test Accuracy:",
    round(
        test_accuracy
        * 100,
        2
    ),
    "%"
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


# =====================================================
# SAVE REPORT
# =====================================================

with open(
    REPORT_SAVE_PATH,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "IMAGE-ONLY SEA STATE "
        "CLASSIFICATION REPORT\n"
    )

    f.write(
        "==========================================\n\n"
    )

    f.write(
        "Dataset split method:\n"
    )

    f.write(
        "Original MU-SSiD Train / "
        "Validation / Test folders\n"
    )

    f.write(
        "No random image-level "
        "train_test_split used.\n\n"
    )

    f.write(
        f"Train Size: "
        f"{len(train_records)}\n"
    )

    f.write(
        f"Validation Size: "
        f"{len(val_records)}\n"
    )

    f.write(
        f"Test Size: "
        f"{len(test_records)}\n"
    )

    f.write(
        f"Total Experimental Images: "
        f"{len(train_records) + len(val_records) + len(test_records)}\n\n"
    )


    f.write(
        "Image overlap between "
        "splits: 0\n\n"
    )


    f.write(
        f"Best Validation Accuracy: "
        f"{best_val_accuracy * 100:.2f}%\n"
    )


    f.write(
        f"Final Test Accuracy: "
        f"{test_accuracy * 100:.2f}%\n"
    )


    f.write(
        f"Final Test Loss: "
        f"{test_loss:.4f}\n\n"
    )


    f.write(
        "Training History:\n"
    )


    for item in training_history:

        f.write(
            f"Epoch {item['epoch']}: "
            f"Train Acc="
            f"{item['train_accuracy'] * 100:.2f}% | "
            f"Val Acc="
            f"{item['val_accuracy'] * 100:.2f}% | "
            f"Train Loss="
            f"{item['train_loss']:.4f} | "
            f"Val Loss="
            f"{item['val_loss']:.4f}\n"
        )


    f.write(
        "\nLabel Mapping:\n"
    )


    for (
        label,
        value
    ) in label_map.items():

        f.write(
            f"{value} = {label}\n"
        )


    f.write(
        "\nClassification Report:\n"
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


print(
    "\nReport saved to:",
    REPORT_SAVE_PATH
)

print(
    "Model saved to:",
    MODEL_SAVE_PATH
)

print(
    "Label map saved to:",
    LABEL_MAP_SAVE_PATH
)