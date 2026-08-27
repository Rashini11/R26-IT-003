from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = (
    PROJECT_ROOT
    / "ml"
    / "dataset_v3_grouped"
    / "val"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "ml"
    / "models"
    / "v3_runs"
    / "deepercnn"
    / "deepercnn_v3_grouped_best.pth"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "ml"
    / "evaluation"
    / "v3_validation"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

CLASS_NAMES = [
    "bird",
    "ship",
    "unknown",
]

IMAGE_SIZE = 128
BATCH_SIZE = 32


if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


class DeeperCNN(nn.Module):

    def __init__(
        self,
        num_classes=3,
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(
            3, 16, 3, padding=1
        )

        self.conv2 = nn.Conv2d(
            16, 32, 3, padding=1
        )

        self.conv3 = nn.Conv2d(
            32, 64, 3, padding=1
        )

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(
            64 * 16 * 16,
            256,
        )

        self.fc2 = nn.Linear(
            256,
            num_classes,
        )


    def forward(self, x):

        x = self.pool(
            F.relu(self.conv1(x))
        )

        x = self.pool(
            F.relu(self.conv2(x))
        )

        x = self.pool(
            F.relu(self.conv3(x))
        )

        x = x.view(
            x.size(0),
            -1,
        )

        x = F.relu(
            self.fc1(x)
        )

        return self.fc2(x)


transform = transforms.Compose([
    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),
    transforms.ToTensor(),
])


dataset = datasets.ImageFolder(
    DATA_DIR,
    transform=transform,
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)


model = DeeperCNN(
    num_classes=3
).to(device)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True,
    )
)

model.eval()


actual = []
predicted = []


with torch.inference_mode():

    for images, labels in loader:

        images = images.to(device)

        outputs = model(images)

        predictions = outputs.argmax(
            dim=1
        )

        actual.extend(
            labels.numpy().tolist()
        )

        predicted.extend(
            predictions
            .cpu()
            .numpy()
            .tolist()
        )


print()
print("=" * 60)
print("V3 VALIDATION CLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        actual,
        predicted,
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
    )
)


matrix = confusion_matrix(
    actual,
    predicted,
)


print("Confusion matrix:")
print(matrix)


figure, axis = plt.subplots(
    figsize=(7, 6)
)

image = axis.imshow(
    matrix
)

axis.set_title(
    "DeeperCNN V3 Validation Confusion Matrix"
)

axis.set_xlabel(
    "Predicted"
)

axis.set_ylabel(
    "Actual"
)

axis.set_xticks(
    range(3)
)

axis.set_yticks(
    range(3)
)

axis.set_xticklabels(
    CLASS_NAMES
)

axis.set_yticklabels(
    CLASS_NAMES
)


for row in range(3):

    for column in range(3):

        axis.text(
            column,
            row,
            str(
                matrix[
                    row,
                    column
                ]
            ),
            ha="center",
            va="center",
        )


figure.colorbar(
    image,
    ax=axis,
)

figure.tight_layout()

output_path = (
    OUTPUT_DIR
    / "deepercnn_validation_confusion_matrix.png"
)

figure.savefig(
    output_path,
    dpi=250,
)

plt.close(
    figure
)


print()
print(
    "Saved:",
    output_path,
)

print("=" * 60)
