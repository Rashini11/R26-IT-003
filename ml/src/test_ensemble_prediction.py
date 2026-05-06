from pathlib import Path
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from ultralytics import YOLO
import matplotlib.pyplot as plt

# =========================
# IMAGE TO TEST
# =========================
RAW_IMAGE_PATH = "ml/dataset/test/custom/000121_jpg.rf.887057cf040ac378dfdcec9e4d484a5e.jpg"
HEATMAP_IMAGE_PATH = "ml/dataset/test/custom/temp_heatmap_input.png"

# =========================
# MODEL PATHS
# =========================
YOLO_MODEL_PATH = "runs/classify/ml/models/yolo_runs/YOLO11_Medium/weights/best.pt"
CNN_MODEL_PATH = "ml/models/DeeperCNN_best.pth"

CLASS_NAMES = ["bird", "ship", "unknown"]

# =========================
# DEVICE
# =========================
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

# =========================
# CONVERT RAW IMAGE TO HEATMAP STYLE
# =========================
raw_image = Image.open(RAW_IMAGE_PATH).convert("L")
raw_image = raw_image.resize((128, 128))

plt.imshow(raw_image, cmap="viridis")
plt.axis("off")
plt.savefig(HEATMAP_IMAGE_PATH, bbox_inches="tight", pad_inches=0)
plt.close()

print("Converted raw radar image to heatmap:", HEATMAP_IMAGE_PATH)

# =========================
# CNN MODEL
# =========================
class DeeperCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(DeeperCNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(64 * 16 * 16, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

cnn_model = DeeperCNN(num_classes=3)
cnn_model.load_state_dict(torch.load(CNN_MODEL_PATH, map_location=device))
cnn_model.to(device)
cnn_model.eval()

# =========================
# YOLO MODEL
# =========================
yolo_model = YOLO(YOLO_MODEL_PATH)

# =========================
# CNN PREDICTION
# =========================
image = Image.open(HEATMAP_IMAGE_PATH).convert("RGB")
input_tensor = transform(image).unsqueeze(0).to(device)

with torch.no_grad():
    cnn_output = cnn_model(input_tensor)
    cnn_probs = torch.softmax(cnn_output, dim=1)
    cnn_confidence, cnn_pred = torch.max(cnn_probs, 1)

cnn_label = CLASS_NAMES[cnn_pred.item()]
cnn_conf = cnn_confidence.item() * 100

# =========================
# YOLO PREDICTION
# =========================
yolo_results = yolo_model(HEATMAP_IMAGE_PATH)
yolo_probs = yolo_results[0].probs

yolo_pred_index = int(yolo_probs.top1)
yolo_conf = float(yolo_probs.top1conf) * 100
yolo_label = yolo_results[0].names[yolo_pred_index]

# =========================
# FINAL DECISION LOGIC
# =========================
if yolo_label == cnn_label:
    final_label = yolo_label
    decision_status = "High confidence - both models agree"
else:
    final_label = "uncertain"
    decision_status = "Models disagree - manual review required"

# =========================
# OUTPUT
# =========================
print("\n===== MODEL PREDICTIONS =====")
print(f"YOLO Prediction: {yolo_label} ({yolo_conf:.2f}%)")
print(f"CNN Prediction : {cnn_label} ({cnn_conf:.2f}%)")

print("\n===== FINAL DECISION =====")
print(f"Final Prediction: {final_label}")
print(f"Decision Status : {decision_status}")