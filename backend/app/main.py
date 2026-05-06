from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from ultralytics import YOLO
import numpy as np
import cv2
import shutil
from pathlib import Path
import uuid

app = FastAPI(title="Radar Object Classification API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parents[2]

YOLO_MODEL_PATH = BASE_DIR / "runs/classify/ml/models/yolo_runs/YOLO11_Medium/weights/best.pt"
CNN_MODEL_PATH = BASE_DIR / "ml/models/DeeperCNN_best.pth"

TEMP_DIR = BASE_DIR / "backend/temp"
TEMP_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["bird", "ship", "unknown"]

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


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


def convert_to_heatmap(input_path: Path, output_path: Path):
    image = Image.open(input_path).convert("L")
    image = image.resize((128, 128))

    gray = np.array(image)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

    heatmap = cv2.applyColorMap(gray.astype(np.uint8), cv2.COLORMAP_VIRIDIS)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    Image.fromarray(heatmap).save(output_path)


cnn_model = DeeperCNN(num_classes=3)
cnn_model.load_state_dict(torch.load(CNN_MODEL_PATH, map_location=device))
cnn_model.to(device)
cnn_model.eval()

yolo_model = YOLO(str(YOLO_MODEL_PATH))


@app.get("/")
def root():
    return {
        "message": "Radar Object Classification API is running",
        "classes": CLASS_NAMES
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    file_id = str(uuid.uuid4())

    raw_path = TEMP_DIR / f"{file_id}_{file.filename}"
    heatmap_path = TEMP_DIR / f"{file_id}_heatmap.png"

    with open(raw_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    convert_to_heatmap(raw_path, heatmap_path)

    # CNN prediction
    image = Image.open(heatmap_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        cnn_output = cnn_model(input_tensor)
        cnn_probs = torch.softmax(cnn_output, dim=1)
        cnn_confidence, cnn_pred = torch.max(cnn_probs, 1)

    cnn_label = CLASS_NAMES[cnn_pred.item()]
    cnn_conf = round(cnn_confidence.item() * 100, 2)

    # YOLO prediction
    yolo_results = yolo_model(str(heatmap_path))
    yolo_probs = yolo_results[0].probs

    yolo_pred_index = int(yolo_probs.top1)
    yolo_conf = round(float(yolo_probs.top1conf) * 100, 2)
    yolo_label = yolo_results[0].names[yolo_pred_index]

    # Final decision
    if yolo_label == cnn_label:
        final_prediction = yolo_label
        decision_status = "High confidence - both models agree"
    else:
        final_prediction = "uncertain"
        decision_status = "Models disagree - manual review required"

    return {
        "yolo_prediction": yolo_label,
        "yolo_confidence": yolo_conf,
        "cnn_prediction": cnn_label,
        "cnn_confidence": cnn_conf,
        "final_prediction": final_prediction,
        "decision_status": decision_status
    }