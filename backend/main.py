import os
import json
import shutil
import torch
import torch.nn as nn
import torchvision.models as models

from torchvision.models import MobileNet_V2_Weights
from torchvision import transforms
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

# -----------------------------
# Paths
# -----------------------------
MODEL_PATH = "model/image_only_model.pth"
LABEL_MAP_PATH = "model/label_map.json"
UPLOAD_DIR = "backend/uploads"

os.makedirs(UPLOAD_DIR, exist_ok=True)

# -----------------------------
# FastAPI App
# -----------------------------
app = FastAPI(title="Sea State Classification API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Load label map
# -----------------------------
with open(LABEL_MAP_PATH, "r") as f:
    label_map = json.load(f)

reverse_label_map = {value: key for key, value in label_map.items()}

# -----------------------------
# Image Transform
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# -----------------------------
# Model Class
# -----------------------------
class ImageOnlyMobileNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.cnn = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        self.cnn.classifier[1] = nn.Linear(1280, num_classes)

    def forward(self, image):
        return self.cnn(image)

# -----------------------------
# Load Model
# -----------------------------
model = ImageOnlyMobileNet(num_classes=len(label_map)).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# -----------------------------
# Routes
# -----------------------------
@app.get("/")
def home():
    return {
        "message": "Sea State Classification API is running",
        "model": "MobileNetV2 Image-Only",
        "classes": label_map
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    image = Image.open(file_path).convert("RGB")
    image = transform(image)
    image = image.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted_class = torch.max(probabilities, 1)

    predicted_class = predicted_class.item()
    confidence = confidence.item() * 100
    predicted_label = reverse_label_map[predicted_class]

    class_probabilities = {}

    for i, prob in enumerate(probabilities[0]):
        label = reverse_label_map[i]
        class_probabilities[label] = round(prob.item() * 100, 2)

    return {
        "predicted_sea_state": predicted_label,
        "confidence": round(confidence, 2),
        "probabilities": class_probabilities
    }