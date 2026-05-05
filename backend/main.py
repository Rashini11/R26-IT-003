import os
import json
import shutil
import base64

import tensorflow as tf
import numpy as np
import cv2

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import MobileNet_V2_Weights
from torchvision import transforms
from PIL import Image

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI(title="Marine AI Inspection System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# -----------------------------
# PATHS
# -----------------------------
HULL_MODEL_PATH = os.path.join(BASE_DIR, "model", "hull_model.keras")

SEA_MODEL_PATH = os.path.join(BASE_DIR, "model", "image_only_model.pth")
LABEL_MAP_PATH = os.path.join(BASE_DIR, "model", "label_map.json")

UPLOAD_DIR = os.path.join(BASE_DIR, "backend", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


# =====================================================
# HULL DEFECT MODEL - TENSORFLOW
# =====================================================
print("Loading hull model from:", HULL_MODEL_PATH)
hull_model = tf.keras.models.load_model(HULL_MODEL_PATH, compile=False)

hull_classes = ["biofouling", "corrosion", "cracks"]

recommendations = {
    "biofouling": "Clean hull using high-pressure water or antifouling treatment.",
    "corrosion": "Apply anti-corrosion coating or replace damaged metal.",
    "cracks": "Critical damage. Perform welding repair immediately."
}


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
    conv_outputs = None

    with tf.GradientTape() as tape:
        x = img_tensor

        for layer in model.layers:
            x = layer(x)

            if layer.name == last_conv_layer_name:
                conv_outputs = x
                tape.watch(conv_outputs)

        predictions = x

        if pred_index is None:
            pred_index = tf.argmax(predictions[0])

        loss = predictions[:, pred_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)

    return heatmap.numpy()


@app.post("/predict-hull-defect")
async def predict_hull_defect(file: UploadFile = File(...)):
    contents = await file.read()

    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return {"error": "Invalid image file"}

    original_img = img.copy()

    img_resized = cv2.resize(img, (224, 224)) / 255.0
    img_array = np.expand_dims(img_resized, axis=0)

    preds = hull_model.predict(img_array)

    class_idx = np.argmax(preds)
    prediction = hull_classes[class_idx]
    confidence = float(np.max(preds))

    LAST_CONV_LAYER = "mobilenetv2_1.00_224"
    heatmap = make_gradcam_heatmap(img_array, hull_model, LAST_CONV_LAYER)

    heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed_img = cv2.addWeighted(original_img, 0.6, heatmap, 0.4, 0)

    _, buffer = cv2.imencode(".jpg", superimposed_img)
    gradcam_base64 = base64.b64encode(buffer).decode("utf-8")

    warning = (
        "Low confidence. Manual inspection recommended."
        if confidence < 0.7
        else "Prediction reliable."
    )

    return {
        "prediction": prediction,
        "confidence": confidence,
        "recommendation": recommendations[prediction],
        "warning": warning,
        "gradcam": gradcam_base64,
    }


# =====================================================
# SEA STATE MODEL - PYTORCH
# =====================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open(LABEL_MAP_PATH, "r") as f:
    label_map = json.load(f)

reverse_label_map = {value: key for key, value in label_map.items()}

sea_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


class ImageOnlyMobileNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.cnn = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        self.cnn.classifier[1] = nn.Linear(1280, num_classes)

    def forward(self, image):
        return self.cnn(image)


print("Loading sea-state model from:", SEA_MODEL_PATH)
sea_model = ImageOnlyMobileNet(num_classes=len(label_map)).to(device)
sea_model.load_state_dict(torch.load(SEA_MODEL_PATH, map_location=device))
sea_model.eval()


@app.post("/predict-sea-state")
async def predict_sea_state(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    image = Image.open(file_path).convert("RGB")
    image = sea_transform(image)
    image = image.unsqueeze(0).to(device)

    with torch.no_grad():
        output = sea_model(image)
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
        "probabilities": class_probabilities,
    }


@app.get("/")
def home():
    return {
        "message": "Marine AI Inspection System API is running",
        "endpoints": {
            "hull_defect": "/predict-hull-defect",
            "sea_state": "/predict-sea-state",
        },
    }