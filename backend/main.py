import os
import json
import shutil
import base64
import tempfile
import traceback
import uuid
from pathlib import Path
from datetime import datetime

import tensorflow as tf
import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2
from torchvision import transforms
from PIL import Image, ImageStat, ImageEnhance, ImageFilter

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware

from ultralytics import YOLO

import time


# =====================================================
# FASTAPI APP
# =====================================================
app = FastAPI(title="Marine AI Inspection System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_PATH = Path(BASE_DIR)

UPLOAD_DIR = os.path.join(BASE_DIR, "backend", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

TEMP_DIR = BASE_PATH / "backend" / "temp"
TEMP_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("Using device:", device)


# =====================================================
# PATHS
# =====================================================
HULL_MODEL_PATH = os.path.join(BASE_DIR, "model", "hull_model.keras")

SEA_MODEL_PATH = os.path.join(BASE_DIR, "model", "image_only_model.pth")
LABEL_MAP_PATH = os.path.join(BASE_DIR, "model", "label_map.json")
SEA_HISTORY_FILE = os.path.join(BASE_DIR, "backend", "sea_prediction_history.json")

BOAT_MODEL_PATH = BASE_PATH / "model" / "boat_detection.pt"
if not BOAT_MODEL_PATH.exists():
    BOAT_MODEL_PATH = BASE_PATH / "model" / "boat_detection_last.pt"

RADAR_YOLO_MODEL_PATH = BASE_PATH / "ml" / "models" / "final" / "yolo11_medium_best.pt"
RADAR_CNN_MODEL_PATH = BASE_PATH / "ml" / "models" / "final" / "deepercnn_best.pth"


# =====================================================
# HULL DEFECT MODEL - TENSORFLOW
# =====================================================
hull_model = None
hull_classes = ["biofouling", "corrosion", "cracks"]

recommendations = {
    "biofouling": "Clean hull using high-pressure water or antifouling treatment.",
    "corrosion": "Apply anti-corrosion coating or replace damaged metal.",
    "cracks": "Critical damage. Perform welding repair immediately."
}

try:
    if os.path.exists(HULL_MODEL_PATH):
        print("Loading hull model from:", HULL_MODEL_PATH)
        hull_model = tf.keras.models.load_model(HULL_MODEL_PATH, compile=False)
        print("Hull model loaded successfully")
    else:
        print("WARNING: Hull model not found:", HULL_MODEL_PATH)
except Exception as e:
    print("ERROR loading hull model:", e)
    hull_model = None


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
    if hull_model is None:
        return {"error": "Hull defect model not loaded"}

    try:
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

    except Exception as e:
        traceback.print_exc()
        return {"error": f"Hull defect prediction failed: {str(e)}"}


# =====================================================
# SEA STATE MODEL - PYTORCH
# =====================================================
sea_model = None
label_map = {}
reverse_label_map = {}

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


try:
    if os.path.exists(SEA_MODEL_PATH) and os.path.exists(LABEL_MAP_PATH):
        with open(LABEL_MAP_PATH, "r") as f:
            label_map = json.load(f)

        reverse_label_map = {value: key for key, value in label_map.items()}

        print("Loading sea-state model from:", SEA_MODEL_PATH)
        sea_model = ImageOnlyMobileNet(num_classes=len(label_map)).to(device)
        sea_model.load_state_dict(torch.load(SEA_MODEL_PATH, map_location=device))
        sea_model.eval()
        print("Sea-state model loaded successfully")
    else:
        print("WARNING: Sea-state model or label map not found")

except Exception as e:
    print("ERROR loading sea-state model:", e)
    sea_model = None

# =====================================================
# IMAGE VALIDATION MODEL (ImageNet MobileNet)
# =====================================================

validation_weights = MobileNet_V2_Weights.DEFAULT
validation_model = mobilenet_v2(
    weights=validation_weights
).to(device)

validation_model.eval()

validation_transform = validation_weights.transforms()

imagenet_classes = validation_weights.meta["categories"]

print("Image validation model loaded successfully")


# -----------------------------
# Sea State Extra Features
# -----------------------------
def analyze_sea_image_quality(image: Image.Image):
    grayscale = image.convert("L")

    stat = ImageStat.Stat(grayscale)
    brightness = stat.mean[0]
    contrast = stat.stddev[0]

    edges = grayscale.filter(ImageFilter.FIND_EDGES)
    edge_stat = ImageStat.Stat(edges)
    sharpness = edge_stat.stddev[0]

    if brightness < 50:
        brightness_status = "Low light"
    elif brightness > 210:
        brightness_status = "Overexposed"
    else:
        brightness_status = "Normal"

    if contrast < 25:
        contrast_status = "Low contrast / possible haze or fog"
    elif contrast < 45:
        contrast_status = "Moderate contrast"
    else:
        contrast_status = "Good contrast"

    if sharpness < 8:
        sharpness_status = "Blurry / low detail"
    elif sharpness < 18:
        sharpness_status = "Moderate sharpness"
    else:
        sharpness_status = "Good sharpness"

    if brightness < 50 or contrast < 25 or sharpness < 8:
        visibility_status = "Poor visibility"
    elif brightness > 210 or contrast < 45 or sharpness < 18:
        visibility_status = "Moderate visibility"
    else:
        visibility_status = "Clear visibility"

    return {
        "brightness_value": round(brightness, 2),
        "contrast_value": round(contrast, 2),
        "sharpness_value": round(sharpness, 2),
        "brightness_status": brightness_status,
        "contrast_status": contrast_status,
        "sharpness_status": sharpness_status,
        "visibility_status": visibility_status
    }


def enhance_sea_image(image: Image.Image):
    image = ImageEnhance.Contrast(image).enhance(1.3)
    image = ImageEnhance.Sharpness(image).enhance(1.2)
    image = ImageEnhance.Brightness(image).enhance(1.05)
    return image

def validate_sea_image(image: Image.Image):
    """
    Validate whether the uploaded image is suitable
    for sea-state classification.

    Combines:
    1. MobileNet scene/object understanding
    2. Blue-color heuristic
    """

    if validation_model is None:
        return {
            "is_valid": True,
            "message": "Image validator unavailable."
        }

    # ---------------------------------
    # MobileNet Prediction
    # ---------------------------------

    input_tensor = validation_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = validation_model(input_tensor)
        probabilities = torch.softmax(output, dim=1)

    top5_prob, top5_catid = torch.topk(probabilities, 5)

    labels = MobileNet_V2_Weights.DEFAULT.meta["categories"]

    top_predictions = []

    for i in range(5):
        label = labels[top5_catid[0][i].item()].lower()
        score = top5_prob[0][i].item()

        top_predictions.append((label, score))

    # ---------------------------------
    # Keywords that usually indicate
    # marine/ocean scenes
    # ---------------------------------

    marine_keywords = [
        "sea",
        "ocean",
        "coast",
        "shore",
        "beach",
        "lakeside",
        "harbor",
        "pier",
        "ship",
        "boat",
        "submarine",
        "container ship",
        "aircraft carrier",
        "canoe",
        "kayak",
        "lifeboat",
        "sailboat"
    ]

    mobilenet_detected = False

    for label, score in top_predictions:
        if any(word in label for word in marine_keywords):
            mobilenet_detected = True
            break

    # ---------------------------------
    # Blue Pixel Heuristic
    # ---------------------------------

    img = np.array(image.resize((224, 224)))

    r = img[:, :, 0]
    g = img[:, :, 1]
    b = img[:, :, 2]

    blue_pixels = np.sum(
        (b > r + 20) &
        (b > g + 20)
    )

    blue_ratio = blue_pixels / (224 * 224)

    heuristic_pass = blue_ratio > 0.18

    # ---------------------------------
    # Final Decision
    # ---------------------------------

    if mobilenet_detected or heuristic_pass:
        return {
            "is_valid": True,
            "message": "Ocean surface detected."
        }

    return {
        "is_valid": False,
        "message": "Uploaded image does not appear to contain a sea or ocean surface."
    }

def get_sea_state_recommendation(sea_state: str):
    recommendations_map = {
        "calm": {
            "risk_level": "Low",
            "message": "Normal sea condition detected. Navigation can continue under standard monitoring."
        },
        "moderate": {
            "risk_level": "Medium",
            "message": "Moderate sea condition detected. Continue navigation with regular monitoring of wave changes."
        },
        "rough": {
            "risk_level": "High",
            "message": "Rough sea condition detected. Navigation officers should reduce speed and monitor vessel stability."
        },
        "very_rough": {
            "risk_level": "Very High",
            "message": "Very rough sea condition detected. High-risk condition. Extra caution and operational alerts are recommended."
        }
    }

    return recommendations_map.get(
        sea_state,
        {
            "risk_level": "Unknown",
            "message": "No recommendation available for this sea state."
        }
    )

def get_weather_suitability(sea_state: str, confidence: float, visibility: str):
    """
    Calculates whether current sea conditions are suitable for marine operations.
    """

    if sea_state == "calm":
        operations = [
            "Navigation",
            "Cargo Transport",
            "Fishing",
            "Patrol Boats"
        ]
        score = 96
        condition = "Favorable"

    elif sea_state == "moderate":
        operations = [
            "Navigation",
            "Cargo Transport",
            "Fishing"
        ]
        score = 74
        condition = "Moderate"

    elif sea_state == "rough":
        operations = [
            "Navigation with caution",
            "Limited cargo operations"
        ]
        score = 42
        condition = "Unfavorable"

    else:   # very_rough
        operations = [
            "Emergency Operations Only"
        ]
        score = 12
        condition = "Unfavorable"

    # Reduce score based on image visibility
    if visibility == "Moderate visibility":
        score -= 10

    elif visibility == "Poor visibility":
        score -= 20

    # Reduce score if AI confidence is low
    if confidence < 70:
        score -= 10

    score = max(0, min(100, score))

    if score >= 80:
        condition = "Favorable"
    elif score >= 55:
        condition = "Moderate"
    else:
        condition = "Unfavorable"

    if sea_state == "calm":
        reason = (
            "Calm sea conditions with minimal wave activity. "
            "Marine operations can be performed safely."
        )

    elif sea_state == "moderate":
        reason = (
            "Moderate waves detected. Most marine operations remain possible "
            "with normal precautions."
        )

    elif sea_state == "rough":
        reason = (
            "High wave activity detected. Operations should be limited due to "
            "increased safety risks."
        )

    else:
        reason = (
            "Very rough sea conditions detected. Only emergency operations "
            "should be considered."
        )

    if visibility == "Poor visibility":
        reason += " Image visibility is poor, reducing assessment reliability."

    elif visibility == "Moderate visibility":
        reason += " Image visibility is moderate."

    if confidence < 70:
        reason += " AI prediction confidence is relatively low."

    return {
        "condition": condition,
        "score": score,
        "operations": operations,
        "reason": reason
    }

def calculate_risk_indicator(sea_state, confidence, image_quality):
    """
    Calculates overall operational risk using
    sea state + confidence + image quality.
    """

    # Base score from sea state
    base_scores = {
        "calm": 20,
        "moderate": 45,
        "rough": 75,
        "very_rough": 95
    }

    score = base_scores.get(sea_state, 50)
    reasons = []

    # -------------------------
    # Confidence
    # -------------------------

    if confidence < 50:
        score += 20
        reasons.append("Low prediction confidence")

    elif confidence < 70:
        score += 10
        reasons.append("Moderate prediction confidence")

    # -------------------------
    # Visibility
    # -------------------------

    visibility = image_quality["visibility_status"]

    if visibility == "Poor visibility":
        score += 10
        reasons.append("Poor visibility")

    # -------------------------
    # Brightness
    # -------------------------

    if image_quality["brightness_status"] != "Normal":
        score += 5
        reasons.append("Suboptimal brightness")

    # -------------------------
    # Sharpness
    # -------------------------

    if image_quality["sharpness_status"] != "Good sharpness":
        score += 5
        reasons.append("Low image sharpness")

    score = min(score, 100)

    if score <= 25:
        level = "Low"

    elif score <= 50:
        level = "Medium"

    elif score <= 75:
        level = "High"

    else:
        level = "Very High"

    return {
        "score": score,
        "level": level,
        "reasons": reasons
    }

def get_sea_confidence_warnings(confidence: float, visibility_status: str):
    warnings = []

    if confidence < 70:
        warnings.append("Prediction confidence is low. Use a clearer image or verify with manual observation.")

    if visibility_status == "Poor visibility":
        warnings.append("Image quality indicates poor visibility. Prediction may be less reliable.")

    if visibility_status == "Moderate visibility":
        warnings.append("Image quality is moderate. Prediction should be interpreted with caution.")

    if len(warnings) == 0:
        warnings.append("Prediction confidence and image quality are acceptable.")

    return warnings


def load_sea_history():
    if not os.path.exists(SEA_HISTORY_FILE):
        return []

    try:
        with open(SEA_HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def save_sea_history_record(record):
    history = load_sea_history()
    history.append(record)

    with open(SEA_HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=4)


@app.post("/predict-sea-state")
async def predict_sea_state(
    file: UploadFile = File(...),
    apply_enhancement: bool = Form(False)
):
    if sea_model is None:
        return {"error": "Sea-state model not loaded"}

    try:
        start_time = time.perf_counter()

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        file_path = os.path.join(UPLOAD_DIR, file.filename)

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        original_image = Image.open(file_path).convert("RGB")

        # -----------------------------------------
        # Validate uploaded image
        # -----------------------------------------
        validation = validate_sea_image(original_image)

        if not validation["is_valid"]:
            return {
                "error": "Invalid image. Please upload an image that clearly shows the sea surface.",
                "validation": validation
            }

        quality_report = analyze_sea_image_quality(original_image)

        if apply_enhancement:
            prediction_image = enhance_sea_image(original_image)
            enhanced_path = os.path.join(UPLOAD_DIR, "enhanced_" + file.filename)
            prediction_image.save(enhanced_path)
        else:
            prediction_image = original_image

        image = sea_transform(prediction_image)
        image = image.unsqueeze(0).to(device)

        with torch.no_grad():
            output = sea_model(image)
            probabilities = torch.softmax(output, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)

        predicted_class = predicted_class.item()
        predicted_label = reverse_label_map[predicted_class]

        confidence = confidence.item()
        confidence_percent = round(confidence * 100, 2)

        class_probabilities = {}

        for i, prob in enumerate(probabilities[0]):
            label = reverse_label_map[i]
            class_probabilities[label] = round(prob.item() * 100, 2)

        weather_suitability = get_weather_suitability(
            predicted_label,
            confidence_percent,
            quality_report["visibility_status"]
        )

        risk_indicator = calculate_risk_indicator(
            predicted_label,
            confidence_percent,
            quality_report
        )

        recommendation = get_sea_state_recommendation(predicted_label)

        warnings = get_sea_confidence_warnings(
            confidence_percent,
            quality_report["visibility_status"]
        )

        processing_time = round(time.perf_counter() - start_time, 3)

        result = {
                    "timestamp": timestamp,
                    "filename": file.filename,
                    "predicted_sea_state": predicted_label,
                    "confidence": confidence_percent,
                    "validation": validation,
                    "processing_time": processing_time,
                    "probabilities": class_probabilities,
                    "image_quality": quality_report,
                    "enhancement_applied": apply_enhancement,
                    "recommendation": recommendation,
                    "weather_suitability": weather_suitability,
                    "risk_indicator": risk_indicator,
                    "warnings": warnings
                }
        

        save_sea_history_record(result)

        return result

    except Exception as e:
        traceback.print_exc()
        return {"error": f"Sea-state prediction failed: {str(e)}"}


@app.get("/sea-state-history")
def get_sea_state_history():
    return {
        "history": load_sea_history()
    }


@app.delete("/sea-state-history")
def clear_sea_state_history():
    with open(SEA_HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump([], f, indent=4)

    return {
        "message": "Sea-state prediction history cleared successfully"
    }


# =====================================================
# BOAT DETECTION MODEL - YOLOv8
# =====================================================
boat_model = None

try:
    if not BOAT_MODEL_PATH.exists():
        print("ERROR: No boat detection weights file found!")
        boat_model = None
    else:
        print(f"Loading boat detection model from: {BOAT_MODEL_PATH}")
        boat_model = YOLO(str(BOAT_MODEL_PATH))
        print("Boat detection model loaded successfully")

except Exception as e:
    print(f"ERROR loading boat model: {e}")
    boat_model = None


@app.post("/predict-boat-detection")
async def predict_boat_detection(file: UploadFile = File(...)):
    if boat_model is None:
        return {"error": "Boat detection model not loaded"}

    try:
        if not file.filename:
            return {"error": "No filename"}

        filename_lower = file.filename.lower()

        if not filename_lower.endswith((".png", ".jpg", ".jpeg", ".bmp")):
            return {"error": f"Invalid file type: {filename_lower}"}

        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            contents = await file.read()
            tmp.write(contents)
            tmp_path = tmp.name

        results = boat_model(tmp_path, save=False, verbose=False)

        detections = []

        if results and len(results) > 0:
            result = results[0]

            if result.boxes is not None and len(result.boxes) > 0:
                for box in result.boxes:
                    try:
                        cls_id = int(box.cls.item()) if box.cls is not None else 0
                        conf = float(box.conf.item()) if box.conf is not None else 0
                        label = boat_model.names.get(cls_id, f"class_{cls_id}")

                        detections.append({
                            "label": label,
                            "confidence": round(conf * 100, 1)
                        })

                    except Exception as e:
                        print(f"Error parsing box: {e}")

        Path(tmp_path).unlink(missing_ok=True)

        return {
            "image": file.filename,
            "results": detections,
            "count": len(detections)
        }

    except Exception as e:
        print(f"ERROR: {e}")
        traceback.print_exc()
        return {"error": f"Inference failed: {str(e)}"}


# =====================================================
# RADAR OBJECT CLASSIFICATION MODEL - YOLO + CNN ENSEMBLE
# =====================================================
radar_yolo_model = None
radar_cnn_model = None

RADAR_CLASSES = ["bird", "ship", "unknown"]

radar_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])


class RadarDeeperCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(RadarDeeperCNN, self).__init__()

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


def convert_radar_to_heatmap(input_path: Path, output_path: Path):
    image = Image.open(input_path).convert("L")
    image = image.resize((128, 128))

    gray = np.array(image)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

    heatmap = cv2.applyColorMap(gray.astype(np.uint8), cv2.COLORMAP_VIRIDIS)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    Image.fromarray(heatmap).save(output_path)


try:
    if RADAR_YOLO_MODEL_PATH.exists():
        print("Loading radar YOLO model from:", RADAR_YOLO_MODEL_PATH)
        radar_yolo_model = YOLO(str(RADAR_YOLO_MODEL_PATH))
        print("Radar YOLO model loaded successfully")
    else:
        print("ERROR: Radar YOLO model not found:", RADAR_YOLO_MODEL_PATH)

except Exception as e:
    print("ERROR loading radar YOLO model:", e)
    radar_yolo_model = None


try:
    if RADAR_CNN_MODEL_PATH.exists():
        print("Loading radar CNN model from:", RADAR_CNN_MODEL_PATH)
        radar_cnn_model = RadarDeeperCNN(num_classes=3)
        radar_cnn_model.load_state_dict(torch.load(RADAR_CNN_MODEL_PATH, map_location=device))
        radar_cnn_model.to(device)
        radar_cnn_model.eval()
        print("Radar CNN model loaded successfully")
    else:
        print("ERROR: Radar CNN model not found:", RADAR_CNN_MODEL_PATH)

except Exception as e:
    print("ERROR loading radar CNN model:", e)
    radar_cnn_model = None


@app.post("/predict-radar-object")
async def predict_radar_object(file: UploadFile = File(...)):
    if radar_yolo_model is None:
        return {"error": "Radar YOLO model not loaded"}

    if radar_cnn_model is None:
        return {"error": "Radar CNN model not loaded"}

    try:
        if not file.filename:
            return {"error": "No filename provided"}

        filename_lower = file.filename.lower()

        if not filename_lower.endswith((".png", ".jpg", ".jpeg", ".bmp")):
            return {"error": f"Invalid file type: {filename_lower}"}

        file_id = str(uuid.uuid4())

        raw_path = TEMP_DIR / f"{file_id}_{file.filename}"
        heatmap_path = TEMP_DIR / f"{file_id}_heatmap.png"

        with open(raw_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        convert_radar_to_heatmap(raw_path, heatmap_path)

        image = Image.open(heatmap_path).convert("RGB")
        input_tensor = radar_transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            cnn_output = radar_cnn_model(input_tensor)
            cnn_probs = torch.softmax(cnn_output, dim=1)
            cnn_confidence, cnn_pred = torch.max(cnn_probs, 1)

        cnn_label = RADAR_CLASSES[cnn_pred.item()]
        cnn_conf = round(cnn_confidence.item() * 100, 2)

        yolo_results = radar_yolo_model(str(heatmap_path), verbose=False)
        yolo_probs = yolo_results[0].probs

        yolo_pred_index = int(yolo_probs.top1)
        yolo_conf = round(float(yolo_probs.top1conf) * 100, 2)
        yolo_label = yolo_results[0].names[yolo_pred_index]

        if yolo_label == cnn_label:
            final_prediction = yolo_label
            decision_status = "High confidence - both models agree"
        else:
            final_prediction = "uncertain"
            decision_status = "Models disagree - manual review required"

        return {
            "image": file.filename,
            "yolo_prediction": yolo_label,
            "yolo_confidence": yolo_conf,
            "cnn_prediction": cnn_label,
            "cnn_confidence": cnn_conf,
            "final_prediction": final_prediction,
            "decision_status": decision_status
        }

    except Exception as e:
        print(f"ERROR in radar object prediction: {e}")
        traceback.print_exc()
        return {"error": f"Radar object prediction failed: {str(e)}"}


# =====================================================
# HOME ROUTE
# =====================================================
@app.get("/")
def home():
    return {
        "message": "Marine AI Inspection System API is running",
        "endpoints": {
            "hull_defect": "/predict-hull-defect",
            "sea_state": "/predict-sea-state",
            "sea_state_history": "/sea-state-history",
            "boat_detection": "/predict-boat-detection",
            "radar_object_classification": "/predict-radar-object",
        },
    }