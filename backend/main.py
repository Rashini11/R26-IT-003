import os
import json
import shutil
import base64
import tempfile
import traceback
import uuid
import subprocess
from datetime import datetime
from pathlib import Path
from datetime import datetime

import tensorflow as tf
import numpy as np
import cv2

try:
    import imageio_ffmpeg
except ImportError:
    imageio_ffmpeg = None

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import MobileNet_V2_Weights
from torchvision import transforms
from PIL import Image, ImageStat, ImageEnhance, ImageFilter

from fastapi import FastAPI, UploadFile, File, Form, Depends
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from ultralytics import YOLO

import time

try:
    from dotenv import load_dotenv
    from pymongo import MongoClient
except ImportError:
    load_dotenv = None
    MongoClient = None


# =====================================================
# FASTAPI APP
# =====================================================
# This is now the ONLY FastAPI application that needs
# to be started for OceanIQ.
# =====================================================
app = FastAPI(
    title="OceanIQ Marine AI Intelligence Platform",
    version="2.1.0",
)

FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "").strip()

ALLOWED_ORIGINS = [
    "http://localhost:5174",
    "http://127.0.0.1:5174",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

if FRONTEND_ORIGIN:
    ALLOWED_ORIGINS.append(FRONTEND_ORIGIN)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================
# BACKEND AUTHENTICATION
# MongoDB-backed users + server-side sessions.
# The browser receives only an HttpOnly session cookie.
# =====================================================
from auth import (
    router as auth_router,
    require_authenticated_request,
)

app.include_router(auth_router)

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)
BASE_PATH = Path(BASE_DIR)

UPLOAD_DIR = os.path.join(
    BASE_DIR,
    "backend",
    "uploads",
)
os.makedirs(
    UPLOAD_DIR,
    exist_ok=True,
)

TEMP_DIR = (
    BASE_PATH
    / "backend"
    / "temp"
)
TEMP_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

device = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Using device:", device)


# =====================================================
# PATHS
# =====================================================
HULL_MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "hull_model.keras",
)

HULL_VALIDATOR_MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "hull_validator.keras",
)

SEA_MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "image_only_model.pth",
)

LABEL_MAP_PATH = os.path.join(
    BASE_DIR,
    "model",
    "label_map.json",
)

BOAT_MODEL_PATH = BASE_PATH / "backend" / "best.pt"

if not BOAT_MODEL_PATH.exists():
    BOAT_MODEL_PATH = BASE_PATH / "model" / "boat_detection.pt"

if not BOAT_MODEL_PATH.exists():
    BOAT_MODEL_PATH = BASE_PATH / "model" / "boat_detection_last.pt"

RADAR_MODEL_PATH = (
    BASE_PATH
    / "ml"
    / "models"
    / "v6_runs"
    / "radar_unknown3"
    / "radar_unknown3_best.pth"
)

RADAR_MODEL_VERSION = "radar_unknown3_v1"

# The model has an explicit unknown output class.
# Direct three-class argmax performed better than calibration.
RADAR_UNKNOWN_THRESHOLD = None


# =====================================================
# SEA-STATE HISTORY STORAGE
# MongoDB is used when MONGO_URI is configured. The JSON
# file is a local fallback so the integrated backend still
# starts and the history feature still works without MongoDB.
# =====================================================
SEA_HISTORY_FILE = BASE_PATH / "backend" / "sea_prediction_history.json"
HULL_HISTORY_FILE = BASE_PATH / "backend" / "hull_prediction_history.json"
VESSEL_HISTORY_FILE = BASE_PATH / "backend" / "vessel_prediction_history.json"
sea_state_collection = None
hull_prediction_collection = None
radar_prediction_collection = None
simulation_runs_collection = None
simulation_events_collection = None
vessel_prediction_collection = None
sea_history_backend = "json"

if load_dotenv is not None:
    # Load Sea-State/OceanIQ database settings from backend/.env.
    load_dotenv(BASE_PATH / "backend" / ".env")

MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "marine_ai_db")

if MONGO_URI and MongoClient is not None:
    try:
        mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=2000)
        mongo_client.admin.command("ping")
        mongo_db = mongo_client[MONGO_DB_NAME]
        sea_state_collection = mongo_db["sea_state_predictions"]
        hull_prediction_collection = mongo_db["hull_predictions"]
        radar_prediction_collection = mongo_db["radar_predictions"]
        simulation_runs_collection = mongo_db["simulation_runs"]
        simulation_events_collection = mongo_db["simulation_events"]
        vessel_prediction_collection = mongo_db["vessel_predictions"]
        vessel_prediction_collection.create_index([("timestamp", -1)])

        sea_history_backend = "mongodb"

        print("MongoDB connected successfully for sea-state history")
        print("MongoDB connected successfully for hull prediction history")
        print("MongoDB connected successfully for radar predictions")
        print("MongoDB connected successfully for simulation persistence")

    except Exception as e:
        print(
            "MongoDB unavailable; using JSON history fallback:",
            e
        )
        sea_state_collection = None
        hull_prediction_collection = None
        radar_prediction_collection = None
        simulation_runs_collection = None
        simulation_events_collection = None
        vessel_prediction_collection = None


def _load_local_sea_history():
    try:
        if not SEA_HISTORY_FILE.exists():
            return []
        with open(SEA_HISTORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception as e:
        print("Error loading local sea-state history:", e)
        return []


def _save_local_sea_history(history):
    try:
        SEA_HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(SEA_HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
        return True
    except Exception as e:
        print("Error saving local sea-state history:", e)
        return False

def _load_local_hull_history():
    try:
        if not HULL_HISTORY_FILE.exists():
            return []

        with open(HULL_HISTORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        return data if isinstance(data, list) else []

    except Exception as e:
        print("Error loading local hull history:", e)
        return []


def _save_local_hull_history(history):
    try:
        HULL_HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)

        with open(HULL_HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        return True

    except Exception as e:
        print("Error saving local hull history:", e)
        return False


def _load_local_vessel_history():
    try:
        if not VESSEL_HISTORY_FILE.exists():
            return []
        with open(VESSEL_HISTORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception as e:
        print("Error loading local vessel history:", e)
        return []


def _save_local_vessel_history(history):
    try:
        VESSEL_HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(VESSEL_HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
        return True
    except Exception as e:
        print("Error saving local vessel history:", e)
        return False



# =====================================================
# HULL DEFECT MODEL - TENSORFLOW
# =====================================================
hull_model = None
hull_classes = [
    "biofouling",
    "corrosion",
    "cracks",
    "paint_damage",
    
]

# =====================================================
# HULL IMAGE VALIDATION SETTINGS
# =====================================================

HULL_CONFIDENCE_THRESHOLD = 0.70


recommendations = {
    "biofouling": "Clean hull using high-pressure water or antifouling treatment.",
    "corrosion": "Apply anti-corrosion coating or replace damaged metal.",
    "cracks": "Critical damage. Perform welding repair immediately.",
    "paint_damage": "Inspect the affected area and repair or reapply marine-grade protective coating.",
}

try:
    if os.path.exists(HULL_MODEL_PATH):
        print(
            "Loading hull model from:",
            HULL_MODEL_PATH,
        )

        hull_model = (
            tf.keras.models.load_model(
                HULL_MODEL_PATH,
                compile=False,
            )
        )

        print(
            "Hull model loaded successfully"
        )

        print("\n========== MOBILENETV2 LAYERS ==========")

        for layer in hull_model.layers:

            if isinstance(layer, tf.keras.Model):

                print("Nested model:", layer.name)

                for sublayer in layer.layers:
                    print(
                        sublayer.name,
                        "->",
                        sublayer.__class__.__name__
                    )

        print("========================================\n")

    else:
        print(
            "WARNING: Hull model not found:",
            HULL_MODEL_PATH,
        )

except Exception as e:
    print(
        "ERROR loading hull model:",
        e,
    )
    hull_model = None


# =====================================================
# HULL / NON-HULL VALIDATOR MODEL
# =====================================================

hull_validator_model = None

try:
    if os.path.exists(HULL_VALIDATOR_MODEL_PATH):

        print(
            "Loading hull validator from:",
            HULL_VALIDATOR_MODEL_PATH,
        )

        hull_validator_model = tf.keras.models.load_model(
            HULL_VALIDATOR_MODEL_PATH,
            compile=False,
        )

        print(
            "Hull validator loaded successfully"
        )

    else:

        print(
            "WARNING: Hull validator model not found:",
            HULL_VALIDATOR_MODEL_PATH,
        )

except Exception as e:

    print(
        "ERROR loading hull validator:",
        e,
    )

    hull_validator_model = None


def validate_hull_image(image):
    """
    Basic image-quality validation before hull classification.
    This does NOT determine whether the image is a hull.
    """

    if image is None:
        return {
            "is_valid": False,
            "reason": "Invalid image.",
            "contrast": 0,
        }

    height, width = image.shape[:2]

    if height < 100 or width < 100:
        return {
            "is_valid": False,
            "reason": "Image resolution is too low.",
            "contrast": 0,
        }

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    contrast = float(np.std(gray))

    if contrast < 12:
        return {
            "is_valid": False,
            "reason": "Image has very low visual detail.",
            "contrast": round(contrast, 2),
        }

    return {
        "is_valid": True,
        "reason": "Image quality is sufficient for analysis.",
        "contrast": round(contrast, 2),
        "width": width,
        "height": height,
    }


# =====================================================
# HULL / NON-HULL CLASSIFICATION
# =====================================================

HULL_VALIDATOR_THRESHOLD = 0.70


def classify_hull_or_non_hull(image):

    if hull_validator_model is None:

        return {
            "is_hull": False,
            "hull_confidence": 0,
            "non_hull_confidence": 0,
            "message": "Hull validator model is not available.",
        }

    resized = cv2.resize(
        image,
        (224, 224),
    )

    resized = cv2.cvtColor(
        resized,
        cv2.COLOR_BGR2RGB,
    )

    image_array = np.expand_dims(
        resized.astype(np.float32),
        axis=0,
    )

    # IMPORTANT:
    # The validator model already contains
    # MobileNetV2 preprocessing in its input pipeline.
    prediction = hull_validator_model.predict(
        image_array,
        verbose=0,
    )[0][0]

    non_hull_probability = float(prediction)

    hull_probability = 1.0 - non_hull_probability

    print("\n========== HULL VALIDATOR ==========")
    print(
        f"Hull: {hull_probability * 100:.2f}%"
    )
    print(
        f"Non-Hull: {non_hull_probability * 100:.2f}%"
    )
    print("====================================\n")

    if non_hull_probability >= HULL_VALIDATOR_THRESHOLD:

        return {
            "is_hull": False,
            "hull_confidence": round(
                hull_probability * 100,
                2,
            ),
            "non_hull_confidence": round(
                non_hull_probability * 100,
                2,
            ),
            "message": (
                "The uploaded image does not appear "
                "to show an underwater ship hull."
            ),
        }

    return {
        "is_hull": True,
        "hull_confidence": round(
            hull_probability * 100,
            2,
        ),
        "non_hull_confidence": round(
            non_hull_probability * 100,
            2,
        ),
        "message": (
            "Underwater hull image accepted."
        ),
    }

def make_gradcam_heatmap(
    img_array,
    model,
    last_conv_layer_name,
    pred_index=None,
):
    """
    Grad-CAM for the nested MobileNetV2 hull model.
    """

    # Get the nested MobileNetV2
    base_model = model.get_layer(
        "mobilenetv2_1.00_224"
    )

    # Get the target convolutional layer
    last_conv_layer = base_model.get_layer(
        last_conv_layer_name
    )

    # Create a model that gives us:
    # 1. Conv feature maps
    # 2. MobileNetV2 output
    grad_model = tf.keras.models.Model(
        inputs=base_model.input,
        outputs=[
            last_conv_layer.output,
            base_model.output,
        ],
    )

    img_tensor = tf.convert_to_tensor(
        img_array,
        dtype=tf.float32,
    )

    with tf.GradientTape() as tape:

        conv_outputs, features = grad_model(
            img_tensor,
            training=False,
        )

        # IMPORTANT:
        # Get the classifier layers after MobileNetV2
        x = features

        for layer in model.layers:
            if layer.name == "mobilenetv2_1.00_224":
                continue

            if layer.name in [
                "random_flip",
                "random_rotation",
                "random_zoom",
                "random_contrast",
            ]:
                continue

            # Only apply actual classifier layers
            if isinstance(
                layer,
                (
                    tf.keras.layers.GlobalAveragePooling2D,
                    tf.keras.layers.BatchNormalization,
                    tf.keras.layers.Dense,
                    tf.keras.layers.Dropout,
                ),
            ):
                x = layer(
                    x,
                    training=False,
                )

        predictions = x

        if pred_index is None:
            pred_index = tf.argmax(
                predictions[0]
            )

        class_channel = predictions[
            :,
            pred_index,
        ]

    # Calculate gradients
    grads = tape.gradient(
        class_channel,
        conv_outputs,
    )

    if grads is None:
        raise ValueError(
            "Grad-CAM gradients are None."
        )

    # Average gradients
    pooled_grads = tf.reduce_mean(
        grads,
        axis=(1, 2),
    )

    conv_outputs = conv_outputs[0]
    pooled_grads = pooled_grads[0]

    # Create heatmap
    heatmap = tf.reduce_sum(
        conv_outputs * pooled_grads,
        axis=-1,
    )

    # ReLU
    heatmap = tf.maximum(
        heatmap,
        0,
    )

    # Normalize
    heatmap = heatmap / (
        tf.reduce_max(heatmap)
        + 1e-8
    )

    return heatmap.numpy()


@app.post("/predict-hull-defect", dependencies=[Depends(require_authenticated_request)])
async def predict_hull_defect(
    file: UploadFile = File(...),
):
    if hull_model is None:
        return {
            "error": (
                "Hull defect model not loaded"
            )
        }

    try:
        contents = await file.read()

        nparr = np.frombuffer(
            contents,
            np.uint8,
        )

        img = cv2.imdecode(
            nparr,
            cv2.IMREAD_COLOR,
        )

        if img is None:
            return {
                "error": (
                    "Invalid image file"
                )
            }

        original_img = img.copy()

        # =====================================================
        # HULL IMAGE VALIDATION
        # =====================================================

        validation = validate_hull_image(
            original_img
        )

        if not validation["is_valid"]:

            return {
                "prediction": None,
                "confidence": 0,
                "valid_hull_image": False,
                "detected_defects": [],
                "multiple_defects": False,
                "recommendation": None,
                "warning": validation["reason"],
                "validation": validation,
                "hull_validation": None,
                "gradcam": None,
            }


        # =====================================================
        # HULL / NON-HULL VALIDATION
        # =====================================================

        hull_validation = classify_hull_or_non_hull(
            original_img
        )

        print(
            "Hull validation result:",
            hull_validation
        )


        # =====================================================
        # REJECT NON-HULL IMAGE
        # =====================================================

        if not hull_validation["is_hull"]:

            return {
                "prediction": None,
                "confidence": 0,
                "valid_hull_image": False,
                "detected_defects": [],
                "multiple_defects": False,
                "recommendation": (
                    "Please upload a clear underwater "
                    "ship-hull inspection image."
                ),
                "warning": (
                    "Non-hull image detected. "
                    "Defect classification was not performed."
                ),
                "validation": validation,
                "hull_validation": hull_validation,
                "gradcam": None,
            }


        # =====================================================
        # PREPROCESS IMAGE FOR HULL DEFECT MODEL
        # =====================================================

        img_resized = cv2.resize(
            img,
            (224, 224),
        )

        img_resized = cv2.cvtColor(
            img_resized,
            cv2.COLOR_BGR2RGB,
        )

        img_array = np.expand_dims(
            img_resized.astype(np.float32),
            axis=0,
        )

        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(
            img_array
        )

        preds = hull_model.predict(
            img_array,
            verbose=0
        )

        # Get probabilities for every defect class
        probabilities = preds[0]

        if len(probabilities) != len(hull_classes):
            return {
                "prediction": None,
                "confidence": 0,
                "valid_hull_image": False,
                "detected_defects": [],
                "multiple_defects": False,
                "recommendation": None,
                "warning": (
                    f"Model output mismatch: "
                    f"model returned {len(probabilities)} outputs, "
                    f"but backend expects {len(hull_classes)} classes."
                ),
                "validation": validation,
                "gradcam": None,
            }

        print("\n========== HULL PREDICTION DEBUG ==========")

        for i, prob in enumerate(probabilities):
            print(
                f"{hull_classes[i]}: {float(prob) * 100:.2f}%"
            )

        print("===========================================\n")

        

        # =====================================================
        # PRIMARY PREDICTION
        # =====================================================

        class_idx = int(np.argmax(probabilities))
        prediction = hull_classes[class_idx]

        confidence = float(probabilities[class_idx])

       

        # =====================================================
        # CONFIDENCE VALIDATION
        # =====================================================

        if confidence < HULL_CONFIDENCE_THRESHOLD:

            return {
                "prediction": "uncertain",
                "confidence": round(confidence * 100, 2),
                "valid_hull_image": True,
                "detected_defects": [],
                "multiple_defects": False,
                "recommendation": (
                    "The image appears to show a hull, "
                    "but no defect could be classified reliably. "
                    "Please upload a clearer hull image."
                ),
                "warning": (
                    "Low confidence. Manual inspection recommended."
                ),
                "validation": validation,
                "gradcam": None,
            } 

        # =====================================================
        # HULL DEFECT PROBABILITIES
        # =====================================================

        # Show probability for EVERY defect class
        detected_defects = []

        for i, prob in enumerate(probabilities):

            detected_defects.append({
                "defect": hull_classes[i],
                "confidence": round(
                    float(prob) * 100,
                    2
                )
            })

        # Sort highest probability first
        detected_defects.sort(
            key=lambda x: x["confidence"],
            reverse=True
        )

        # =====================================================
        # TOP DEFECTS
        # =====================================================

        # Highest probability defect
        primary_defect = detected_defects[0]

        # Only consider another defect if its probability
        # is high enough to be meaningful.
        SECONDARY_THRESHOLD = 15.0  # 15%

        secondary_defects = [
            item
            for item in detected_defects[1:]
            if item["confidence"] >= SECONDARY_THRESHOLD
        ]

        # Only report multiple defects when there are
        # actually meaningful secondary predictions.
        multiple_defects = False

        reported_defects = [
            primary_defect
        ] 

        # =====================================================
        # RECOMMENDATION
        # =====================================================

        if multiple_defects:

            recommendation = (
                "Multiple possible defects detected: "
                + ", ".join(
                    item["defect"]
                    for item in reported_defects
                )
                + ". Inspect the affected area carefully "
                "and perform appropriate repair for each defect."
            )

        else:

            recommendation = recommendations[prediction]

        LAST_CONV_LAYER = "Conv_1"

        heatmap = (
            make_gradcam_heatmap(
                img_array,
                hull_model,
                LAST_CONV_LAYER,
            )
        )

        heatmap = cv2.resize(
            heatmap,
            (
                original_img.shape[1],
                original_img.shape[0],
            ),
        )

        heatmap = np.uint8(
            255 * heatmap
        )

        heatmap = cv2.applyColorMap(
            heatmap,
            cv2.COLORMAP_JET,
        )

        superimposed_img = (
            cv2.addWeighted(
                original_img,
                0.6,
                heatmap,
                0.4,
                0,
            )
        )

        _, buffer = cv2.imencode(
            ".jpg",
            superimposed_img,
        )

        gradcam_base64 = (
            base64.b64encode(
                buffer
            ).decode("utf-8")
        )

        warning = (
            "Low confidence. Manual "
            "inspection recommended."
            if confidence < HULL_CONFIDENCE_THRESHOLD
            else "Prediction reliable."
        )

        # =====================================================
        # SAVE HULL PREDICTION TO MONGODB
        # =====================================================
        hull_record = {
        "timestamp": datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
        "filename": file.filename,
        "prediction": prediction,
        "confidence": confidence,
        "detected_defects": reported_defects,
        "multiple_defects": multiple_defects,
        "recommendation": recommendation,
        "warning": warning,
    }

        save_hull_prediction_record(hull_record)

        return {
            "prediction": prediction,
            "confidence": round(confidence * 100, 2),
            "valid_hull_image": True,
            "detected_defects": detected_defects,
            "multiple_defects": multiple_defects,
            "recommendation": recommendation,
            "warning": warning,
            "gradcam": gradcam_base64,
        }

    except Exception as e:
        traceback.print_exc()

        return {
            "error": (
                "Hull defect prediction "
                f"failed: {str(e)}"
            )
        }


# =====================================================
# SEA STATE MODEL - PYTORCH
# =====================================================
sea_model = None
label_map = {}
reverse_label_map = {}

sea_transform = transforms.Compose([
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
])


class ImageOnlyMobileNet(nn.Module):
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
        return self.cnn(image)


try:
    if (
        os.path.exists(
            SEA_MODEL_PATH
        )
        and os.path.exists(
            LABEL_MAP_PATH
        )
    ):
        with open(
            LABEL_MAP_PATH,
            "r",
        ) as f:
            label_map = json.load(f)

        reverse_label_map = {
            value: key
            for key, value
            in label_map.items()
        }

        print(
            "Loading sea-state "
            "model from:",
            SEA_MODEL_PATH,
        )

        sea_model = (
            ImageOnlyMobileNet(
                num_classes=len(
                    label_map
                )
            ).to(device)
        )

        sea_model.load_state_dict(
            torch.load(
                SEA_MODEL_PATH,
                map_location=device,
                weights_only=True,
            )
        )

        sea_model.eval()

        print(
            "Sea-state model "
            "loaded successfully"
        )

    else:
        print(
            "WARNING: Sea-state model "
            "or label map not found"
        )

except Exception as e:
    print(
        "ERROR loading "
        "sea-state model:",
        e,
    )
    sea_model = None


# =====================================================
# SEA-STATE IMAGE VALIDATION + DECISION-SUPPORT FEATURES
# =====================================================
validation_weights = MobileNet_V2_Weights.DEFAULT
validation_model = None
validation_transform = validation_weights.transforms()

try:
    validation_model = models.mobilenet_v2(
        weights=validation_weights
    ).to(device)
    validation_model.eval()
    print("Sea-state image validation model loaded successfully")
except Exception as e:
    print("WARNING: Sea-state image validation model unavailable:", e)
    validation_model = None


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
        "visibility_status": visibility_status,
    }


def enhance_sea_image(image: Image.Image):
    image = ImageEnhance.Contrast(image).enhance(1.3)
    image = ImageEnhance.Sharpness(image).enhance(1.2)
    image = ImageEnhance.Brightness(image).enhance(1.05)
    return image


def validate_sea_image(image: Image.Image):
    """Validate that an uploaded image plausibly contains a marine surface."""
    if validation_model is None:
        return {
            "is_valid": True,
            "message": "Image validator unavailable; classification allowed.",
        }

    input_tensor = validation_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = validation_model(input_tensor)
        probabilities = torch.softmax(output, dim=1)

    top5_prob, top5_catid = torch.topk(probabilities, 5)
    labels = validation_weights.meta["categories"]
    top_predictions = []

    for i in range(5):
        label = labels[top5_catid[0][i].item()].lower()
        score = top5_prob[0][i].item()
        top_predictions.append((label, score))

    marine_keywords = [
        "sea", "ocean", "coast", "shore", "beach", "lakeside",
        "harbor", "pier", "ship", "boat", "submarine",
        "container ship", "aircraft carrier", "canoe", "kayak",
        "lifeboat", "sailboat",
    ]

    mobilenet_detected = any(
        any(word in label for word in marine_keywords)
        for label, _score in top_predictions
    )

    # Cast before arithmetic to avoid uint8 overflow in the blue heuristic.
    img = np.array(image.resize((224, 224))).astype(np.int16)
    r = img[:, :, 0]
    g = img[:, :, 1]
    b = img[:, :, 2]
    blue_pixels = np.sum((b > r + 20) & (b > g + 20))
    blue_ratio = blue_pixels / (224 * 224)
    heuristic_pass = blue_ratio > 0.18

    if mobilenet_detected or heuristic_pass:
        return {
            "is_valid": True,
            "message": "Ocean surface detected.",
        }

    return {
        "is_valid": False,
        "message": "Uploaded image does not appear to contain a sea or ocean surface.",
    }


def _sea_state_key(sea_state: str):
    return str(sea_state or "").strip().lower().replace(" ", "_").replace("-", "_")


def get_sea_state_recommendation(sea_state: str):
    recommendations_map = {
        "calm": {
            "risk_level": "Low",
            "message": "Normal sea condition detected. Navigation can continue under standard monitoring.",
        },
        "moderate": {
            "risk_level": "Medium",
            "message": "Moderate sea condition detected. Continue navigation with regular monitoring of wave changes.",
        },
        "rough": {
            "risk_level": "High",
            "message": "Rough sea condition detected. Navigation officers should reduce speed and monitor vessel stability.",
        },
        "very_rough": {
            "risk_level": "Very High",
            "message": "Very rough sea condition detected. High-risk condition. Extra caution and operational alerts are recommended.",
        },
    }

    return recommendations_map.get(
        _sea_state_key(sea_state),
        {
            "risk_level": "Unknown",
            "message": "No recommendation available for this sea state.",
        },
    )


def get_weather_suitability(sea_state: str, confidence: float, visibility: str):
    sea_key = _sea_state_key(sea_state)

    if sea_key == "calm":
        operations = ["Navigation", "Cargo Transport", "Fishing", "Patrol Boats"]
        score = 96
    elif sea_key == "moderate":
        operations = ["Navigation", "Cargo Transport", "Fishing"]
        score = 74
    elif sea_key == "rough":
        operations = ["Navigation with caution", "Limited cargo operations"]
        score = 42
    else:
        operations = ["Emergency Operations Only"]
        score = 12

    if visibility == "Moderate visibility":
        score -= 10
    elif visibility == "Poor visibility":
        score -= 20

    if confidence < 70:
        score -= 10

    score = max(0, min(100, score))

    if score >= 80:
        condition = "Favorable"
    elif score >= 55:
        condition = "Moderate"
    else:
        condition = "Unfavorable"

    if sea_key == "calm":
        reason = "Calm sea conditions with minimal wave activity. Marine operations can be performed safely."
    elif sea_key == "moderate":
        reason = "Moderate waves detected. Most marine operations remain possible with normal precautions."
    elif sea_key == "rough":
        reason = "High wave activity detected. Operations should be limited due to increased safety risks."
    else:
        reason = "Very rough sea conditions detected. Only emergency operations should be considered."

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
        "reason": reason,
    }


def calculate_risk_indicator(sea_state, confidence, image_quality):
    base_scores = {
        "calm": 20,
        "moderate": 45,
        "rough": 75,
        "very_rough": 95,
    }

    score = base_scores.get(_sea_state_key(sea_state), 50)
    reasons = []

    if confidence < 50:
        score += 20
        reasons.append("Low prediction confidence")
    elif confidence < 70:
        score += 10
        reasons.append("Moderate prediction confidence")

    visibility = image_quality["visibility_status"]
    if visibility == "Poor visibility":
        score += 10
        reasons.append("Poor visibility")

    if image_quality["brightness_status"] != "Normal":
        score += 5
        reasons.append("Suboptimal brightness")

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

    return {"score": score, "level": level, "reasons": reasons}


def get_sea_confidence_warnings(confidence: float, visibility_status: str):
    warnings = []

    if confidence < 70:
        warnings.append(
            "Prediction confidence is low. Use a clearer image or verify with manual observation."
        )

    if visibility_status == "Poor visibility":
        warnings.append(
            "Image quality indicates poor visibility. Prediction may be less reliable."
        )
    elif visibility_status == "Moderate visibility":
        warnings.append(
            "Image quality is moderate. Prediction should be interpreted with caution."
        )

    if not warnings:
        warnings.append("Prediction confidence and image quality are acceptable.")

    return warnings


def save_sea_history_record(record):
    if sea_state_collection is not None:
        try:
            record_to_save = record.copy()
            result = sea_state_collection.insert_one(record_to_save)
            return str(result.inserted_id)
        except Exception as e:
            print("Error saving sea-state prediction to MongoDB; using JSON fallback:", e)

    history = _load_local_sea_history()
    history.append(record)
    # Keep local history bounded for a lightweight research prototype.
    history = history[-500:]
    _save_local_sea_history(history)
    return None


def save_hull_prediction_record(record):
    if hull_prediction_collection is not None:
        try:
            result = hull_prediction_collection.insert_one(record)
            return str(result.inserted_id)

        except Exception as e:
            print(
                "Error saving hull prediction to MongoDB; "
                "using JSON fallback:",
                e
            )

    history = _load_local_hull_history()
    history.append(record)

    # Keep local history bounded.
    history = history[-500:]

    _save_local_hull_history(history)

    return None


def save_vessel_prediction_record(record):
    if vessel_prediction_collection is not None:
        try:
            result = vessel_prediction_collection.insert_one(record.copy())
            return str(result.inserted_id)
        except Exception as e:
            print("Error saving vessel prediction to MongoDB; using JSON fallback:", e)

    history = _load_local_vessel_history()
    history.append(record)
    _save_local_vessel_history(history[-500:])
    return None


@app.get("/vessel-detection-history", dependencies=[Depends(require_authenticated_request)])
def get_vessel_detection_history():
    if vessel_prediction_collection is not None:
        try:
            history = list(
                vessel_prediction_collection.find({}, {"_id": 0}).sort("timestamp", -1)
            )
            return {"history": history, "storage": "mongodb"}
        except Exception as e:
            print("Error loading vessel history from MongoDB; using JSON fallback:", e)

    return {
        "history": list(reversed(_load_local_vessel_history())),
        "storage": "json",
    }


@app.delete("/vessel-detection-history", dependencies=[Depends(require_authenticated_request)])
def clear_vessel_detection_history():
    deleted_count = 0
    if vessel_prediction_collection is not None:
        try:
            deleted_count = vessel_prediction_collection.delete_many({}).deleted_count
        except Exception as e:
            print("Error clearing vessel history from MongoDB:", e)

    local_history = _load_local_vessel_history()
    deleted_count += len(local_history)
    _save_local_vessel_history([])
    return {"message": "Vessel prediction history cleared successfully", "deleted_count": deleted_count}

@app.get(
    "/hull-prediction-history",
    dependencies=[Depends(require_authenticated_request)]
)
def get_hull_prediction_history():

    if hull_prediction_collection is not None:
        try:
            history = list(
                hull_prediction_collection.find(
                    {},
                    {"_id": 0}
                ).sort(
                    "timestamp",
                    -1
                )
            )

            return {
                "history": history,
                "storage": "mongodb"
            }

        except Exception as e:
            print(
                "Error loading hull prediction history from MongoDB; "
                "using JSON fallback:",
                e
            )

    history = list(
        reversed(
            _load_local_hull_history()
        )
    )

    return {
        "history": history,
        "storage": "json"
    }

@app.delete(
    "/hull-prediction-history",
    dependencies=[Depends(require_authenticated_request)]
)
def clear_hull_prediction_history():

    deleted_count = 0

    if hull_prediction_collection is not None:
        try:
            result = hull_prediction_collection.delete_many({})

            deleted_count = result.deleted_count

        except Exception as e:
            print(
                "Error clearing hull prediction history from MongoDB:",
                e
            )

    local_history = _load_local_hull_history()

    if local_history:
        deleted_count += len(local_history)

    _save_local_hull_history([])

    return {
        "message": "Hull prediction history cleared successfully",
        "deleted_count": deleted_count
    }


@app.post("/predict-sea-state", dependencies=[Depends(require_authenticated_request)])
async def predict_sea_state(
    file: UploadFile = File(...),
    apply_enhancement: bool = Form(False),
):
    if sea_model is None:
        return {"error": "Sea-state model not loaded"}

    try:
        start_time = time.perf_counter()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        safe_filename = Path(file.filename or "sea_image").name
        file_path = os.path.join(UPLOAD_DIR, safe_filename)

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        original_image = Image.open(file_path).convert("RGB")

        validation = validate_sea_image(original_image)
        if not validation["is_valid"]:
            return {
                "error": "Invalid image. Please upload an image that clearly shows the sea surface.",
                "validation": validation,
            }

        quality_report = analyze_sea_image_quality(original_image)

        if apply_enhancement:
            prediction_image = enhance_sea_image(original_image)
            enhanced_path = os.path.join(UPLOAD_DIR, "enhanced_" + safe_filename)
            prediction_image.save(enhanced_path)
        else:
            prediction_image = original_image

        image = sea_transform(prediction_image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = sea_model(image)
            probabilities = torch.softmax(output, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)

        predicted_class = predicted_class.item()
        predicted_label = reverse_label_map[predicted_class]
        confidence_percent = round(confidence.item() * 100, 2)

        class_probabilities = {}
        for i, prob in enumerate(probabilities[0]):
            label = reverse_label_map[i]
            class_probabilities[label] = round(prob.item() * 100, 2)

        weather_suitability = get_weather_suitability(
            predicted_label,
            confidence_percent,
            quality_report["visibility_status"],
        )
        risk_indicator = calculate_risk_indicator(
            predicted_label,
            confidence_percent,
            quality_report,
        )
        recommendation = get_sea_state_recommendation(predicted_label)
        warnings = get_sea_confidence_warnings(
            confidence_percent,
            quality_report["visibility_status"],
        )

        processing_time = round(time.perf_counter() - start_time, 3)

        result = {
            "timestamp": timestamp,
            "filename": safe_filename,
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
            "warnings": warnings,
        }

        save_sea_history_record(result)
        return result

    except Exception as e:
        traceback.print_exc()
        return {"error": f"Sea-state prediction failed: {str(e)}"}


@app.get("/sea-state-history", dependencies=[Depends(require_authenticated_request)])
def get_sea_state_history():
    if sea_state_collection is not None:
        try:
            history = list(
                sea_state_collection.find({}, {"_id": 0}).sort("timestamp", -1)
            )
            return {"history": history, "storage": "mongodb"}
        except Exception as e:
            print("Error loading sea-state history from MongoDB; using JSON fallback:", e)

    history = list(reversed(_load_local_sea_history()))
    return {"history": history, "storage": "json"}


@app.delete("/sea-state-history", dependencies=[Depends(require_authenticated_request)])
def clear_sea_state_history():
    deleted_count = 0

    if sea_state_collection is not None:
        try:
            result = sea_state_collection.delete_many({})
            deleted_count = result.deleted_count
        except Exception as e:
            print("Error clearing MongoDB sea-state history:", e)

    local_history = _load_local_sea_history()
    deleted_count += len(local_history)
    _save_local_sea_history([])

    return {
        "message": "Sea-state prediction history cleared successfully",
        "deleted_count": deleted_count,
    }


# =====================================================
# VESSEL DETECTION MODEL - YOLO
# Class order for the retrained model: boat, ship, sl_flag.
# =====================================================
boat_model = None

VESSEL_CLASS_BOAT = 0
VESSEL_CLASS_SHIP = 1
VESSEL_CLASS_SL_FLAG = 2


def infer_vessel_origin(detections):
    if not detections:
        return "Unknown"

    top_detection = max(detections, key=lambda item: item.get("confidence", 0))
    label = (top_detection.get("label") or "").lower()

    if "local ship" in label:
        return "Local Ship"
    if "foreign ship" in label:
        return "Foreign Ship"
    if "local boat" in label:
        return "Local Boat"
    if "foreign boat" in label:
        return "Foreign Boat"
    return "Unknown"


def infer_vessel_classifications(detections):
    classifications = []
    for detection in detections or []:
        label = detection.get("label") or ""
        normalized = label.strip().lower()
        if normalized in {"local boat", "foreign boat", "local ship", "foreign ship"}:
            classification = " ".join(word.capitalize() for word in normalized.split())
            if classification not in classifications:
                classifications.append(classification)
    return classifications or ([infer_vessel_origin(detections)] if detections else [])


def infer_estimated_size(detections):
    if not detections:
        return "No Vessel"

    top_detection = max(detections, key=lambda item: item.get("confidence", 0))
    label = (top_detection.get("label") or "").lower()
    confidence = top_detection.get("confidence", 0)

    if confidence >= 85 and any(keyword in label for keyword in ["cargo", "container", "ship", "vessel"]):
        return "Large Vessel"

    if any(keyword in label for keyword in ["small", "fishing"]):
        return "Small Vessel"

    return "Medium Vessel"


def ship_has_local_flag(ship_box, flag_boxes, min_overlap=0.25):
    """Match an sl_flag to a ship using the fraction of flag area overlapped."""
    flag_x1, flag_y1, flag_x2, flag_y2 = flag_boxes
    flag_area = max(0, flag_x2 - flag_x1) * max(0, flag_y2 - flag_y1)
    if flag_area <= 0:
        return False

    ship_x1, ship_y1, ship_x2, ship_y2 = ship_box
    intersection = max(0, min(ship_x2, flag_x2) - max(ship_x1, flag_x1)) * max(
        0,
        min(ship_y2, flag_y2) - max(ship_y1, flag_y1),
    )
    return intersection / flag_area >= min_overlap


def build_demo_boat_detection_response():
    return {
        "image": "demo-drone-scene.jpg",
        "status": "Detected",
        "confidence": 89.4,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source": "Drone",
        "estimated_size": "Large Vessel",
        "vessel_origin": "Local Boat",
        "results": [
            {
                "label": "Cargo Vessel",
                "confidence": 89.4,
            }
        ],
        "count": 1,
        "demo": True,
    }


try:
    if not BOAT_MODEL_PATH.exists():
        print(
            "ERROR: No boat detection "
            "weights file found!"
        )
        boat_model = None

    else:
        print(
            "Loading boat detection "
            "model from:",
            BOAT_MODEL_PATH,
        )

        boat_model = YOLO(
            str(BOAT_MODEL_PATH)
        )

        print(
            "Boat detection model "
            "loaded successfully"
        )

except Exception as e:
    print(
        "ERROR loading "
        "boat model:",
        e,
    )
    boat_model = None


@app.get("/demo-boat-detection")
async def get_demo_boat_detection():
    return build_demo_boat_detection_response()


@app.post("/predict-boat-detection", dependencies=[Depends(require_authenticated_request)])
async def predict_boat_detection(
    file: UploadFile = File(...),
):
    if boat_model is None:
        return {
            "error": (
                "Boat detection model "
                "not loaded"
            )
        }
    
    try:
        if not file.filename:
            return {
                "error": "No filename"
            }

        filename_lower = (
            file.filename.lower()
        )

        if not filename_lower.endswith(
            (
                ".png",
                ".jpg",
                ".jpeg",
                ".bmp",
            )
        ):
            return {
                "error": (
                    "Invalid file type: "
                    f"{filename_lower}"
                )
            }

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".png",
        ) as tmp:
            contents = (
                await file.read()
            )

            tmp.write(contents)
            tmp_path = tmp.name

        results = boat_model(
            tmp_path,
            conf=0.50,
            iou=0.45,
            save=False,
            verbose=False,
        )

        detections = []

        if (
            results
            and len(results) > 0
        ):
            result = results[0]

            if (
                result.boxes is not None
                and len(
                    result.boxes
                ) > 0
            ):
                for box in result.boxes:
                    try:
                        cls_id = (
                            int(
                                box.cls.item()
                            )
                            if box.cls
                            is not None
                            else 0
                        )

                        conf = (
                            float(
                                box.conf.item()
                            )
                            if box.conf
                            is not None
                            else 0
                        )

                        if conf < 0.50:
                            continue

                        label = (
                            boat_model.names.get(
                                cls_id,
                                f"class_{cls_id}",
                            )
                        )

                        detections.append({
                            "class_id": cls_id,
                            "label": label,
                            "confidence": round(conf * 100, 1),
                            "box": [float(value) for value in box.xyxy[0].tolist()],
                        })

                    except Exception as e:
                        print(
                            "Error parsing box:",
                            e,
                        )

        Path(
            tmp_path
        ).unlink(
            missing_ok=True
        )

        flag_boxes = [
            detection["box"]
            for detection in detections
            if detection["class_id"] == VESSEL_CLASS_SL_FLAG
        ]
        image_size = None
        if results and len(results) > 0 and results[0].orig_shape:
            image_height, image_width = results[0].orig_shape[:2]
            image_size = [int(image_width), int(image_height)]
        vessel_detections = []
        flag_detections = []
        for detection in detections:
            if detection["class_id"] == VESSEL_CLASS_SL_FLAG:
                flag_detections.append({
                    "label": "SL Flag",
                    "confidence": detection["confidence"],
                    "box": detection["box"],
                    "detection_type": "flag",
                })
                continue
            if detection["class_id"] not in (VESSEL_CLASS_BOAT, VESSEL_CLASS_SHIP):
                continue

            is_local = any(
                ship_has_local_flag(detection["box"], flag_box)
                for flag_box in flag_boxes
            )
            vessel_type = (
                "Boat"
                if detection["class_id"] == VESSEL_CLASS_BOAT
                else "Ship"
            )
            vessel_detections.append({
                "label": f"{'Local' if is_local else 'Foreign'} {vessel_type}",
                "confidence": detection["confidence"],
                "sl_flag_detected": is_local,
                "box": detection["box"],
                "detection_type": vessel_type.lower(),
            })

        vessel_detections = sorted(
            vessel_detections,
            key=lambda detection: detection["confidence"],
            reverse=True,
        )[:1]
        flag_detections = sorted(
            flag_detections,
            key=lambda detection: detection["confidence"],
            reverse=True,
        )[:1]
        all_detections = vessel_detections + flag_detections

        confidence_value = round(max((d["confidence"] for d in vessel_detections), default=0), 1)

        response = {
            "image": file.filename,
            "status": "Detected" if all_detections else "No Vessel Detected",
            "confidence": confidence_value,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "source": "Drone",
            "estimated_size": infer_estimated_size(vessel_detections),
            "vessel_origin": infer_vessel_origin(vessel_detections),
            "results": all_detections,
            "count": len(vessel_detections),
            "image_size": image_size,
            "demo": False,
        }
        save_vessel_prediction_record({
            **response,
            "module": "vessel",
            "mode": "image",
        })
        return response

    except Exception as e:
        print(
            "ERROR:",
            e,
        )

        traceback.print_exc()

        return {
            "error": (
                "Inference failed: "
                f"{str(e)}"
            )
        }


@app.post("/predict-boat-video", dependencies=[Depends(require_authenticated_request)])
async def predict_boat_video(file: UploadFile = File(...)):
    """Detect vessels in every video frame and return an annotated video."""
    if boat_model is None:
        return {"error": "Boat detection model not loaded"}

    allowed_extensions = (".mp4", ".avi", ".mov", ".mkv", ".webm")
    filename = Path(file.filename or "").name
    if not filename.lower().endswith(allowed_extensions):
        return {"error": "Invalid video file type"}

    input_path = TEMP_DIR / f"{uuid.uuid4().hex}_{filename}"
    output_name = f"boat_detection_{uuid.uuid4().hex}.mp4"
    output_path = TEMP_DIR / output_name

    try:
        input_path.write_bytes(await file.read())
        capture = cv2.VideoCapture(str(input_path))
        if not capture.isOpened():
            return {"error": "Unable to read video file"}

        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = capture.get(cv2.CAP_PROP_FPS)
        fps = fps if fps and fps > 0 else 25.0
        writer = cv2.VideoWriter(
            str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )
        if not writer.isOpened():
            capture.release()
            return {"error": "Unable to create annotated video"}

        detected_labels = {}
        frame_count = 0
        while True:
            success, frame = capture.read()
            if not success:
                break
            frame_results = boat_model(
                frame, conf=0.50, iou=0.45, save=False, verbose=False
            )
            if frame_results:
                frame_result = frame_results[0]
                if frame_result.boxes is not None:
                    frame_detections = []
                    for box in frame_result.boxes:
                        confidence = float(box.conf.item())
                        if confidence < 0.50:
                            continue
                        class_id = int(box.cls.item())
                        frame_detections.append({
                            "class_id": class_id,
                            "confidence": confidence * 100,
                            "box": [float(value) for value in box.xyxy[0].tolist()],
                        })

                    flag_boxes = [
                        detection["box"]
                        for detection in frame_detections
                        if detection["class_id"] == VESSEL_CLASS_SL_FLAG
                    ]
                    for detection in frame_detections:
                        if detection["class_id"] not in (VESSEL_CLASS_BOAT, VESSEL_CLASS_SHIP):
                            continue

                        is_local = any(
                            ship_has_local_flag(detection["box"], flag_box)
                            for flag_box in flag_boxes
                        )
                        vessel_type = (
                            "Boat"
                            if detection["class_id"] == VESSEL_CLASS_BOAT
                            else "Ship"
                        )
                        label = f"{'Local' if is_local else 'Foreign'} {vessel_type}"
                        detected_labels[label] = detected_labels.get(label, 0) + 1
                frame = frame_result.plot()
            writer.write(frame)
            frame_count += 1

        capture.release()
        writer.release()
        if frame_count == 0 or not output_path.exists():
            return {"error": "Video contains no readable frames"}

        # OpenCV's mp4v output is not decoded by some browsers. Re-encode it
        # as H.264 with a browser-friendly pixel format and metadata order.
        browser_output_path = TEMP_DIR / f"browser_{output_name}"
        try:
            if imageio_ffmpeg is None:
                raise RuntimeError("imageio-ffmpeg is not installed")
            subprocess.run(
                [
                    imageio_ffmpeg.get_ffmpeg_exe(),
                    "-y",
                    "-i",
                    str(output_path),
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    "-an",
                    str(browser_output_path),
                ],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            output_path.unlink(missing_ok=True)
            browser_output_path.replace(output_path)
        except (OSError, subprocess.CalledProcessError) as conversion_error:
            browser_output_path.unlink(missing_ok=True)
            print("Browser video conversion failed; keeping OpenCV output:", conversion_error)

        results = [
            {"label": label, "frames_detected": count}
            for label, count in sorted(
                detected_labels.items(), key=lambda item: item[1], reverse=True
            )
        ]
        response = {
            "image": filename,
            "status": "Detected" if results else "No Vessel Detected",
            "results": results,
            "video_detections": results,
            "count": len(results),
            "frame_count": frame_count,
            "video_url": f"/boat-detection-video/{output_name}",
        }
        response["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        response["vessel_origin"] = (
            "Local" if any(label.startswith("Local") for label in detected_labels)
            else "Foreign" if detected_labels else "Unknown"
        )
        response["vessel_classifications"] = infer_vessel_classifications(
            [{"label": label} for label in detected_labels]
        )
        save_vessel_prediction_record({
            **response,
            "module": "vessel",
            "mode": "video",
        })
        return response
    except Exception as e:
        traceback.print_exc()
        output_path.unlink(missing_ok=True)
        return {"error": f"Video inference failed: {str(e)}"}
    finally:
        input_path.unlink(missing_ok=True)


@app.get("/boat-detection-video/{filename}", dependencies=[Depends(require_authenticated_request)])
async def get_boat_detection_video(filename: str):
    safe_name = Path(filename).name
    video_path = TEMP_DIR / safe_name
    if not video_path.exists() or video_path.suffix.lower() != ".mp4":
        return {"error": "Annotated video not found"}
    return FileResponse(video_path, media_type="video/mp4", filename=safe_name)


# =====================================================
# RADAR OBJECT CLASSIFICATION MODEL â€” V4 RAW
# MobileNetV3-Small
#
# Input:
#   Raw RGB radar screenshot
#
# Classes:
#   bird / ship
#
# Unknown:
#   Confidence-based abstention using the validation-frozen
#   threshold of 0.85.
#
# IMPORTANT:
#   No Viridis/heatmap conversion is used.
# =====================================================

RADAR_CLASSES = [
    "bird",
    "ship",
    "unknown",
]

radar_model = None


# ============================================================
# FINAL RADAR PREPROCESSING
#
# Must match train_radar_unknown3.py / evaluation:
#   RGB
#   Resize 64 x 64
#   ToTensor
#
# No Viridis heatmaps.
# No ImageNet normalisation.
# ============================================================

radar_transform = transforms.Compose(
    [
        transforms.Resize(
            (64, 64)
        ),
        transforms.ToTensor(),
    ]
)


class RadarTargetCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(
                3,
                8,
                kernel_size=5,
                stride=2,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(
                8,
                12,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((4, 4)),
        )

        self.dropout = nn.Dropout(
            0.30
        )

        self.fc = nn.Linear(
            12 * 4 * 4,
            3,
        )

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        x = self.dropout(x)
        return self.fc(x)


def build_radar_final_model():
    return RadarTargetCNN()


try:
    if RADAR_MODEL_PATH.exists():

        print(
            "Loading final Radar model from:",
            RADAR_MODEL_PATH,
        )

        radar_model = (
            build_radar_final_model()
        )

        radar_model.load_state_dict(
            torch.load(
                RADAR_MODEL_PATH,
                map_location=device,
                weights_only=True,
            )
        )

        radar_model.to(device)
        radar_model.eval()

        print(
            "Final Radar model loaded successfully"
        )

        print(
            "Radar model version:",
            RADAR_MODEL_VERSION,
        )

    else:
        print(
            "ERROR: Final Radar model not found:",
            RADAR_MODEL_PATH,
        )

except Exception as e:
    print(
        "ERROR loading final Radar model:",
        e,
    )

    traceback.print_exc()

    radar_model = None


def classify_radar_image_path(
    image_path: Path,
    display_name: str | None = None,
):
    """
    Shared final Radar inference pipeline.

    Used by:
      1. Authenticated /predict-radar-object
      2. Internal Live Maritime Simulation

    The model receives the original RGB radar screenshot.
    No heatmap preprocessing is performed.
    """

    if radar_model is None:
        raise RuntimeError(
            "Final Radar model not loaded"
        )

    image_path = Path(
        image_path
    )

    filename = (
        display_name
        if display_name
        else image_path.name
    )

    filename_lower = (
        filename.lower()
    )

    if not filename_lower.endswith(
        (
            ".png",
            ".jpg",
            ".jpeg",
            ".bmp",
        )
    ):
        raise ValueError(
            "Invalid Radar image type: "
            f"{filename_lower}"
        )

    image = (
        Image.open(
            image_path
        )
        .convert("RGB")
    )

    input_tensor = (
        radar_transform(
            image
        )
        .unsqueeze(0)
        .to(device)
    )

    with torch.inference_mode():

        output = radar_model(
            input_tensor
        )

        probabilities = (
            torch.softmax(
                output,
                dim=1,
            )
        )

        (
            confidence,
            prediction,
        ) = probabilities.max(
            dim=1
        )

    predicted_index = int(
        prediction.item()
    )

    confidence_value = float(
        confidence.item()
    )

    model_prediction = (
        RADAR_CLASSES[
            predicted_index
        ]
    )

    # Legacy field retained for frontend, history,
    # simulation and PDF-report compatibility.
    binary_prediction = model_prediction

    bird_probability = float(
        probabilities[
            0,
            0,
        ].item()
    )

    ship_probability = float(
        probabilities[
            0,
            1,
        ].item()
    )

    unknown_probability = float(
        probabilities[
            0,
            2,
        ].item()
    )

    final_prediction = model_prediction

    if final_prediction == "unknown":
        decision_status = (
            "Three-class model identified "
            "the input as unknown"
        )
    else:
        decision_status = (
            "Three-class Radar "
            "classification accepted"
        )

    return {
        "image":
            filename,

        "binary_prediction":
            binary_prediction,

        "confidence":
            round(
                confidence_value
                * 100,
                2,
            ),

        "bird_probability":
            round(
                bird_probability
                * 100,
                2,
            ),

        "ship_probability":
            round(
                ship_probability
                * 100,
                2,
            ),

        "unknown_probability":
            round(
                unknown_probability
                * 100,
                2,
            ),
        "final_prediction":
            final_prediction,

        "decision_status":
            decision_status,

        "model_name":
            "RadarTargetCNN",
        "model_version":
            RADAR_MODEL_VERSION,
        "unknown_threshold":
            None,
        "heatmap_preprocessing":
            False,

        # Final RadarTargetCNN evaluation metrics.
        "model_accuracy":
            72.07,
        "validation_accuracy":
            70.91,
        "macro_precision":
            0.7316,
        "macro_recall":
            0.7207,
        "macro_f1":
            0.6955,
        "test_samples":
            333,
        "unknown_recall":
            32.43,
        "test_coverage":
            100.00,
    }


@app.post(
    "/predict-radar-object",
    dependencies=[
        Depends(
            require_authenticated_request
        )
    ],
)
async def predict_radar_object(
    file: UploadFile = File(...),
):
    """
    Authenticated external Radar classification endpoint.

    Uploaded files are saved temporarily and then passed through
    the same internal classifier used by Live Simulation.
    """

    raw_path = None

    try:
        if not file.filename:
            return {
                "error": (
                    "No filename provided"
                )
            }

        safe_filename = Path(
            file.filename
        ).name

        raw_path = (
            TEMP_DIR
            / (
                f"{uuid.uuid4()}_"
                f"{safe_filename}"
            )
        )

        with open(
            raw_path,
            "wb",
        ) as buffer:
            shutil.copyfileobj(
                file.file,
                buffer,
            )

        result = classify_radar_image_path(
            raw_path,
            display_name=file.filename,
        )

        # -------------------------------------------------
        # RADAR PREDICTION PERSISTENCE
        # MongoDB failure must never break inference.
        # -------------------------------------------------
        if radar_prediction_collection is not None:
            try:
                from datetime import datetime, timezone

                radar_record = {
                    "timestamp": datetime.now(
                        timezone.utc
                    ),
                    "source": "manual_upload",
                    "filename": safe_filename,
                    "model_name": result.get(
                        "model_name"
                    ),
                    "model_version": result.get(
                        "model_version"
                    ),
                    "binary_prediction": result.get(
                        "binary_prediction"
                    ),
                    "final_prediction": result.get(
                        "final_prediction"
                    ),
                    "confidence": result.get(
                        "confidence"
                    ),
                    "bird_probability": result.get(
                        "bird_probability"
                    ),
                    "ship_probability": result.get(
                        "ship_probability"
                    ),
                    "unknown_probability": result.get(
                        "unknown_probability"
                    ),
                    "decision_status": result.get(
                        "decision_status"
                    ),
                    "validation_accuracy": result.get(
                        "validation_accuracy"
                    ),
                    "test_accuracy": result.get(
                        "model_accuracy"
                    ),
                    "macro_f1": result.get(
                        "macro_f1"
                    ),
                }

                insert_result = (
                    radar_prediction_collection
                    .insert_one(
                        radar_record
                    )
                )

                result[
                    "database_record_id"
                ] = str(
                    insert_result.inserted_id
                )

            except Exception as db_error:
                print(
                    "Radar MongoDB save failed:",
                    db_error,
                )

        return result

    except Exception as e:
        print(
            "ERROR in radar "
            "object prediction:",
            e,
        )

        traceback.print_exc()

        return {
            "error": (
                "Radar object prediction "
                f"failed: {str(e)}"
            )
        }

    finally:
        if raw_path is not None:
            Path(raw_path).unlink(
                missing_ok=True
            )



@app.get(
    "/radar/history",
    dependencies=[
        Depends(require_authenticated_request)
    ],
)
def radar_history(
    limit: int = 20,
):
    if radar_prediction_collection is None:
        return {
            "database_available": False,
            "records": [],
        }

    try:
        limit = max(
            1,
            min(int(limit), 200),
        )

        records = list(
            radar_prediction_collection
            .find(
                {},
                {"_id": 0},
            )
            .sort(
                "timestamp",
                -1,
            )
            .limit(limit)
        )

        for record in records:
            timestamp = record.get(
                "timestamp"
            )

            if hasattr(
                timestamp,
                "isoformat",
            ):
                record["timestamp"] = (
                    timestamp.isoformat()
                )

        return {
            "database_available": True,
            "count": len(records),
            "records": records,
        }

    except Exception as error:
        print(
            "Radar history MongoDB read failed:",
            error,
        )

        return {
            "database_available": False,
            "records": [],
            "error": str(error),
        }


# Register the shared Radar classifier with the Live Simulation.
# The simulation therefore performs trusted in-process inference
# and does NOT bypass or weaken the HTTP authentication layer.
from simulation.sar_streamer import (
    configure_internal_radar_classifier,
)

configure_internal_radar_classifier(
    classify_radar_image_path
)


# =====================================================
# SIMULATION / AIS / GRU / COLLISION-RISK ROUTER
# =====================================================
# Imported only after the existing OceanIQ prediction
# routes are defined. backend/simulation/app.py no longer
# creates another FastAPI application.
# =====================================================
from simulation.app import (
    router as simulation_router,
    configure_simulation_persistence,
)

configure_simulation_persistence(
    simulation_runs_collection,
    simulation_events_collection,
)

app.include_router(
    simulation_router,
    dependencies=[Depends(require_authenticated_request)],
)


# =====================================================
# UNIFIED HEALTH ROUTE
# =====================================================
@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": (
            "OceanIQ Marine AI "
            "Intelligence Platform"
        ),
        "backend_mode": (
            "single-integrated-backend"
        ),
        "models": {
            "hull_model_loaded": (
                hull_model is not None
            ),
            "hull_validator_loaded": (
                hull_validator_model is not None
            ),
            "sea_model_loaded": (
                sea_model is not None
            ),
            "boat_model_loaded": (
                boat_model is not None
            ),
            "radar_yolo_loaded": (
                radar_yolo_model
                is not None
            ),
            "radar_cnn_loaded": (
                radar_cnn_model
                is not None
            ),
        },
        "simulation_health": (
            "/simulation/health"
        ),
    }


# =====================================================
# HOME ROUTE
# =====================================================
@app.get("/api-info")
def home():
    return {
        "message": (
            "OceanIQ Marine AI "
            "Intelligence Platform "
            "API is running"
        ),
        "backend": (
            "single integrated FastAPI "
            "application"
        ),
        "endpoints": {
            "health": "/health",
            "hull_defect": (
                "/predict-hull-defect"
            ),
            "sea_state": (
                "/predict-sea-state"
            ),
            "boat_detection": (
                "/predict-boat-detection"
            ),
            "radar_object_classification": (
                "/predict-radar-object"
            ),
            "simulation_health": (
                "/simulation/health"
            ),
            "simulation_start": (
                "/simulation/start"
            ),
            "simulation_stop": (
                "/simulation/stop"
            ),
            "simulation_status": (
                "/simulation/status"
            ),
            "simulation_latest": (
                "/simulation/latest"
            ),
            "simulation_history": (
                "/simulation/history"
            ),
            "simulation_current_image": (
                "/simulation/current-image"
            ),
            "vessel_motion": (
                "/predict-vessel-motion"
            ),
            "collision_risk": (
                "/predict-collision-risk"
            ),
        },
    }


# ============================================================
# PRODUCTION REACT FRONTEND
# Keep this section LAST so existing API routes take priority.
# ============================================================

FRONTEND_DIST = BASE_PATH / "frontend" / "dist"

if FRONTEND_DIST.exists():
    assets_dir = FRONTEND_DIST / "assets"

    if assets_dir.exists():
        app.mount(
            "/assets",
            StaticFiles(directory=str(assets_dir)),
            name="frontend-assets",
        )

    @app.get("/{full_path:path}", include_in_schema=False)
    async def serve_react_app(full_path: str):
        requested_file = FRONTEND_DIST / full_path

        if full_path and requested_file.is_file():
            return FileResponse(str(requested_file))

        return FileResponse(
            str(FRONTEND_DIST / "index.html")
        )
