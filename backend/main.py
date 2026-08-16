import os
import json
import shutil
import base64
import tempfile
import traceback
import uuid
from pathlib import Path

import tensorflow as tf
import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import MobileNet_V2_Weights
from torchvision import transforms
from PIL import Image

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from ultralytics import YOLO


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

BOAT_MODEL_PATH = (
    BASE_PATH
    / "model"
    / "boat_detection.pt"
)

if not BOAT_MODEL_PATH.exists():
    BOAT_MODEL_PATH = (
        BASE_PATH
        / "model"
        / "boat_detection_last.pt"
    )

RADAR_YOLO_MODEL_PATH = (
    BASE_PATH
    / "ml"
    / "models"
    / "final"
    / "yolo11_medium_best.pt"
)

RADAR_CNN_MODEL_PATH = (
    BASE_PATH
    / "ml"
    / "models"
    / "final"
    / "deepercnn_best.pth"
)


# =====================================================
# HULL DEFECT MODEL - TENSORFLOW
# =====================================================
hull_model = None

hull_classes = [
    "biofouling",
    "corrosion",
    "cracks",
]

recommendations = {
    "biofouling": (
        "Clean hull using high-pressure water "
        "or antifouling treatment."
    ),
    "corrosion": (
        "Apply anti-corrosion coating or "
        "replace damaged metal."
    ),
    "cracks": (
        "Critical damage. Perform welding "
        "repair immediately."
    ),
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


def make_gradcam_heatmap(
    img_array,
    model,
    last_conv_layer_name,
    pred_index=None,
):
    img_tensor = tf.convert_to_tensor(
        img_array,
        dtype=tf.float32,
    )

    conv_outputs = None

    with tf.GradientTape() as tape:
        x = img_tensor

        for layer in model.layers:
            x = layer(x)

            if (
                layer.name
                == last_conv_layer_name
            ):
                conv_outputs = x
                tape.watch(conv_outputs)

        predictions = x

        if pred_index is None:
            pred_index = tf.argmax(
                predictions[0]
            )

        loss = predictions[
            :,
            pred_index,
        ]

    grads = tape.gradient(
        loss,
        conv_outputs,
    )

    pooled_grads = tf.reduce_mean(
        grads,
        axis=(0, 1, 2),
    )

    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(
        conv_outputs
        * pooled_grads,
        axis=-1,
    )

    heatmap = tf.maximum(
        heatmap,
        0,
    )

    heatmap = heatmap / (
        tf.reduce_max(heatmap)
        + 1e-8
    )

    return heatmap.numpy()


@app.post("/predict-hull-defect")
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

        img_resized = (
            cv2.resize(
                img,
                (224, 224),
            )
            / 255.0
        )

        img_array = np.expand_dims(
            img_resized,
            axis=0,
        )

        preds = hull_model.predict(
            img_array
        )

        class_idx = np.argmax(preds)
        prediction = hull_classes[
            class_idx
        ]

        confidence = float(
            np.max(preds)
        )

        LAST_CONV_LAYER = (
            "mobilenetv2_1.00_224"
        )

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
            if confidence < 0.7
            else "Prediction reliable."
        )

        return {
            "prediction": prediction,
            "confidence": confidence,
            "recommendation": (
                recommendations[
                    prediction
                ]
            ),
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

sea_transform = transforms.Compose(
    [
        transforms.Resize(
            (224, 224)
        ),
        transforms.ToTensor(),
    ]
)


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


@app.post("/predict-sea-state")
async def predict_sea_state(
    file: UploadFile = File(...),
):
    if sea_model is None:
        return {
            "error": (
                "Sea-state model "
                "not loaded"
            )
        }

    try:
        file_path = os.path.join(
            UPLOAD_DIR,
            file.filename,
        )

        with open(
            file_path,
            "wb",
        ) as buffer:
            shutil.copyfileobj(
                file.file,
                buffer,
            )

        image = (
            Image.open(
                file_path
            )
            .convert("RGB")
        )

        image = sea_transform(
            image
        )

        image = (
            image.unsqueeze(0)
            .to(device)
        )

        with torch.no_grad():
            output = sea_model(
                image
            )

            probabilities = (
                torch.softmax(
                    output,
                    dim=1,
                )
            )

            confidence, predicted_class = (
                torch.max(
                    probabilities,
                    1,
                )
            )

        predicted_class = (
            predicted_class.item()
        )

        confidence = (
            confidence.item()
            * 100
        )

        predicted_label = (
            reverse_label_map[
                predicted_class
            ]
        )

        class_probabilities = {}

        for i, prob in enumerate(
            probabilities[0]
        ):
            label = (
                reverse_label_map[i]
            )

            class_probabilities[
                label
            ] = round(
                prob.item() * 100,
                2,
            )

        return {
            "predicted_sea_state": (
                predicted_label
            ),
            "confidence": round(
                confidence,
                2,
            ),
            "probabilities": (
                class_probabilities
            ),
        }

    except Exception as e:
        traceback.print_exc()

        return {
            "error": (
                "Sea-state prediction "
                f"failed: {str(e)}"
            )
        }


# =====================================================
# BOAT DETECTION MODEL - YOLOv8
# =====================================================
boat_model = None

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


@app.post("/predict-boat-detection")
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

                        label = (
                            boat_model.names.get(
                                cls_id,
                                f"class_{cls_id}",
                            )
                        )

                        detections.append(
                            {
                                "label": label,
                                "confidence": round(
                                    conf * 100,
                                    1,
                                ),
                            }
                        )

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

        return {
            "image": file.filename,
            "results": detections,
            "count": len(
                detections
            ),
        }

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


# =====================================================
# RADAR OBJECT CLASSIFICATION MODEL
# YOLO + CNN ENSEMBLE
# =====================================================
radar_yolo_model = None
radar_cnn_model = None

RADAR_CLASSES = [
    "bird",
    "ship",
    "unknown",
]

radar_transform = (
    transforms.Compose(
        [
            transforms.Resize(
                (128, 128)
            ),
            transforms.ToTensor(),
        ]
    )
)


class RadarDeeperCNN(nn.Module):
    def __init__(
        self,
        num_classes=3,
    ):
        super(
            RadarDeeperCNN,
            self,
        ).__init__()

        self.conv1 = nn.Conv2d(
            3,
            16,
            3,
            padding=1,
        )

        self.conv2 = nn.Conv2d(
            16,
            32,
            3,
            padding=1,
        )

        self.conv3 = nn.Conv2d(
            32,
            64,
            3,
            padding=1,
        )

        self.pool = nn.MaxPool2d(
            2,
            2,
        )

        self.fc1 = nn.Linear(
            64 * 16 * 16,
            256,
        )

        self.fc2 = nn.Linear(
            256,
            num_classes,
        )

    def forward(
        self,
        x,
    ):
        x = self.pool(
            F.relu(
                self.conv1(x)
            )
        )

        x = self.pool(
            F.relu(
                self.conv2(x)
            )
        )

        x = self.pool(
            F.relu(
                self.conv3(x)
            )
        )

        x = x.view(
            x.size(0),
            -1,
        )

        x = F.relu(
            self.fc1(x)
        )

        return self.fc2(x)


def convert_radar_to_heatmap(
    input_path: Path,
    output_path: Path,
):
    image = (
        Image.open(
            input_path
        )
        .convert("L")
    )

    image = image.resize(
        (128, 128)
    )

    gray = np.array(
        image
    )

    gray = cv2.normalize(
        gray,
        None,
        0,
        255,
        cv2.NORM_MINMAX,
    )

    heatmap = cv2.applyColorMap(
        gray.astype(np.uint8),
        cv2.COLORMAP_VIRIDIS,
    )

    heatmap = cv2.cvtColor(
        heatmap,
        cv2.COLOR_BGR2RGB,
    )

    Image.fromarray(
        heatmap
    ).save(
        output_path
    )


try:
    if RADAR_YOLO_MODEL_PATH.exists():
        print(
            "Loading radar YOLO "
            "model from:",
            RADAR_YOLO_MODEL_PATH,
        )

        radar_yolo_model = YOLO(
            str(
                RADAR_YOLO_MODEL_PATH
            )
        )

        print(
            "Radar YOLO model "
            "loaded successfully"
        )

    else:
        print(
            "ERROR: Radar YOLO "
            "model not found:",
            RADAR_YOLO_MODEL_PATH,
        )

except Exception as e:
    print(
        "ERROR loading radar "
        "YOLO model:",
        e,
    )
    radar_yolo_model = None


try:
    if RADAR_CNN_MODEL_PATH.exists():
        print(
            "Loading radar CNN "
            "model from:",
            RADAR_CNN_MODEL_PATH,
        )

        radar_cnn_model = (
            RadarDeeperCNN(
                num_classes=3
            )
        )

        radar_cnn_model.load_state_dict(
            torch.load(
                RADAR_CNN_MODEL_PATH,
                map_location=device,
                weights_only=True,
            )
        )

        radar_cnn_model.to(
            device
        )

        radar_cnn_model.eval()

        print(
            "Radar CNN model "
            "loaded successfully"
        )

    else:
        print(
            "ERROR: Radar CNN "
            "model not found:",
            RADAR_CNN_MODEL_PATH,
        )

except Exception as e:
    print(
        "ERROR loading radar "
        "CNN model:",
        e,
    )
    radar_cnn_model = None


@app.post("/predict-radar-object")
async def predict_radar_object(
    file: UploadFile = File(...),
):
    if radar_yolo_model is None:
        return {
            "error": (
                "Radar YOLO model "
                "not loaded"
            )
        }

    if radar_cnn_model is None:
        return {
            "error": (
                "Radar CNN model "
                "not loaded"
            )
        }

    raw_path = None
    heatmap_path = None

    try:
        if not file.filename:
            return {
                "error": (
                    "No filename provided"
                )
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

        file_id = str(
            uuid.uuid4()
        )

        raw_path = (
            TEMP_DIR
            / f"{file_id}_{file.filename}"
        )

        heatmap_path = (
            TEMP_DIR
            / f"{file_id}_heatmap.png"
        )

        with open(
            raw_path,
            "wb",
        ) as buffer:
            shutil.copyfileobj(
                file.file,
                buffer,
            )

        convert_radar_to_heatmap(
            raw_path,
            heatmap_path,
        )

        image = (
            Image.open(
                heatmap_path
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

        with torch.no_grad():
            cnn_output = (
                radar_cnn_model(
                    input_tensor
                )
            )

            cnn_probs = (
                torch.softmax(
                    cnn_output,
                    dim=1,
                )
            )

            (
                cnn_confidence,
                cnn_pred,
            ) = torch.max(
                cnn_probs,
                1,
            )

        cnn_label = (
            RADAR_CLASSES[
                cnn_pred.item()
            ]
        )

        cnn_conf = round(
            cnn_confidence.item()
            * 100,
            2,
        )

        yolo_results = (
            radar_yolo_model(
                str(
                    heatmap_path
                ),
                verbose=False,
            )
        )

        yolo_probs = (
            yolo_results[0].probs
        )

        yolo_pred_index = int(
            yolo_probs.top1
        )

        yolo_conf = round(
            float(
                yolo_probs.top1conf
            )
            * 100,
            2,
        )

        yolo_label = (
            yolo_results[0]
            .names[
                yolo_pred_index
            ]
        )

        if (
            yolo_label
            == cnn_label
        ):
            final_prediction = (
                yolo_label
            )

            decision_status = (
                "High confidence - "
                "both models agree"
            )

        else:
            final_prediction = (
                "uncertain"
            )

            decision_status = (
                "Models disagree - "
                "manual review required"
            )

        return {
            "image": file.filename,
            "yolo_prediction": (
                yolo_label
            ),
            "yolo_confidence": (
                yolo_conf
            ),
            "cnn_prediction": (
                cnn_label
            ),
            "cnn_confidence": (
                cnn_conf
            ),
            "final_prediction": (
                final_prediction
            ),
            "decision_status": (
                decision_status
            ),
        }

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
        # Prevent backend/temp from growing forever.
        if raw_path is not None:
            Path(raw_path).unlink(
                missing_ok=True
            )

        if heatmap_path is not None:
            Path(heatmap_path).unlink(
                missing_ok=True
            )


# =====================================================
# SIMULATION / AIS / GRU / COLLISION-RISK ROUTER
# =====================================================
# Imported only after the existing OceanIQ prediction
# routes are defined. backend/simulation/app.py no longer
# creates another FastAPI application.
# =====================================================
from backend.simulation.app import (
    router as simulation_router,
)

app.include_router(
    simulation_router
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
@app.get("/")
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
