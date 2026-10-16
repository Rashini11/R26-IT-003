from pathlib import Path
import threading

import torch
from ultralytics import YOLO


BASE_PATH = Path(__file__).resolve().parents[3]
VESSEL_DETECTION_MODEL_PATH = BASE_PATH / "model" / "ship-detection-model.pt"
VESSEL_INFO_MODEL_PATH = BASE_PATH / "model" / "Vessel-info-detector-model.pt"

vessel_info_reader = None
vessel_info_reader_lock = threading.Lock()


def load_vessel_model(model_path, model_label):
    """Load a YOLO model from its weights file."""
    try:
        if not model_path.exists():
            raise FileNotFoundError(f"Model weights not found: {model_path}")
        print(f"Loading {model_label} model from:", model_path)
        model = YOLO(str(model_path))
        print(f"{model_label} model loaded successfully")
        return model
    except Exception as error:
        print(f"ERROR loading {model_label} model:", error)
        return None


ship_detection_model = load_vessel_model(
    VESSEL_DETECTION_MODEL_PATH,
    "ship detection",
)
vessel_info_model = load_vessel_model(
    VESSEL_INFO_MODEL_PATH,
    "vessel information",
)


def get_vessel_info_reader():
    """Create the OCR reader once and reuse it for later images."""
    global vessel_info_reader
    if vessel_info_reader is None:
        with vessel_info_reader_lock:
            if vessel_info_reader is None:
                import easyocr

                vessel_info_reader = easyocr.Reader(
                    ["en"],
                    gpu=torch.cuda.is_available(),
                    verbose=False,
                )
    return vessel_info_reader
