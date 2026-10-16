import re
from pathlib import Path

import cv2
import numpy as np
from fastapi import UploadFile

from .model_loader import get_vessel_info_reader


def infer_estimated_size(detections):
    """Estimate vessel size from the strongest detection."""
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


def normalize_imo_number(text):
    """Find and return a valid seven-digit IMO number."""
    digits = re.sub(r"\D", "", text or "")
    for start in range(max(0, len(digits) - 6)):
        candidate = digits[start : start + 7]
        if len(candidate) != 7:
            continue
        checksum = sum(
            int(digit) * weight
            for digit, weight in zip(candidate[:6], range(7, 1, -1))
        ) % 10
        if checksum == int(candidate[-1]):
            return f"IMO {candidate}"
    return None


async def read_uploaded_image(file: UploadFile):
    """Check the upload type and decode the image."""
    filename = Path(file.filename or "").name
    if not filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp")):
        raise ValueError(f"Invalid image file type: {filename}")

    contents = await file.read()
    image = cv2.imdecode(
        np.frombuffer(contents, dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )
    if image is None:
        raise ValueError(f"Unable to read image: {filename}")
    return image


def extract_ship_detections(result, model, confidence_threshold=0.25):
    """Convert YOLO boxes into vessel results for the API."""
    detections = []
    if result.boxes is None:
        return detections

    names = model.names
    for box in result.boxes:
        confidence = float(box.conf.item())
        if confidence < confidence_threshold:
            continue
        class_id = int(box.cls.item())
        label = (
            names.get(class_id, f"class_{class_id}")
            if isinstance(names, dict)
            else names[class_id]
        )
        detections.append({
            "class_id": class_id,
            "label": label,
            "vessel_kind": (
                "Ship"
                if label.lower() in {
                    "merchant ship",
                    "military ship",
                    "submarine",
                }
                else "Boat"
            ),
            "confidence": round(confidence * 100, 1),
            "box": [float(value) for value in box.xyxy[0].tolist()],
            "detection_type": "vessel",
        })
    return detections


def detect_vessel_info(image, vessel_info_model):
    """Read vessel name, port, and IMO text from the backside image."""
    results = vessel_info_model(
        image,
        conf=0.20,
        iou=0.45,
        save=False,
        verbose=False,
    )
    fields = {
        "vessel_name": None,
        "port": None,
        "imo_number": None,
    }
    field_scores = {}
    detections = []
    image_height, image_width = image.shape[:2]

    if (
        not results
        or results[0].boxes is None
        or len(results[0].boxes) == 0
    ):
        return {"fields": fields, "detections": detections}

    for box in results[0].boxes:
        class_id = int(box.cls.item())
        names = vessel_info_model.names
        class_name = (
            names.get(class_id, f"class_{class_id}")
            if isinstance(names, dict)
            else names[class_id]
        )
        normalized_name = re.sub(r"[^a-z]", "", class_name.lower())
        field_name = {
            "vesselname": "vessel_name",
            "port": "port",
            "imonumber": "imo_number",
        }.get(normalized_name)
        if field_name is None:
            continue

        x1, y1, x2, y2 = [int(value) for value in box.xyxy[0].tolist()]
        padding_x = max(4, int((x2 - x1) * 0.08))
        padding_y = max(4, int((y2 - y1) * 0.12))
        x1 = max(0, x1 - padding_x)
        y1 = max(0, y1 - padding_y)
        x2 = min(image_width, x2 + padding_x)
        y2 = min(image_height, y2 + padding_y)
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        enlarged = cv2.resize(
            crop,
            None,
            fx=3,
            fy=3,
            interpolation=cv2.INTER_CUBIC,
        )
        grayscale = cv2.cvtColor(enlarged, cv2.COLOR_BGR2GRAY)
        enhanced = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8),
        ).apply(grayscale)
        blurred = cv2.GaussianBlur(enhanced, (0, 0), 1.0)
        enhanced = cv2.addWeighted(enhanced, 1.6, blurred, -0.6, 0)
        allowlist = (
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
            if field_name == "imo_number"
            else None
        )
        ocr_results = get_vessel_info_reader().readtext(
            enhanced,
            detail=1,
            paragraph=False,
            allowlist=allowlist,
        )
        recognized = sorted(
            (
                (entry[0][0][1], entry[0][0][0], entry[1].strip(), float(entry[2]))
                for entry in ocr_results
                if len(entry) >= 3 and entry[1].strip()
            ),
            key=lambda entry: (entry[0], entry[1]),
        )
        text = " ".join(entry[2] for entry in recognized).strip() or None
        if field_name == "imo_number":
            text = normalize_imo_number(text)
        ocr_confidence = (
            sum(entry[3] for entry in recognized) / len(recognized)
            if recognized
            else 0.0
        )
        detection_confidence = float(box.conf.item())
        combined_score = detection_confidence * ocr_confidence
        if text and combined_score > field_scores.get(field_name, -1):
            fields[field_name] = text
            field_scores[field_name] = combined_score

        detections.append({
            "field": field_name,
            "text": text,
            "confidence": round(detection_confidence * 100, 1),
            "ocr_confidence": round(ocr_confidence * 100, 1),
            "box": [x1, y1, x2, y2],
        })

    return {"fields": fields, "detections": detections}
