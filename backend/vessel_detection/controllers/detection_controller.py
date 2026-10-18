import hashlib
import json
import os
import time
import traceback
import uuid
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np
import requests
from fastapi import Body, File, Form, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from ..services.image_detection_service import (
    detect_vessel_info,
    extract_ship_detections,
    infer_estimated_size,
    read_uploaded_image,
)
from ..services.model_loader import ship_detection_model, vessel_info_model
from ..services.port_country_service import lookup_port_country
from ..services.video_detection_service import predict_boat_video


VESSEL_HISTORY_FILE = (
    Path(__file__).resolve().parents[3]
    / "backend"
    / "vessel_prediction_history.json"
)


# Database and local file storage helpers
def _get_vessel_collection():
    """Get the vessel records collection configured by the main app."""
    from ... import main as main_module

    return main_module.vessel_prediction_collection


def _load_local_vessel_history():
    """Load saved vessel records from the local JSON backup."""
    try:
        if not VESSEL_HISTORY_FILE.exists():
            return []
        with open(VESSEL_HISTORY_FILE, "r", encoding="utf-8") as history_file:
            data = json.load(history_file)
        return data if isinstance(data, list) else []
    except Exception as error:
        print("Error loading local vessel history:", error)
        return []


def _save_local_vessel_history(history):
    """Write vessel records to the local JSON backup."""
    try:
        VESSEL_HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(VESSEL_HISTORY_FILE, "w", encoding="utf-8") as history_file:
            json.dump(history, history_file, indent=2)
        return True
    except Exception as error:
        print("Error saving local vessel history:", error)
        return False


def _ensure_vessel_history_record_ids(history):
    """Add record IDs to older saved records that do not have one."""
    normalized_history = []
    changed = False
    for record in history:
        if isinstance(record, dict) and not record.get("record_id"):
            record = {**record, "record_id": uuid.uuid4().hex}
            changed = True
        normalized_history.append(record)
    return normalized_history, changed


def save_vessel_prediction_record(record):
    """Save a new vessel record to MongoDB or the local JSON backup."""
    record = {**record}
    record_id = record.setdefault("record_id", uuid.uuid4().hex)
    vessel_collection = _get_vessel_collection()
    if vessel_collection is not None:
        try:
            vessel_collection.insert_one(record)
            return record_id, "mongodb"
        except Exception as error:
            print("Error saving vessel prediction to MongoDB; using JSON fallback:", error)

    history = _load_local_vessel_history()
    history.append(record)
    _save_local_vessel_history(history[-500:])
    return record_id, "json"


def update_vessel_prediction_record(record_id, updates):
    """Update a saved vessel record in MongoDB or the local JSON backup."""
    vessel_collection = _get_vessel_collection()
    if vessel_collection is not None:
        try:
            existing = vessel_collection.find_one(
                {"record_id": record_id},
                {"_id": 0},
            )
            if existing:
                merged = {**existing, **updates, "record_id": record_id}
                vessel_collection.update_one(
                    {"record_id": record_id},
                    {"$set": merged},
                )
                return merged
        except Exception as error:
            print("Error updating vessel prediction in MongoDB; using JSON fallback:", error)

    history = _load_local_vessel_history()
    for index, existing in enumerate(history):
        if existing.get("record_id") == record_id:
            merged = {**existing, **updates, "record_id": record_id}
            history[index] = merged
            _save_local_vessel_history(history[-500:])
            return merged
    return None


# Cloudinary image storage helpers
def get_cloudinary_credentials():
    """Read Cloudinary credentials from the environment."""
    cloudinary_url = os.getenv("CLOUDINARY_URL", "")
    parsed = urlparse(cloudinary_url)
    cloud_name = os.getenv("CLOUDINARY_CLOUD_NAME") or parsed.hostname
    api_key = os.getenv("CLOUDINARY_API_KEY") or parsed.username
    api_secret = os.getenv("CLOUDINARY_API_SECRET") or parsed.password
    if not cloud_name or not api_key or not api_secret:
        raise RuntimeError(
            "Cloudinary is not configured. Set CLOUDINARY_URL or "
            "CLOUDINARY_CLOUD_NAME, CLOUDINARY_API_KEY, and "
            "CLOUDINARY_API_SECRET in backend/.env."
        )
    return cloud_name, api_key, api_secret


def upload_vessel_image_to_cloudinary(image_bytes, filename, field_name):
    """Upload one vessel image and return its URL and Cloudinary ID."""
    cloud_name, api_key, api_secret = get_cloudinary_credentials()
    timestamp = int(time.time())
    public_id = f"oceaniq/vessels/{uuid.uuid4().hex}_{field_name}"
    params = {
        "folder": "oceaniq/vessels",
        "public_id": public_id.rsplit("/", 1)[-1],
        "timestamp": timestamp,
    }
    signature_source = "&".join(
        f"{key}={params[key]}" for key in sorted(params)
    ) + api_secret
    params["signature"] = hashlib.sha1(
        signature_source.encode("utf-8")
    ).hexdigest()
    params["api_key"] = api_key
    response = requests.post(
        f"https://api.cloudinary.com/v1_1/{cloud_name}/image/upload",
        data=params,
        files={"file": (Path(filename).name, image_bytes)},
        timeout=45,
    )
    response.raise_for_status()
    payload = response.json()
    if not payload.get("secure_url") or not payload.get("public_id"):
        raise RuntimeError("Cloudinary did not return an image URL.")
    return {
        "url": payload["secure_url"],
        "public_id": payload["public_id"],
    }


def delete_cloudinary_image(public_id):
    """Delete a vessel image from Cloudinary by its public ID."""
    cloud_name, api_key, api_secret = get_cloudinary_credentials()
    params = {
        "public_id": public_id,
        "timestamp": int(time.time()),
    }
    signature_source = "&".join(
        f"{key}={params[key]}" for key in sorted(params)
    ) + api_secret
    params["signature"] = hashlib.sha1(
        signature_source.encode("utf-8")
    ).hexdigest()
    params["api_key"] = api_key
    response = requests.post(
        f"https://api.cloudinary.com/v1_1/{cloud_name}/image/destroy",
        data=params,
        timeout=30,
    )
    response.raise_for_status()
    result = response.json().get("result")
    if result not in {"ok", "not found"}:
        raise RuntimeError("Cloudinary could not delete the stored image.")


# Delete a vessel record together with its stored images.
def delete_vessel_record(record_id):
    """Delete a vessel record and its uploaded images."""
    record = None
    vessel_collection = _get_vessel_collection()
    if vessel_collection is not None:
        try:
            record = vessel_collection.find_one(
                {"record_id": record_id},
                {"_id": 0},
            )
        except Exception as error:
            print("Error loading vessel record before delete:", error)

    local_history = _load_local_vessel_history()
    if record is None:
        record = next(
            (item for item in local_history if item.get("record_id") == record_id),
            None,
        )
    if record is None:
        return False

    for image in (record.get("cloudinary_images") or {}).values():
        public_id = image.get("public_id") if isinstance(image, dict) else None
        if public_id:
            delete_cloudinary_image(public_id)

    deleted = False
    if vessel_collection is not None:
        try:
            deleted = vessel_collection.delete_one(
                {"record_id": record_id}
            ).deleted_count > 0
        except Exception as error:
            print("Error deleting vessel record from MongoDB; trying JSON fallback:", error)

    filtered = [item for item in local_history if item.get("record_id") != record_id]
    if len(filtered) != len(local_history):
        _save_local_vessel_history(filtered)
        deleted = True
    return deleted


# Vessel detection API endpoints
def build_demo_boat_detection_response():
    """Build the sample response shown by the vessel demo endpoint."""
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


async def get_demo_boat_detection():
    """Return the sample vessel detection result."""
    return build_demo_boat_detection_response()


async def predict_boat_detection(
    file: UploadFile | None = File(None),
    vessel_info_file: UploadFile | None = File(None),
):
    """Detect vessels in an image and read details from a backside image."""
    if file is None and vessel_info_file is None:
        return {"error": "Upload a vessel image or a vessel backside image"}
    if file is not None and ship_detection_model is None:
        return {"error": "Ship detection model is not loaded"}
    if vessel_info_file is not None and vessel_info_model is None:
        return {"error": "Vessel information model is not loaded"}

    try:
        detections = []
        info = None
        image_size = None
        if file is not None:
            image = await read_uploaded_image(file)
            predictions = ship_detection_model(
                image,
                conf=0.25,
                iou=0.45,
                save=False,
                verbose=False,
            )
            if predictions:
                detections = extract_ship_detections(
                    predictions[0],
                    ship_detection_model,
                )
            image_height, image_width = image.shape[:2]
            image_size = [int(image_width), int(image_height)]
        if vessel_info_file is not None:
            info_image = await read_uploaded_image(vessel_info_file)
            info = detect_vessel_info(info_image, vessel_info_model)
            ocr_port = (info["fields"].get("port") or "").strip()
            if ocr_port:
                port_match = None
                try:
                    port_match = await run_in_threadpool(
                        lookup_port_country,
                        ocr_port,
                    )
                    info["fields"]["port_ocr"] = ocr_port
                    info["fields"]["port_match_status"] = port_match["status"]
                    info["fields"]["port_match_score"] = port_match["match_score"]
                    info["fields"]["port_type"] = port_match["port_type"]
                    if port_match["status"] == "found":
                        info["fields"]["port"] = port_match["port"]
                        info["fields"]["port_country"] = port_match["country"]
                        info["fields"]["port_country_code"] = port_match["country_code"]
                    elif port_match["status"] == "ambiguous":
                        info["fields"]["port_match_candidates"] = port_match["candidates"]
                except ValueError:
                    info["fields"]["port_ocr"] = ocr_port
                    info["fields"]["port_match_status"] = "not_found"
                    info["fields"]["port_type"] = "Foreign"

                if port_match and port_match["status"] == "found":
                    for detection in info["detections"]:
                        if detection["field"] == "port":
                            detection["matched_text"] = port_match["port"]

        detections.sort(key=lambda detection: detection["confidence"], reverse=True)

        return {
            "image": Path(file.filename or "").name if file else None,
            "vessel_info_image": (
                Path(vessel_info_file.filename or "").name
                if vessel_info_file
                else None
            ),
            "vessel_detection_performed": file is not None,
            "vessel_info_performed": vessel_info_file is not None,
            "status": (
                "Detected"
                if detections
                else "No Vessel Detected" if file is not None
                else "Vessel information analyzed"
            ),
            "confidence": max(
                (detection["confidence"] for detection in detections),
                default=0,
            ),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "source": "Drone",
            "estimated_size": infer_estimated_size(detections),
            "vessel_origin": "Unknown",
            "vessel_classifications": list(dict.fromkeys(
                detection["label"] for detection in detections
            )),
            "vessel_kinds": list(dict.fromkeys(
                detection["vessel_kind"] for detection in detections
            )),
            "results": detections,
            "count": len(detections),
            "image_size": image_size,
            "vessel_info": info["fields"] if info else None,
            "vessel_info_detections": info["detections"] if info else [],
            "demo": False,
        }
    except Exception as error:
        print("ERROR:", error)
        traceback.print_exc()
        return {"error": f"Inference failed: {str(error)}"}


async def create_vessel_detection_record(
    record_json: str = Form(...),
    ship_image: UploadFile | None = File(None),
    vessel_info_image: UploadFile | None = File(None),
):
    """Upload analyzed images and save a new vessel record."""
    try:
        record = json.loads(record_json)
        if not isinstance(record, dict):
            return {"error": "Record details must be a JSON object"}
        if ship_image is None and vessel_info_image is None:
            return {"error": "At least one analyzed image is required"}

        images = {}
        for field_name, uploaded_file in (
            ("ship_image", ship_image),
            ("vessel_info_image", vessel_info_image),
        ):
            if uploaded_file is None:
                continue
            contents = await uploaded_file.read()
            decoded_image = cv2.imdecode(
                np.frombuffer(contents, dtype=np.uint8),
                cv2.IMREAD_COLOR,
            )
            if decoded_image is None:
                return {"error": f"{field_name} is not a valid image"}
            images[field_name] = (
                contents,
                Path(uploaded_file.filename or field_name).name,
            )

        cloudinary_images = {}
        for field_name, (contents, filename) in images.items():
            cloudinary_images[field_name] = upload_vessel_image_to_cloudinary(
                contents,
                filename,
                field_name,
            )

        record.update({
            "record_id": uuid.uuid4().hex,
            "module": "vessel",
            "status": record.get("status", "Saved"),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "cloudinary_images": cloudinary_images,
            "image_url": cloudinary_images.get("ship_image", {}).get("url"),
            "vessel_info_image_url": cloudinary_images.get(
                "vessel_info_image", {}
            ).get("url"),
        })
        record["record_id"], storage = save_vessel_prediction_record(record)
        return {"record": record, "storage": storage}
    except requests.RequestException as error:
        print("Cloudinary image upload failed:", error)
        return {"error": "Cloudinary image upload failed. Check Cloudinary settings and try again."}
    except RuntimeError as error:
        return {"error": str(error)}
    except (json.JSONDecodeError, TypeError, ValueError) as error:
        return {"error": f"Invalid vessel record: {error}"}


def get_vessel_detection_history():
    """Return saved vessel records from MongoDB and the JSON backup."""
    local_history = _load_local_vessel_history()
    local_history, ids_added = _ensure_vessel_history_record_ids(local_history)
    if ids_added and not _save_local_vessel_history(local_history):
        raise HTTPException(
            status_code=500,
            detail="Could not prepare saved vessel records for display.",
        )

    history = []
    storage_sources = []
    vessel_collection = _get_vessel_collection()
    if vessel_collection is not None:
        try:
            mongo_history = list(
                vessel_collection.find({}, {"_id": 0}).sort("timestamp", -1)
            )
            history.extend(mongo_history)
            storage_sources.append("mongodb")
        except Exception as error:
            print("Error loading vessel history from MongoDB; using JSON fallback:", error)

    if local_history:
        storage_sources.append("json")
        existing_ids = {
            record.get("record_id")
            for record in history
            if isinstance(record, dict) and record.get("record_id")
        }
        for record in local_history:
            record_id = record.get("record_id") if isinstance(record, dict) else None
            if record_id and record_id in existing_ids:
                continue
            history.append(record)
            if record_id:
                existing_ids.add(record_id)

    history.sort(
        key=lambda record: str(record.get("timestamp") or "")
        if isinstance(record, dict)
        else "",
        reverse=True,
    )
    return {
        "history": history,
        "storage": " + ".join(storage_sources) if storage_sources else "json",
    }


def edit_vessel_detection_record(record_id: str, updates: dict = Body(...)):
    """Update allowed fields on a saved vessel record."""
    editable_fields = {
        "vessel_info",
        "vessel_classifications",
        "results",
        "notes",
    }
    safe_updates = {
        key: value for key, value in updates.items() if key in editable_fields
    }
    if not safe_updates:
        return {"error": "No editable vessel fields were supplied"}

    record = update_vessel_prediction_record(record_id, safe_updates)
    if not record or not record.get("timestamp"):
        return {"error": "Vessel record not found"}
    return {"record": record}


def delete_vessel_detection_record(record_id: str):
    """Delete one saved record and its uploaded images."""
    try:
        if not delete_vessel_record(record_id):
            return {"error": "Vessel record not found"}
        return {"message": "Vessel record and uploaded images deleted", "record_id": record_id}
    except requests.RequestException as error:
        print("Cloudinary image deletion failed:", error)
        return {"error": "Could not delete the Cloudinary image. The database record was kept."}
    except RuntimeError as error:
        return {"error": str(error)}


async def predict_vessel_video(video_file: UploadFile = File(...)):
    """Run vessel detection on an uploaded video."""
    return await predict_boat_video(video_file, ship_detection_model)


def get_vessel_port_country(port: str):
    """Look up a port name in the bundled world port list."""
    try:
        return lookup_port_country(port)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except requests.RequestException as error:
        raise HTTPException(
            status_code=502,
            detail="The port-country service is temporarily unavailable. Try again later.",
        ) from error
