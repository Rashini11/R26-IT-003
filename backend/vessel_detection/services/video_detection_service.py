import math
import os
import tempfile
import traceback
from pathlib import Path

import cv2
from fastapi import UploadFile

from .image_detection_service import extract_ship_detections


async def predict_boat_video(video_file: UploadFile, ship_detection_model):
    """Check a video and detect vessels in one frame per second."""
    if ship_detection_model is None:
        return {"error": "Ship detection model is not loaded"}

    filename = Path(video_file.filename or "").name
    suffix = Path(filename).suffix.lower()
    allowed_video_extensions = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
    if suffix not in allowed_video_extensions:
        return {"error": "Upload a supported video file (MP4, MOV, AVI, MKV, or WEBM)"}

    max_video_bytes = 100 * 1024 * 1024
    max_video_duration_seconds = 300
    temp_path = None
    capture = None
    try:
        with tempfile.NamedTemporaryFile(
            suffix=suffix,
            prefix="oceaniq-vessel-",
            delete=False,
        ) as temporary_video:
            temp_path = temporary_video.name
            bytes_written = 0
            while chunk := await video_file.read(1024 * 1024):
                bytes_written += len(chunk)
                if bytes_written > max_video_bytes:
                    return {"error": "Video is too large. Maximum upload size is 100 MB."}
                temporary_video.write(chunk)

        capture = cv2.VideoCapture(temp_path)
        if not capture.isOpened():
            return {"error": "Could not read this video. Try an MP4, MOV, AVI, MKV, or WEBM file."}

        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if fps <= 0 or frame_count <= 0:
            return {"error": "Video duration could not be determined. Try another video."}

        duration_seconds = frame_count / fps
        if duration_seconds > max_video_duration_seconds:
            return {"error": "Video is too long. Maximum duration is 5 minutes."}

        sampled_frames = []
        all_detections = []
        for second in range(math.ceil(duration_seconds)):
            capture.set(cv2.CAP_PROP_POS_MSEC, second * 1000)
            success, frame = capture.read()
            if not success:
                continue

            predictions = ship_detection_model(
                frame,
                conf=0.25,
                iou=0.45,
                save=False,
                verbose=False,
            )
            detections = (
                extract_ship_detections(predictions[0], ship_detection_model)
                if predictions
                else []
            )
            for detection in detections:
                all_detections.append({
                    **detection,
                    "frame_time_seconds": second,
                })
            sampled_frames.append({
                "time_seconds": second,
                "detections": detections,
            })

        classifications = list(dict.fromkeys(
            detection["label"] for detection in all_detections
        ))
        vessel_kinds = list(dict.fromkeys(
            detection["vessel_kind"] for detection in all_detections
        ))
        return {
            "video": filename,
            "video_analysis_performed": True,
            "vessel_detection_performed": True,
            "vessel_info_performed": False,
            "status": "Detected" if all_detections else "No Vessel Detected",
            "video_duration_seconds": round(duration_seconds, 1),
            "sample_interval_seconds": 1,
            "frames_analyzed": len(sampled_frames),
            "frames": sampled_frames,
            "results": all_detections,
            "count": len(all_detections),
            "vessel_classifications": classifications,
            "vessel_kinds": vessel_kinds,
            "demo": False,
            "saved_to_database": False,
        }
    except Exception as error:
        print("Vessel video inference failed:", error)
        traceback.print_exc()
        return {"error": "Video analysis failed. Check the video file and try again."}
    finally:
        if capture is not None:
            capture.release()
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)
        await video_file.close()
