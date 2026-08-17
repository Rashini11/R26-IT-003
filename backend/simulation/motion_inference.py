from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from .config import MOTION_METADATA_PATH, MOTION_MODEL_PATH
from .geo import local_xy_metres, offset_lat_lon


class MultiTaskMotionModel(nn.Module):
    def __init__(
        self,
        input_size: int,
        model_type: str,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        class_count: int = 4,
    ):
        super().__init__()
        recurrent_class = nn.GRU if model_type == "gru" else nn.LSTM
        self.recurrent = recurrent_class(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.shared = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.position_head = nn.Linear(64, 2)
        self.speed_head = nn.Linear(64, 1)
        self.class_head = nn.Linear(64, class_count)

    def forward(self, inputs):
        recurrent_output, _ = self.recurrent(inputs)
        shared = self.shared(recurrent_output[:, -1, :])
        return (
            self.position_head(shared),
            self.speed_head(shared).squeeze(1),
            self.class_head(shared),
        )


def _as_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


class MotionPredictor:
    def __init__(
        self,
        model_path: Path = MOTION_MODEL_PATH,
        metadata_path: Path = MOTION_METADATA_PATH,
    ):
        self.model_path = model_path
        self.metadata_path = metadata_path
        self.device = torch.device(
            "mps"
            if torch.backends.mps.is_available()
            else "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
        self.model: MultiTaskMotionModel | None = None
        self.checkpoint: dict[str, Any] | None = None
        self.feature_mean: np.ndarray | None = None
        self.feature_std: np.ndarray | None = None
        self.class_names: list[str] = ["Stopped", "Slow", "Moderate", "Fast"]
        self.load_error: str | None = None

    @property
    def ready(self) -> bool:
        return self.model is not None and self.load_error is None

    def load(self) -> None:
        if self.model is not None or self.load_error is not None:
            return
        try:
            if not self.model_path.exists():
                raise FileNotFoundError(f"GRU checkpoint not found: {self.model_path}")
            checkpoint = torch.load(
                self.model_path,
                map_location=self.device,
                weights_only=True,
            )
            metadata = self._load_feature_metadata()
            self.feature_mean = np.asarray(metadata["feature_mean"], dtype=np.float32)
            self.feature_std = np.asarray(metadata["feature_std"], dtype=np.float32)
            self.feature_std = np.where(self.feature_std < 1e-6, 1.0, self.feature_std)
            input_size = int(checkpoint["input_size"])
            if self.feature_mean.shape != (input_size,) or self.feature_std.shape != (input_size,):
                raise ValueError(
                    "Motion metadata feature shape does not match the checkpoint input size."
                )
            self.class_names = list(checkpoint.get("class_names", metadata.get("class_names", self.class_names)))
            model = MultiTaskMotionModel(
                input_size=input_size,
                model_type=str(checkpoint.get("model_type", "gru")),
                hidden_size=int(checkpoint.get("hidden_size", 128)),
                num_layers=int(checkpoint.get("num_layers", 2)),
                dropout=float(checkpoint.get("dropout", 0.2)),
                class_count=len(self.class_names),
            ).to(self.device)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            self.model = model
            self.checkpoint = checkpoint
        except Exception as error:
            self.load_error = str(error)

    def _load_feature_metadata(self) -> dict[str, Any]:
        if self.metadata_path.exists():
            return json.loads(self.metadata_path.read_text())
        raise FileNotFoundError(
            "Motion input-normalisation metadata is missing. Expected "
            f"{self.metadata_path}. Re-run prepare_ais_motion_sequences_5min.py "
            "if that file is unavailable."
        )

    @staticmethod
    def _derive_course(history: list[dict[str, Any]], index: int) -> float:
        supplied = history[index].get("course_degrees")
        if supplied is not None and math.isfinite(float(supplied)):
            return float(supplied) % 360.0
        if index == 0 and len(history) > 1:
            first, second = history[0], history[1]
        elif index > 0:
            first, second = history[index - 1], history[index]
        else:
            return 0.0
        east, north = local_xy_metres(
            [second["latitude"]],
            [second["longitude"]],
            first["latitude"],
            first["longitude"],
        )
        return (math.degrees(math.atan2(float(east[0]), float(north[0]))) + 360.0) % 360.0

    def predict(self, history: list[dict[str, Any]]) -> dict[str, Any]:
        self.load()
        if not self.ready or self.model is None or self.checkpoint is None:
            raise RuntimeError(self.load_error or "Motion model is unavailable.")
        if len(history) != 10:
            raise ValueError("Exactly 10 one-minute observations are required.")
        history = sorted(history, key=lambda item: item["timestamp"])
        if len({int(item["mmsi"]) for item in history}) != 1:
            raise ValueError("All history rows must have the same MMSI.")
        timestamps = [item["timestamp"] for item in history]
        if len(set(timestamps)) != len(timestamps):
            raise ValueError("Duplicate motion-history timestamps are not allowed.")
        gaps = [
            (timestamps[index] - timestamps[index - 1]).total_seconds()
            for index in range(1, len(timestamps))
        ]
        if any(gap < 30 or gap > 90 for gap in gaps):
            raise ValueError("Motion-history rows must be approximately one minute apart.")

        east, north = local_xy_metres(
            [item["latitude"] for item in history],
            [item["longitude"] for item in history],
            history[0]["latitude"],
            history[0]["longitude"],
        )
        feature_rows = []
        for index, item in enumerate(history):
            explicit_course_missing = item.get("course_missing")
            is_course_missing = (
                item.get("course_degrees") is None
                if explicit_course_missing is None
                else bool(explicit_course_missing)
            )
            course_missing = 1.0 if is_course_missing else 0.0
            course = self._derive_course(history, index)
            radians = math.radians(course)
            feature_rows.append([
                float(east[index]),
                float(north[index]),
                float(item["speed_knots"]),
                math.sin(radians),
                math.cos(radians),
                1.0 if item.get("observed", True) else 0.0,
                course_missing,
            ])
        features = np.asarray(feature_rows, dtype=np.float32)
        normalised = (features - self.feature_mean) / self.feature_std
        tensor = torch.from_numpy(normalised).unsqueeze(0).to(self.device)

        start = time.perf_counter()
        with torch.inference_mode():
            position_norm, speed_norm, class_logits = self.model(tensor)
            probabilities = torch.softmax(class_logits, dim=1)[0]
        inference_ms = (time.perf_counter() - start) * 1000.0

        position_mean = _as_numpy(self.checkpoint["position_mean"]).astype(np.float32)
        position_std = _as_numpy(self.checkpoint["position_std"]).astype(np.float32)
        position = position_norm[0].detach().cpu().numpy() * position_std + position_mean
        speed = (
            float(speed_norm[0].detach().cpu()) * float(self.checkpoint["speed_std"])
            + float(self.checkpoint["speed_mean"])
        )
        predicted_index = int(torch.argmax(probabilities).item())
        last = history[-1]
        predicted_latitude, predicted_longitude = offset_lat_lon(
            last["latitude"], last["longitude"], float(position[0]), float(position[1])
        )
        return {
            "mmsi": int(last["mmsi"]),
            "history_minutes": 10,
            "forecast_horizon_minutes": 5,
            "predicted_east_displacement_metres": round(float(position[0]), 2),
            "predicted_north_displacement_metres": round(float(position[1]), 2),
            "predicted_latitude": predicted_latitude,
            "predicted_longitude": predicted_longitude,
            "predicted_speed_knots": round(max(0.0, speed), 3),
            "predicted_motion_class": self.class_names[predicted_index],
            "class_probabilities": {
                name: round(float(probabilities[index].detach().cpu()), 6)
                for index, name in enumerate(self.class_names)
            },
            "inference_time_ms": round(inference_ms, 3),
        }
