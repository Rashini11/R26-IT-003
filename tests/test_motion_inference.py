from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import torch

from backend.simulation.motion_inference import MotionPredictor, MultiTaskMotionModel


def test_motion_predictor_loads_safe_checkpoint_and_predicts(tmp_path):
    model = MultiTaskMotionModel(
        input_size=7,
        model_type="gru",
        hidden_size=128,
        num_layers=2,
        dropout=0.2,
        class_count=4,
    )
    model_path = tmp_path / "motion.pth"
    metadata_path = tmp_path / "metadata.json"
    torch.save({
        "model_type": "gru",
        "model_state_dict": model.state_dict(),
        "input_size": 7,
        "hidden_size": 128,
        "num_layers": 2,
        "dropout": 0.2,
        "class_names": ["Stopped", "Slow", "Moderate", "Fast"],
        "position_mean": torch.tensor([0.0, 0.0]),
        "position_std": torch.tensor([1.0, 1.0]),
        "speed_mean": 5.0,
        "speed_std": 1.0,
    }, model_path)
    metadata_path.write_text(json.dumps({
        "feature_mean": [0.0] * 7,
        "feature_std": [1.0] * 7,
        "class_names": ["Stopped", "Slow", "Moderate", "Fast"],
    }))

    start = datetime(2022, 1, 1, tzinfo=timezone.utc)
    history = [
        {
            "mmsi": 123456789,
            "timestamp": start + timedelta(minutes=index),
            "latitude": 6.0,
            "longitude": 79.0 + index * 0.0001,
            "speed_knots": 5.0,
            "course_degrees": 90.0,
            "observed": True,
        }
        for index in range(10)
    ]

    predictor = MotionPredictor(model_path=model_path, metadata_path=metadata_path)
    result = predictor.predict(history)

    assert predictor.ready
    assert result["mmsi"] == 123456789
    assert result["forecast_horizon_minutes"] == 5
    assert result["predicted_motion_class"] in {"Stopped", "Slow", "Moderate", "Fast"}
    assert set(result["class_probabilities"]) == {"Stopped", "Slow", "Moderate", "Fast"}
