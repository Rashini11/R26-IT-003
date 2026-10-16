"""Image-type suggestions. Separate from component predictions."""
from io import BytesIO
from pathlib import Path
from threading import Lock

import torch
from PIL import Image, ImageOps
from torch import nn
from torchvision.models import (
    mobilenet_v3_small,
    MobileNet_V3_Small_Weights,
)

CHECKPOINT = (
    Path(__file__).resolve().parents[2]
    / "ml/models/image_router_v1/image_router_best.pth"
)

_LOCK = Lock()
_MODEL = None
_METADATA = None
_TRANSFORM = None

LABELS = {
    "radar": "Radar Objects",
    "vessel": "Drone Vessel Detection",
    "sea": "Sea State",
    "hull": "Hull Condition",
    "unsupported": "Unsupported image",
}

def suggest_image_component(image_bytes):
    global _MODEL, _METADATA, _TRANSFORM

    if not CHECKPOINT.is_file():
        return {
            "status": "unavailable",
            "automatic_routing_available": False,
            "message": "Image router is unavailable. Please select an analysis.",
        }

    # CPU avoids contention with the existing component models.
    with _LOCK:
        if _MODEL is None:
            saved = torch.load(
                CHECKPOINT, map_location="cpu", weights_only=True
            )
            metadata = saved["metadata"]
            classes = metadata["classes"]

            if (
                metadata["architecture"] != "mobilenet_v3_small"
                or len(classes) != 5
                or set(classes) != set(LABELS)
            ):
                raise RuntimeError("Incompatible image-router checkpoint")

            model = mobilenet_v3_small(weights=None)
            model.classifier[3] = nn.Linear(
                model.classifier[3].in_features, len(classes)
            )
            model.load_state_dict(saved["model_state_dict"], strict=True)
            model.eval()

            _TRANSFORM = (
                MobileNet_V3_Small_Weights.IMAGENET1K_V1.transforms()
            )
            _METADATA = metadata
            _MODEL = model

        with Image.open(BytesIO(image_bytes)) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            inputs = _TRANSFORM(image).unsqueeze(0)

        with torch.inference_mode():
            scores = _MODEL(inputs).softmax(dim=1)[0]

        classes = _METADATA["classes"]
        index = int(scores.argmax())
        prediction = classes[index]
        score = float(scores[index])

        return {
            "status": (
                "unsupported" if prediction == "unsupported"
                else "needs_confirmation"
            ),
            "automatic_routing_available": True,
            "suggested_component": (
                None if prediction == "unsupported" else prediction
            ),
            "predicted_category": prediction,
            "label": LABELS[prediction],
            "score": round(score * 100, 2),
            "probabilities": {
                name: round(float(scores[i]) * 100, 2)
                for i, name in enumerate(classes)
            },
            "requires_user_confirmation": True,
            "model_version": _METADATA["model_version"],
            "message": (
                "This image may be unsupported. Choose a component manually "
                "if you want to continue."
                if prediction == "unsupported"
                else f"Suggested analysis: {LABELS[prediction]}. "
                     "Confirm the component before running analysis."
            ),
            "limitations": [
                "Image-type scores are not component prediction scores.",
                "Experimental router; unsupported-image detection is limited.",
                "User confirmation is required for every suggestion.",
            ],
        }
