from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import requests

from .config import CLASSIFICATION_ENDPOINT, SAR_ROOT, VALID_IMAGE_EXTENSIONS


def resolve_sar_source(source: str) -> Path:
    mapping = {
        "ship": SAR_ROOT / "ship",
        "bird": SAR_ROOT / "bird",
        "unknown": SAR_ROOT / "unknown",
        "all": SAR_ROOT,
    }
    if source not in mapping:
        raise ValueError(f"Unsupported SAR source: {source}")
    candidate = mapping[source].resolve()
    try:
        candidate.relative_to(SAR_ROOT)
    except ValueError as error:
        raise ValueError("SAR source escaped the configured dataset root.") from error
    return candidate


def collect_sar_images(source: str) -> list[Path]:
    directory = resolve_sar_source(source)
    if not directory.exists():
        raise FileNotFoundError(f"SAR image directory not found: {directory}")
    images = sorted(
        path.resolve()
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in VALID_IMAGE_EXTENSIONS
    )
    images = [path for path in images if SAR_ROOT in path.parents]
    if not images:
        raise FileNotFoundError(f"No SAR images found in: {directory}")
    return images


class SARImageStreamer:
    def __init__(self, source: str, seed: int = 42, shuffle: bool = True):
        self.images = collect_sar_images(source)
        self.random = random.Random(seed)
        self.shuffle = shuffle
        self.index = 0
        if self.shuffle:
            self.random.shuffle(self.images)

    def next_image(self) -> Path:
        if self.index >= len(self.images):
            self.index = 0
            if self.shuffle:
                self.random.shuffle(self.images)
        image = self.images[self.index]
        self.index += 1
        return image

    @staticmethod
    def classify(image_path: Path, timeout_seconds: float = 120.0) -> dict[str, Any]:
        with image_path.open("rb") as image_file:
            response = requests.post(
                CLASSIFICATION_ENDPOINT,
                files={
                    "file": (
                        image_path.name,
                        image_file,
                        "application/octet-stream",
                    )
                },
                timeout=timeout_seconds,
            )
        response.raise_for_status()
        payload = response.json()
        if payload.get("error"):
            raise RuntimeError(payload["error"])
        return payload
