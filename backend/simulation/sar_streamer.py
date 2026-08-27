from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable

from .config import (
    SAR_ROOT,
    VALID_IMAGE_EXTENSIONS,
)


# ============================================================
# INTERNAL RADAR CLASSIFIER
# ============================================================

_INTERNAL_RADAR_CLASSIFIER: (
    Callable[[Path], dict[str, Any]] | None
) = None


def configure_internal_radar_classifier(
    classifier: Callable[
        [Path],
        dict[str, Any],
    ],
) -> None:

    global _INTERNAL_RADAR_CLASSIFIER

    _INTERNAL_RADAR_CLASSIFIER = classifier


# ============================================================
# NATURAL SORT
#
# ship1, ship2, ..., ship10
# instead of:
# ship1, ship10, ship2, ...
# ============================================================

def natural_sort_key(path: Path):

    parts = re.split(
        r"(\d+)",
        str(path).lower(),
    )

    return [
        int(part)
        if part.isdigit()
        else part
        for part in parts
    ]


# ============================================================
# LIVE SIMULATION RADAR SOURCE
#
# Current dataset:
#
# ml/live_simulation_radar/
#   ship1/
#   ship2/
#   ...
#   ship10/
#
# Only these folders are allowed.
# ============================================================

def collect_sar_images(
    source: str,
) -> list[Path]:

    if not SAR_ROOT.exists():

        raise FileNotFoundError(
            f"Live Radar root not found: "
            f"{SAR_ROOT}"
        )

    # The current Live Simulation dataset
    # contains ship radar images only.
    #
    # Keep "all" for compatibility with the
    # existing frontend dropdown.
    if source not in {
        "ship",
        "all",
    }:

        raise ValueError(
            "Live Simulation currently "
            "supports ship Radar images only."
        )

    ship_folders = sorted(
        [
            path
            for path in SAR_ROOT.iterdir()
            if (
                path.is_dir()
                and re.fullmatch(
                    r"ship\d+",
                    path.name.lower(),
                )
            )
        ],
        key=natural_sort_key,
    )

    if not ship_folders:

        raise FileNotFoundError(
            "No ship1 ... ship10 folders "
            f"found under {SAR_ROOT}"
        )

    images: list[Path] = []

    for folder in ship_folders:

        folder_images = sorted(
            [
                path.resolve()
                for path in folder.rglob("*")
                if (
                    path.is_file()
                    and path.suffix.lower()
                    in VALID_IMAGE_EXTENSIONS
                )
            ],
            key=natural_sort_key,
        )

        images.extend(
            folder_images
        )

    if not images:

        raise FileNotFoundError(
            "No Radar images found in "
            "ship1 ... ship10 folders."
        )

    print(
        f"Live Simulation Radar source: "
        f"{len(ship_folders)} ship folders, "
        f"{len(images)} images"
    )

    return images


# ============================================================
# IMAGE STREAMER
#
# Deterministic sequence:
#
# ship1 images
#   ->
# ship2 images
#   ->
# ...
#   ->
# ship10 images
#   ->
# repeat from ship1
#
# ============================================================

class SARImageStreamer:

    def __init__(
        self,
        source: str,
        seed: int = 42,
        shuffle: bool = False,
    ):

        # seed kept for backward compatibility.
        del seed

        self.images = (
            collect_sar_images(
                source
            )
        )

        self.index = 0

        # Force deterministic cycling so every
        # Radar frame is eventually used.
        self.shuffle = False

    def next_image(
        self,
    ) -> Path:

        if self.index >= len(
            self.images
        ):
            self.index = 0

        image = self.images[
            self.index
        ]

        self.index += 1

        print(
            "[LIVE RADAR FRAME]",
            image
        )

        return image


    @staticmethod
    def classify(
        image_path: Path,
        timeout_seconds: float = 120.0,
    ) -> dict[str, Any]:

        # Kept only for compatibility.
        del timeout_seconds

        if (
            _INTERNAL_RADAR_CLASSIFIER
            is None
        ):

            raise RuntimeError(
                "Internal Radar classifier "
                "has not been configured."
            )

        return (
            _INTERNAL_RADAR_CLASSIFIER(
                image_path
            )
        )
