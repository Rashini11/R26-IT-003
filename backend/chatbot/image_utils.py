from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Optional

from PIL import Image, UnidentifiedImageError


MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_IMAGE_SIDE = 4096
MAX_IMAGE_PIXELS = 4096 * 4096
ALLOWED_FORMATS = {"JPEG", "PNG", "BMP", "WEBP"}


@dataclass
class ImageValidationResult:
    ok: bool
    reason: str = ""
    width: int = 0
    height: int = 0
    format: Optional[str] = None
    rgb_bytes: Optional[bytes] = None


def validate_image_bytes(data: bytes, filename: str | None = None) -> ImageValidationResult:
    if not data:
        return ImageValidationResult(ok=False, reason="Empty upload.")
    if len(data) > MAX_UPLOAD_BYTES:
        return ImageValidationResult(
            ok=False,
            reason=f"Image exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB upload limit.",
        )

    try:
        with Image.open(io.BytesIO(data)) as image:
            image.load()
            width, height = image.size
            fmt = (image.format or "").upper() or None
            if width < 16 or height < 16:
                return ImageValidationResult(
                    ok=False,
                    reason="Image resolution is too low for analysis.",
                    width=width,
                    height=height,
                    format=fmt,
                )
            if width > MAX_IMAGE_SIDE or height > MAX_IMAGE_SIDE:
                return ImageValidationResult(
                    ok=False,
                    reason=f"Image dimensions must be at most {MAX_IMAGE_SIDE}×{MAX_IMAGE_SIDE} pixels.",
                    width=width,
                    height=height,
                    format=fmt,
                )
            if width * height > MAX_IMAGE_PIXELS:
                return ImageValidationResult(
                    ok=False,
                    reason="Decoded image exceeds the maximum allowed pixel count.",
                    width=width,
                    height=height,
                    format=fmt,
                )
            if fmt and fmt not in ALLOWED_FORMATS:
                return ImageValidationResult(
                    ok=False,
                    reason=f"Unsupported image format: {fmt}. Use JPEG, PNG, BMP, or WEBP.",
                    width=width,
                    height=height,
                    format=fmt,
                )
            rgb = image.convert("RGB")
            buffer = io.BytesIO()
            rgb.save(buffer, format="JPEG", quality=92)
            return ImageValidationResult(
                ok=True,
                reason="Image is readable.",
                width=width,
                height=height,
                format=fmt,
                rgb_bytes=buffer.getvalue(),
            )
    except UnidentifiedImageError:
        return ImageValidationResult(
            ok=False,
            reason="The file could not be decoded as an image.",
        )
    except OSError:
        return ImageValidationResult(
            ok=False,
            reason="The image file appears corrupt or unsupported.",
        )
    except Exception:
        return ImageValidationResult(
            ok=False,
            reason="Unable to validate the uploaded image.",
        )
