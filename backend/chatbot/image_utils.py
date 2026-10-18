from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Optional

from PIL import Image, ImageOps, UnidentifiedImageError


MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_IMAGE_SIDE = 4096
MAX_IMAGE_PIXELS = 40_000_000
ALLOWED_FORMATS = {"JPEG", "PNG", "BMP", "WEBP"}


@dataclass
class ImageValidationResult:
    ok: bool
    reason: str = ""
    width: int = 0
    height: int = 0
    format: Optional[str] = None
    rgb_bytes: Optional[bytes] = None
    original_width: int = 0
    original_height: int = 0
    resized: bool = False


def validate_image_bytes(
    data: bytes, filename: str | None = None
) -> ImageValidationResult:
    if not data:
        return ImageValidationResult(ok=False, reason="Empty upload.")

    if len(data) > MAX_UPLOAD_BYTES:
        return ImageValidationResult(
            ok=False,
            reason=f"Image exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB upload limit.",
        )

    try:
        with Image.open(io.BytesIO(data)) as image:
            width, height = image.size
            fmt = (image.format or "").upper() or None

            # Check header dimensions before decoding the full image.
            if fmt not in ALLOWED_FORMATS:
                return ImageValidationResult(
                    ok=False,
                    reason="Unsupported image format. Use JPEG, PNG, BMP, or WEBP.",
                    format=fmt,
                )

            if width < 16 or height < 16:
                return ImageValidationResult(
                    ok=False,
                    reason="Image resolution is too low for analysis.",
                    width=width,
                    height=height,
                    format=fmt,
                )

            if width * height > MAX_IMAGE_PIXELS:
                return ImageValidationResult(
                    ok=False,
                    reason="Image exceeds the 40-million-pixel input limit.",
                    width=width,
                    height=height,
                    format=fmt,
                )

            image.load()
            oriented = ImageOps.exif_transpose(image)
            original_width, original_height = oriented.size
            rgb = oriented.convert("RGB")

            # Fit inside the processing boundary, retaining the whole frame.
            # thumbnail preserves aspect ratio and never enlarges images.
            rgb.thumbnail(
                (MAX_IMAGE_SIDE, MAX_IMAGE_SIDE),
                Image.Resampling.LANCZOS,
            )

            processed_width, processed_height = rgb.size
            resized = rgb.size != (original_width, original_height)

            buffer = io.BytesIO()
            # Lossless encoding avoids adding JPEG artifacts to radar images.
            rgb.save(buffer, format="PNG")

            return ImageValidationResult(
                ok=True,
                reason=(
                    f"Image resized from {original_width}×{original_height} "
                    f"to {processed_width}×{processed_height}; "
                    "aspect ratio preserved, no cropping."
                    if resized else "Image is readable."
                ),
                width=processed_width,
                height=processed_height,
                format=fmt,
                rgb_bytes=buffer.getvalue(),
                original_width=original_width,
                original_height=original_height,
                resized=resized,
            )

    except (Image.DecompressionBombError, Image.DecompressionBombWarning):
        return ImageValidationResult(
            ok=False, reason="Image exceeds safe decoding limits."
        )
    except UnidentifiedImageError:
        return ImageValidationResult(
            ok=False, reason="The file could not be decoded as an image."
        )
    except OSError:
        return ImageValidationResult(
            ok=False, reason="The image file appears corrupt or unsupported."
        )
    except Exception:
        return ImageValidationResult(
            ok=False, reason="Unable to validate the uploaded image."
        )
