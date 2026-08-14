"""TIFF와 JPEG 파일을 원본 NumPy 배열로 읽는다."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError
import tifffile

from .image_frame import ImageFrame

SUPPORTED_EXTENSIONS = {".tif", ".tiff", ".jpg", ".jpeg", ".png"}
SUPPORTED_DTYPES = {np.dtype(np.uint8), np.dtype(np.uint16)}


class ImageLoadError(ValueError):
    """지원하지 않거나 손상된 영상 파일을 읽을 때 발생한다."""


def _bit_depth(image: np.ndarray) -> int:
    if image.dtype not in SUPPORTED_DTYPES:
        raise ImageLoadError(
            f"지원하지 않는 픽셀 형식입니다: {image.dtype}. "
            "uint8 또는 uint16 영상만 지원합니다."
        )
    return image.dtype.itemsize * 8


def _dimensions(image: np.ndarray) -> tuple[int, int, int]:
    if image.ndim == 2:
        height, width = image.shape
        return width, height, 1
    if image.ndim == 3 and image.shape[2] in (3, 4):
        height, width, channels = image.shape
        return width, height, channels
    raise ImageLoadError(f"지원하지 않는 영상 배열 형태입니다: {image.shape}")


def _load_tiff(path: Path) -> tuple[np.ndarray, int, dict[str, object]]:
    with tifffile.TiffFile(path) as tif:
        page_count = len(tif.pages)
        if page_count == 0:
            raise ImageLoadError("TIFF 파일에 영상 페이지가 없습니다.")

        page = tif.pages[0]
        image = page.asarray()
        compression = getattr(page.compression, "name", str(page.compression))
        metadata: dict[str, object] = {
            "format": "TIFF",
            "compression": compression,
            "selected_page": 0,
        }
    return image, page_count, metadata


def _load_pillow_image(path: Path) -> tuple[np.ndarray, int, dict[str, object]]:
    with Image.open(path) as pil_image:
        pil_image.load()
        original_mode = pil_image.mode
        if pil_image.mode not in {"L", "RGB", "I;16", "I;16L", "I;16B"}:
            pil_image = pil_image.convert("RGB")
        image = np.asarray(pil_image).copy()
        if image.dtype.byteorder == ">":
            image = image.astype(np.uint16)
        metadata: dict[str, object] = {
            "format": path.suffix.lstrip(".").upper(),
            "mode": pil_image.mode,
            "original_mode": original_mode,
        }
    return image, 1, metadata


def load_image(file_path: str | Path) -> ImageFrame:
    """영상 파일을 읽고 원본 배열을 변경하지 않은 ImageFrame을 반환한다."""
    path = Path(file_path).expanduser().resolve()
    if not path.is_file():
        raise ImageLoadError(f"영상 파일을 찾을 수 없습니다: {path}")

    extension = path.suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise ImageLoadError(
            f"지원하지 않는 확장자입니다: {extension or '(없음)'}. 지원 형식: {supported}"
        )

    try:
        if extension in {".tif", ".tiff"}:
            image, page_count, metadata = _load_tiff(path)
        else:
            image, page_count, metadata = _load_pillow_image(path)
    except (OSError, ValueError, tifffile.TiffFileError, UnidentifiedImageError) as exc:
        raise ImageLoadError(f"영상 파일을 읽을 수 없습니다: {path}") from exc

    if image.size == 0:
        raise ImageLoadError("영상 데이터가 비어 있습니다.")

    bit_depth = _bit_depth(image)
    width, height, channels = _dimensions(image)

    return ImageFrame(
        image=image,
        width=width,
        height=height,
        bit_depth=bit_depth,
        channels=channels,
        source_type="file",
        timestamp=datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc),
        source_path=path,
        page_count=page_count,
        metadata=metadata,
    )
