"""분석 원본을 변경하지 않고 화면 표시용 uint8 영상을 생성한다."""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray


def to_display_uint8(
    image: NDArray[np.generic],
    *,
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.0,
) -> NDArray[np.uint8]:
    """원본 범위를 percentile stretch하여 독립된 uint8 배열로 반환한다."""
    if image.size == 0:
        raise ValueError("빈 영상은 표시용으로 변환할 수 없습니다.")
    if not 0.0 <= lower_percentile < upper_percentile <= 100.0:
        raise ValueError("Percentile 범위가 올바르지 않습니다.")
    if image.dtype not in (np.uint8, np.uint16):
        raise ValueError(f"지원하지 않는 픽셀 형식입니다: {image.dtype}")

    if image.dtype == np.uint8:
        return image.copy()

    low, high = np.percentile(image, (lower_percentile, upper_percentile))
    if high <= low:
        return np.zeros(image.shape, dtype=np.uint8)

    scaled = np.clip((image.astype(np.float32) - low) * (255.0 / (high - low)), 0, 255)
    return scaled.astype(np.uint8)


def window_to_uint8(
    image: NDArray[np.generic],
    minimum: int | float,
    maximum: int | float,
) -> NDArray[np.uint8]:
    """지정한 표시 Min/Max 범위를 uint8로 변환하며 원본은 변경하지 않는다."""
    if image.size == 0:
        raise ValueError("빈 영상은 표시용으로 변환할 수 없습니다.")
    if image.dtype not in (np.uint8, np.uint16):
        raise ValueError(f"지원하지 않는 픽셀 형식입니다: {image.dtype}")
    if maximum <= minimum:
        raise ValueError("표시 최댓값은 최솟값보다 커야 합니다.")

    scaled = np.clip(
        (image.astype(np.float32) - float(minimum))
        * (255.0 / (float(maximum) - float(minimum))),
        0,
        255,
    )
    return scaled.astype(np.uint8)


def to_display_gray_uint8(image: NDArray[np.generic]) -> NDArray[np.uint8]:
    """컬러 영상이면 Gray8로 변환하고, 단일 채널이면 표시 변환만 수행한다."""
    display = to_display_uint8(image)
    if display.ndim == 2:
        return display
    if display.shape[2] == 3:
        return cv2.cvtColor(display, cv2.COLOR_RGB2GRAY)
    if display.shape[2] == 4:
        return cv2.cvtColor(display, cv2.COLOR_RGBA2GRAY)
    raise ValueError(f"지원하지 않는 채널 구조입니다: {display.shape}")
