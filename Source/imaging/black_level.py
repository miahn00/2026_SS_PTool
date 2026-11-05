"""Image-estimated black-level helpers for optional uint16 measurements."""

from __future__ import annotations

import numpy as np

RECOMMENDED_BLACK_LEVEL_MIN = 20_000
RECOMMENDED_BLACK_LEVEL_MAX = 30_000
DEFAULT_BLACK_LEVEL = 25_000


def estimate_black_level(image: np.ndarray) -> int:
    """Estimate the lower signal floor from an unmodified uint16 frame."""
    array = np.asarray(image)
    if array.dtype != np.uint16:
        raise ValueError("Black Level 추정은 uint16 원본 영상에서만 지원합니다.")
    return int(round(float(np.percentile(array, 1.0))))


def recommended_black_level(estimated_level: int) -> int:
    """Return a conservative recommendation inside the configured soft band."""
    return min(
        RECOMMENDED_BLACK_LEVEL_MAX,
        max(RECOMMENDED_BLACK_LEVEL_MIN, int(estimated_level)),
    )


def temporal_black_level(estimates: list[int]) -> int:
    """Return a 10%-trimmed mean for a sequence of raw-frame estimates."""
    if not estimates:
        raise ValueError("Black Level 시간 평균에 사용할 프레임이 없습니다.")
    values = np.sort(np.asarray(estimates, dtype=np.float64))
    trim = int(values.size * 0.1)
    if trim and values.size - trim * 2 >= 1:
        values = values[trim:-trim]
    return int(round(float(values.mean())))


def apply_black_level(image: np.ndarray, level: int) -> np.ndarray:
    """Subtract a scalar level without modifying the original uint16 image."""
    array = np.asarray(image)
    if array.dtype != np.uint16:
        raise ValueError("Black Level 보정은 uint16 원본 영상에서만 지원합니다.")
    if not 0 <= int(level) <= 65_535:
        raise ValueError("Black Level은 0~65535 범위여야 합니다.")
    corrected = np.maximum(array.astype(np.int32) - int(level), 0)
    return corrected.astype(np.uint16)
