from __future__ import annotations

import numpy as np

from imaging.roi import RoiData
from inspection import analyze_roi, extract_roi


def test_extract_roi_is_independent_copy() -> None:
    source = np.arange(100 * 120, dtype=np.uint16).reshape(100, 120)
    roi = RoiData(1, "ROI 1", 10, 20, 30, 40)

    extracted = extract_roi(source, roi)
    extracted[0, 0] = 65535

    assert extracted.shape == (40, 30)
    assert source[20, 10] != 65535
    assert not np.shares_memory(source, extracted)


def test_analyze_vertical_bar_pattern() -> None:
    columns = np.arange(128)
    line = np.where((columns // 8) % 2 == 0, 1000, 50000).astype(np.uint16)
    image = np.tile(line, (96, 1))
    roi = RoiData(1, "Bars", 0, 0, 128, 96, direction="Auto")

    result = analyze_roi(image, roi)

    assert result.status == "READY"
    assert result.detected_direction == "V"
    assert result.direction_confidence > 0.9
    assert result.raw_profile.shape == (128,)
    assert result.profile.shape == (128,)
    assert result.fft_frequency.shape == result.fft_magnitude.shape
    assert result.fft_magnitude.max() == 1.0


def test_analyze_flat_roi_returns_invalid() -> None:
    image = np.full((64, 64), 1000, dtype=np.uint16)
    roi = RoiData(1, "Flat", 0, 0, 64, 64)

    result = analyze_roi(image, roi)

    assert result.status == "INVALID"
    assert "대비" in result.message


def test_analyze_inactive_roi_returns_invalid() -> None:
    image = np.arange(64 * 64, dtype=np.uint16).reshape(64, 64)
    roi = RoiData(1, "Inactive", 0, 0, 64, 64, active=False)

    result = analyze_roi(image, roi)

    assert result.status == "INVALID"
    assert "비활성화" in result.message
