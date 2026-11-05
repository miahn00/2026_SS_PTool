from __future__ import annotations

import numpy as np

from imaging.black_level import (
    apply_black_level,
    estimate_black_level,
    recommended_black_level,
    temporal_black_level,
)


def test_apply_black_level_preserves_original_and_clips_to_zero() -> None:
    image = np.array([[19000, 25000, 65535]], dtype=np.uint16)
    original = image.copy()

    corrected = apply_black_level(image, 25000)

    assert corrected.dtype == np.uint16
    assert corrected.tolist() == [[0, 0, 40535]]
    assert np.array_equal(image, original)


def test_estimate_and_recommendation_use_uint16_raw_values() -> None:
    image = np.full((100, 100), 33000, dtype=np.uint16)
    image[:10] = 32000

    estimated = estimate_black_level(image)

    assert estimated == 32000
    assert recommended_black_level(estimated) == 30000
    assert recommended_black_level(15000) == 20000


def test_temporal_black_level_trims_outliers() -> None:
    estimates = [24990, 25000, 25010, 25020, 40000, 1000, 25000, 25010, 24980, 25000]

    assert 24990 <= temporal_black_level(estimates) <= 25010
