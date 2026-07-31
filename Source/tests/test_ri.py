from __future__ import annotations

import numpy as np
import pytest

from inspection.ri import (
    evaluate_minimum_ri,
    generate_ri_grid_rois,
    measure_grid_relative_illumination,
    measure_relative_illumination,
)


def test_relative_illumination_uses_center_as_100_percent() -> None:
    image = np.zeros((100, 100), dtype=np.uint16)
    image[40:60, 40:60] = 1000
    image[0:20, 0:20] = 800
    image[0:20, 80:100] = 700

    result = measure_relative_illumination(
        image,
        center_roi=(40, 40, 20, 20),
        measurement_rois=[(0, 0, 20, 20), (80, 0, 20, 20)],
    )

    assert result.center_mean == 1000
    assert result.relative_percent == pytest.approx((80, 70))
    assert result.minimum_percent == 70
    assert result.maximum_asymmetry_percent == 10


def test_generate_five_by_five_grid_uses_inner_half() -> None:
    rois = generate_ri_grid_rois(
        100,
        100,
        rows=5,
        columns=5,
        inner_fraction=0.5,
    )

    assert len(rois) == 25
    assert rois[0] == (5, 5, 10, 10)
    assert rois[12] == (45, 45, 10, 10)
    assert rois[-1] == (85, 85, 10, 10)


def test_grid_ri_reports_center_corners_minimum_and_asymmetry() -> None:
    levels = np.array(
        [
            [70, 80, 85, 78, 60],
            [80, 90, 95, 88, 75],
            [85, 95, 100, 92, 80],
            [78, 88, 94, 86, 72],
            [65, 75, 80, 70, 50],
        ],
        dtype=np.uint16,
    )
    image = np.kron(levels, np.ones((20, 20), dtype=np.uint16))

    result = measure_grid_relative_illumination(image)

    assert result.center_mean == 100
    assert result.minimum_percent == 50
    assert result.minimum_position == (4, 4)
    assert result.corner_percent == pytest.approx((70, 60, 65, 50))
    assert result.left_right_asymmetry_percent == 15
    assert result.top_bottom_asymmetry_percent == 10
    assert len(result.cells) == 25
    assert result.cells[12].region_type == "CENTER"
    assert result.cells[0].region_type == "CORNER"
    assert result.cells[1].region_type == "EDGE"


def test_grid_ri_rejects_even_grid_and_zero_center() -> None:
    with pytest.raises(ValueError, match="홀수"):
        generate_ri_grid_rois(100, 100, rows=4, columns=5)

    with pytest.raises(ValueError, match="중앙"):
        measure_grid_relative_illumination(
            np.zeros((100, 100), dtype=np.uint16)
        )


def test_minimum_ri_evaluation_pass_fail_and_unset() -> None:
    image = np.full((100, 100), 1000, dtype=np.uint16)
    image[:20, :20] = 800
    result = measure_grid_relative_illumination(image)

    assert evaluate_minimum_ri(result, 75).status == "PASS"
    assert evaluate_minimum_ri(result, 85).status == "FAIL"
    assert evaluate_minimum_ri(result, 0).status == "UNSET"
