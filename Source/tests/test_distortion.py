from __future__ import annotations

import numpy as np
import pytest
import cv2

from inspection.distortion import (
    analyze_checkerboard,
    estimate_checkerboard_size,
    measure_grid_distortion,
)


def test_identical_grid_has_zero_distortion() -> None:
    points = np.array(
        [
            [10, 10],
            [90, 10],
            [10, 90],
            [90, 90],
            [50, 50],
            [25, 50],
        ],
        dtype=np.float64,
    )

    result = measure_grid_distortion(points, points.copy(), image_center=(50, 50))

    assert result.maximum_absolute_percent == pytest.approx(0, abs=1e-10)
    assert result.mean_absolute_percent == pytest.approx(0, abs=1e-10)
    assert np.allclose(result.displacement_vectors, 0)


def test_checkerboard_image_is_detected_with_small_rotation() -> None:
    square = 28
    rows, columns = 10, 14
    chart = np.zeros((rows * square, columns * square), dtype=np.uint8)
    for row in range(rows):
        for column in range(columns):
            if (row + column) % 2:
                chart[
                    row * square : (row + 1) * square,
                    column * square : (column + 1) * square,
                ] = 255
    canvas = np.full((420, 560), 127, dtype=np.uint8)
    y = (canvas.shape[0] - chart.shape[0]) // 2
    x = (canvas.shape[1] - chart.shape[1]) // 2
    canvas[y : y + chart.shape[0], x : x + chart.shape[1]] = chart
    matrix = cv2.getRotationMatrix2D(
        (canvas.shape[1] / 2, canvas.shape[0] / 2), 3.0, 1.0
    )
    rotated = cv2.warpAffine(canvas, matrix, canvas.shape[::-1])

    result = analyze_checkerboard(rotated)

    assert result.status == "VALID"
    assert result.valid_point_count >= 60
    assert result.smia_tv_distortion_percent == pytest.approx(0, abs=0.2)


def test_checkerboard_size_is_estimated_from_repeating_grid() -> None:
    square = 32
    rows, columns = 16, 20
    chart = np.zeros((rows * square, columns * square), dtype=np.uint8)
    for row in range(rows):
        for column in range(columns):
            chart[
                row * square : (row + 1) * square,
                column * square : (column + 1) * square,
            ] = 192 if (row + column) % 2 else 64

    assert estimate_checkerboard_size(chart) == (19, 15)
