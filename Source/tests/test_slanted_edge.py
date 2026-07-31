from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from inspection.slanted_edge import (
    calculate_slanted_edge_mtf_curve,
    measure_slanted_edge,
)


def _slanted_edge(sigma: float, size: int = 128) -> np.ndarray:
    yy, xx = np.mgrid[:size, :size]
    edge_x = size / 2 + 0.0875 * (yy - size / 2)
    image = np.where(xx >= edge_x, 50000, 1000).astype(np.float32)
    image = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return np.clip(image, 0, 65535).astype(np.uint16)


def test_slanted_edge_creates_object_side_mtf_curve() -> None:
    curve, orientation, angle, r_squared, contrast = (
        calculate_slanted_edge_mtf_curve(
            _slanted_edge(1.2),
            pixel_pitch_x_um=10,
            pixel_pitch_y_um=10,
            magnification=1,
        )
    )

    assert orientation == "VERTICAL"
    assert 3.0 <= angle <= 7.0
    assert r_squared > 0.95
    assert contrast > 80
    assert curve.source_method == "SLANTED_EDGE"
    assert curve.frequency_range_lpmm[1] <= 50.0
    assert curve.mtf_percent[0] > curve.mtf_percent[-1]


def test_slanted_edge_returns_mtf_at_reference_and_pass() -> None:
    result = measure_slanted_edge(
        _slanted_edge(1.2),
        pixel_pitch_x_um=10,
        pixel_pitch_y_um=10,
        magnification=1,
        reference_frequency_lpmm=15,
        target_mtf_percent=20,
    )

    assert result.status == "PASS"
    assert result.evaluation is not None
    assert result.evaluation.mtf_at_reference_frequency_percent is not None
    assert result.evaluation.mtf_at_reference_frequency_percent >= 20
    assert result.nyquist_frequency_lpmm == 50.0
    assert result.reference_to_nyquist_ratio == 0.3
    assert result.quality_grade in {"GOOD", "WARNING"}


def test_reference_near_nyquist_has_quality_warning() -> None:
    result = measure_slanted_edge(
        _slanted_edge(1.2),
        pixel_pitch_x_um=15,
        pixel_pitch_y_um=15,
        magnification=0.5,
        reference_frequency_lpmm=15,
        target_mtf_percent=20,
    )

    assert result.status in {"PASS", "FAIL"}
    assert result.quality_grade == "WARNING"
    assert result.reference_to_nyquist_ratio is not None
    assert result.reference_to_nyquist_ratio >= 0.8
    assert "Nyquist" in result.quality_message


def test_more_blur_reduces_slanted_edge_mtf() -> None:
    sharp = measure_slanted_edge(
        _slanted_edge(0.8), 10, 10, 1, 15, 0
    )
    blurred = measure_slanted_edge(
        _slanted_edge(2.0), 10, 10, 1, 15, 0
    )

    assert sharp.evaluation is not None
    assert blurred.evaluation is not None
    assert (
        sharp.evaluation.mtf_at_reference_frequency_percent
        > blurred.evaluation.mtf_at_reference_frequency_percent
    )


@pytest.mark.parametrize("sigma", [0.8, 1.2, 2.0])
def test_gaussian_edge_matches_theoretical_mtf(sigma: float) -> None:
    """합성 Gaussian PSF의 이론 MTF와 계산 MTF를 비교한다."""
    reference_lpmm = 15.0
    pixel_pitch_um = 10.0
    result = measure_slanted_edge(
        _slanted_edge(sigma),
        pixel_pitch_um,
        pixel_pitch_um,
        1.0,
        reference_lpmm,
        0.0,
    )

    assert result.evaluation is not None
    measured = result.evaluation.mtf_at_reference_frequency_percent
    assert measured is not None
    frequency_cycles_per_pixel = reference_lpmm * pixel_pitch_um / 1000.0
    theoretical = (
        math.exp(
            -2.0
            * math.pi**2
            * sigma**2
            * frequency_cycles_per_pixel**2
        )
        * 100.0
    )
    assert measured == pytest.approx(theoretical, abs=2.0)


def test_sensor_noise_keeps_mtf_stable() -> None:
    clean_image = _slanted_edge(1.2)
    generator = np.random.default_rng(20260731)
    noisy_image = np.clip(
        clean_image.astype(np.float64)
        + generator.normal(0.0, 500.0, clean_image.shape),
        0,
        65535,
    ).astype(np.uint16)

    clean = measure_slanted_edge(clean_image, 10, 10, 1, 15, 0)
    noisy = measure_slanted_edge(noisy_image, 10, 10, 1, 15, 0)

    assert clean.evaluation is not None
    assert noisy.evaluation is not None
    clean_mtf = clean.evaluation.mtf_at_reference_frequency_percent
    noisy_mtf = noisy.evaluation.mtf_at_reference_frequency_percent
    assert clean_mtf is not None and noisy_mtf is not None
    assert abs(clean_mtf - noisy_mtf) < 5.0


def test_low_contrast_edge_is_invalid() -> None:
    source = _slanted_edge(1.2).astype(np.float64)
    low_contrast = (
        10000.0 + (source - source.min()) / (source.max() - source.min()) * 1000.0
    ).astype(np.uint16)

    result = measure_slanted_edge(low_contrast, 10, 10, 1, 15, 0)

    assert result.status == "INVALID"
    assert "대비" in result.quality_message


def test_multiple_edges_are_invalid() -> None:
    size = 128
    yy, xx = np.mgrid[:size, :size]
    left = size / 2 - 18 + 0.0875 * (yy - size / 2)
    right = size / 2 + 18 + 0.0875 * (yy - size / 2)
    image = np.where((xx >= left) & (xx <= right), 50000, 1000).astype(np.float32)
    image = cv2.GaussianBlur(image, (0, 0), sigmaX=1.2, sigmaY=1.2)

    result = measure_slanted_edge(image.astype(np.uint16), 10, 10, 1, 15, 0)

    assert result.status == "INVALID"
    assert "복수 Edge" in result.quality_message


def test_axis_aligned_edge_is_invalid() -> None:
    image = np.zeros((128, 128), dtype=np.uint16)
    image[:, 64:] = 50000

    result = measure_slanted_edge(image, 10, 10, 1, 15, 20)

    assert result.status == "INVALID"
    assert "기울기" in result.message
    assert result.quality_grade == "INVALID"
    assert result.quality_message == result.message
