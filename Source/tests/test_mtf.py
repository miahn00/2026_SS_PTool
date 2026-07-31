from __future__ import annotations

import math

import numpy as np
import pytest

from inspection.mtf import (
    MtfCurve,
    MtfMeasurementSettings,
    MtfOutOfRangeError,
    calculate_ctf,
    convert_sensor_frequency_to_object_lpmm,
    ctf_to_mtf_percent,
    dominant_frequency,
    evaluate_mtf,
    evaluate_mtf_curve_at_reference,
    interpolate_mtf_at_frequency,
    measure_bar_target_mtf,
    validate_frequency_range,
    usaf_frequency_lpmm,
    usaf_elements_around_frequency,
)


def test_usaf_group_element_frequency() -> None:
    assert usaf_frequency_lpmm(3, 6) == pytest.approx(14.254379, rel=1e-6)
    assert usaf_frequency_lpmm(4, 1) == 16.0


def test_usaf_elements_bracket_evaluation_frequency() -> None:
    elements = usaf_elements_around_frequency(6.0, 2)

    assert [(group, element) for group, element, _ in elements] == [
        (2, 4),
        (2, 5),
    ]
    assert elements[0][2] < 6.0 < elements[1][2]


def _square_profile(length: int = 256) -> np.ndarray:
    pixels = np.arange(length)
    return np.where((pixels // 8) % 2 == 0, 50000.0, 1000.0)


def _settings(reference: float = 6.25, target: float = 70) -> MtfMeasurementSettings:
    return MtfMeasurementSettings(
        reference_frequency_lpmm=reference,
        target_mtf_percent=target,
        pixel_pitch_um=10,
        magnification=1,
        pattern_frequency_tolerance_percent=5,
    )


def test_dominant_frequency_detects_line_pair_period() -> None:
    frequency, confidence = dominant_frequency(_square_profile())
    assert frequency == pytest.approx(1 / 16, rel=0.02)
    assert confidence > 0.2


def test_ctf_and_first_order_coltman_conversion() -> None:
    _, _, contrast, ctf = calculate_ctf(_square_profile())
    assert contrast == pytest.approx(49_000 / 51_000)
    assert ctf == pytest.approx(96.078431, rel=1e-5)
    assert ctf_to_mtf_percent(ctf) == pytest.approx(math.pi / 4 * ctf)


def test_frequency_conversion_uses_pitch_and_magnification() -> None:
    converted = convert_sensor_frequency_to_object_lpmm(1 / 16, 10, 0.5)
    assert converted == pytest.approx(3.125)


def test_curve_interpolates_only_inside_measured_range() -> None:
    curve = MtfCurve(
        np.array([5.0, 10.0, 20.0]),
        np.array([80.0, 60.0, 20.0]),
        "TEST",
    )
    assert validate_frequency_range(curve, 15.0)
    expected = 60.0 + (
        (math.log(15.0) - math.log(10.0))
        / (math.log(20.0) - math.log(10.0))
    ) * (20.0 - 60.0)
    assert interpolate_mtf_at_frequency(curve, 15.0) == pytest.approx(expected)
    with pytest.raises(MtfOutOfRangeError):
        interpolate_mtf_at_frequency(curve, 25.0)


def test_usaf_log_frequency_interpolation_example() -> None:
    curve = MtfCurve(
        np.array([14.30, 16.00]),
        np.array([34.00, 27.00]),
        "USAF_MULTI_POINT",
    )

    result = evaluate_mtf_curve_at_reference(curve, 15.00, 30.00)

    expected_weight = (
        math.log(15.00) - math.log(14.30)
    ) / (
        math.log(16.00) - math.log(14.30)
    )
    assert result.mtf_at_reference_frequency_percent == pytest.approx(
        34.00 + expected_weight * (27.00 - 34.00)
    )
    assert result.status == "PASS"
    assert result.lower_frequency_lpmm == 14.30
    assert result.upper_frequency_lpmm == 16.00
    assert result.interpolation_method == "log_frequency_linear"


def test_evaluate_mtf_uses_only_target_value() -> None:
    assert evaluate_mtf(34.5, 30.0) == "PASS"
    assert evaluate_mtf(24.2, 30.0) == "FAIL"


def test_single_bar_matching_reference_can_pass() -> None:
    result = measure_bar_target_mtf(_square_profile(), _settings())
    assert result.status == "PASS"
    assert result.pattern_validation_status == "MATCH"
    assert result.detected_frequency_lpmm == pytest.approx(6.25, rel=0.02)
    assert result.mtf_at_reference_frequency_percent is not None
    assert result.mtf_at_reference_frequency_percent >= 70


def test_single_bar_matching_reference_can_fail_target() -> None:
    result = measure_bar_target_mtf(_square_profile(), _settings(target=90))
    assert result.status == "FAIL"
    assert result.mtf_at_reference_frequency_percent is not None
    assert "목표 MTF" in result.message


def test_single_bar_does_not_extrapolate_to_other_reference() -> None:
    result = measure_bar_target_mtf(
        _square_profile(),
        _settings(reference=15.0),
    )
    assert result.status == "OUT_OF_RANGE"
    assert result.pattern_validation_status == "MISMATCH"
    assert result.mtf_at_reference_frequency_percent is None
    assert result.mtf_at_detected_frequency_percent > 70


def test_ctf_uses_raw_profile_separately_from_fft_profile() -> None:
    raw = _square_profile()
    detrended = raw - 2000.0
    result = measure_bar_target_mtf(
        detrended,
        _settings(),
        contrast_profile=raw,
    )
    assert result.ctf_at_detected_frequency_percent == pytest.approx(
        96.078431,
        rel=1e-5,
    )
