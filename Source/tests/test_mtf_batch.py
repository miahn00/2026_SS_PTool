from __future__ import annotations

import numpy as np

from imaging.roi import RoiData
from inspection.mtf_batch import measure_rois_mtf


def _two_pattern_image() -> np.ndarray:
    image = np.zeros((128, 256), dtype=np.uint16)
    x = np.arange(128)
    vertical = np.where((x // 8) % 2 == 0, 50000, 1000).astype(np.uint16)
    image[:, :128] = np.tile(vertical, (128, 1))

    y = np.arange(128)
    horizontal = np.where((y // 4) % 2 == 0, 50000, 1000).astype(np.uint16)
    image[:, 128:] = np.tile(horizontal[:, None], (1, 128))
    return image


def test_batch_mtf_all_pass() -> None:
    rois = [
        RoiData(1, "Vertical", 0, 0, 128, 128, usaf_frequency_lpmm=6.25),
        RoiData(2, "Horizontal", 128, 0, 128, 128, usaf_frequency_lpmm=12.5),
    ]

    result = measure_rois_mtf(
        _two_pattern_image(),
        rois,
        reference_frequency_lpmm=8.0,
        target_mtf_percent=70,
        pixel_pitch_x_um=10,
        pixel_pitch_y_um=10,
        magnification=1,
        pattern_frequency_tolerance_percent=100,
    )

    assert result.overall_status == "PASS"
    assert [item.status for item in result.roi_results] == ["PASS", "PASS"]
    assert all(
        item.measurement is not None
        for item in result.roi_results
    )
    assert result.curve_evaluation is None
    assert all(item.chart_frequency_lpmm == 8.0 for item in result.roi_results)
    assert [item.detected_direction for item in result.roi_results] == ["V", "H"]


def test_batch_mtf_invalid_included_roi_makes_overall_invalid() -> None:
    image = _two_pattern_image()
    image[32:96, 32:96] = 1000
    rois = [RoiData(1, "Flat", 32, 32, 64, 64)]

    result = measure_rois_mtf(
        image,
        rois,
        reference_frequency_lpmm=6.25,
        target_mtf_percent=70,
        pixel_pitch_x_um=10,
        pixel_pitch_y_um=10,
        magnification=1,
        pattern_frequency_tolerance_percent=5,
    )

    assert result.overall_status == "INVALID"
    assert result.roi_results[0].status == "INVALID"


def test_batch_mtf_uses_reference_frequency_for_every_roi() -> None:
    result = measure_rois_mtf(
        _two_pattern_image(),
        [RoiData(1, "No frequency", 0, 0, 128, 128)],
        reference_frequency_lpmm=6.0,
        target_mtf_percent=30,
        pixel_pitch_x_um=10,
        pixel_pitch_y_um=10,
        magnification=1,
        pattern_frequency_tolerance_percent=10,
    )

    assert result.roi_results[0].chart_frequency_lpmm == 6.0
    assert result.roi_results[0].measurement is not None


def test_batch_mtf_excludes_unjudged_invalid_roi() -> None:
    image = _two_pattern_image()
    rois = [
        RoiData(
            1,
            "Valid",
            0,
            0,
            128,
            128,
            usaf_frequency_lpmm=6.25,
        ),
        RoiData(
            2,
            "Ignored flat",
            160,
            32,
            32,
            32,
            include_in_judgment=False,
        ),
    ]
    image[32:64, 160:192] = 1000

    result = measure_rois_mtf(
        image,
        rois,
        reference_frequency_lpmm=6.25,
        target_mtf_percent=70,
        pixel_pitch_x_um=10,
        pixel_pitch_y_um=10,
        magnification=1,
        pattern_frequency_tolerance_percent=5,
    )

    assert result.overall_status == "PASS"
    assert result.roi_results[1].status == "INVALID"
    assert not result.roi_results[1].included_in_judgment


def test_batch_mtf_reports_out_of_range_without_extrapolation() -> None:
    result = measure_rois_mtf(
        _two_pattern_image(),
        [
            RoiData(
                1,
                "Low frequency",
                0,
                0,
                128,
                128,
                usaf_frequency_lpmm=6.25,
            )
        ],
        reference_frequency_lpmm=15.0,
        target_mtf_percent=30,
        pixel_pitch_x_um=10,
        pixel_pitch_y_um=10,
        magnification=1,
        pattern_frequency_tolerance_percent=5,
    )

    assert result.overall_status == "OUT_OF_RANGE"
    measurement = result.roi_results[0].measurement
    assert measurement is not None
    assert result.curve_evaluation is None
    assert measurement.mtf_at_reference_frequency_percent is None
