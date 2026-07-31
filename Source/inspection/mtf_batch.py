"""여러 USAF/Bar ROI 측정점으로 MTF 곡선을 구성하고 평가한다."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from imaging.roi import RoiData
from .mtf import (
    MtfCurveEvaluationResult,
    MtfMeasurementResult,
    MtfMeasurementSettings,
    measure_bar_target_mtf,
)
from .roi_analysis import analyze_roi


@dataclass(slots=True, frozen=True)
class RoiMtfResult:
    roi_number: int
    roi_name: str
    included_in_judgment: bool
    status: str
    message: str
    detected_direction: str
    direction_confidence: float
    chart_frequency_lpmm: float | None = None
    measurement: MtfMeasurementResult | None = None
    usaf_group: int | None = None
    usaf_element: int | None = None


@dataclass(slots=True, frozen=True)
class BatchMtfResult:
    overall_status: str
    message: str
    roi_results: tuple[RoiMtfResult, ...]
    curve_evaluation: MtfCurveEvaluationResult | None = None


def measure_rois_mtf(
    image: NDArray[np.generic],
    rois: list[RoiData],
    *,
    reference_frequency_lpmm: float,
    target_mtf_percent: float,
    pixel_pitch_x_um: float,
    pixel_pitch_y_um: float,
    magnification: float,
    pattern_frequency_tolerance_percent: float,
    interpolation_method: str = "log_frequency_linear",
) -> BatchMtfResult:
    """ROI별 MTF 점을 모아 기준 주파수에서 보간하고 판정한다."""
    results: list[RoiMtfResult] = []
    for roi in sorted(rois, key=lambda value: value.number):
        if not roi.active:
            results.append(
                RoiMtfResult(
                    roi.number,
                    roi.name,
                    False,
                    "INACTIVE",
                    "비활성화된 ROI입니다.",
                    "Unknown",
                    0.0,
                )
            )
            continue

        analysis = analyze_roi(image, roi)
        if analysis.status != "READY":
            results.append(
                RoiMtfResult(
                    roi.number,
                    roi.name,
                    roi.include_in_judgment,
                    "INVALID",
                    analysis.message,
                    analysis.detected_direction,
                    analysis.direction_confidence,
                )
            )
            continue

        direction = (
            roi.direction
            if roi.direction in {"H", "V"}
            else analysis.detected_direction
        )
        pixel_pitch = pixel_pitch_x_um if direction == "V" else pixel_pitch_y_um

        # 모든 ROI는 공통 평가 주파수의 USAF 패턴을 반복 측정한다.
        point_settings = MtfMeasurementSettings(
            reference_frequency_lpmm=reference_frequency_lpmm,
            target_mtf_percent=target_mtf_percent,
            pixel_pitch_um=pixel_pitch,
            magnification=magnification,
            pattern_frequency_tolerance_percent=pattern_frequency_tolerance_percent,
        )
        try:
            measurement = measure_bar_target_mtf(
                analysis.profile,
                point_settings,
                contrast_profile=analysis.raw_profile,
            )
        except ValueError as exc:
            results.append(
                RoiMtfResult(
                    roi.number,
                    roi.name,
                    roi.include_in_judgment,
                    "INVALID",
                    str(exc),
                    direction,
                    analysis.direction_confidence,
                )
            )
            continue

        results.append(
            RoiMtfResult(
                roi.number,
                roi.name,
                roi.include_in_judgment,
                measurement.status,
                (
                    "패턴 주파수 MATCH | " + measurement.message
                    if measurement.pattern_validation_status == "MATCH"
                    else "패턴 주파수 오차가 허용범위를 초과했습니다."
                ),
                direction,
                analysis.direction_confidence,
                reference_frequency_lpmm,
                measurement,
                roi.usaf_group,
                roi.usaf_element,
            )
        )

    judged = [item for item in results if item.included_in_judgment]
    if not judged:
        return BatchMtfResult(
            "INVALID",
            "판정에 포함된 활성 ROI가 없습니다.",
            tuple(results),
        )
    if any(item.status == "INVALID" for item in judged):
        status, message = "INVALID", "분석할 수 없는 USAF ROI가 있습니다."
    elif any(item.status == "OUT_OF_RANGE" for item in judged):
        status, message = (
            "OUT_OF_RANGE",
            "평가 주파수와 일치하지 않는 USAF 패턴 ROI가 있습니다.",
        )
    elif any(item.status == "FAIL" for item in judged):
        status, message = "FAIL", "목표 MTF에 미달한 ROI가 있습니다."
    else:
        status, message = "PASS", "판정 포함 ROI가 모두 목표 MTF를 만족합니다."
    return BatchMtfResult(status, message, tuple(results))
