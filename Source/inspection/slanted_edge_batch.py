"""현재 ROI 목록의 Slanted Edge MTF 일괄 측정."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from imaging.roi import RoiData
from .roi_analysis import extract_roi
from .slanted_edge import SlantedEdgeResult, measure_slanted_edge


@dataclass(slots=True, frozen=True)
class RoiSlantedEdgeResult:
    roi_number: int
    roi_name: str
    included_in_judgment: bool
    result: SlantedEdgeResult


@dataclass(slots=True, frozen=True)
class BatchSlantedEdgeResult:
    overall_status: str
    message: str
    roi_results: tuple[RoiSlantedEdgeResult, ...]


def measure_rois_slanted_edge(
    image: NDArray[np.generic],
    rois: list[RoiData],
    *,
    pixel_pitch_x_um: float,
    pixel_pitch_y_um: float,
    magnification: float,
    reference_frequency_lpmm: float,
    target_mtf_percent: float,
) -> BatchSlantedEdgeResult:
    results: list[RoiSlantedEdgeResult] = []
    for roi in sorted(rois, key=lambda value: value.number):
        if not roi.active:
            result = SlantedEdgeResult(
                "INACTIVE",
                "비활성화된 ROI입니다.",
                "Unknown",
                0,
                0,
                0,
                0,
                None,
                0,
                0,
                0,
                0,
                "INACTIVE",
                "비활성화된 ROI입니다.",
                None,
                None,
            )
        else:
            result = measure_slanted_edge(
                extract_roi(image, roi),
                pixel_pitch_x_um,
                pixel_pitch_y_um,
                magnification,
                reference_frequency_lpmm,
                target_mtf_percent,
            )
        results.append(
            RoiSlantedEdgeResult(
                roi.number,
                roi.name,
                roi.include_in_judgment and roi.active,
                result,
            )
        )

    judged = [item.result for item in results if item.included_in_judgment]
    if not judged:
        status, message = "INVALID", "판정에 포함된 활성 ROI가 없습니다."
    elif any(item.status == "INVALID" for item in judged):
        status, message = "INVALID", "분석할 수 없는 Slanted Edge ROI가 있습니다."
    elif any(item.status == "OUT_OF_RANGE" for item in judged):
        status, message = "OUT_OF_RANGE", "평가 주파수가 측정 범위 밖인 ROI가 있습니다."
    elif any(item.status == "FAIL" for item in judged):
        status, message = "FAIL", "목표 MTF에 미달한 ROI가 있습니다."
    else:
        status, message = "PASS", "판정 포함 ROI가 모두 목표 MTF를 만족합니다."
    return BatchSlantedEdgeResult(status, message, tuple(results))
