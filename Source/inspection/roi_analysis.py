"""MTF 계산 전 ROI 추출과 기초 신호 분석."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from imaging.roi import RoiData


@dataclass(slots=True)
class RoiAnalysisResult:
    status: str
    message: str
    roi_image: NDArray[np.generic]
    processed_image: NDArray[np.float32]
    raw_profile: NDArray[np.float64]
    profile: NDArray[np.float64]
    fft_frequency: NDArray[np.float64]
    fft_magnitude: NDArray[np.float64]
    detected_direction: str
    direction_confidence: float
    minimum: float
    maximum: float
    mean: float
    standard_deviation: float
    saturation_percent: float


def extract_roi(image: NDArray[np.generic], roi: RoiData) -> NDArray[np.generic]:
    """원본 영상에서 ROI를 독립된 연속 배열로 복사한다."""
    height, width = image.shape[:2]
    roi.validate(width, height)
    return np.ascontiguousarray(
        image[roi.y : roi.y + roi.height, roi.x : roi.x + roi.width]
    ).copy()


def _to_gray_float(image: NDArray[np.generic]) -> NDArray[np.float32]:
    if image.ndim == 2:
        return image.astype(np.float32)
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY).astype(np.float32)
    raise ValueError(f"지원하지 않는 ROI 영상 형태입니다: {image.shape}")


def _remove_plane(image: NDArray[np.float32]) -> NDArray[np.float32]:
    rows, columns = image.shape
    yy, xx = np.mgrid[:rows, :columns]
    design = np.column_stack((xx.ravel(), yy.ravel(), np.ones(image.size)))
    coefficients, *_ = np.linalg.lstsq(design, image.ravel(), rcond=None)
    plane = (coefficients[0] * xx + coefficients[1] * yy + coefficients[2])
    return (image - plane + float(image.mean())).astype(np.float32)


def _detect_direction(image: NDArray[np.float32]) -> tuple[str, float]:
    gradient_x = float(np.mean(np.abs(cv2.Sobel(image, cv2.CV_32F, 1, 0))))
    gradient_y = float(np.mean(np.abs(cv2.Sobel(image, cv2.CV_32F, 0, 1))))
    total = gradient_x + gradient_y
    if total <= np.finfo(np.float32).eps:
        return "Unknown", 0.0
    direction = "V" if gradient_x >= gradient_y else "H"
    confidence = abs(gradient_x - gradient_y) / total
    return direction, float(confidence)


def _empty_result(
    status: str,
    message: str,
    roi_image: NDArray[np.generic],
    *,
    minimum: float = 0.0,
    maximum: float = 0.0,
    mean: float = 0.0,
    standard_deviation: float = 0.0,
    saturation_percent: float = 0.0,
) -> RoiAnalysisResult:
    empty = np.array([], dtype=np.float64)
    return RoiAnalysisResult(
        status=status,
        message=message,
        roi_image=roi_image,
        processed_image=np.empty((0, 0), dtype=np.float32),
        raw_profile=empty,
        profile=empty,
        fft_frequency=empty,
        fft_magnitude=empty,
        detected_direction="Unknown",
        direction_confidence=0.0,
        minimum=minimum,
        maximum=maximum,
        mean=mean,
        standard_deviation=standard_deviation,
        saturation_percent=saturation_percent,
    )


def analyze_roi(
    image: NDArray[np.generic],
    roi: RoiData,
) -> RoiAnalysisResult:
    """선택 ROI의 품질, 방향, 평균 프로파일과 FFT를 계산한다."""
    roi_image = extract_roi(image, roi)
    if not roi.active:
        return _empty_result("INVALID", "비활성화된 ROI입니다.", roi_image)
    if roi.width < 8 or roi.height < 8:
        return _empty_result("INVALID", "ROI 크기가 너무 작습니다.", roi_image)

    gray = _to_gray_float(roi_image)
    minimum = float(gray.min())
    maximum = float(gray.max())
    mean = float(gray.mean())
    standard_deviation = float(gray.std())
    dtype_max = float(np.iinfo(roi_image.dtype).max)
    saturation_percent = float(
        np.count_nonzero((gray <= 0) | (gray >= dtype_max)) / gray.size * 100.0
    )
    if standard_deviation < 1.0:
        return _empty_result(
            "INVALID",
            "ROI 대비가 너무 낮습니다.",
            roi_image,
            minimum=minimum,
            maximum=maximum,
            mean=mean,
            standard_deviation=standard_deviation,
            saturation_percent=saturation_percent,
        )

    processed = _remove_plane(gray)
    detected_direction, confidence = _detect_direction(processed)
    direction = roi.direction if roi.direction in {"H", "V"} else detected_direction
    if direction == "Unknown" or (roi.direction == "Auto" and confidence < 0.1):
        return _empty_result(
            "INVALID",
            "패턴 방향 신뢰도가 낮습니다.",
            roi_image,
            minimum=minimum,
            maximum=maximum,
            mean=mean,
            standard_deviation=standard_deviation,
            saturation_percent=saturation_percent,
        )

    raw_profile = (
        gray.mean(axis=0, dtype=np.float64)
        if direction == "V"
        else gray.mean(axis=1, dtype=np.float64)
    )
    profile = (
        processed.mean(axis=0, dtype=np.float64)
        if direction == "V"
        else processed.mean(axis=1, dtype=np.float64)
    )
    centered = profile - profile.mean()
    windowed = centered * np.hanning(centered.size)
    magnitude = np.abs(np.fft.rfft(windowed))
    if magnitude.size and magnitude.max() > 0:
        magnitude = magnitude / magnitude.max()
    frequency = np.fft.rfftfreq(centered.size, d=1.0)

    return RoiAnalysisResult(
        status="READY",
        message="MTF 계산 전처리가 완료되었습니다.",
        roi_image=roi_image,
        processed_image=processed,
        raw_profile=raw_profile,
        profile=profile,
        fft_frequency=frequency,
        fft_magnitude=magnitude,
        detected_direction=detected_direction,
        direction_confidence=confidence,
        minimum=minimum,
        maximum=maximum,
        mean=mean,
        standard_deviation=standard_deviation,
        saturation_percent=saturation_percent,
    )
