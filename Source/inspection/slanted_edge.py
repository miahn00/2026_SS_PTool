"""Slanted Edge ROI에서 연속 MTF 곡선을 계산한다.

UI와 영상 입력 방식에 의존하지 않는 1차 구현이다. Edge 위치를 행/열별
최대 기울기로 구하고 직선 피팅한 뒤, Edge 법선 거리로 oversampled ESF를
구성한다. LSF FFT는 sensor Nyquist(0.5 cycles/pixel)까지만 사용한다.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np
from numpy.typing import NDArray

from .mtf import (
    MtfCurve,
    MtfCurveEvaluationResult,
    convert_sensor_frequency_to_object_lpmm,
    evaluate_mtf_curve_at_reference,
)


@dataclass(slots=True, frozen=True)
class SlantedEdgeResult:
    status: str
    message: str
    edge_orientation: str
    edge_angle_degrees: float
    edge_fit_r_squared: float
    contrast_percent: float
    nyquist_frequency_lpmm: float
    reference_to_nyquist_ratio: float | None
    esf_bin_coverage_percent: float
    subpixel_phase_bins: int
    saturation_percent: float
    secondary_edge_ratio: float
    quality_grade: str
    quality_message: str
    curve: MtfCurve | None
    evaluation: MtfCurveEvaluationResult | None


def sample_mtf_curve_at_1_lpmm(
    curve: MtfCurve,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Interpolate a measured curve at integer object-side lp/mm points.

    Only points inside the measured curve are interpolated.  The normalized
    DC point (0 lp/mm, 100%) is included explicitly; no values beyond the
    measured upper frequency are extrapolated.
    """
    lower, upper = curve.frequency_range_lpmm
    integer_frequency = np.arange(1.0, math.floor(upper) + 1.0, dtype=np.float64)
    integer_frequency = integer_frequency[integer_frequency >= lower]
    integer_mtf = np.interp(
        integer_frequency,
        curve.frequency_lpmm,
        curve.mtf_percent,
    )
    return (
        np.r_[0.0, integer_frequency],
        np.r_[100.0, integer_mtf],
    )


def _quality_assessment(
    r_squared: float,
    contrast_percent: float,
    reference_to_nyquist_ratio: float,
    esf_bin_coverage_percent: float,
    subpixel_phase_bins: int,
    saturation_percent: float,
    secondary_edge_ratio: float,
) -> tuple[str, str]:
    """MTF 합격 판정과 별도로 측정 조건의 신뢰도를 평가한다."""
    reasons: list[str] = []
    if r_squared < 0.95:
        return "INVALID", f"Edge 직선 적합도 R²가 낮습니다 ({r_squared:.3f} < 0.950)."
    if contrast_percent < 10.0:
        return "INVALID", f"Edge 대비가 너무 낮습니다 ({contrast_percent:.1f}% < 10%)."
    if esf_bin_coverage_percent < 75.0:
        return "INVALID", (
            f"ESF bin 충족률이 부족합니다 ({esf_bin_coverage_percent:.1f}% < 75%)."
        )
    if subpixel_phase_bins < 3:
        return "INVALID", (
            f"Subpixel phase 분포가 부족합니다 ({subpixel_phase_bins}/4 bins)."
        )
    if secondary_edge_ratio >= 0.5:
        return "INVALID", (
            "ROI에 복수 Edge가 포함된 것으로 판단됩니다 "
            f"(보조/주 Edge={secondary_edge_ratio:.2f})."
        )
    if r_squared < 0.98:
        reasons.append(
            f"Edge 직선 적합도 R²가 권장값 0.980 미만입니다 ({r_squared:.3f})."
        )
    if contrast_percent < 20.0:
        reasons.append(f"Edge 대비가 낮습니다 ({contrast_percent:.1f}%).")
    if esf_bin_coverage_percent < 90.0:
        reasons.append(
            f"ESF bin 충족률이 권장값보다 낮습니다 ({esf_bin_coverage_percent:.1f}%)."
        )
    if subpixel_phase_bins < 4:
        reasons.append(
            f"Subpixel phase가 일부 부족합니다 ({subpixel_phase_bins}/4 bins)."
        )
    if secondary_edge_ratio >= 0.3:
        reasons.append(
            f"보조 Edge 성분이 감지되었습니다 (비율 {secondary_edge_ratio:.2f})."
        )
    if reference_to_nyquist_ratio >= 0.8:
        reasons.append(
            "평가 주파수가 Nyquist의 "
            f"{reference_to_nyquist_ratio * 100:.1f}%로 상한에 가깝습니다."
        )
    if reasons:
        return "WARNING", " ".join(reasons)
    return "GOOD", (
        "측정 조건 양호: "
        f"ESF {esf_bin_coverage_percent:.1f}%, "
        f"phase {subpixel_phase_bins}/4, "
        f"포화 {saturation_percent:.1f}%."
    )


def _edge_quality_metrics(
    roi_image: NDArray[np.generic],
) -> tuple[float, int, float, float]:
    """ESF sampling, clipping and multiple-edge indicators."""
    gray = _gray_float(roi_image)
    orientation, slope, intercept, _, angle = _fit_edge(gray)
    independent_size = gray.shape[0] if orientation == "VERTICAL" else gray.shape[1]
    independent = np.arange(independent_size, dtype=np.float64)
    phases = np.mod(slope * independent + intercept, 1.0)
    phase_counts, _ = np.histogram(phases, bins=np.linspace(0.0, 1.0, 5))
    phase_threshold = max(2, int(independent_size * 0.02))
    phase_bins = int(np.count_nonzero(phase_counts >= phase_threshold))

    yy, xx = np.mgrid[: gray.shape[0], : gray.shape[1]]
    if orientation == "VERTICAL":
        distance = (xx - (slope * yy + intercept)) / math.sqrt(1 + slope**2)
        gradient = np.mean(np.abs(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)), axis=0)
    else:
        distance = (yy - (slope * xx + intercept)) / math.sqrt(1 + slope**2)
        gradient = np.mean(np.abs(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)), axis=1)
    indices = np.floor((distance.ravel() - distance.min()) * 4).astype(int)
    counts = np.bincount(indices)
    esf_coverage = float(np.count_nonzero(counts) / max(1, counts.size) * 100.0)

    main_index = int(np.argmax(gradient))
    main_strength = float(gradient[main_index])
    remaining = gradient.copy()
    edge_span = int(
        math.ceil(math.tan(math.radians(angle)) * independent_size)
    )
    radius = max(4, edge_span + 4)
    remaining[max(0, main_index - radius): main_index + radius + 1] = 0
    secondary_ratio = (
        float(np.max(remaining)) / main_strength
        if main_strength > np.finfo(float).eps else 1.0
    )

    if np.issubdtype(roi_image.dtype, np.integer):
        limits = np.iinfo(roi_image.dtype)
        saturated = (roi_image <= limits.min) | (roi_image >= limits.max)
        if roi_image.ndim == 3:
            saturated = np.any(saturated, axis=2)
        saturation = float(np.mean(saturated) * 100.0)
    else:
        saturation = 0.0
    return esf_coverage, phase_bins, saturation, secondary_ratio


def _gray_float(image: NDArray[np.generic]) -> NDArray[np.float64]:
    if image.ndim == 2:
        gray = image.astype(np.float64)
    elif image.ndim == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float64)
    elif image.ndim == 3 and image.shape[2] == 4:
        gray = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY).astype(np.float64)
    else:
        raise ValueError(f"지원하지 않는 Edge 영상 형태입니다: {image.shape}")
    return gray


def _fit_edge(
    gray: NDArray[np.float64],
) -> tuple[str, float, float, float, float]:
    gradient_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    energy_x = float(np.mean(np.abs(gradient_x)))
    energy_y = float(np.mean(np.abs(gradient_y)))
    orientation = "VERTICAL" if energy_x >= energy_y else "HORIZONTAL"

    if orientation == "VERTICAL":
        positions = np.argmax(np.abs(gradient_x), axis=1).astype(np.float64)
        strengths = np.max(np.abs(gradient_x), axis=1)
        independent = np.arange(gray.shape[0], dtype=np.float64)
    else:
        positions = np.argmax(np.abs(gradient_y), axis=0).astype(np.float64)
        strengths = np.max(np.abs(gradient_y), axis=0)
        independent = np.arange(gray.shape[1], dtype=np.float64)

    threshold = max(float(np.percentile(strengths, 25)), np.finfo(float).eps)
    valid = strengths >= threshold
    if np.count_nonzero(valid) < 16:
        raise ValueError("Edge 직선 피팅에 필요한 유효 위치가 부족합니다.")
    slope, intercept = np.polyfit(independent[valid], positions[valid], 1)
    predicted = slope * independent[valid] + intercept
    residual = positions[valid] - predicted
    total = positions[valid] - positions[valid].mean()
    denominator = float(np.sum(total**2))
    residual_sum = float(np.sum(residual**2))
    r_squared = (
        1.0 - residual_sum / denominator
        if denominator > np.finfo(float).eps
        else 1.0 if residual_sum <= np.finfo(float).eps
        else 0.0
    )
    angle = abs(math.degrees(math.atan(float(slope))))
    return orientation, float(slope), float(intercept), r_squared, angle


def _oversampled_esf(
    gray: NDArray[np.float64],
    orientation: str,
    slope: float,
    intercept: float,
    oversampling: int,
) -> tuple[NDArray[np.float64], float]:
    yy, xx = np.mgrid[: gray.shape[0], : gray.shape[1]]
    if orientation == "VERTICAL":
        distance = (xx - (slope * yy + intercept)) / math.sqrt(1 + slope**2)
    else:
        distance = (yy - (slope * xx + intercept)) / math.sqrt(1 + slope**2)

    bin_width = 1.0 / oversampling
    minimum = math.floor(float(distance.min()) / bin_width) * bin_width
    indices = np.floor((distance.ravel() - minimum) / bin_width).astype(int)
    count = np.bincount(indices)
    total = np.bincount(indices, weights=gray.ravel())
    valid = count > 0
    if np.count_nonzero(valid) < 32:
        raise ValueError("Oversampled ESF를 구성할 데이터가 부족합니다.")
    centers = minimum + (np.arange(count.size) + 0.5) * bin_width
    esf = total[valid] / count[valid]
    valid_centers = centers[valid]
    uniform_centers = np.arange(valid_centers[0], valid_centers[-1], bin_width)
    esf = np.interp(uniform_centers, valid_centers, esf)
    if esf[-1] < esf[0]:
        esf = esf[::-1]
    return esf, bin_width


def calculate_slanted_edge_mtf_curve(
    roi_image: NDArray[np.generic],
    pixel_pitch_x_um: float,
    pixel_pitch_y_um: float,
    magnification: float,
    *,
    oversampling: int = 4,
) -> tuple[MtfCurve, str, float, float, float]:
    """ROI에서 object-side lp/mm MTF 곡선과 Edge 품질 정보를 반환한다."""
    if roi_image.shape[0] < 32 or roi_image.shape[1] < 32:
        raise ValueError("Slanted Edge ROI는 최소 32×32 픽셀이어야 합니다.")
    if oversampling < 2:
        raise ValueError("ESF oversampling은 2 이상이어야 합니다.")
    gray = _gray_float(roi_image)
    low, high = np.percentile(gray, (10, 90))
    dynamic = float(gray.max() - gray.min())
    if dynamic <= 0:
        raise ValueError("Edge ROI에 밝기 대비가 없습니다.")
    contrast = float((high - low) / max(high + low, np.finfo(float).eps) * 100)
    if high - low < max(1.0, dynamic * 0.1):
        raise ValueError("Edge ROI 대비가 너무 낮습니다.")

    orientation, slope, intercept, r_squared, angle = _fit_edge(gray)
    if r_squared < 0.8:
        raise ValueError("Edge 직선 피팅 신뢰도가 낮습니다.")
    if not 1.0 <= angle <= 15.0:
        raise ValueError(
            f"Edge 기울기 {angle:.2f}°가 권장 분석 범위 1~15° 밖에 있습니다."
        )

    esf, sample_spacing = _oversampled_esf(
        gray,
        orientation,
        slope,
        intercept,
        oversampling,
    )
    lsf = np.gradient(esf)
    lsf *= np.hamming(lsf.size)
    spectrum = np.abs(np.fft.rfft(lsf))
    if spectrum.size == 0 or spectrum[0] <= np.finfo(float).eps:
        raise ValueError("LSF에서 유효한 MTF를 계산할 수 없습니다.")
    mtf = spectrum / spectrum[0] * 100.0
    sensor_frequency = np.fft.rfftfreq(lsf.size, d=sample_spacing)
    valid = (sensor_frequency > 0) & (sensor_frequency <= 0.5)
    sensor_frequency = sensor_frequency[valid]
    mtf = np.clip(mtf[valid], 0, 100)
    if sensor_frequency.size < 2:
        raise ValueError("Sensor Nyquist 범위의 MTF 측정점이 부족합니다.")

    pixel_pitch = (
        pixel_pitch_x_um if orientation == "VERTICAL" else pixel_pitch_y_um
    )
    object_frequency = convert_sensor_frequency_to_object_lpmm(
        sensor_frequency,
        pixel_pitch,
        magnification,
    )
    curve = MtfCurve(
        np.asarray(object_frequency, dtype=np.float64),
        np.asarray(mtf, dtype=np.float64),
        "SLANTED_EDGE",
    )
    return curve, orientation, angle, r_squared, contrast


def measure_slanted_edge(
    roi_image: NDArray[np.generic],
    pixel_pitch_x_um: float,
    pixel_pitch_y_um: float,
    magnification: float,
    reference_frequency_lpmm: float,
    target_mtf_percent: float,
) -> SlantedEdgeResult:
    """Slanted Edge MTF 곡선을 계산하고 평가 주파수에서 판정한다."""
    try:
        if roi_image.shape[0] < 32 or roi_image.shape[1] < 32:
            raise ValueError("Slanted Edge ROI는 최소 32×32 픽셀이어야 합니다.")
        (
            esf_bin_coverage,
            subpixel_phase_bins,
            saturation,
            secondary_edge_ratio,
        ) = _edge_quality_metrics(roi_image)
        if secondary_edge_ratio >= 0.5:
            message = (
                "ROI에 복수 Edge가 포함된 것으로 판단됩니다 "
                f"(보조/주 Edge={secondary_edge_ratio:.2f})."
            )
            return SlantedEdgeResult(
                "INVALID",
                message,
                "Unknown",
                0.0,
                0.0,
                0.0,
                0.0,
                None,
                esf_bin_coverage,
                subpixel_phase_bins,
                saturation,
                secondary_edge_ratio,
                "INVALID",
                message,
                None,
                None,
            )
        curve, orientation, angle, r_squared, contrast = (
            calculate_slanted_edge_mtf_curve(
                roi_image,
                pixel_pitch_x_um,
                pixel_pitch_y_um,
                magnification,
            )
        )
        pixel_pitch = (
            pixel_pitch_x_um if orientation == "VERTICAL" else pixel_pitch_y_um
        )
        nyquist_frequency = float(
            convert_sensor_frequency_to_object_lpmm(
                0.5, pixel_pitch, magnification
            )
        )
        reference_to_nyquist_ratio = (
            reference_frequency_lpmm / nyquist_frequency
            if nyquist_frequency > 0
            else None
        )
        evaluation = evaluate_mtf_curve_at_reference(
            curve,
            reference_frequency_lpmm,
            target_mtf_percent,
            interpolation_method="linear_frequency",
        )
        quality_grade, quality_message = _quality_assessment(
            r_squared,
            contrast,
            reference_to_nyquist_ratio or 0.0,
            esf_bin_coverage,
            subpixel_phase_bins,
            saturation,
            secondary_edge_ratio,
        )
        if quality_grade == "INVALID":
            return SlantedEdgeResult(
                "INVALID",
                quality_message,
                orientation,
                angle,
                r_squared,
                contrast,
                nyquist_frequency,
                reference_to_nyquist_ratio,
                esf_bin_coverage,
                subpixel_phase_bins,
                saturation,
                secondary_edge_ratio,
                quality_grade,
                quality_message,
                curve,
                evaluation,
            )
    except ValueError as exc:
        return SlantedEdgeResult(
            "INVALID",
            str(exc),
            "Unknown",
            0.0,
            0.0,
            0.0,
            0.0,
            None,
            0.0,
            0,
            0.0,
            0.0,
            "INVALID",
            str(exc),
            None,
            None,
        )
    return SlantedEdgeResult(
        evaluation.status,
        evaluation.message,
        orientation,
        angle,
        r_squared,
        contrast,
        nyquist_frequency,
        reference_to_nyquist_ratio,
        esf_bin_coverage,
        subpixel_phase_bins,
        saturation,
        secondary_edge_ratio,
        quality_grade,
        quality_message,
        curve,
        evaluation,
    )
