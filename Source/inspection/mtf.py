"""Bar target 측정점과 MTF 곡선의 평가 함수.

Bar/USAF 단일 주파수 ROI는 CTF 및 환산 MTF 측정점 하나만 제공한다.
측정 범위 밖의 기준 주파수로 외삽하지 않는다. ``magnification``은
image size / object size로 정의한다.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import NDArray


class MtfOutOfRangeError(ValueError):
    """기준 주파수가 실제 MTF 측정 범위를 벗어난 경우."""


def usaf_frequency_lpmm(group: int, element: int) -> float:
    """USAF 1951 Group/Element의 공칭 공간주파수를 반환한다."""
    if not 1 <= element <= 6:
        raise ValueError("USAF Element는 1~6 범위여야 합니다.")
    return float(2.0 ** (group + (element - 1) / 6.0))


def usaf_elements_around_frequency(
    reference_lpmm: float,
    count: int,
    *,
    minimum_group: int = -2,
    maximum_group: int = 9,
) -> list[tuple[int, int, float]]:
    """평가 주파수를 둘러싸는 연속 USAF 요소를 낮은 순서로 반환한다."""
    if reference_lpmm <= 0:
        raise ValueError("평가 주파수는 0보다 커야 합니다.")
    if count <= 0:
        return []
    catalog = [
        (group, element, usaf_frequency_lpmm(group, element))
        for group in range(minimum_group, maximum_group + 1)
        for element in range(1, 7)
    ]
    count = min(count, len(catalog))
    if count == 1:
        return [min(catalog, key=lambda item: abs(item[2] - reference_lpmm))]
    upper_index = next(
        (
            index
            for index, item in enumerate(catalog)
            if item[2] >= reference_lpmm
        ),
        len(catalog) - 1,
    )
    start = upper_index - count // 2
    start = max(0, min(start, len(catalog) - count))
    return catalog[start:start + count]


@dataclass(slots=True, frozen=True)
class MtfCurve:
    frequency_lpmm: NDArray[np.float64]
    mtf_percent: NDArray[np.float64]
    source_method: str

    def __post_init__(self) -> None:
        frequency = np.asarray(self.frequency_lpmm, dtype=np.float64)
        mtf = np.asarray(self.mtf_percent, dtype=np.float64)
        if frequency.ndim != 1 or mtf.ndim != 1 or frequency.size != mtf.size:
            raise ValueError("MTF 주파수와 값은 길이가 같은 1D 배열이어야 합니다.")
        if frequency.size == 0:
            raise ValueError("MTF 곡선에 측정점이 없습니다.")
        if np.any(frequency <= 0) or np.any(np.diff(frequency) <= 0):
            raise ValueError("MTF 주파수는 양수이며 오름차순이어야 합니다.")
        if np.any((mtf < 0) | (mtf > 100)):
            raise ValueError("MTF 값은 0~100% 범위여야 합니다.")
        object.__setattr__(self, "frequency_lpmm", frequency)
        object.__setattr__(self, "mtf_percent", mtf)

    @property
    def frequency_range_lpmm(self) -> tuple[float, float]:
        return float(self.frequency_lpmm[0]), float(self.frequency_lpmm[-1])


@dataclass(slots=True, frozen=True)
class MtfMeasurementSettings:
    reference_frequency_lpmm: float
    target_mtf_percent: float
    pixel_pitch_um: float
    magnification: float = 1.0
    pattern_frequency_tolerance_percent: float = 10.0

    def validate(self) -> None:
        if self.reference_frequency_lpmm <= 0:
            raise ValueError("기준 공간주파수는 0보다 커야 합니다.")
        if not 0 <= self.target_mtf_percent <= 100:
            raise ValueError("목표 MTF는 0~100% 범위여야 합니다.")
        if self.pixel_pitch_um <= 0:
            raise ValueError("Pixel pitch는 0보다 커야 합니다.")
        if self.magnification <= 0:
            raise ValueError("렌즈 배율은 0보다 커야 합니다.")
        if self.pattern_frequency_tolerance_percent < 0:
            raise ValueError("패턴 주파수 검증 허용오차는 음수일 수 없습니다.")


@dataclass(slots=True, frozen=True)
class MtfMeasurementResult:
    status: str
    message: str
    detected_frequency_lpmm: float
    reference_frequency_lpmm: float
    pattern_frequency_error_percent: float
    pattern_validation_status: str
    cycles_per_pixel: float
    pixels_per_line_pair: float
    bright_level: float
    dark_level: float
    raw_contrast: float
    ctf_at_detected_frequency_percent: float
    mtf_at_detected_frequency_percent: float
    mtf_at_reference_frequency_percent: float | None
    target_mtf_percent: float
    frequency_range_lpmm: tuple[float, float]
    peak_confidence: float


@dataclass(slots=True, frozen=True)
class MtfCurveEvaluationResult:
    status: str
    message: str
    reference_frequency_lpmm: float
    target_mtf_percent: float
    mtf_at_reference_frequency_percent: float | None
    lower_frequency_lpmm: float | None
    lower_mtf_percent: float | None
    upper_frequency_lpmm: float | None
    upper_mtf_percent: float | None
    frequency_range_lpmm: tuple[float, float]
    interpolation_method: str


def dominant_frequency(
    profile: NDArray[np.generic],
    *,
    minimum_cycles: float = 1.5,
) -> tuple[float, float]:
    """1D 프로파일의 대표 공간주파수(cycles/pixel)와 신뢰도를 반환한다."""
    values = np.asarray(profile, dtype=np.float64)
    if values.ndim != 1 or values.size < 8:
        raise ValueError("주파수 검출에는 8개 이상의 1D 프로파일이 필요합니다.")
    centered = values - values.mean()
    if float(centered.std()) <= np.finfo(np.float64).eps:
        raise ValueError("프로파일 대비가 없어 주파수를 검출할 수 없습니다.")

    spectrum = np.abs(np.fft.rfft(centered * np.hanning(values.size)))
    frequencies = np.fft.rfftfreq(values.size, d=1.0)
    valid = frequencies >= minimum_cycles / values.size
    if not np.any(valid):
        raise ValueError("검출 가능한 주파수 구간이 없습니다.")
    valid_indices = np.flatnonzero(valid)
    peak_index = int(valid_indices[np.argmax(spectrum[valid])])
    peak_value = float(spectrum[peak_index])
    if peak_value <= 0:
        raise ValueError("유효한 주파수 피크가 없습니다.")

    interpolated_index = float(peak_index)
    if 0 < peak_index < spectrum.size - 1:
        left, center, right = spectrum[peak_index - 1 : peak_index + 2]
        denominator = left - 2.0 * center + right
        if abs(denominator) > np.finfo(np.float64).eps:
            interpolated_index += float(0.5 * (left - right) / denominator)
    frequency = interpolated_index / values.size
    spectral_sum = float(spectrum[valid].sum())
    confidence = peak_value / spectral_sum if spectral_sum > 0 else 0.0
    return frequency, confidence


def calculate_ctf(profile: NDArray[np.generic]) -> tuple[float, float, float, float]:
    """원본 프로파일 상·하위 20% 평균으로 CTF를 계산한다."""
    values = np.asarray(profile, dtype=np.float64)
    if values.ndim != 1 or values.size < 8:
        raise ValueError("CTF 계산에는 8개 이상의 1D 프로파일이 필요합니다.")
    sorted_values = np.sort(values)
    count = max(1, int(math.ceil(values.size * 0.2)))
    dark = float(sorted_values[:count].mean())
    bright = float(sorted_values[-count:].mean())
    denominator = bright + dark
    if denominator <= 0 or bright <= dark:
        raise ValueError("Bright/Dark 대비가 올바르지 않습니다.")
    contrast = (bright - dark) / denominator
    ctf = contrast * 100.0
    if not 0 <= ctf <= 100:
        raise ValueError("계산된 CTF가 0~100% 범위를 벗어났습니다.")
    return bright, dark, contrast, ctf


def ctf_to_mtf_percent(ctf_percent: float) -> float:
    """단일 CTF에 1차 Coltman 근사를 적용한다."""
    if not 0 <= ctf_percent <= 100:
        raise ValueError("CTF는 0~100% 범위여야 합니다.")
    return min(100.0, math.pi / 4.0 * ctf_percent)


def convert_sensor_frequency_to_object_lpmm(
    sensor_frequency_cycles_per_pixel: float | NDArray[np.generic],
    pixel_pitch_um: float,
    magnification: float,
) -> float | NDArray[np.float64]:
    """sensor cycles/pixel을 object-side lp/mm로 환산한다."""
    if pixel_pitch_um <= 0 or magnification <= 0:
        raise ValueError("Pixel pitch와 렌즈 배율은 0보다 커야 합니다.")
    frequency = np.asarray(sensor_frequency_cycles_per_pixel, dtype=np.float64)
    if np.any(frequency < 0):
        raise ValueError("공간주파수는 음수일 수 없습니다.")
    converted = frequency * 1000.0 / pixel_pitch_um * magnification
    return float(converted) if converted.ndim == 0 else converted


def validate_frequency_range(curve: MtfCurve, reference_lpmm: float) -> bool:
    """기준 주파수가 측정점 범위 안에 있는지 반환한다."""
    minimum, maximum = curve.frequency_range_lpmm
    return minimum <= reference_lpmm <= maximum


def interpolate_mtf_at_frequency(
    curve: MtfCurve,
    reference_lpmm: float,
    *,
    method: str = "log_frequency_linear",
) -> float:
    """측정 범위 안에서만 보간하며 외삽은 거부한다."""
    if not validate_frequency_range(curve, reference_lpmm):
        minimum, maximum = curve.frequency_range_lpmm
        raise MtfOutOfRangeError(
            f"기준 주파수 {reference_lpmm:.3f} lp/mm가 측정 범위 "
            f"{minimum:.3f}~{maximum:.3f} lp/mm 밖에 있습니다."
        )
    if curve.frequency_lpmm.size == 1:
        return float(curve.mtf_percent[0])
    if method == "log_frequency_linear":
        return float(
            np.interp(
                math.log(reference_lpmm),
                np.log(curve.frequency_lpmm),
                curve.mtf_percent,
            )
        )
    if method == "linear_frequency":
        return float(
            np.interp(reference_lpmm, curve.frequency_lpmm, curve.mtf_percent)
        )
    raise ValueError(f"지원하지 않는 MTF 보간 방식입니다: {method}")


def evaluate_mtf(measured_mtf_percent: float, target_mtf_percent: float) -> str:
    """기준 주파수 MTF만으로 PASS/FAIL을 판정한다."""
    if not 0 <= measured_mtf_percent <= 100:
        raise ValueError("측정 MTF는 0~100% 범위여야 합니다.")
    if not 0 <= target_mtf_percent <= 100:
        raise ValueError("목표 MTF는 0~100% 범위여야 합니다.")
    return "PASS" if measured_mtf_percent >= target_mtf_percent else "FAIL"


def build_mtf_curve_from_points(
    frequencies_lpmm: NDArray[np.generic] | list[float],
    mtf_percent: NDArray[np.generic] | list[float],
    *,
    source_method: str = "USAF_MULTI_POINT",
) -> MtfCurve:
    """여러 USAF/Bar 측정점을 정렬하여 MTF 곡선을 구성한다."""
    frequencies = np.asarray(frequencies_lpmm, dtype=np.float64)
    values = np.asarray(mtf_percent, dtype=np.float64)
    if frequencies.size != values.size or frequencies.size == 0:
        raise ValueError("MTF 측정점의 주파수와 값 개수가 일치해야 합니다.")
    order = np.argsort(frequencies)
    frequencies = frequencies[order]
    values = values[order]
    if np.any(np.diff(frequencies) <= 0):
        raise ValueError("MTF 곡선에 중복된 USAF 주파수가 있습니다.")
    return MtfCurve(frequencies, values, source_method)


def evaluate_mtf_curve_at_reference(
    curve: MtfCurve,
    reference_frequency_lpmm: float,
    target_mtf_percent: float,
    *,
    interpolation_method: str = "log_frequency_linear",
) -> MtfCurveEvaluationResult:
    """기준 주파수를 둘러싼 측정점으로 보간하고 MTF를 판정한다."""
    minimum, maximum = curve.frequency_range_lpmm
    if not validate_frequency_range(curve, reference_frequency_lpmm):
        return MtfCurveEvaluationResult(
            status="OUT_OF_RANGE",
            message="기준 주파수의 하한·상한 측정점이 모두 존재하지 않습니다.",
            reference_frequency_lpmm=reference_frequency_lpmm,
            target_mtf_percent=target_mtf_percent,
            mtf_at_reference_frequency_percent=None,
            lower_frequency_lpmm=None,
            lower_mtf_percent=None,
            upper_frequency_lpmm=None,
            upper_mtf_percent=None,
            frequency_range_lpmm=(minimum, maximum),
            interpolation_method=interpolation_method,
        )

    frequencies = curve.frequency_lpmm
    exact = np.flatnonzero(np.isclose(frequencies, reference_frequency_lpmm))
    if exact.size:
        lower_index = upper_index = int(exact[0])
    else:
        upper_index = int(np.searchsorted(frequencies, reference_frequency_lpmm))
        lower_index = upper_index - 1
    measured = interpolate_mtf_at_frequency(
        curve,
        reference_frequency_lpmm,
        method=interpolation_method,
    )
    status = evaluate_mtf(measured, target_mtf_percent)
    return MtfCurveEvaluationResult(
        status=status,
        message=(
            "목표 MTF를 만족합니다."
            if status == "PASS"
            else "목표 MTF에 미달합니다."
        ),
        reference_frequency_lpmm=reference_frequency_lpmm,
        target_mtf_percent=target_mtf_percent,
        mtf_at_reference_frequency_percent=measured,
        lower_frequency_lpmm=float(frequencies[lower_index]),
        lower_mtf_percent=float(curve.mtf_percent[lower_index]),
        upper_frequency_lpmm=float(frequencies[upper_index]),
        upper_mtf_percent=float(curve.mtf_percent[upper_index]),
        frequency_range_lpmm=(minimum, maximum),
        interpolation_method=interpolation_method,
    )


def calculate_mtf_curve(
    profile: NDArray[np.generic],
    pixel_pitch_um: float,
    magnification: float,
    *,
    contrast_profile: NDArray[np.generic] | None = None,
) -> tuple[MtfCurve, float, float, float, float, float]:
    """단일 Bar ROI에서 한 점짜리 MTF 곡선과 CTF 관련 값을 생성한다."""
    cycles_per_pixel, confidence = dominant_frequency(profile)
    source = profile if contrast_profile is None else contrast_profile
    bright, dark, contrast, ctf = calculate_ctf(source)
    mtf = ctf_to_mtf_percent(ctf)
    detected_lpmm = convert_sensor_frequency_to_object_lpmm(
        cycles_per_pixel,
        pixel_pitch_um,
        magnification,
    )
    curve = MtfCurve(
        np.array([detected_lpmm], dtype=np.float64),
        np.array([mtf], dtype=np.float64),
        "BAR_SINGLE",
    )
    return curve, cycles_per_pixel, confidence, bright, dark, contrast


def measure_bar_target_mtf(
    profile: NDArray[np.generic],
    settings: MtfMeasurementSettings,
    *,
    contrast_profile: NDArray[np.generic] | None = None,
) -> MtfMeasurementResult:
    """단일 Bar 측정점이 기준 주파수와 일치할 때만 MTF를 판정한다."""
    settings.validate()
    curve, cycles_per_pixel, confidence, bright, dark, contrast = (
        calculate_mtf_curve(
            profile,
            settings.pixel_pitch_um,
            settings.magnification,
            contrast_profile=contrast_profile,
        )
    )
    detected = curve.frequency_range_lpmm[0]
    error = (
        (detected - settings.reference_frequency_lpmm)
        / settings.reference_frequency_lpmm
        * 100.0
    )
    pattern_match = abs(error) <= settings.pattern_frequency_tolerance_percent
    pattern_status = "MATCH" if pattern_match else "MISMATCH"
    ctf = contrast * 100.0
    mtf_detected = float(curve.mtf_percent[0])

    # 단일 Bar 측정점은 허용오차 내에서 기준 주파수용 패턴으로 확인된 경우에만
    # 그 측정값을 기준 주파수 MTF로 인정한다. 다른 주파수로 외삽하지 않는다.
    if pattern_match:
        mtf_at_reference = mtf_detected
        status = evaluate_mtf(
            mtf_at_reference,
            settings.target_mtf_percent,
        )
        message = (
            "목표 MTF를 만족합니다."
            if status == "PASS"
            else "목표 MTF에 미달합니다."
        )
    else:
        mtf_at_reference = None
        status = "OUT_OF_RANGE"
        message = (
            "단일 Bar 검출 주파수가 기준 주파수와 일치하지 않아 "
            "MTF를 외삽하지 않았습니다."
        )

    return MtfMeasurementResult(
        status=status,
        message=message,
        detected_frequency_lpmm=detected,
        reference_frequency_lpmm=settings.reference_frequency_lpmm,
        pattern_frequency_error_percent=error,
        pattern_validation_status=pattern_status,
        cycles_per_pixel=cycles_per_pixel,
        pixels_per_line_pair=1.0 / cycles_per_pixel,
        bright_level=bright,
        dark_level=dark,
        raw_contrast=contrast,
        ctf_at_detected_frequency_percent=ctf,
        mtf_at_detected_frequency_percent=mtf_detected,
        mtf_at_reference_frequency_percent=mtf_at_reference,
        target_mtf_percent=settings.target_mtf_percent,
        frequency_range_lpmm=curve.frequency_range_lpmm,
        peak_confidence=confidence,
    )
