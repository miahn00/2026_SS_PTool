"""Checkerboard-based geometric distortion measurements.

The low-level ``measure_grid_distortion`` API is retained for callers that
already know the ideal/measured point correspondence.  ``analyze_checkerboard``
is the application-facing API: it detects a usable (possibly partial)
checkerboard, separates planar pose from radial distortion, and reports a
single signed SMIA-style TV distortion summary.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares


@dataclass(slots=True, frozen=True)
class DistortionMeasurementResult:
    aligned_ideal_points: NDArray[np.float64]
    displacement_vectors: NDArray[np.float64]
    distortion_percent: NDArray[np.float64]
    mean_absolute_percent: float
    maximum_absolute_percent: float


@dataclass(slots=True, frozen=True)
class CheckerboardDistortionResult:
    status: str
    message: str
    smia_tv_distortion_percent: float | None
    distortion_type: str
    model_name: str
    pattern_columns: int
    pattern_rows: int
    detected_points: NDArray[np.float64]
    fitted_points: NDArray[np.float64]
    rejected_points: NDArray[np.float64]
    rms_residual_pixels: float | None
    rotation_degrees: float | None
    distortion_center: tuple[float, float] | None
    k1: float | None
    k2: float | None

    @property
    def valid_point_count(self) -> int:
        return int(self.detected_points.shape[0])


def _empty_result(message: str) -> CheckerboardDistortionResult:
    empty = np.empty((0, 2), dtype=np.float64)
    return CheckerboardDistortionResult(
        status="INVALID",
        message=message,
        smia_tv_distortion_percent=None,
        distortion_type="-",
        model_name="5th order radial with decentering",
        pattern_columns=0,
        pattern_rows=0,
        detected_points=empty,
        fitted_points=empty,
        rejected_points=empty,
        rms_residual_pixels=None,
        rotation_degrees=None,
        distortion_center=None,
        k1=None,
        k2=None,
    )


def _to_gray_u8(image: NDArray[np.generic]) -> NDArray[np.uint8]:
    array = np.asarray(image)
    if array.ndim == 3:
        if array.shape[2] == 4:
            array = cv2.cvtColor(array, cv2.COLOR_RGBA2GRAY)
        else:
            array = cv2.cvtColor(array[..., :3], cv2.COLOR_RGB2GRAY)
    if array.ndim != 2:
        raise ValueError("왜곡 분석 영상은 2차원 또는 RGB 영상이어야 합니다.")
    finite = np.asarray(array, dtype=np.float64)
    low, high = np.percentile(finite, (0.5, 99.5))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        raise ValueError("명암 범위가 없어 체커보드를 검출할 수 없습니다.")
    scaled = np.clip((finite - low) * (255.0 / (high - low)), 0, 255)
    return scaled.astype(np.uint8)


def _candidate_pattern_sizes(width: int, height: int) -> list[tuple[int, int]]:
    """Return likely inner-corner sizes without assuming a fixed chart."""
    image_ratio = width / max(height, 1)
    candidates = [
        (columns, rows)
        for rows in range(5, 15)
        for columns in range(5, 21)
        if columns * rows >= 30
    ]
    # Full-board detection is expensive.  Aspect agreement is a strong prior,
    # while both landscape and portrait charts are retained.
    ordered = sorted(
        candidates,
        key=lambda size: (
            abs(np.log((size[0] / size[1]) / image_ratio)),
            -size[0] * size[1],
        ),
    )
    common_partial_sizes = [
        (12, 8),
        (15, 11),
        (14, 10),
        (16, 12),
        (13, 9),
        (15, 9),
        (17, 11),
        (11, 8),
        (13, 10),
    ]
    return common_partial_sizes + [
        size for size in ordered if size not in common_partial_sizes
    ]


def detect_partial_checkerboard(
    image: NDArray[np.generic],
) -> tuple[tuple[int, int], NDArray[np.float64]]:
    """Detect the largest reliable rectangular subset of a checkerboard.

    Missing central markings are tolerated when a clean rectangular subset
    remains elsewhere.  Pattern dimensions refer to inner intersections.
    """
    gray = _to_gray_u8(image)
    height, width = gray.shape
    flags_fast = (
        cv2.CALIB_CB_ADAPTIVE_THRESH
        | cv2.CALIB_CB_NORMALIZE_IMAGE
        | cv2.CALIB_CB_FAST_CHECK
    )
    flags_sb = (
        cv2.CALIB_CB_EXHAUSTIVE
        | cv2.CALIB_CB_ACCURACY
        | cv2.CALIB_CB_NORMALIZE_IMAGE
    )
    candidates = _candidate_pattern_sizes(width, height)
    best: tuple[tuple[int, int], NDArray[np.float64]] | None = None

    # Probe the most likely sizes with both methods first.  This avoids a slow
    # exhaustive pass for the common case while retaining a full fallback.
    priority_count = min(12, len(candidates))
    for size in candidates[:priority_count]:
        found, corners = cv2.findChessboardCorners(gray, size, flags_fast)
        if not found or corners is None:
            found, corners = cv2.findChessboardCornersSB(gray, size, flags_sb)
        if found and corners is not None:
            return size, corners.reshape(-1, 2).astype(np.float64)

    # The classic detector cheaply narrows remaining complete-grid cases.
    for size in candidates[priority_count:]:
        found, corners = cv2.findChessboardCorners(gray, size, flags_fast)
        if not found or corners is None:
            continue
        points = corners.reshape(-1, 2).astype(np.float64)
        if best is None or len(points) > len(best[1]):
            best = (size, points)
        if len(points) >= 60:
            break

    if best is None:
        # SB is more tolerant of blur, perspective and local occlusion.  Try
        # the most plausible dimensions first, then the full set if needed.
        for size in candidates[priority_count:]:
            found, corners = cv2.findChessboardCornersSB(gray, size, flags_sb)
            if not found or corners is None:
                continue
            points = corners.reshape(-1, 2).astype(np.float64)
            best = (size, points)
            # 60+ intersections provide enough spatial support for a stable
            # fifth-order fit; retain responsiveness once such a subset exists.
            if len(points) >= 60:
                break

    if best is None:
        raise ValueError(
            "체커보드의 연속된 부분 격자를 찾지 못했습니다. "
            "차트 외곽을 포함하고 반사·흐림을 줄여 주세요."
        )

    size, points = best
    refined = cv2.cornerSubPix(
        gray,
        points.astype(np.float32).reshape(-1, 1, 2),
        (5, 5),
        (-1, -1),
        (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
            40,
            0.01,
        ),
    )
    return size, refined.reshape(-1, 2).astype(np.float64)


def _project_lattice(
    lattice: NDArray[np.float64],
    parameters: NDArray[np.float64],
    radius_scale: float,
) -> NDArray[np.float64]:
    h = np.array(
        [
            [parameters[0], parameters[1], parameters[2]],
            [parameters[3], parameters[4], parameters[5]],
            [parameters[6], parameters[7], 1.0],
        ],
        dtype=np.float64,
    )
    undistorted = cv2.perspectiveTransform(
        lattice.reshape(1, -1, 2), h
    ).reshape(-1, 2)
    center = parameters[10:12]
    delta = (undistorted - center) / radius_scale
    radius2 = np.sum(delta * delta, axis=1)
    scale = 1.0 + parameters[8] * radius2 + parameters[9] * radius2**2
    return center + (undistorted - center) * scale[:, None]


def _fit_checkerboard_model(
    points: NDArray[np.float64],
    size: tuple[int, int],
    image_shape: tuple[int, int],
) -> tuple[NDArray[np.float64], NDArray[np.bool_], float]:
    columns, rows = size
    lattice = np.array(
        [(column, row) for row in range(rows) for column in range(columns)],
        dtype=np.float64,
    )
    homography, _ = cv2.findHomography(lattice, points, method=0)
    if homography is None:
        raise ValueError("체커보드 평면 자세를 계산할 수 없습니다.")
    homography /= homography[2, 2]
    height, width = image_shape
    initial = np.r_[
        homography[0, :],
        homography[1, :],
        homography[2, :2],
        0.0,
        0.0,
        width / 2.0,
        height / 2.0,
    ]
    radius_scale = 0.5 * np.hypot(width, height)

    def residual(parameters: NDArray[np.float64]) -> NDArray[np.float64]:
        return (_project_lattice(lattice, parameters, radius_scale) - points).ravel()

    lower = np.full(initial.shape, -np.inf)
    upper = np.full(initial.shape, np.inf)
    lower[8:10] = -2.0
    upper[8:10] = 2.0
    lower[10:12] = (0.1 * width, 0.1 * height)
    upper[10:12] = (0.9 * width, 0.9 * height)
    first = least_squares(
        residual,
        initial,
        bounds=(lower, upper),
        loss="soft_l1",
        f_scale=1.0,
        max_nfev=1500,
    )
    point_error = np.linalg.norm(
        residual(first.x).reshape(-1, 2), axis=1
    )
    median = float(np.median(point_error))
    threshold = max(1.25, 3.0 * median)
    inliers = point_error <= threshold
    if np.count_nonzero(inliers) < max(20, int(0.65 * len(points))):
        raise ValueError("유효 격자점이 부족하여 왜곡 모델을 안정적으로 계산할 수 없습니다.")

    def inlier_residual(parameters: NDArray[np.float64]) -> NDArray[np.float64]:
        predicted = _project_lattice(lattice, parameters, radius_scale)
        return (predicted[inliers] - points[inliers]).ravel()

    final = least_squares(
        inlier_residual,
        first.x,
        bounds=(lower, upper),
        loss="soft_l1",
        f_scale=0.5,
        max_nfev=2000,
    )
    rms = float(np.sqrt(np.mean(inlier_residual(final.x) ** 2)))
    return final.x, inliers, rms


def analyze_checkerboard(
    image: NDArray[np.generic],
) -> CheckerboardDistortionResult:
    """Analyze a checkerboard and return a signed SMIA-style TV summary."""
    try:
        gray = _to_gray_u8(image)
        size, points = detect_partial_checkerboard(gray)
        parameters, inliers, rms = _fit_checkerboard_model(
            points, size, gray.shape
        )
    except ValueError as exc:
        return _empty_result(str(exc))

    columns, rows = size
    lattice = np.array(
        [(column, row) for row in range(rows) for column in range(columns)],
        dtype=np.float64,
    )
    radius_scale = 0.5 * np.hypot(gray.shape[1], gray.shape[0])
    fitted = _project_lattice(lattice, parameters, radius_scale)

    # SMIA TV distortion is evaluated near the image boundary.  The fitted
    # radial scale at 98% image height is used so central occlusion does not
    # become the reported metric.  No extrapolation is allowed unless detected
    # inliers actually support that radius.
    center = parameters[10:12]
    supported_radius = np.max(
        np.linalg.norm(points[inliers] - center, axis=1)
    )
    evaluation_radius = 0.98 * gray.shape[0] / 2.0
    if supported_radius < 0.85 * evaluation_radius:
        return CheckerboardDistortionResult(
            status="INVALID",
            message="영상 외곽의 격자점이 부족하여 SMIA TV 왜곡을 외삽하지 않습니다.",
            smia_tv_distortion_percent=None,
            distortion_type="-",
            model_name="5th order radial with decentering",
            pattern_columns=columns,
            pattern_rows=rows,
            detected_points=points[inliers],
            fitted_points=fitted[inliers],
            rejected_points=points[~inliers],
            rms_residual_pixels=rms,
            rotation_degrees=None,
            distortion_center=(float(center[0]), float(center[1])),
            k1=float(parameters[8]),
            k2=float(parameters[9]),
        )

    normalized_radius = evaluation_radius / radius_scale
    radial_scale = (
        1.0
        + parameters[8] * normalized_radius**2
        + parameters[9] * normalized_radius**4
    )
    smia = float((radial_scale - 1.0) * 100.0)
    if smia < -0.005:
        distortion_type = "Barrel"
    elif smia > 0.005:
        distortion_type = "Pincushion"
    else:
        distortion_type = "Near zero"

    first_row = fitted[:columns]
    direction = first_row[-1] - first_row[0]
    rotation = float(np.degrees(np.arctan2(direction[1], direction[0])))
    status = "VALID" if rms <= 1.5 else "INVALID"
    message = (
        "SMIA TV 왜곡 계산이 완료되었습니다."
        if status == "VALID"
        else "모델 잔차가 커 결과 신뢰 조건을 만족하지 못했습니다."
    )
    return CheckerboardDistortionResult(
        status=status,
        message=message,
        smia_tv_distortion_percent=smia if status == "VALID" else None,
        distortion_type=distortion_type if status == "VALID" else "-",
        model_name="5th order radial with decentering",
        pattern_columns=columns,
        pattern_rows=rows,
        detected_points=points[inliers],
        fitted_points=fitted[inliers],
        rejected_points=points[~inliers],
        rms_residual_pixels=rms,
        rotation_degrees=rotation,
        distortion_center=(float(center[0]), float(center[1])),
        k1=float(parameters[8]),
        k2=float(parameters[9]),
    )


def measure_grid_distortion(
    ideal_points: NDArray[np.generic],
    measured_points: NDArray[np.generic],
    image_center: tuple[float, float],
) -> DistortionMeasurementResult:
    """Calculate displacement after projectively aligning known grid points."""
    ideal = np.asarray(ideal_points, dtype=np.float64).reshape(-1, 2)
    measured = np.asarray(measured_points, dtype=np.float64).reshape(-1, 2)
    if ideal.shape != measured.shape or ideal.shape[0] < 4:
        raise ValueError("동일 개수의 이상/측정 점이 최소 4개 필요합니다.")
    homography, mask = cv2.findHomography(ideal, measured, method=0)
    if homography is None or mask is None:
        raise ValueError("차트 정렬 변환을 계산할 수 없습니다.")
    aligned = cv2.perspectiveTransform(
        ideal.reshape(1, -1, 2).astype(np.float64),
        homography,
    ).reshape(-1, 2)
    vectors = measured - aligned
    center = np.asarray(image_center, dtype=np.float64)
    reference_radius = np.linalg.norm(aligned - center, axis=1)
    radial_unit = np.divide(
        aligned - center,
        reference_radius[:, None],
        out=np.zeros_like(aligned),
        where=reference_radius[:, None] > 0,
    )
    radial_displacement = np.sum(vectors * radial_unit, axis=1)
    distortion = np.divide(
        radial_displacement * 100.0,
        reference_radius,
        out=np.zeros_like(radial_displacement),
        where=reference_radius > 0,
    )
    absolute = np.abs(distortion)
    return DistortionMeasurementResult(
        aligned_ideal_points=aligned,
        displacement_vectors=vectors,
        distortion_percent=distortion,
        mean_absolute_percent=float(absolute.mean()),
        maximum_absolute_percent=float(absolute.max()),
    )
