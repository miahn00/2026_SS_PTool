"""상대조도(Relative Illumination) 계산 함수."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray


@dataclass(slots=True, frozen=True)
class RiMeasurementResult:
    center_mean: float
    region_means: tuple[float, ...]
    relative_percent: tuple[float, ...]
    minimum_percent: float
    maximum_asymmetry_percent: float


@dataclass(slots=True, frozen=True)
class RiGridCellResult:
    row: int
    column: int
    region_type: str
    roi: tuple[int, int, int, int]
    mean_level: float
    relative_percent: float


@dataclass(slots=True, frozen=True)
class RiGridMeasurementResult:
    rows: int
    columns: int
    center_row: int
    center_column: int
    center_mean: float
    cells: tuple[RiGridCellResult, ...]
    average_percent: float
    minimum_percent: float
    minimum_position: tuple[int, int]
    corner_percent: tuple[float, float, float, float]
    left_average_percent: float
    right_average_percent: float
    top_average_percent: float
    bottom_average_percent: float
    left_right_asymmetry_percent: float
    top_bottom_asymmetry_percent: float
    saturation_percent: float


@dataclass(slots=True, frozen=True)
class RiEvaluationResult:
    status: str
    minimum_ri_percent: float
    minimum_required_percent: float
    message: str


def evaluate_minimum_ri(
    result: RiGridMeasurementResult,
    minimum_required_percent: float,
) -> RiEvaluationResult:
    """실측 최소 RI를 사용자가 설정한 단일 하한 기준과 비교한다."""
    if not 0 <= minimum_required_percent <= 100:
        raise ValueError("최소 RI 판정 기준은 0~100% 범위여야 합니다.")
    if minimum_required_percent == 0:
        return RiEvaluationResult(
            "UNSET",
            result.minimum_percent,
            minimum_required_percent,
            "최소 RI 판정 기준이 설정되지 않았습니다.",
        )
    status = (
        "PASS"
        if result.minimum_percent >= minimum_required_percent
        else "FAIL"
    )
    message = (
        "측정 최소 RI가 판정 기준을 만족합니다."
        if status == "PASS"
        else "측정 최소 RI가 판정 기준에 미달합니다."
    )
    return RiEvaluationResult(
        status,
        result.minimum_percent,
        minimum_required_percent,
        message,
    )


def _gray(image: NDArray[np.generic]) -> NDArray[np.float64]:
    if image.ndim == 2:
        return image.astype(np.float64)
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float64)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY).astype(np.float64)
    raise ValueError(f"지원하지 않는 RI 영상 형태입니다: {image.shape}")


def generate_ri_grid_rois(
    image_width: int,
    image_height: int,
    *,
    rows: int = 5,
    columns: int = 5,
    inner_fraction: float = 0.5,
) -> tuple[tuple[int, int, int, int], ...]:
    """영상 전체를 분할하고 각 cell 중앙의 측정 ROI를 행 우선으로 반환한다."""
    if image_width <= 0 or image_height <= 0:
        raise ValueError("RI 영상 크기는 0보다 커야 합니다.")
    if rows < 3 or columns < 3 or rows % 2 == 0 or columns % 2 == 0:
        raise ValueError("RI Grid 행과 열은 3 이상의 홀수여야 합니다.")
    if not 0 < inner_fraction <= 1:
        raise ValueError("RI cell 내부 측정 비율은 0 초과 1 이하여야 합니다.")
    if image_width < columns or image_height < rows:
        raise ValueError("영상 크기가 RI Grid보다 작습니다.")

    x_edges = np.rint(np.linspace(0, image_width, columns + 1)).astype(int)
    y_edges = np.rint(np.linspace(0, image_height, rows + 1)).astype(int)
    regions: list[tuple[int, int, int, int]] = []
    for row in range(rows):
        for column in range(columns):
            cell_x0, cell_x1 = int(x_edges[column]), int(x_edges[column + 1])
            cell_y0, cell_y1 = int(y_edges[row]), int(y_edges[row + 1])
            cell_width = cell_x1 - cell_x0
            cell_height = cell_y1 - cell_y0
            roi_width = max(1, int(round(cell_width * inner_fraction)))
            roi_height = max(1, int(round(cell_height * inner_fraction)))
            x = cell_x0 + (cell_width - roi_width) // 2
            y = cell_y0 + (cell_height - roi_height) // 2
            regions.append((x, y, roi_width, roi_height))
    return tuple(regions)


def _grid_region_type(
    row: int,
    column: int,
    rows: int,
    columns: int,
) -> str:
    center = (rows // 2, columns // 2)
    if (row, column) == center:
        return "CENTER"
    if row in {0, rows - 1} and column in {0, columns - 1}:
        return "CORNER"
    if row in {0, rows - 1} or column in {0, columns - 1}:
        return "EDGE"
    return "OTHER"


def measure_grid_relative_illumination(
    image: NDArray[np.generic],
    *,
    rows: int = 5,
    columns: int = 5,
    inner_fraction: float = 0.5,
) -> RiGridMeasurementResult:
    """자동 Grid에서 중앙을 100%로 정규화한 상대조도를 계산한다."""
    gray = _gray(image)
    height, width = gray.shape
    if not np.all(np.isfinite(gray)):
        raise ValueError("RI 영상에 NaN 또는 무한대 값이 있습니다.")
    rois = generate_ri_grid_rois(
        width,
        height,
        rows=rows,
        columns=columns,
        inner_fraction=inner_fraction,
    )
    means = np.asarray(
        [
            float(gray[y:y + roi_height, x:x + roi_width].mean())
            for x, y, roi_width, roi_height in rois
        ],
        dtype=np.float64,
    ).reshape(rows, columns)
    center_row, center_column = rows // 2, columns // 2
    center_mean = float(means[center_row, center_column])
    if center_mean <= np.finfo(float).eps:
        raise ValueError("RI 중앙 Grid 평균은 0보다 커야 합니다.")
    relative = means / center_mean * 100.0

    cells = tuple(
        RiGridCellResult(
            row=row,
            column=column,
            region_type=_grid_region_type(row, column, rows, columns),
            roi=rois[row * columns + column],
            mean_level=float(means[row, column]),
            relative_percent=float(relative[row, column]),
        )
        for row in range(rows)
        for column in range(columns)
    )
    minimum_index = np.unravel_index(int(np.argmin(relative)), relative.shape)
    corners = (
        float(relative[0, 0]),
        float(relative[0, -1]),
        float(relative[-1, 0]),
        float(relative[-1, -1]),
    )
    left_right = np.abs(relative - np.fliplr(relative))
    top_bottom = np.abs(relative - np.flipud(relative))

    if np.issubdtype(image.dtype, np.integer):
        limits = np.iinfo(image.dtype)
        saturated = (image <= limits.min) | (image >= limits.max)
        if image.ndim == 3:
            saturated = np.any(saturated, axis=2)
        saturation = float(np.mean(saturated) * 100.0)
    else:
        saturation = 0.0

    return RiGridMeasurementResult(
        rows=rows,
        columns=columns,
        center_row=center_row,
        center_column=center_column,
        center_mean=center_mean,
        cells=cells,
        average_percent=float(relative.mean()),
        minimum_percent=float(relative[minimum_index]),
        minimum_position=(int(minimum_index[0]), int(minimum_index[1])),
        corner_percent=corners,
        left_average_percent=float(relative[:, 0].mean()),
        right_average_percent=float(relative[:, -1].mean()),
        top_average_percent=float(relative[0, :].mean()),
        bottom_average_percent=float(relative[-1, :].mean()),
        left_right_asymmetry_percent=float(left_right.max()),
        top_bottom_asymmetry_percent=float(top_bottom.max()),
        saturation_percent=saturation,
    )


def measure_relative_illumination(
    image: NDArray[np.generic],
    center_roi: tuple[int, int, int, int],
    measurement_rois: list[tuple[int, int, int, int]],
) -> RiMeasurementResult:
    """중앙 ROI 평균을 100%로 정규화한 측정 ROI들의 상대조도를 계산한다."""
    gray = _gray(image)
    height, width = gray.shape

    def region_mean(region: tuple[int, int, int, int]) -> float:
        x, y, region_width, region_height = region
        if (
            x < 0
            or y < 0
            or region_width <= 0
            or region_height <= 0
            or x + region_width > width
            or y + region_height > height
        ):
            raise ValueError("RI ROI가 영상 경계를 벗어납니다.")
        return float(gray[y : y + region_height, x : x + region_width].mean())

    center_mean = region_mean(center_roi)
    if center_mean <= 0:
        raise ValueError("중앙 ROI 평균은 0보다 커야 합니다.")
    means = tuple(region_mean(region) for region in measurement_rois)
    relative = tuple(value / center_mean * 100.0 for value in means)
    minimum = min(relative) if relative else 100.0
    asymmetry = max(relative) - min(relative) if len(relative) >= 2 else 0.0
    return RiMeasurementResult(center_mean, means, relative, minimum, asymmetry)
