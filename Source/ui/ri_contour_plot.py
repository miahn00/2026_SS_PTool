"""RI 등고선 Figure 렌더링."""

from __future__ import annotations

import numpy as np
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from scipy.ndimage import gaussian_filter

from inspection import RiEvaluationResult, RiGridMeasurementResult


def render_ri_contour(
    figure: Figure,
    result: RiGridMeasurementResult,
    *,
    evaluation: RiEvaluationResult | None = None,
    analyzed_at: str = "",
    source_filename: str = "",
) -> None:
    """실측 Grid 통계는 바꾸지 않고 표시용 등고선만 평활한다."""
    heatmap = np.asarray(
        [cell.relative_percent for cell in result.cells],
        dtype=float,
    ).reshape(result.rows, result.columns)
    display_map = gaussian_filter(heatmap, sigma=0.8, mode="nearest")

    figure.clear()
    axes = figure.add_subplot(111)
    korean_font = FontProperties(family="Malgun Gothic")
    header = (
        "RI 분석 결과\n"
        f"분석 일시: {analyzed_at or '-'}    "
        f"파일명: {source_filename or '-'}\n"
        f"Center: {result.center_mean:.2f} (100%)    "
        f"평균 RI: {result.average_percent:.2f}%    "
        f"최소 RI: {result.minimum_percent:.2f}% "
        f"(R{result.minimum_position[0] + 1}, "
        f"C{result.minimum_position[1] + 1})    "
        f"포화: {result.saturation_percent:.2f}%"
    )
    corners = ", ".join(f"{value:.2f}%" for value in result.corner_percent)
    footer = (
        f"코너 RI (좌상/우상/좌하/우하): {corners}    "
        f"좌우 비대칭: {result.left_right_asymmetry_percent:.2f}%    "
        f"상하 비대칭: {result.top_bottom_asymmetry_percent:.2f}%"
    )
    if evaluation is not None:
        criterion = (
            f"{evaluation.minimum_required_percent:.2f}%"
            if evaluation.minimum_required_percent > 0
            else "미설정"
        )
        header += (
            "\n"
            f"판정 방식: 측정 최소 RI ≥ 설정 기준    "
            f"측정 최소: {evaluation.minimum_ri_percent:.2f}%    "
            f"판정 기준: {criterion}    "
            f"결과: {evaluation.status}"
        )
        footer += f"\n판정 사유: {evaluation.message}"
    figure.text(
        0.04,
        0.965,
        header,
        va="top",
        fontsize=10,
        fontproperties=korean_font,
    )
    figure.text(
        0.04,
        0.035,
        footer,
        va="bottom",
        fontsize=10,
        fontproperties=korean_font,
    )
    x = np.arange(1, result.columns + 1)
    y = np.arange(1, result.rows + 1)
    minimum = float(display_map.min())
    maximum = float(display_map.max())
    if np.isclose(minimum, maximum):
        levels = np.linspace(minimum - 0.5, maximum + 0.5, 8)
        line_levels = [maximum]
    else:
        levels = np.linspace(minimum, maximum, 16)
        line_levels = np.linspace(minimum, maximum, 7)

    filled = axes.contourf(x, y, display_map, levels=levels, cmap="turbo")
    lines = axes.contour(
        x,
        y,
        display_map,
        levels=line_levels,
        colors="black",
        linewidths=0.55,
    )
    axes.clabel(lines, inline=True, fontsize=8, fmt="%.1f%%")
    axes.scatter(
        [result.center_column + 1],
        [result.center_row + 1],
        marker="s",
        facecolors="none",
        edgecolors="red",
        linewidths=1.3,
        label="Center",
    )
    axes.invert_yaxis()
    axes.set_aspect("equal")
    axes.set_title("Relative Illumination Contours (Center = 100%)")
    axes.set_xlabel("Grid column")
    axes.set_ylabel("Grid row")
    axes.legend(loc="lower right", fontsize=8)
    colorbar = figure.colorbar(filled, ax=axes)
    colorbar.set_label("RI (%)")
    figure.subplots_adjust(left=0.08, right=0.88, top=0.72, bottom=0.16)
