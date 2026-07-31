"""Scrollable and saveable checkerboard distortion result popup."""

from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QScrollArea, QVBoxLayout, QWidget

from inspection import CheckerboardDistortionResult


class _ScrollableCanvas(FigureCanvasQTAgg):
    def wheelEvent(self, event) -> None:  # noqa: N802
        event.ignore()


class DistortionResultDialog(QDialog):
    def __init__(
        self,
        image: np.ndarray,
        result: CheckerboardDistortionResult,
        *,
        analyzed_at: str,
        source_filename: str,
        limit_percent: float,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Distortion 분석 결과")
        self.resize(1050, 780)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)
        self.figure = Figure(figsize=(11, 8.5))
        self.canvas = _ScrollableCanvas(self.figure)
        self.canvas.setMinimumSize(900, 650)
        self._render(
            image, result, analyzed_at, source_filename, limit_percent
        )
        self.canvas.draw()

        scroll = QScrollArea()
        scroll.setWidgetResizable(False)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setWidget(self.canvas)
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.addWidget(NavigationToolbar2QT(self.canvas, self))
        layout.addWidget(scroll)
        self.setLayout(layout)

    def _render(
        self,
        image: np.ndarray,
        result: CheckerboardDistortionResult,
        analyzed_at: str,
        source_filename: str,
        limit_percent: float,
    ) -> None:
        self.figure.clear()
        axes = self.figure.add_subplot(111)
        axes.imshow(image if image.ndim == 2 else image[..., :3], cmap="gray")
        if result.fitted_points.size:
            axes.scatter(
                result.fitted_points[:, 0], result.fitted_points[:, 1],
                s=14, facecolors="none", edgecolors="#00b0ff",
                linewidths=0.8, label="Fitted grid",
            )
        if result.detected_points.size:
            axes.scatter(
                result.detected_points[:, 0], result.detected_points[:, 1],
                s=10, color="#ff00cc", label="Detected points",
            )
        if result.rejected_points.size:
            axes.scatter(
                result.rejected_points[:, 0], result.rejected_points[:, 1],
                marker="x", s=20, color="#ff3d00", label="Rejected",
            )
        if result.distortion_center is not None:
            axes.scatter(
                [result.distortion_center[0]], [result.distortion_center[1]],
                marker="x", s=70, linewidths=2, color="red",
                label="Distortion center",
            )
        axes.set_axis_off()
        axes.legend(loc="upper right", fontsize=8)
        value = (
            f"{result.smia_tv_distortion_percent:+.3f}%"
            if result.smia_tv_distortion_percent is not None else "-"
        )
        rms = (
            f"{result.rms_residual_pixels:.3f} px"
            if result.rms_residual_pixels is not None else "-"
        )
        rotation = (
            f"{result.rotation_degrees:+.2f}°"
            if result.rotation_degrees is not None else "-"
        )
        judgment = (
            "INVALID"
            if result.smia_tv_distortion_percent is None
            else "PASS"
            if abs(result.smia_tv_distortion_percent) <= limit_percent
            else "FAIL"
        )
        header = (
            "Checkerboard Distortion 분석\n"
            f"분석 일시: {analyzed_at}    파일명: {source_filename}\n"
            f"SMIA TV Distortion: {value}    Type: {result.distortion_type}    "
            f"판정: {judgment}\n"
            f"판정 기준: |SMIA TV Distortion| ≤ {limit_percent:.2f}%\n"
            f"부분 격자: {result.pattern_columns} × {result.pattern_rows} points    "
            f"유효점: {result.valid_point_count}    회전/원근 보정: 적용 ({rotation})\n"
            f"계산 모델: {result.model_name}    RMS 잔차: {rms}\n"
            f"결과: {result.message}"
        )
        self.figure.text(
            0.04, 0.97, header, va="top", fontsize=10,
            fontproperties=FontProperties(family="Malgun Gothic"),
        )
        self.figure.subplots_adjust(left=0.03, right=0.97, top=0.78, bottom=0.03)
