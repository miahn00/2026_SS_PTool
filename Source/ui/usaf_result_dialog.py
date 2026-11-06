"""USAF ROI images and measured brightness profiles in a tabbed result popup."""

from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QScrollArea, QTabWidget, QVBoxLayout, QWidget

from imaging.roi import RoiData
from inspection import BatchMtfResult, RoiMtfResult, extract_roi

KOREAN_FONT = FontProperties(family="Malgun Gothic")


class _ScrollableCanvas(FigureCanvasQTAgg):
    def wheelEvent(self, event) -> None:  # noqa: N802 - Qt API
        event.ignore()


class UsafResultDialog(QDialog):
    """Show every USAF ROI without recalculating its measurement profile."""

    def __init__(
        self,
        image: np.ndarray,
        rois: list[RoiData],
        result: BatchMtfResult,
        *,
        analyzed_at: str,
        source_filename: str,
        reference_frequency_lpmm: float,
        target_mtf_percent: float,
        black_level: int | None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("USAF 측정 결과")
        self.resize(1120, 800)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)
        self.tabs = QTabWidget()
        roi_by_number = {roi.number: roi for roi in rois}

        for item in result.roi_results:
            figure = Figure(figsize=(10.5, 6.6))
            canvas = _ScrollableCanvas(figure)
            canvas.setMinimumSize(920, 590)
            self._render_detail(
                figure,
                image,
                roi_by_number.get(item.roi_number),
                item,
                analyzed_at=analyzed_at,
                source_filename=source_filename,
                reference_frequency_lpmm=reference_frequency_lpmm,
                target_mtf_percent=target_mtf_percent,
                black_level=black_level,
            )
            canvas.draw()
            scroll = QScrollArea()
            scroll.setWidgetResizable(False)
            scroll.setWidget(canvas)
            tab = QWidget()
            layout = QVBoxLayout(tab)
            layout.addWidget(NavigationToolbar2QT(canvas, self))
            layout.addWidget(scroll)
            self.tabs.addTab(tab, f"ROI {item.roi_number} {item.status}")

        root = QVBoxLayout(self)
        root.addWidget(self.tabs)

        # The archive stores one page containing every ROI, while the UI uses tabs.
        self.figure = Figure(figsize=(11, max(5.5, 2.7 * len(result.roi_results))))
        self._render_summary(
            self.figure,
            result,
            analyzed_at=analyzed_at,
            source_filename=source_filename,
            reference_frequency_lpmm=reference_frequency_lpmm,
            target_mtf_percent=target_mtf_percent,
            black_level=black_level,
        )

    @staticmethod
    def _status_color(status: str) -> str:
        return {
            "PASS": "#2e7d32",
            "FAIL": "#c62828",
            "INVALID": "#ef6c00",
            "OUT_OF_RANGE": "#ef6c00",
            "INACTIVE": "#757575",
        }.get(status, "#455a64")

    @classmethod
    def _plot_profile(cls, axes, item: RoiMtfResult) -> None:
        profile = item.raw_profile
        if profile is None or not profile.size:
            axes.text(
                0.5,
                0.5,
                f"밝기 프로파일 없음\n{item.message}",
                ha="center",
                va="center",
                transform=axes.transAxes,
                color=cls._status_color(item.status),
                fontproperties=KOREAN_FONT,
            )
            axes.set_xticks([])
            axes.set_yticks([])
            return
        pixels = np.arange(profile.size)
        axes.plot(pixels, profile, color="#1565c0", linewidth=1.4, label="Average profile")
        if item.measurement is not None:
            axes.axhline(
                item.measurement.bright_level,
                color="#d32f2f",
                linestyle="--",
                linewidth=1.0,
                label=f"Bright {item.measurement.bright_level:.1f}",
            )
            axes.axhline(
                item.measurement.dark_level,
                color="#616161",
                linestyle=":",
                linewidth=1.0,
                label=f"Dark {item.measurement.dark_level:.1f}",
            )
        axes.set_xlabel("ROI pixel position")
        axes.set_ylabel("Brightness level")
        axes.grid(True, alpha=0.25)
        axes.legend(loc="best", fontsize=8)

    @classmethod
    def _render_detail(
        cls,
        figure: Figure,
        image: np.ndarray,
        roi: RoiData | None,
        item: RoiMtfResult,
        *,
        analyzed_at: str,
        source_filename: str,
        reference_frequency_lpmm: float,
        target_mtf_percent: float,
        black_level: int | None,
    ) -> None:
        preview_axes = figure.add_subplot(121)
        profile_axes = figure.add_subplot(122)
        if roi is not None:
            roi_image = extract_roi(image, roi)
            preview_axes.imshow(
                roi_image if roi_image.ndim == 2 else roi_image[..., :3],
                cmap="gray",
            )
        preview_axes.set_title(
            f"ROI {item.roi_number} 측정 영상", fontproperties=KOREAN_FONT
        )
        preview_axes.set_axis_off()
        cls._plot_profile(profile_axes, item)
        profile_axes.set_title(
            f"ROI {item.roi_number} 밝기 프로파일", fontproperties=KOREAN_FONT
        )

        measurement = item.measurement
        values = (
            f"Bright: {measurement.bright_level:.2f}    "
            f"Dark: {measurement.dark_level:.2f}    "
            f"CTF: {measurement.ctf_at_detected_frequency_percent:.2f}%    "
            f"MTF: {measurement.mtf_at_detected_frequency_percent:.2f}%"
            if measurement is not None else "Bright: -    Dark: -    CTF: -    MTF: -"
        )
        header = (
            f"USAF ROI {item.roi_number} 측정 결과\n"
            f"분석 일시: {analyzed_at}    파일명: {source_filename}\n"
            f"방향: {item.detected_direction}    입력: {image.dtype.name}    "
            f"Black Level: {black_level if black_level is not None else 'OFF'}\n"
            f"평가 주파수: {reference_frequency_lpmm:.3f} lp/mm    "
            f"목표 MTF: {target_mtf_percent:.2f}%    판정: {item.status}\n"
            f"{values}\n결과: {item.message}"
        )
        figure.text(
            0.04,
            0.97,
            header,
            va="top",
            fontsize=10,
            color=cls._status_color(item.status),
            fontproperties=KOREAN_FONT,
        )
        figure.subplots_adjust(left=0.07, right=0.97, top=0.69, bottom=0.11, wspace=0.24)

    @classmethod
    def _render_summary(
        cls,
        figure: Figure,
        result: BatchMtfResult,
        *,
        analyzed_at: str,
        source_filename: str,
        reference_frequency_lpmm: float,
        target_mtf_percent: float,
        black_level: int | None,
    ) -> None:
        items = result.roi_results
        for index, item in enumerate(items, start=1):
            axes = figure.add_subplot(max(1, len(items)), 1, index)
            cls._plot_profile(axes, item)
            measurement = item.measurement
            value = (
                f"CTF {measurement.ctf_at_detected_frequency_percent:.2f}% / "
                f"MTF {measurement.mtf_at_detected_frequency_percent:.2f}%"
                if measurement is not None else item.message
            )
            axes.set_title(
                f"ROI {item.roi_number} · {item.status} · {item.detected_direction} · {value}",
                color=cls._status_color(item.status),
                fontsize=9,
                fontproperties=KOREAN_FONT,
            )
        header = (
            "USAF 전체 ROI 밝기 프로파일\n"
            f"분석 일시: {analyzed_at}    파일명: {source_filename}\n"
            f"평가 주파수: {reference_frequency_lpmm:.3f} lp/mm    "
            f"목표 MTF: {target_mtf_percent:.2f}%    "
            f"Black Level: {black_level if black_level is not None else 'OFF'}    "
            f"전체 판정: {result.overall_status}"
        )
        figure.text(
            0.04,
            0.99,
            header,
            va="top",
            fontsize=10,
            fontproperties=KOREAN_FONT,
        )
        figure.subplots_adjust(
            left=0.09,
            right=0.97,
            top=max(0.72, 0.91 - 0.02 * len(items)),
            bottom=0.06,
            hspace=0.75,
        )
