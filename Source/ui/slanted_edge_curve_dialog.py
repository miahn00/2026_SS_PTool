"""Combined 1 lp/mm Slanted Edge MTF curve popup."""

from __future__ import annotations

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import MultipleLocator
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QScrollArea, QVBoxLayout, QWidget

from inspection import BatchSlantedEdgeResult, sample_mtf_curve_at_1_lpmm


class _ScrollableCanvas(FigureCanvasQTAgg):
    def wheelEvent(self, event) -> None:  # noqa: N802 - Qt API
        event.ignore()


class SlantedEdgeCurveDialog(QDialog):
    def __init__(
        self,
        result: BatchSlantedEdgeResult,
        *,
        reference_frequency_lpmm: float,
        target_mtf_percent: float,
        analyzed_at: str,
        source_filename: str,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Slanted Edge 전체 MTF 곡선")
        self.resize(1100, 780)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)

        self.figure = Figure(figsize=(11, 8.5))
        self.canvas = _ScrollableCanvas(self.figure)
        self.canvas.setMinimumSize(950, 650)
        self._render(
            result,
            reference_frequency_lpmm,
            target_mtf_percent,
            analyzed_at,
            source_filename,
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
        result: BatchSlantedEdgeResult,
        reference_frequency_lpmm: float,
        target_mtf_percent: float,
        analyzed_at: str,
        source_filename: str,
    ) -> None:
        self.figure.clear()
        axes = self.figure.add_subplot(111)
        colors = ("#1565c0", "#2e7d32", "#ef6c00", "#8e24aa")
        maximum_frequency = 0.0
        curve_count = 0
        for index, item in enumerate(result.roi_results):
            curve = item.result.curve
            if curve is None:
                continue
            frequency, mtf = sample_mtf_curve_at_1_lpmm(curve)
            color = colors[index % len(colors)]
            axes.plot(
                curve.frequency_lpmm,
                curve.mtf_percent,
                color=color,
                linewidth=1.2,
                alpha=0.55,
            )
            axes.plot(
                frequency,
                mtf,
                color=color,
                marker="o",
                markersize=3.2,
                linewidth=1.5,
                label=f"{item.roi_name} (1 lp/mm)",
            )
            maximum_frequency = max(maximum_frequency, curve.frequency_range_lpmm[1])
            curve_count += 1

        axes.axvline(
            reference_frequency_lpmm,
            color="#d32f2f",
            linestyle="--",
            linewidth=1.2,
            label=f"평가 주파수 {reference_frequency_lpmm:.2f} lp/mm",
        )
        axes.axhline(
            target_mtf_percent,
            color="#616161",
            linestyle=":",
            linewidth=1.2,
            label=f"목표 MTF {target_mtf_percent:.2f}%",
        )
        axes.set_xlim(0, max(1.0, maximum_frequency))
        axes.set_ylim(0, 105)
        axes.set_xlabel("Object-side spatial frequency (lp/mm)")
        axes.set_ylabel("MTF (%)")
        axes.set_title("Slanted Edge MTF Curves (1 lp/mm interpolated points)")
        axes.xaxis.set_minor_locator(MultipleLocator(1.0))
        axes.grid(True, which="major", alpha=0.25)
        axes.grid(True, which="minor", axis="x", alpha=0.10)
        axes.legend(loc="best", fontsize=8)

        header = (
            "Slanted Edge 전체 MTF 분석\n"
            f"분석 일시: {analyzed_at}    파일명: {source_filename}\n"
            f"평가 주파수: {reference_frequency_lpmm:.2f} lp/mm    "
            f"목표 MTF: {target_mtf_percent:.2f}%    "
            f"전체 판정: {result.overall_status}\n"
            f"표시 곡선: {curve_count}개    "
            "원형 마커: 측정 범위 안의 1 lp/mm 선형 보간값"
        )
        self.figure.text(
            0.04,
            0.97,
            header,
            va="top",
            fontsize=10,
            fontproperties=FontProperties(family="Malgun Gothic"),
        )
        self.figure.subplots_adjust(left=0.09, right=0.97, top=0.76, bottom=0.10)
