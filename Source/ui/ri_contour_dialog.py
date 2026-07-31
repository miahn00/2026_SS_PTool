"""확대 가능한 RI 등고선 팝업."""

from __future__ import annotations

from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg,
    NavigationToolbar2QT,
)
from matplotlib.figure import Figure
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QScrollArea, QVBoxLayout, QWidget

from inspection import RiEvaluationResult, RiGridMeasurementResult
from ui.ri_contour_plot import render_ri_contour


class _ScrollableFigureCanvas(FigureCanvasQTAgg):
    def wheelEvent(self, event) -> None:  # noqa: N802 - Qt API
        event.ignore()


class RiContourDialog(QDialog):
    def __init__(
        self,
        result: RiGridMeasurementResult,
        *,
        evaluation: RiEvaluationResult,
        analyzed_at: str,
        source_filename: str,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("RI 변환 이미지 - Relative Illumination Contours")
        self.resize(1050, 760)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)

        self.figure = Figure(figsize=(11, 8.5))
        self.canvas = _ScrollableFigureCanvas(self.figure)
        self.canvas.setMinimumSize(900, 620)
        render_ri_contour(
            self.figure,
            result,
            evaluation=evaluation,
            analyzed_at=analyzed_at,
            source_filename=source_filename,
        )
        self.canvas.draw()

        scroll = QScrollArea()
        scroll.setWidgetResizable(False)
        scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setWidget(self.canvas)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.addWidget(NavigationToolbar2QT(self.canvas, self))
        layout.addWidget(scroll)
        self.setLayout(layout)
