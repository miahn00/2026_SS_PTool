"""선택 ROI의 전처리 결과와 프로파일/FFT 표시."""

from __future__ import annotations

import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QLabel,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from inspection import (
    BatchMtfResult,
    BatchSlantedEdgeResult,
    CheckerboardDistortionResult,
    MtfMeasurementResult,
    RiEvaluationResult,
    RiGridMeasurementResult,
    RoiAnalysisResult,
    SlantedEdgeResult,
)


class AnalysisPanel(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.result_label = QLabel("전체 ROI 검사를 실행하면 결과가 표시됩니다.")
        self.mtf_result_label = QLabel("MTF 측정 결과 없음")
        self.mtf_result_label.setStyleSheet(
            "QLabel { font-size: 13px; font-weight: bold; padding: 6px; "
            "border: 1px solid #666; }"
        )
        self.profile_plot = pg.PlotWidget(title="ROI 평균 프로파일")
        self.profile_plot.addLegend()
        self.fft_plot = pg.PlotWidget(title="FFT 진단 (MTF 곡선 아님)")
        self.profile_plot.setLabel("bottom", "Pixel")
        self.profile_plot.setLabel("left", "Raw level")
        self.fft_plot.setLabel("bottom", "cycles/pixel")
        self.fft_plot.setLabel("left", "Normalized magnitude")
        self.profile_plot.setMinimumHeight(90)
        self.fft_plot.setMinimumHeight(90)
        self.overall_label = QLabel("전체 판정: -")
        self.overall_label.setStyleSheet(
            "QLabel { font-size: 15px; font-weight: bold; padding: 6px; }"
        )
        self.result_table = QTableWidget(0, 9)
        self.result_table.setHorizontalHeaderLabels(
            [
                "ROI",
                "USAF 공칭 lp/mm",
                "검출 패턴 lp/mm",
                "CTF @ 검출 %",
                "MTF @ 검출 %",
                "패턴 오차 %",
                "패턴 검증",
                "판정 포함",
                "측정점 상태",
            ]
        )
        self.result_table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.result_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.result_table.verticalHeader().setDefaultSectionSize(24)
        self.result_table.horizontalHeader().setFixedHeight(26)
        self.result_table.horizontalHeader().setStretchLastSection(True)
        table_height = 26 + (24 * 4) + (self.result_table.frameWidth() * 2) + 6
        self.result_table.setFixedHeight(table_height)

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.addWidget(self.overall_label)
        content_layout.addWidget(self.result_table)
        content_layout.addWidget(self.result_label)
        content_layout.addWidget(self.mtf_result_label)
        content_layout.addWidget(self.profile_plot)
        content_layout.addWidget(self.fft_plot)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self.scroll_area.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self.scroll_area.setWidget(content)
        outer_layout = QVBoxLayout()
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.addWidget(self.scroll_area)
        self.setLayout(outer_layout)

    def show_result(self, result: RoiAnalysisResult) -> None:
        self.profile_plot.setVisible(True)
        self.fft_plot.setVisible(True)
        self.profile_plot.setTitle("ROI 평균 프로파일")
        self.profile_plot.setLabel("bottom", "Pixel")
        self.profile_plot.setLabel("left", "Raw level")
        self.fft_plot.setTitle("FFT 진단 (MTF 곡선 아님)")
        self.profile_plot.clear()
        self.fft_plot.clear()
        self.result_label.setText(
            f"상태: {result.status} | {result.message} | "
            f"방향: {result.detected_direction} "
            f"(신뢰도 {result.direction_confidence:.2f}) | "
            f"Mean: {result.mean:.2f} | Std: {result.standard_deviation:.2f} | "
            f"포화: {result.saturation_percent:.2f}%"
        )
        if result.raw_profile.size:
            self.profile_plot.plot(
                result.raw_profile,
                pen="#40c4ff",
                name="원본 (CTF)",
            )
        if result.profile.size:
            self.profile_plot.plot(
                result.profile,
                pen="#ff4081",
                name="전처리 (FFT)",
            )
        if result.fft_frequency.size:
            self.fft_plot.plot(
                result.fft_frequency,
                result.fft_magnitude,
                pen="#ffd740",
            )

    def show_mtf_result(self, result: MtfMeasurementResult) -> None:
        color = {
            "PASS": "#2e7d32",
            "FAIL": "#c62828",
            "INVALID": "#ef6c00",
            "OUT_OF_RANGE": "#ef6c00",
        }.get(result.status, "#555555")
        self.mtf_result_label.setStyleSheet(
            f"QLabel {{ font-size: 13px; font-weight: bold; padding: 6px; "
            f"border: 2px solid {color}; }}"
        )
        self.mtf_result_label.setText(
            f"상태: {result.status} | "
            f"검출 패턴: {result.detected_frequency_lpmm:.3f} lp/mm | "
            f"기준: {result.reference_frequency_lpmm:.3f} lp/mm | "
            f"패턴 검증: {result.pattern_validation_status} "
            f"({result.pattern_frequency_error_percent:+.2f}%)\n"
            f"CTF @ 검출: {result.ctf_at_detected_frequency_percent:.2f}% | "
            f"MTF @ 검출: {result.mtf_at_detected_frequency_percent:.2f}% | "
            f"MTF @ 기준: "
            f"{f'{result.mtf_at_reference_frequency_percent:.2f}%' if result.mtf_at_reference_frequency_percent is not None else 'OUT OF RANGE'} | "
            f"목표: {result.target_mtf_percent:.2f}% | "
            f"주기: {result.pixels_per_line_pair:.3f} pixel/LP | "
            f"Bright: {result.bright_level:.2f} | Dark: {result.dark_level:.2f} | "
            f"피크 신뢰도: {result.peak_confidence:.3f} | {result.message}"
        )

    def show_mtf_error(self, message: str) -> None:
        self.mtf_result_label.setStyleSheet(
            "QLabel { font-size: 13px; font-weight: bold; padding: 6px; "
            "border: 2px solid #ef6c00; }"
        )
        self.mtf_result_label.setText(f"판정: INVALID | {message}")

    def show_distortion_result(
        self,
        result: CheckerboardDistortionResult,
        limit_percent: float,
    ) -> None:
        self.profile_plot.setVisible(False)
        self.fft_plot.setVisible(False)
        self.result_table.setVisible(False)
        measured = result.smia_tv_distortion_percent
        judgment = (
            "INVALID"
            if measured is None
            else "PASS"
            if abs(measured) <= limit_percent
            else "FAIL"
        )
        color = {
            "PASS": "#2e7d32",
            "FAIL": "#c62828",
            "INVALID": "#ef6c00",
        }[judgment]
        self.overall_label.setStyleSheet(
            f"QLabel {{ font-size: 15px; font-weight: bold; padding: 8px; "
            f"border: 2px solid {color}; }}"
        )
        value = (
            f"{result.smia_tv_distortion_percent:+.3f}%"
            if result.smia_tv_distortion_percent is not None
            else "-"
        )
        self.overall_label.setText(
            f"Distortion 판정: {judgment} | SMIA TV Distortion {value} | "
            f"{result.distortion_type} | 기준: |왜곡률| ≤ {limit_percent:.2f}%"
        )
        rms = (
            f"{result.rms_residual_pixels:.3f} px"
            if result.rms_residual_pixels is not None
            else "-"
        )
        self.result_label.setText(
            f"검출 부분 격자: {result.pattern_columns} × "
            f"{result.pattern_rows} | 유효점: {result.valid_point_count} | "
            f"RMS 잔차: {rms} | {result.message}"
        )
        self.mtf_result_label.setText(
            f"계산 모델: {result.model_name} | "
            "회전·원근 및 왜곡 중심 자동 보정"
        )

    def show_batch_result(
        self,
        result: BatchMtfResult,
        reference_frequency_lpmm: float,
        target_mtf_percent: float,
    ) -> None:
        self.profile_plot.setVisible(True)
        self.fft_plot.setVisible(True)
        self.result_table.setColumnCount(10)
        self.result_table.setHorizontalHeaderLabels(
            [
                "ROI",
                "평가 주파수 lp/mm",
                "검출 패턴 lp/mm",
                "CTF @ 검출 %",
                "MTF @ 검출 %",
                "패턴 오차 %",
                "패턴 검증",
                "판정 포함",
                "측정점 상태",
                "사유",
            ]
        )
        colors = {
            "PASS": QColor("#2e7d32"),
            "FAIL": QColor("#c62828"),
            "INVALID": QColor("#ef6c00"),
            "OUT_OF_RANGE": QColor("#ef6c00"),
            "INACTIVE": QColor("#757575"),
            "MEASURED": QColor("#1565c0"),
        }
        overall_color = colors[result.overall_status].name()
        self.overall_label.setStyleSheet(
            f"QLabel {{ font-size: 15px; font-weight: bold; padding: 6px; "
            f"border: 2px solid {overall_color}; }}"
        )
        self.overall_label.setText(
            f"전체 판정: {result.overall_status} | {result.message}"
        )
        self.mtf_result_label.setText(
            f"평가 주파수: {reference_frequency_lpmm:.2f} lp/mm | "
            f"목표 MTF: {target_mtf_percent:.2f}% | "
            "판정: 패턴 MATCH 후 MTF @ 검출값을 목표와 비교"
        )
        self.result_table.setRowCount(len(result.roi_results))
        for row, roi_result in enumerate(result.roi_results):
            measurement = roi_result.measurement
            values = [
                roi_result.roi_name,
                f"{roi_result.chart_frequency_lpmm:.3f}"
                if roi_result.chart_frequency_lpmm is not None else "-",
                f"{measurement.detected_frequency_lpmm:.3f}" if measurement else "-",
                f"{measurement.ctf_at_detected_frequency_percent:.2f}"
                if measurement else "-",
                f"{measurement.mtf_at_detected_frequency_percent:.2f}"
                if measurement else "-",
                f"{measurement.pattern_frequency_error_percent:+.2f}"
                if measurement else "-",
                measurement.pattern_validation_status if measurement else "-",
                "Yes" if roi_result.included_in_judgment else "No",
                roi_result.status,
                roi_result.message,
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if column == 8:
                    item.setForeground(colors.get(roi_result.status, QColor("white")))
                item.setToolTip(roi_result.message)
                self.result_table.setItem(row, column, item)
        self.result_table.resizeColumnsToContents()

    def show_slanted_edge_result(self, result: SlantedEdgeResult) -> None:
        colors = {
            "PASS": "#2e7d32",
            "FAIL": "#c62828",
            "INVALID": "#ef6c00",
            "OUT_OF_RANGE": "#ef6c00",
        }
        color = colors.get(result.status, "#555555")
        evaluation = result.evaluation
        measured = (
            f"{evaluation.mtf_at_reference_frequency_percent:.2f}%"
            if evaluation
            and evaluation.mtf_at_reference_frequency_percent is not None
            else "OUT OF RANGE" if evaluation else "-"
        )
        frequency = (
            f"{evaluation.reference_frequency_lpmm:.2f} lp/mm"
            if evaluation
            else "-"
        )
        self.result_label.setText(
            f"Slanted Edge | 방향: {result.edge_orientation} | "
            f"기울기: {result.edge_angle_degrees:.2f}° | "
            f"직선 적합도 R²: {result.edge_fit_r_squared:.3f} | "
            f"대비: {result.contrast_percent:.2f}% | "
            f"측정 품질: {result.quality_grade}"
        )
        self.mtf_result_label.setStyleSheet(
            f"QLabel {{ font-size: 13px; font-weight: bold; padding: 6px; "
            f"border: 2px solid {color}; }}"
        )
        self.mtf_result_label.setText(
            f"상태: {result.status} | MTF @ {frequency}: {measured} | "
            f"{result.message} | {result.quality_message}"
        )
        self.profile_plot.clear()
        self.profile_plot.setVisible(True)
        self.profile_plot.setTitle("Slanted Edge MTF 곡선")
        self.profile_plot.setLabel("bottom", "Object-side frequency", "lp/mm")
        self.profile_plot.setLabel("left", "MTF", "%")
        self.fft_plot.setVisible(False)
        if result.curve is not None:
            self.profile_plot.plot(
                result.curve.frequency_lpmm,
                result.curve.mtf_percent,
                pen="#40c4ff",
            )

    def show_slanted_edge_batch_result(
        self,
        result: BatchSlantedEdgeResult,
        reference_frequency_lpmm: float,
        target_mtf_percent: float,
    ) -> None:
        colors = {
            "PASS": QColor("#2e7d32"),
            "FAIL": QColor("#c62828"),
            "INVALID": QColor("#ef6c00"),
            "OUT_OF_RANGE": QColor("#ef6c00"),
            "INACTIVE": QColor("#757575"),
        }
        color = colors[result.overall_status].name()
        self.overall_label.setStyleSheet(
            f"QLabel {{ font-size: 15px; font-weight: bold; padding: 6px; "
            f"border: 2px solid {color}; }}"
        )
        self.overall_label.setText(
            f"전체 판정: {result.overall_status} | {result.message}"
        )
        headers = [
            "ROI",
            "Edge 방향",
            "Edge 각도 °",
            "R²",
            "측정 범위 (Nyquist) lp/mm",
            "평가/Nyquist %",
            "ESF 충족/Phase",
            "포화 %",
            "측정 품질",
            f"MTF @ {reference_frequency_lpmm:.2f} lp/mm %",
            "목표 MTF %",
            "판정 포함",
            "상태",
            "품질/오류 사유",
        ]
        self.result_table.setColumnCount(len(headers))
        self.result_table.setHorizontalHeaderLabels(headers)
        self.result_table.setRowCount(len(result.roi_results))
        for row, item_result in enumerate(result.roi_results):
            edge = item_result.result
            evaluation = edge.evaluation
            frequency_range = (
                f"0 (정규화)~{edge.nyquist_frequency_lpmm:.2f}"
                if edge.nyquist_frequency_lpmm > 0 else "-"
            )
            nyquist_ratio = (
                f"{edge.reference_to_nyquist_ratio * 100:.1f}"
                if edge.reference_to_nyquist_ratio is not None else "-"
            )
            measured = (
                f"{evaluation.mtf_at_reference_frequency_percent:.2f}"
                if evaluation
                and evaluation.mtf_at_reference_frequency_percent is not None
                else "OUT OF RANGE" if evaluation else "-"
            )
            values = [
                item_result.roi_name,
                edge.edge_orientation,
                f"{edge.edge_angle_degrees:.2f}",
                f"{edge.edge_fit_r_squared:.3f}",
                frequency_range,
                nyquist_ratio,
                (
                    f"{edge.esf_bin_coverage_percent:.1f}%/"
                    f"{edge.subpixel_phase_bins}/4"
                ),
                f"{edge.saturation_percent:.1f}",
                edge.quality_grade,
                measured,
                f"{target_mtf_percent:.2f}",
                "Yes" if item_result.included_in_judgment else "No",
                edge.status,
                edge.quality_message,
            ]
            for column, value in enumerate(values):
                table_item = QTableWidgetItem(value)
                table_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if column == 12:
                    table_item.setForeground(
                        colors.get(edge.status, QColor("white"))
                    )
                table_item.setToolTip(
                    f"{edge.message} | 품질: {edge.quality_message}"
                )
                self.result_table.setItem(row, column, table_item)
        self.result_table.resizeColumnsToContents()
        self.result_table.setColumnWidth(13, min(
            420, max(240, self.result_table.columnWidth(13))
        ))
        self.mtf_result_label.setText(
            f"평가 주파수: {reference_frequency_lpmm:.2f} lp/mm | "
            f"목표 MTF: {target_mtf_percent:.2f}% | Slanted Edge 직접 MTF"
        )
        self.profile_plot.setVisible(False)
        self.fft_plot.setVisible(False)

    def clear_batch_result(self) -> None:
        self.result_table.setVisible(True)
        self.overall_label.setText("전체 판정: -")
        self.overall_label.setStyleSheet(
            "QLabel { font-size: 15px; font-weight: bold; padding: 6px; }"
        )
        self.result_table.setRowCount(0)

    def show_ri_result(
        self,
        result: RiGridMeasurementResult,
        evaluation: RiEvaluationResult,
    ) -> None:
        colors = {
            "PASS": "#2e7d32",
            "FAIL": "#c62828",
            "UNSET": "#1565c0",
        }
        color = colors[evaluation.status]
        self.overall_label.setStyleSheet(
            "QLabel { font-size: 15px; font-weight: bold; padding: 6px; "
            f"border: 2px solid {color}; }}"
        )
        self.overall_label.setText(
            f"RI 판정: {evaluation.status} | {evaluation.message}"
        )
        headers = ["Grid", "영역", "평균 밝기", "RI %", "측정 ROI"]
        self.result_table.setColumnCount(len(headers))
        self.result_table.setHorizontalHeaderLabels(headers)
        cell_map = {(cell.row, cell.column): cell for cell in result.cells}
        positions = [
            (0, 0),
            (0, result.columns // 2),
            (0, result.columns - 1),
            (result.rows // 2, 0),
            (result.center_row, result.center_column),
            (result.rows // 2, result.columns - 1),
            (result.rows - 1, 0),
            (result.rows - 1, result.columns // 2),
            (result.rows - 1, result.columns - 1),
            result.minimum_position,
        ]
        summary_cells = []
        for position in positions:
            if position not in {
                (cell.row, cell.column) for cell in summary_cells
            }:
                summary_cells.append(cell_map[position])

        self.result_table.setRowCount(len(summary_cells))
        for row, cell in enumerate(summary_cells):
            x, y, width, height = cell.roi
            values = [
                f"R{cell.row + 1} C{cell.column + 1}",
                cell.region_type,
                f"{cell.mean_level:.2f}",
                f"{cell.relative_percent:.2f}",
                f"X{x}, Y{y}, {width}×{height}",
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.result_table.setItem(row, column, item)
        self.result_table.resizeColumnsToContents()

        corners = ", ".join(f"{value:.2f}%" for value in result.corner_percent)
        self.result_label.setText(
            f"Center: {result.center_mean:.2f} (100%) | "
            f"평균 RI: {result.average_percent:.2f}% | "
            f"최소 RI: {result.minimum_percent:.2f}% "
            f"(R{result.minimum_position[0] + 1}, "
            f"C{result.minimum_position[1] + 1}) | "
            f"포화: {result.saturation_percent:.2f}%"
        )
        self.mtf_result_label.setStyleSheet(
            "QLabel { font-size: 13px; font-weight: bold; padding: 6px; "
            "border: 1px solid #1565c0; }"
        )
        self.mtf_result_label.setText(
            f"판정 방식: 측정 최소 RI ≥ 설정 기준 | "
            f"측정 최소: {evaluation.minimum_ri_percent:.2f}% | "
            f"기준: "
            f"{f'{evaluation.minimum_required_percent:.2f}%' if evaluation.minimum_required_percent > 0 else '미설정'}\n"
            f"코너 RI (좌상/우상/좌하/우하): {corners} | "
            f"좌우 비대칭: {result.left_right_asymmetry_percent:.2f}% | "
            f"상하 비대칭: {result.top_bottom_asymmetry_percent:.2f}%"
        )

        self.profile_plot.setVisible(False)
        self.fft_plot.setVisible(False)
