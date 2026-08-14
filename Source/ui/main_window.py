"""영상 없이 시작할 수 있는 기본 메인 화면."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, QThread, Qt, Signal, Slot
from PySide6.QtGui import QAction, QColor, QImage, QPixmap
from PySide6.QtWidgets import (
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QComboBox,
    QDoubleSpinBox,
    QDockWidget,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from imaging import (
    ImageFrame,
    ImageLoadError,
    load_image,
    to_display_uint8,
    window_to_uint8,
)
from imaging.roi import RoiData, load_rois, save_rois
from inspection import (
    MtfMeasurementSettings,
    analyze_roi,
    measure_bar_target_mtf,
    measure_rois_mtf,
    measure_rois_slanted_edge,
    measure_slanted_edge,
    measure_grid_relative_illumination,
    evaluate_minimum_ri,
    analyze_checkerboard,
)
from models import OpticalSettings, load_optical_settings, save_optical_settings
from ui.analysis_panel import AnalysisPanel
from ui.image_viewer import ImageViewer
from ui.optical_settings_dialog import OpticalSettingsDialog
from ui.ri_contour_dialog import RiContourDialog
from ui.distortion_result_dialog import DistortionResultDialog
from ui.slanted_edge_curve_dialog import SlantedEdgeCurveDialog


class _DistortionWorker(QObject):
    finished = Signal(object)

    def __init__(self, image: np.ndarray) -> None:
        super().__init__()
        self._image = image

    @Slot()
    def run(self) -> None:
        self.finished.emit(analyze_checkerboard(self._image))


class MainWindow(QMainWindow):
    """파일 입력 기반 광학 성능 측정 프로그램의 초기 화면."""

    def __init__(self, optical_settings_path: str | Path | None = None) -> None:
        super().__init__()
        self._frame: ImageFrame | None = None
        self._pixmap: QPixmap | None = None
        self._updating_window_controls = False
        self._optical_settings_path = (
            Path(optical_settings_path)
            if optical_settings_path is not None
            else Path.cwd() / "optical_settings.json"
        )
        self._optical_settings, settings_notice = self._load_or_create_settings()
        self._ri_contour_dialog: RiContourDialog | None = None
        self._distortion_result_dialog: DistortionResultDialog | None = None
        self._slanted_edge_curve_dialog: SlantedEdgeCurveDialog | None = None
        self._distortion_thread: QThread | None = None
        self._distortion_worker: _DistortionWorker | None = None

        self.setWindowTitle("SS Optical Performance Tool")
        self.resize(1280, 900)
        self._build_menu()
        self._build_content()
        self._build_analysis_dock()
        self.statusBar().showMessage(settings_notice)

    def _load_or_create_settings(self) -> tuple[OpticalSettings, str]:
        if self._optical_settings_path.exists():
            try:
                settings = load_optical_settings(self._optical_settings_path)
                return settings, (
                    f"광학 설정 자동 불러오기 완료: "
                    f"{self._optical_settings_path.name}"
                )
            except ValueError as exc:
                return OpticalSettings(), (
                    f"광학 설정을 읽지 못해 기본값을 사용합니다: {exc}"
                )

        settings = OpticalSettings()
        try:
            save_optical_settings(self._optical_settings_path, settings)
            return settings, (
                f"초기 광학 설정 파일 생성 완료: "
                f"{self._optical_settings_path.name}"
            )
        except (OSError, ValueError) as exc:
            return settings, f"광학 설정 파일을 생성하지 못했습니다: {exc}"

    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("파일")
        open_action = QAction("열기...", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.open_image)
        file_menu.addAction(open_action)

        exit_action = QAction("종료", self)
        exit_action.triggered.connect(self.close)
        file_menu.addSeparator()
        file_menu.addAction(exit_action)

    def _build_content(self) -> None:
        open_button = QPushButton("영상 파일 열기")
        open_button.clicked.connect(self.open_image)

        fit_button = QPushButton("화면 맞춤")
        fit_button.clicked.connect(self._fit_image)
        actual_button = QPushButton("100%")
        actual_button.clicked.connect(self._actual_size)
        auto_button = QPushButton("자동 Contrast")
        auto_button.clicked.connect(self._auto_contrast)
        optical_button = QPushButton("광학 설정")
        optical_button.clicked.connect(self._open_optical_settings)
        self.analyze_all_button = QPushButton("전체 ROI 검사")
        self.analyze_all_button.clicked.connect(self._analyze_all_rois)
        self.measurement_mode_combo = QComboBox()
        self.measurement_mode_combo.addItems(["Slanted Edge", "USAF 차트", "RI"])
        self.measurement_mode_combo.addItem("Distortion")
        self.measurement_mode_combo.currentIndexChanged.connect(
            self._measurement_mode_changed
        )

        toolbar = QHBoxLayout()
        toolbar.addWidget(open_button)
        toolbar.addWidget(fit_button)
        toolbar.addWidget(actual_button)
        toolbar.addWidget(auto_button)
        toolbar.addWidget(optical_button)
        toolbar.addWidget(self.analyze_all_button)
        toolbar.addWidget(QLabel("측정 모드"))
        toolbar.addWidget(self.measurement_mode_combo)
        toolbar.addStretch(1)

        self.viewer = ImageViewer()
        self.viewer.setMinimumSize(640, 480)
        self.viewer.pixelHovered.connect(self._show_pixel)
        self.viewer.zoomChanged.connect(self._show_zoom)
        self.viewer.roisChanged.connect(self._update_roi_count)

        self.info_view = QPlainTextEdit()
        self.info_view.setReadOnly(True)
        self.info_view.setPlaceholderText("영상 정보가 여기에 표시됩니다.")
        self.info_view.setMinimumWidth(280)

        self.minimum_spin = QSpinBox()
        self.maximum_spin = QSpinBox()
        for spin in (self.minimum_spin, self.maximum_spin):
            spin.setRange(0, 65535)
            spin.setEnabled(False)
        self.minimum_spin.valueChanged.connect(self._apply_display_window)
        self.maximum_spin.valueChanged.connect(self._apply_display_window)

        window_form = QFormLayout()
        window_form.addRow("표시 Min", self.minimum_spin)
        window_form.addRow("표시 Max", self.maximum_spin)
        window_frame = QFrame()
        window_frame.setLayout(window_form)

        self.clear_roi_button = QPushButton("전체 ROI 삭제")
        self.clear_roi_button.clicked.connect(self._clear_rois)
        self.save_roi_button = QPushButton("ROI 저장")
        self.save_roi_button.clicked.connect(self._save_rois)
        self.load_roi_button = QPushButton("ROI 불러오기")
        self.load_roi_button.clicked.connect(self._load_rois)

        roi_buttons = QHBoxLayout()
        roi_buttons.addWidget(self.clear_roi_button)
        roi_file_buttons = QHBoxLayout()
        roi_file_buttons.addWidget(self.save_roi_button)
        roi_file_buttons.addWidget(self.load_roi_button)

        self.roi_count_label = QLabel("ROI: 0 / 4")
        self.roi_help_label = QLabel(
            "빈 영상 영역 드래그: ROI 생성\nROI 선택 후 Delete: 개별 삭제"
        )
        self.global_lp_spin = QDoubleSpinBox()
        self.global_lp_spin.setRange(0.001, 100000)
        self.global_lp_spin.setDecimals(3)
        self.global_lp_spin.setValue(
            self._optical_settings.evaluation_frequency_lpmm
        )
        self.global_mtf_spin = QDoubleSpinBox()
        self.global_mtf_spin.setRange(0, 100)
        self.global_mtf_spin.setDecimals(2)
        self.global_mtf_spin.setValue(self._optical_settings.target_mtf_percent)
        self.global_frequency_tolerance_spin = QDoubleSpinBox()
        self.global_frequency_tolerance_spin.setRange(0, 100)
        self.global_frequency_tolerance_spin.setDecimals(2)
        self.global_frequency_tolerance_spin.setValue(
            self._optical_settings.pattern_frequency_tolerance_percent
        )
        self.global_lp_spin.valueChanged.connect(self._apply_global_roi_settings)
        self.global_mtf_spin.valueChanged.connect(self._apply_global_roi_settings)
        self.global_frequency_tolerance_spin.valueChanged.connect(
            self._apply_global_roi_settings
        )
        self.ri_minimum_spin = QDoubleSpinBox()
        self.ri_minimum_spin.setRange(0, 100)
        self.ri_minimum_spin.setDecimals(2)
        self.ri_minimum_spin.setSuffix(" %")
        self.ri_minimum_spin.setSpecialValueText("미설정")
        self.ri_minimum_spin.setValue(self._optical_settings.ri_minimum_percent)
        self.ri_minimum_spin.valueChanged.connect(
            self._apply_global_roi_settings
        )
        self.distortion_limit_spin = QDoubleSpinBox()
        self.distortion_limit_spin.setRange(0, 100)
        self.distortion_limit_spin.setDecimals(2)
        self.distortion_limit_spin.setSuffix(" % 이하")
        self.distortion_limit_spin.setValue(
            self._optical_settings.distortion_limit_percent
        )
        self.distortion_limit_spin.valueChanged.connect(
            self._apply_global_roi_settings
        )
        global_form = QFormLayout()
        global_form.addRow("평가 주파수 lp/mm", self.global_lp_spin)
        global_form.addRow("목표 MTF @ 평가주파수 %", self.global_mtf_spin)
        global_form.addRow(
            "패턴 주파수 검증 허용오차 %",
            self.global_frequency_tolerance_spin,
        )
        global_form.addRow("최소 RI 판정 기준", self.ri_minimum_spin)
        global_form.addRow(
            "SMIA TV Distortion 판정 기준",
            self.distortion_limit_spin,
        )
        self.global_form = global_form
        global_frame = QFrame()
        global_frame.setFrameShape(QFrame.Shape.StyledPanel)
        global_frame.setLayout(global_form)

        right_layout = QVBoxLayout()
        right_layout.addWidget(window_frame)
        right_layout.addWidget(global_frame)
        right_layout.addWidget(self.roi_count_label)
        right_layout.addWidget(self.roi_help_label)
        right_layout.addLayout(roi_buttons)
        right_layout.addLayout(roi_file_buttons)
        right_layout.addWidget(self.info_view, stretch=1)
        right_widget = QWidget()
        right_widget.setLayout(right_layout)

        left_layout = QVBoxLayout()
        left_layout.addLayout(toolbar)
        left_layout.addWidget(self.viewer, stretch=1)
        left_widget = QWidget()
        left_widget.setLayout(left_layout)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 1)

        root_layout = QHBoxLayout()
        root_layout.addWidget(splitter)
        root = QWidget()
        root.setLayout(root_layout)
        self.setCentralWidget(root)
        self._update_measurement_mode_ui()

    def _build_analysis_dock(self) -> None:
        self.analysis_panel = AnalysisPanel()
        dock = QDockWidget("ROI 분석", self)
        dock.setObjectName("roi_analysis_dock")
        dock.setWidget(self.analysis_panel)
        dock.setMinimumHeight(430)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, dock)

    def open_image(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "영상 파일 열기",
            "",
            "영상 파일 (*.tif *.tiff *.jpg *.jpeg *.png);;TIFF (*.tif *.tiff);;JPEG (*.jpg *.jpeg);;PNG (*.png)",
        )
        if not file_path:
            return

        try:
            frame = load_image(file_path)
            display = to_display_uint8(frame.image)
            pixmap = self._make_pixmap(display)
        except (ImageLoadError, ValueError) as exc:
            QMessageBox.critical(self, "영상 열기 오류", str(exc))
            self.statusBar().showMessage("영상 파일을 열지 못했습니다.")
            return

        self._frame = frame
        self._set_window_controls(frame)
        self._set_pixmap(pixmap)
        self._update_info(frame)
        self.statusBar().showMessage(f"열기 완료: {Path(file_path).name}")

    @staticmethod
    def _make_pixmap(image: np.ndarray) -> QPixmap:
        contiguous = np.ascontiguousarray(image)
        if contiguous.ndim == 2:
            height, width = contiguous.shape
            qimage = QImage(
                contiguous.data,
                width,
                height,
                contiguous.strides[0],
                QImage.Format.Format_Grayscale8,
            )
        elif contiguous.ndim == 3 and contiguous.shape[2] == 3:
            height, width, _ = contiguous.shape
            qimage = QImage(
                contiguous.data,
                width,
                height,
                contiguous.strides[0],
                QImage.Format.Format_RGB888,
            )
        elif contiguous.ndim == 3 and contiguous.shape[2] == 4:
            height, width, _ = contiguous.shape
            qimage = QImage(
                contiguous.data,
                width,
                height,
                contiguous.strides[0],
                QImage.Format.Format_RGBA8888,
            )
        else:
            raise ValueError(f"표시할 수 없는 영상 형태입니다: {contiguous.shape}")

        return QPixmap.fromImage(qimage.copy())

    def _set_window_controls(self, frame: ImageFrame) -> None:
        maximum_allowed = 65535 if frame.bit_depth == 16 else 255
        self._updating_window_controls = True
        self.minimum_spin.setRange(0, maximum_allowed)
        self.maximum_spin.setRange(0, maximum_allowed)
        self.minimum_spin.setValue(int(frame.minimum))
        self.maximum_spin.setValue(int(frame.maximum))
        self.minimum_spin.setEnabled(True)
        self.maximum_spin.setEnabled(True)
        self._updating_window_controls = False

    def _set_pixmap(self, pixmap: QPixmap, *, reset_view: bool = True) -> None:
        self._pixmap = pixmap
        if self._frame is None:
            return
        if reset_view:
            self.viewer.set_image(pixmap, self._frame.width, self._frame.height)
        else:
            self.viewer.update_pixmap(pixmap)

    def _apply_display_window(self) -> None:
        if self._updating_window_controls or self._frame is None:
            return
        minimum = self.minimum_spin.value()
        maximum = self.maximum_spin.value()
        if maximum <= minimum:
            return
        display = window_to_uint8(self._frame.image, minimum, maximum)
        self._set_pixmap(self._make_pixmap(display), reset_view=False)

    def _auto_contrast(self) -> None:
        if self._frame is None:
            return
        low, high = np.percentile(self._frame.image, (1.0, 99.0))
        if high <= low:
            low, high = self._frame.minimum, self._frame.maximum
        if high <= low:
            return
        self._updating_window_controls = True
        self.minimum_spin.setValue(int(low))
        self.maximum_spin.setValue(int(high))
        self._updating_window_controls = False
        self._apply_display_window()

    def _fit_image(self) -> None:
        self.viewer.fit_to_window()

    def _actual_size(self) -> None:
        self.viewer.actual_size()

    def _show_zoom(self, zoom: float) -> None:
        self.statusBar().showMessage(f"Zoom: {zoom:.0f}%")

    def _show_pixel(self, x: int, y: int, valid: bool) -> None:
        if not valid or self._frame is None:
            return
        value = self._frame.image[y, x]
        if np.isscalar(value):
            value_text = str(value.item())
        else:
            value_text = ", ".join(str(item) for item in value.tolist())
        self.statusBar().showMessage(
            f"좌표: X={x}, Y={y}  |  원본 값: {value_text}  |  "
            f"Zoom: {self.viewer.zoom_percent:.0f}%"
        )

    def _clear_rois(self) -> None:
        self.viewer.clear_rois()

    def _save_rois(self) -> None:
        if self._frame is None:
            QMessageBox.information(self, "ROI", "먼저 영상 파일을 열어 주세요.")
            return
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "ROI 저장",
            "roi_settings.json",
            "JSON (*.json)",
        )
        if not file_path:
            return
        try:
            save_rois(file_path, self.viewer.roi_data())
        except OSError as exc:
            QMessageBox.critical(self, "ROI 저장 오류", str(exc))
            return
        self.statusBar().showMessage(f"ROI 저장 완료: {Path(file_path).name}")

    def _load_rois(self) -> None:
        if self._frame is None:
            QMessageBox.information(self, "ROI", "먼저 영상 파일을 열어 주세요.")
            return
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "ROI 불러오기",
            "",
            "JSON (*.json)",
        )
        if not file_path:
            return
        try:
            rois = load_rois(file_path, self._frame.width, self._frame.height)
            for roi in rois:
                self._normalize_roi_defaults(roi)
            if rois:
                self.global_lp_spin.setValue(rois[0].reference_frequency_lpmm)
                self.global_mtf_spin.setValue(
                    rois[0].target_mtf_at_reference_percent
                )
            self.viewer.replace_rois(rois)
        except ValueError as exc:
            QMessageBox.critical(self, "ROI 불러오기 오류", str(exc))
            return
        self.statusBar().showMessage(f"ROI 불러오기 완료: {Path(file_path).name}")

    @staticmethod
    def _normalize_roi_defaults(roi: RoiData) -> None:
        roi.name = f"ROI {roi.number}"
        roi.direction = "Auto"
        roi.active = True
        roi.include_in_judgment = True

    def _update_roi_count(self) -> None:
        rois = self.viewer.roi_data()
        for roi in rois:
            self._normalize_roi_defaults(roi)
            roi.reference_frequency_lpmm = self.global_lp_spin.value()
            roi.target_mtf_at_reference_percent = self.global_mtf_spin.value()
            roi.usaf_frequency_lpmm = self.global_lp_spin.value()
            roi.usaf_group = None
            roi.usaf_element = None
        self.roi_count_label.setText(f"ROI: {len(rois)} / 4")
        self.viewer.clear_measurement_results()
        if hasattr(self, "analysis_panel"):
            self.analysis_panel.clear_batch_result()

    def _apply_global_roi_settings(self, *_args) -> None:
        self.viewer.clear_measurement_results()
        if hasattr(self, "analysis_panel"):
            self.analysis_panel.clear_batch_result()
        for roi in self.viewer.roi_data():
            self._normalize_roi_defaults(roi)
            roi.reference_frequency_lpmm = self.global_lp_spin.value()
            roi.target_mtf_at_reference_percent = self.global_mtf_spin.value()
            roi.usaf_frequency_lpmm = self.global_lp_spin.value()
            roi.usaf_group = None
            roi.usaf_element = None
        self._optical_settings.evaluation_frequency_lpmm = (
            self.global_lp_spin.value()
        )
        self._optical_settings.target_mtf_percent = self.global_mtf_spin.value()
        self._optical_settings.pattern_frequency_tolerance_percent = (
            self.global_frequency_tolerance_spin.value()
        )
        self._optical_settings.ri_minimum_percent = self.ri_minimum_spin.value()
        self._optical_settings.distortion_limit_percent = (
            self.distortion_limit_spin.value()
        )
        try:
            save_optical_settings(
                self._optical_settings_path,
                self._optical_settings,
            )
        except (OSError, ValueError) as exc:
            self.statusBar().showMessage(f"공통 평가 설정 자동 저장 실패: {exc}")
        else:
            self.statusBar().showMessage(
                f"공통 평가 설정 자동 저장 완료: "
                f"{self._optical_settings_path.name}"
            )

    def _open_optical_settings(self) -> None:
        dialog = OpticalSettingsDialog(self._optical_settings, self)
        if dialog.exec():
            settings = dialog.settings()
            try:
                save_optical_settings(self._optical_settings_path, settings)
            except (OSError, ValueError) as exc:
                QMessageBox.critical(
                    self,
                    "설정 저장 오류",
                    f"광학 설정을 저장하지 못했습니다.\n{exc}",
                )
                return
            self._optical_settings = settings
            self.statusBar().showMessage(
                f"광학 설정 적용 및 자동 저장 완료: "
                f"{self._optical_settings_path.name}"
            )

    def _analyze_selected_roi(self) -> None:
        if self._frame is None:
            QMessageBox.information(self, "ROI 분석", "먼저 영상 파일을 열어 주세요.")
            return
        item = self.viewer.selected_roi()
        if item is None:
            QMessageBox.information(self, "ROI 분석", "분석할 ROI를 선택해 주세요.")
            return
        if self._is_slanted_edge_mode():
            self._analyze_selected_slanted_edge(item)
            return
        try:
            result = analyze_roi(self._frame.image, item.to_data())
        except ValueError as exc:
            QMessageBox.critical(self, "ROI 분석 오류", str(exc))
            return
        self.analysis_panel.show_result(result)
        if result.status != "READY":
            self.analysis_panel.show_mtf_error(result.message)
            self.statusBar().showMessage(
                f"{item.data.name} 분석: {result.status} - {result.message}"
            )
            return

        measurement_direction = (
            item.data.direction
            if item.data.direction in {"H", "V"}
            else result.detected_direction
        )
        pixel_pitch = (
            self._optical_settings.pixel_pitch_x_um
            if measurement_direction == "V"
            else self._optical_settings.pixel_pitch_y_um
        )
        settings = MtfMeasurementSettings(
            reference_frequency_lpmm=self.global_lp_spin.value(),
            target_mtf_percent=self.global_mtf_spin.value(),
            pixel_pitch_um=pixel_pitch,
            magnification=self._optical_settings.magnification,
            pattern_frequency_tolerance_percent=(
                self.global_frequency_tolerance_spin.value()
            ),
        )
        try:
            mtf_result = measure_bar_target_mtf(
                result.profile,
                settings,
                contrast_profile=result.raw_profile,
            )
        except ValueError as exc:
            self.analysis_panel.show_mtf_error(str(exc))
            self.statusBar().showMessage(f"{item.data.name} 분석: INVALID - {exc}")
            return
        self.analysis_panel.show_mtf_result(mtf_result)
        self.viewer.set_roi_measurement_result(
            item.data.number,
            mtf_result.status,
            (
                f"MTF@{mtf_result.reference_frequency_lpmm:.1f}: "
                f"{mtf_result.mtf_at_reference_frequency_percent:.1f}%"
                if mtf_result.mtf_at_reference_frequency_percent is not None
                else f"MTF@{mtf_result.reference_frequency_lpmm:.1f}: OOR"
            ),
        )
        self.statusBar().showMessage(
            f"{item.data.name}: {mtf_result.status} | "
            f"검출 {mtf_result.detected_frequency_lpmm:.3f} lp/mm | "
            f"MTF@기준 "
            f"{mtf_result.mtf_at_reference_frequency_percent if mtf_result.mtf_at_reference_frequency_percent is not None else 'OUT OF RANGE'}"
        )

    def _analyze_all_rois(self) -> None:
        if self._frame is None:
            QMessageBox.information(self, "전체 ROI 검사", "먼저 영상 파일을 열어 주세요.")
            return
        if self._is_distortion_mode():
            self._analyze_distortion()
            return
        if self._is_ri_mode():
            self._analyze_ri()
            return
        rois = self.viewer.roi_data()
        if not rois:
            QMessageBox.information(self, "전체 ROI 검사", "검사할 ROI가 없습니다.")
            return
        if self._is_slanted_edge_mode():
            self._analyze_all_slanted_edge(rois)
            return

        result = measure_rois_mtf(
            self._frame.image,
            rois,
            reference_frequency_lpmm=self.global_lp_spin.value(),
            target_mtf_percent=self.global_mtf_spin.value(),
            pixel_pitch_x_um=self._optical_settings.pixel_pitch_x_um,
            pixel_pitch_y_um=self._optical_settings.pixel_pitch_y_um,
            magnification=self._optical_settings.magnification,
            pattern_frequency_tolerance_percent=(
                self.global_frequency_tolerance_spin.value()
            ),
        )
        self.analysis_panel.show_batch_result(
            result,
            self.global_lp_spin.value(),
            self.global_mtf_spin.value(),
        )
        self.viewer.clear_measurement_results()
        for roi_result in result.roi_results:
            measurement = roi_result.measurement
            text = (
                f"{roi_result.chart_frequency_lpmm:.2f} lp/mm\n"
                f"MTF {measurement.mtf_at_detected_frequency_percent:.1f}%"
                if measurement
                else roi_result.status
            )
            self.viewer.set_roi_measurement_result(
                roi_result.roi_number,
                roi_result.status,
                text,
            )
        self.statusBar().showMessage(
            f"전체 ROI 검사: {result.overall_status} - {result.message}"
        )

    def _is_slanted_edge_mode(self) -> bool:
        return self.measurement_mode_combo.currentText() == "Slanted Edge"

    def _is_ri_mode(self) -> bool:
        return self.measurement_mode_combo.currentText() == "RI"

    def _is_distortion_mode(self) -> bool:
        return self.measurement_mode_combo.currentText() == "Distortion"

    def _measurement_mode_changed(self, *_args) -> None:
        slanted_edge = self._is_slanted_edge_mode()
        ri_mode = self._is_ri_mode()
        distortion_mode = self._is_distortion_mode()
        self.viewer.clear_measurement_results()
        self.viewer.clear_ri_grid()
        self.viewer.set_roi_drawing_enabled(not (ri_mode or distortion_mode))
        self.analysis_panel.clear_batch_result()
        self._update_measurement_mode_ui()
        if distortion_mode:
            self.statusBar().showMessage("측정 모드: Distortion")
            return
        mode = "RI" if ri_mode else "Slanted Edge" if slanted_edge else "USAF 차트"
        self.statusBar().showMessage(f"측정 모드: {mode}")

    def _update_measurement_mode_ui(self) -> None:
        slanted_edge = self._is_slanted_edge_mode()
        ri_mode = self._is_ri_mode()
        distortion_mode = self._is_distortion_mode()
        mtf_mode = not (ri_mode or distortion_mode)
        self.global_form.setRowVisible(self.global_lp_spin, mtf_mode)
        self.global_form.setRowVisible(self.global_mtf_spin, mtf_mode)
        self.global_form.setRowVisible(
            self.global_frequency_tolerance_spin,
            not slanted_edge and mtf_mode,
        )
        self.global_form.setRowVisible(self.ri_minimum_spin, ri_mode)
        self.global_form.setRowVisible(
            self.distortion_limit_spin, distortion_mode
        )
        roi_mode = not (ri_mode or distortion_mode)
        self.roi_count_label.setVisible(roi_mode)
        self.roi_help_label.setVisible(roi_mode)
        self.clear_roi_button.setVisible(roi_mode)
        self.save_roi_button.setVisible(roi_mode)
        self.load_roi_button.setVisible(roi_mode)

    def _analyze_distortion(self) -> None:
        assert self._frame is not None
        if self._distortion_thread is not None:
            self.statusBar().showMessage("Distortion 분석이 이미 진행 중입니다.")
            return
        self.statusBar().showMessage("Distortion: 체커보드 자동 검출 중...")
        self.analyze_all_button.setEnabled(False)
        self._distortion_thread = QThread(self)
        self._distortion_worker = _DistortionWorker(
            np.array(self._frame.image, copy=True)
        )
        self._distortion_worker.moveToThread(self._distortion_thread)
        self._distortion_thread.started.connect(self._distortion_worker.run)
        self._distortion_worker.finished.connect(self._show_distortion_result)
        self._distortion_worker.finished.connect(self._distortion_thread.quit)
        self._distortion_worker.finished.connect(self._distortion_worker.deleteLater)
        self._distortion_thread.finished.connect(
            self._distortion_analysis_finished
        )
        self._distortion_thread.start()

    @Slot(object)
    def _show_distortion_result(self, result) -> None:
        limit = self.distortion_limit_spin.value()
        self.analysis_panel.show_distortion_result(result, limit)
        if self._distortion_result_dialog is not None:
            self._distortion_result_dialog.close()
        analyzed_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        source_path = getattr(self._frame, "source_path", None)
        source_filename = Path(source_path).name if source_path else "-"
        self._distortion_result_dialog = DistortionResultDialog(
            self._frame.image,
            result,
            analyzed_at=analyzed_at,
            source_filename=source_filename,
            limit_percent=limit,
            parent=self,
        )
        self._distortion_result_dialog.show()
        self._distortion_result_dialog.raise_()
        self._distortion_result_dialog.activateWindow()
        if result.smia_tv_distortion_percent is None:
            judgment = "INVALID"
            value = result.message
        else:
            judgment = (
                "PASS"
                if abs(result.smia_tv_distortion_percent) <= limit
                else "FAIL"
            )
            value = (
                f"{result.smia_tv_distortion_percent:+.3f}% "
                f"{result.distortion_type} | 기준 ≤ {limit:.2f}%"
            )
        self.statusBar().showMessage(f"Distortion {judgment} | {value}")

    @Slot()
    def _distortion_analysis_finished(self) -> None:
        if self._distortion_thread is not None:
            self._distortion_thread.deleteLater()
        self._distortion_thread = None
        self._distortion_worker = None
        self.analyze_all_button.setEnabled(True)

    def _analyze_ri(self) -> None:
        assert self._frame is not None
        try:
            result = measure_grid_relative_illumination(
                self._frame.image,
                rows=25,
                columns=33,
                inner_fraction=0.5,
            )
        except ValueError as exc:
            QMessageBox.critical(self, "RI 분석 오류", str(exc))
            self.statusBar().showMessage(f"RI 분석: INVALID - {exc}")
            return

        evaluation = evaluate_minimum_ri(
            result,
            self.ri_minimum_spin.value(),
        )
        self.analysis_panel.show_ri_result(result, evaluation)
        if self._ri_contour_dialog is not None:
            self._ri_contour_dialog.close()
        analyzed_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        source_path = getattr(self._frame, "source_path", None)
        source_filename = Path(source_path).name if source_path else "-"
        self._ri_contour_dialog = RiContourDialog(
            result,
            evaluation=evaluation,
            analyzed_at=analyzed_at,
            source_filename=source_filename,
            parent=self,
        )
        self._ri_contour_dialog.show()
        self._ri_contour_dialog.raise_()
        self._ri_contour_dialog.activateWindow()
        cell_map = {(cell.row, cell.column): cell for cell in result.cells}
        overlay_positions = [
            (0, 0),
            (0, result.columns // 2),
            (0, result.columns - 1),
            (result.rows // 2, 0),
            (result.center_row, result.center_column),
            (result.rows // 2, result.columns - 1),
            (result.rows - 1, 0),
            (result.rows - 1, result.columns // 2),
            (result.rows - 1, result.columns - 1),
        ]
        grid_items = []
        for position in overlay_positions:
            cell = cell_map[position]
            if cell.region_type == "CENTER":
                color = QColor("#00e676")
            elif cell.relative_percent < 70:
                color = QColor("#ff1744")
            elif cell.relative_percent < 85:
                color = QColor("#ffab00")
            else:
                color = QColor("#40c4ff")
            grid_items.append(
                (
                    cell.roi,
                    f"R{cell.row + 1}C{cell.column + 1}\n"
                    f"{cell.relative_percent:.1f}%",
                    color,
                )
            )
        self.viewer.show_ri_grid(grid_items)
        self.statusBar().showMessage(
            f"RI {evaluation.status} | 최소 {result.minimum_percent:.2f}% "
            f"(R{result.minimum_position[0] + 1}, "
            f"C{result.minimum_position[1] + 1}) | "
            f"기준 {evaluation.minimum_required_percent:.2f}%"
        )

    def _analyze_selected_slanted_edge(self, item) -> None:
        assert self._frame is not None
        from inspection import extract_roi

        result = measure_slanted_edge(
            extract_roi(self._frame.image, item.to_data()),
            self._optical_settings.pixel_pitch_x_um,
            self._optical_settings.pixel_pitch_y_um,
            self._optical_settings.magnification,
            self.global_lp_spin.value(),
            self.global_mtf_spin.value(),
        )
        self.analysis_panel.show_slanted_edge_result(result)
        evaluation = result.evaluation
        text = (
            f"MTF@{self.global_lp_spin.value():.1f}: "
            f"{evaluation.mtf_at_reference_frequency_percent:.1f}%"
            if evaluation
            and evaluation.mtf_at_reference_frequency_percent is not None
            else result.status
        )
        self.viewer.set_roi_measurement_result(
            item.data.number,
            result.status,
            text,
        )
        self.statusBar().showMessage(
            f"{item.data.name} Slanted Edge: {result.status} - {result.message}"
        )

    def _analyze_all_slanted_edge(self, rois: list[RoiData]) -> None:
        assert self._frame is not None
        result = measure_rois_slanted_edge(
            self._frame.image,
            rois,
            pixel_pitch_x_um=self._optical_settings.pixel_pitch_x_um,
            pixel_pitch_y_um=self._optical_settings.pixel_pitch_y_um,
            magnification=self._optical_settings.magnification,
            reference_frequency_lpmm=self.global_lp_spin.value(),
            target_mtf_percent=self.global_mtf_spin.value(),
        )
        self.analysis_panel.show_slanted_edge_batch_result(
            result,
            self.global_lp_spin.value(),
            self.global_mtf_spin.value(),
        )
        if self._slanted_edge_curve_dialog is not None:
            self._slanted_edge_curve_dialog.close()
        source_path = getattr(self._frame, "source_path", None)
        source_filename = Path(source_path).name if source_path else "-"
        self._slanted_edge_curve_dialog = SlantedEdgeCurveDialog(
            result,
            reference_frequency_lpmm=self.global_lp_spin.value(),
            target_mtf_percent=self.global_mtf_spin.value(),
            analyzed_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            source_filename=source_filename,
            parent=self,
        )
        self._slanted_edge_curve_dialog.show()
        self._slanted_edge_curve_dialog.raise_()
        self._slanted_edge_curve_dialog.activateWindow()
        self.viewer.clear_measurement_results()
        for item in result.roi_results:
            evaluation = item.result.evaluation
            text = (
                f"MTF@{self.global_lp_spin.value():.1f}: "
                f"{evaluation.mtf_at_reference_frequency_percent:.1f}%"
                if evaluation
                and evaluation.mtf_at_reference_frequency_percent is not None
                else item.result.status
            )
            self.viewer.set_roi_measurement_result(
                item.roi_number,
                item.result.status,
                text,
            )
        self.statusBar().showMessage(
            f"전체 Slanted Edge 검사: {result.overall_status} - {result.message}"
        )

    def _update_info(self, frame: ImageFrame) -> None:
        lines = [
            f"파일: {frame.source_path}",
            f"해상도: {frame.width} × {frame.height}",
            f"Bit depth: {frame.bit_depth}",
            f"채널: {frame.channels}",
            f"데이터 타입: {frame.dtype_name}",
            f"최솟값: {frame.minimum}",
            f"최댓값: {frame.maximum}",
            f"TIFF 페이지 수: {frame.page_count}",
        ]
        compression = frame.metadata.get("compression")
        if compression:
            lines.append(f"압축 방식: {compression}")
        self.info_view.setPlainText("\n".join(lines))
