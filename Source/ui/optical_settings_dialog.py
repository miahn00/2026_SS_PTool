"""공통 카메라·광학 설정 입력 창."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QLineEdit,
    QMessageBox,
    QSpinBox,
    QVBoxLayout,
)

from models import OpticalSettings


class OpticalSettingsDialog(QDialog):
    def __init__(self, settings: OpticalSettings, parent=None) -> None:
        super().__init__(parent)
        self._evaluation_frequency_lpmm = settings.evaluation_frequency_lpmm
        self._target_mtf_percent = settings.target_mtf_percent
        self._pattern_frequency_tolerance_percent = (
            settings.pattern_frequency_tolerance_percent
        )
        self._ri_minimum_percent = settings.ri_minimum_percent
        self._distortion_limit_percent = settings.distortion_limit_percent
        self.setWindowTitle("카메라 및 광학 설정")
        self.setMinimumWidth(420)

        self.camera_model = QLineEdit()
        self.lens_model = QLineEdit()
        self.product_name = QLineEdit()
        self.serial_number = QLineEdit()
        self.operator = QLineEdit()
        self.image_width = QSpinBox()
        self.image_height = QSpinBox()
        self.image_width.setRange(1, 16384)
        self.image_height.setRange(1, 16384)
        self.pixel_pitch_x = QDoubleSpinBox()
        self.pixel_pitch_y = QDoubleSpinBox()
        self.magnification = QDoubleSpinBox()
        for spin in (self.pixel_pitch_x, self.pixel_pitch_y):
            spin.setRange(0.001, 1000)
            spin.setDecimals(4)
            spin.setSuffix(" µm")
        self.magnification.setRange(0.0001, 10000)
        self.magnification.setDecimals(4)
        self.magnification.setToolTip(
            "렌즈의 광학 배율 M을 입력합니다. "
            "예: 0.5X 렌즈는 0.5, 1X 렌즈는 1.0, 2X 렌즈는 2.0"
        )

        form = QFormLayout()
        form.addRow("카메라 모델", self.camera_model)
        form.addRow("렌즈 모델", self.lens_model)
        form.addRow("제품명", self.product_name)
        form.addRow("제품/렌즈 S/N", self.serial_number)
        form.addRow("작업자", self.operator)
        form.addRow("카메라 입력 해상도 Width", self.image_width)
        form.addRow("카메라 입력 해상도 Height", self.image_height)
        form.addRow("Pixel pitch X", self.pixel_pitch_x)
        form.addRow("Pixel pitch Y", self.pixel_pitch_y)
        form.addRow("렌즈 배율 (Magnification)", self.magnification)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._accept_if_valid)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout()
        layout.addLayout(form)
        layout.addWidget(buttons)
        self.setLayout(layout)
        self.set_settings(settings)

    def settings(self) -> OpticalSettings:
        settings = OpticalSettings(
            camera_model=self.camera_model.text().strip(),
            lens_model=self.lens_model.text().strip(),
            product_name=self.product_name.text().strip(),
            serial_number=self.serial_number.text().strip(),
            operator=self.operator.text().strip(),
            image_width=self.image_width.value(),
            image_height=self.image_height.value(),
            pixel_pitch_x_um=self.pixel_pitch_x.value(),
            pixel_pitch_y_um=self.pixel_pitch_y.value(),
            magnification=self.magnification.value(),
            evaluation_frequency_lpmm=self._evaluation_frequency_lpmm,
            target_mtf_percent=self._target_mtf_percent,
            pattern_frequency_tolerance_percent=(
                self._pattern_frequency_tolerance_percent
            ),
            ri_minimum_percent=self._ri_minimum_percent,
            distortion_limit_percent=self._distortion_limit_percent,
        )
        settings.validate()
        return settings

    def set_settings(self, settings: OpticalSettings) -> None:
        self._evaluation_frequency_lpmm = settings.evaluation_frequency_lpmm
        self._target_mtf_percent = settings.target_mtf_percent
        self._pattern_frequency_tolerance_percent = (
            settings.pattern_frequency_tolerance_percent
        )
        self._ri_minimum_percent = settings.ri_minimum_percent
        self._distortion_limit_percent = settings.distortion_limit_percent
        self.camera_model.setText(settings.camera_model)
        self.lens_model.setText(settings.lens_model)
        self.product_name.setText(settings.product_name)
        self.serial_number.setText(settings.serial_number)
        self.operator.setText(settings.operator)
        self.image_width.setValue(settings.image_width)
        self.image_height.setValue(settings.image_height)
        self.pixel_pitch_x.setValue(settings.pixel_pitch_x_um)
        self.pixel_pitch_y.setValue(settings.pixel_pitch_y_um)
        self.magnification.setValue(settings.magnification)

    def _accept_if_valid(self) -> None:
        try:
            self.settings()
        except ValueError as exc:
            QMessageBox.warning(self, "설정 오류", str(exc))
            return
        self.accept()
