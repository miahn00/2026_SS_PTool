"""Camera Link connection settings dialog."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtGui import QRegularExpressionValidator
from PySide6.QtCore import QRegularExpression
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from camera import CameraConnectionSettings


class CameraConnectionDialog(QDialog):
    def __init__(self, initial: CameraConnectionSettings, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("실시간 카메라 연결 설정")
        self.setModal(True)
        self.setMinimumWidth(560)

        self.cam_file_edit = QLineEdit(str(initial.cam_file))
        browse = QPushButton("찾아보기...")
        browse.clicked.connect(self._browse_cam_file)
        cam_row = QHBoxLayout()
        cam_row.addWidget(self.cam_file_edit, 1)
        cam_row.addWidget(browse)

        self.connector_combo = QComboBox()
        self.connector_combo.setEditable(True)
        self.connector_combo.addItems(["A", "B"])
        self.connector_combo.setCurrentText(initial.connector)
        self.connector_combo.lineEdit().setValidator(
            QRegularExpressionValidator(QRegularExpression("[AaBb]"), self)
        )

        self.pixel_format_combo = QComboBox()
        self.pixel_format_combo.addItems(["Mono16", "Mono8"])
        self.pixel_format_combo.setCurrentText(initial.pixel_format)

        self.interval_combo = QComboBox()
        self.interval_combo.addItem("500 ms", 500)
        self.interval_combo.addItem("1000 ms", 1000)
        index = self.interval_combo.findData(initial.inspection_interval_ms)
        self.interval_combo.setCurrentIndex(max(0, index))

        form = QFormLayout()
        form.addRow("CAM 파일", cam_row)
        form.addRow("보드 커넥터", self.connector_combo)
        form.addRow("입력 픽셀 형식", self.pixel_format_combo)
        form.addRow("실시간 ROI 검사 주기", self.interval_combo)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("확인 및 연결")
        buttons.accepted.connect(self._accept_if_valid)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(buttons)

    def settings(self) -> CameraConnectionSettings:
        return CameraConnectionSettings(
            cam_file=Path(self.cam_file_edit.text().strip()),
            connector=self.connector_combo.currentText().strip().upper(),
            pixel_format=self.pixel_format_combo.currentText(),
            inspection_interval_ms=int(self.interval_combo.currentData()),
        )

    def _accept_if_valid(self) -> None:
        try:
            self.settings().validate()
        except ValueError as exc:
            QMessageBox.warning(self, "카메라 연결 설정", str(exc))
            return
        self.accept()

    def _browse_cam_file(self) -> None:
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "CAM 파일 선택",
            self.cam_file_edit.text(),
            "CAM file (*.cam)",
        )
        if selected:
            self.cam_file_edit.setText(selected)
