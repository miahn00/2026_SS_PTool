from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QPushButton
from PySide6.QtGui import QPixmap
import pytest
import numpy as np

from imaging.roi import RoiData
from ui import MainWindow


def test_main_window_starts_without_image(tmp_path) -> None:
    app = QApplication.instance() or QApplication([])
    settings_path = tmp_path / "optical_settings.json"
    window = MainWindow(settings_path)

    assert window._frame is None
    assert not window.viewer.has_image
    assert window.windowTitle() == "SS Optical Performance Tool"
    assert window.analysis_panel.result_table.height() >= 128
    assert window.measurement_mode_combo.currentText() == "Slanted Edge"
    assert not hasattr(window, "roi_x_spin")
    assert not hasattr(window, "roi_width_spin")
    assert not hasattr(window, "roi_name_edit")
    assert not hasattr(window, "roi_direction_combo")
    assert not hasattr(window, "roi_active_check")
    assert not hasattr(window, "roi_judgment_check")
    assert all(
        button.text() != "선택 ROI 분석"
        for button in window.findChildren(QPushButton)
    )
    assert settings_path.exists()

    window.close()
    app.processEvents()


def test_ri_mode_hides_mtf_controls_and_disables_manual_rois(tmp_path) -> None:
    app = QApplication.instance() or QApplication([])
    window = MainWindow(tmp_path / "optical_settings.json")

    assert window.measurement_mode_combo.findText("RI") >= 0
    window.measurement_mode_combo.setCurrentText("RI")

    assert window.global_lp_spin.isHidden()
    assert window.global_mtf_spin.isHidden()
    assert window.global_frequency_tolerance_spin.isHidden()
    assert not window.ri_minimum_spin.isHidden()
    assert window.distortion_limit_spin.isHidden()
    assert window.roi_count_label.isHidden()
    assert window.clear_roi_button.isHidden()
    assert not window.viewer._roi_drawing_enabled

    window.measurement_mode_combo.setCurrentText("Slanted Edge")
    assert not window.global_lp_spin.isHidden()
    assert window.ri_minimum_spin.isHidden()
    assert window.viewer._roi_drawing_enabled
    window.close()
    app.processEvents()


def test_distortion_mode_shows_default_limit_and_hides_roi_controls(
    tmp_path,
) -> None:
    app = QApplication.instance() or QApplication([])
    window = MainWindow(tmp_path / "optical_settings.json")

    window.measurement_mode_combo.setCurrentText("Distortion")

    assert window.distortion_limit_spin.value() == pytest.approx(2.0)
    assert not window.distortion_limit_spin.isHidden()
    assert window.global_lp_spin.isHidden()
    assert window.ri_minimum_spin.isHidden()
    assert window.roi_count_label.isHidden()
    assert not window.viewer._roi_drawing_enabled
    window.close()
    app.processEvents()


def test_ri_mode_runs_dense_contour_measurement(tmp_path) -> None:
    app = QApplication.instance() or QApplication([])
    window = MainWindow(tmp_path / "optical_settings.json")
    window.viewer.set_image(QPixmap(100, 100), 100, 100)
    image = np.full((100, 100), 1000, dtype=np.uint16)
    image[:20, :20] = 800
    window._frame = type(
        "FrameStub",
        (),
        {
            "width": 100,
            "height": 100,
            "image": image,
            "source_path": tmp_path / "ri_test.tiff",
        },
    )()
    window.measurement_mode_combo.setCurrentText("RI")
    window.ri_minimum_spin.setValue(75.0)

    window._analyze_all_rois()

    assert window.analysis_panel.result_table.rowCount() == 9
    assert len(window.viewer._ri_grid_items) == 18
    assert window._ri_contour_dialog is not None
    assert not window._ri_contour_dialog.isHidden()
    assert window.analysis_panel.scroll_area.widgetResizable()
    figure_text = "\n".join(
        text.get_text() for text in window._ri_contour_dialog.figure.texts
    )
    assert "분석 일시:" in figure_text
    assert "파일명: ri_test.tiff" in figure_text
    assert "코너 RI" in figure_text
    assert "판정 방식:" in figure_text
    assert "결과: PASS" in figure_text
    assert "RI 판정: PASS" in window.analysis_panel.overall_label.text()
    window.close()
    app.processEvents()


def test_usaf_mode_uses_common_evaluation_frequency(
    tmp_path,
) -> None:
    app = QApplication.instance() or QApplication([])
    window = MainWindow(tmp_path / "optical_settings.json")
    window.viewer.set_image(QPixmap(640, 480), 640, 480)
    window._frame = type(
        "FrameStub", (), {"width": 640, "height": 480}
    )()
    window.viewer.add_roi(RoiData(1, "ROI 1", 10, 10, 100, 80))

    window.measurement_mode_combo.setCurrentText("USAF 차트")
    window.global_lp_spin.setValue(6.35)

    roi = window.viewer.selected_roi().data
    assert roi.usaf_group is None
    assert roi.usaf_element is None
    assert roi.usaf_frequency_lpmm == pytest.approx(6.35)
    assert not hasattr(window, "roi_usaf_frequency_spin")
    assert not hasattr(window, "roi_usaf_custom_check")
    assert not hasattr(window, "roi_usaf_group_spin")
    assert not hasattr(window, "roi_usaf_element_spin")
    assert not window.global_frequency_tolerance_spin.isHidden()

    window.measurement_mode_combo.setCurrentText("Slanted Edge")
    assert window.global_frequency_tolerance_spin.isHidden()
    window.close()
    app.processEvents()


def test_evaluation_frequency_updates_all_roi_frequencies(
    tmp_path,
) -> None:
    app = QApplication.instance() or QApplication([])
    window = MainWindow(tmp_path / "optical_settings.json")
    window.viewer.set_image(QPixmap(640, 480), 640, 480)
    window._frame = type(
        "FrameStub", (), {"width": 640, "height": 480}
    )()
    window.viewer.add_roi(
        RoiData(1, "ROI 1", 10, 10, 100, 80, usaf_frequency_lpmm=5.657)
    )
    window.viewer.add_roi(
        RoiData(2, "ROI 2", 140, 10, 100, 80, usaf_frequency_lpmm=6.350)
    )
    window.measurement_mode_combo.setCurrentText("USAF 차트")

    window.global_lp_spin.setValue(6.0)

    rois = sorted(window.viewer.roi_data(), key=lambda roi: roi.number)
    assert rois[0].usaf_frequency_lpmm == pytest.approx(6.0)
    assert rois[1].usaf_frequency_lpmm == pytest.approx(6.0)

    window.close()
    app.processEvents()


def test_roi_options_are_normalized_to_fixed_defaults(tmp_path) -> None:
    app = QApplication.instance() or QApplication([])
    window = MainWindow(tmp_path / "optical_settings.json")
    window.viewer.set_image(QPixmap(640, 480), 640, 480)
    window.viewer.add_roi(
        RoiData(
            1,
            "Custom name",
            10,
            10,
            100,
            80,
            active=False,
            direction="H",
            include_in_judgment=False,
        )
    )

    roi = window.viewer.roi_data()[0]
    assert roi.name == "ROI 1"
    assert roi.direction == "Auto"
    assert roi.active
    assert roi.include_in_judgment
    window.close()
    app.processEvents()


def test_main_window_loads_existing_optical_settings(tmp_path) -> None:
    from models import OpticalSettings, save_optical_settings

    app = QApplication.instance() or QApplication([])
    settings_path = tmp_path / "optical_settings.json"
    save_optical_settings(
        settings_path,
        OpticalSettings(
            camera_model="AUTO LOAD",
            magnification=0.5,
            evaluation_frequency_lpmm=12.0,
            target_mtf_percent=40.0,
            pattern_frequency_tolerance_percent=25.0,
        ),
    )

    window = MainWindow(settings_path)

    assert window._optical_settings.camera_model == "AUTO LOAD"
    assert window._optical_settings.magnification == 0.5
    assert window.global_lp_spin.value() == 12.0
    assert window.global_mtf_spin.value() == 40.0
    assert window.global_frequency_tolerance_spin.value() == 25.0
    window.close()
    app.processEvents()


def test_common_evaluation_settings_are_saved_and_reloaded(tmp_path) -> None:
    from models import load_optical_settings

    app = QApplication.instance() or QApplication([])
    settings_path = tmp_path / "optical_settings.json"
    window = MainWindow(settings_path)

    window.global_lp_spin.setValue(10.0)
    window.global_mtf_spin.setValue(30.0)
    window.global_frequency_tolerance_spin.setValue(30.0)
    window.ri_minimum_spin.setValue(82.0)
    window.distortion_limit_spin.setValue(1.25)

    saved = load_optical_settings(settings_path)
    assert saved.evaluation_frequency_lpmm == 10.0
    assert saved.target_mtf_percent == 30.0
    assert saved.pattern_frequency_tolerance_percent == 30.0
    assert saved.ri_minimum_percent == 82.0
    assert saved.distortion_limit_percent == 1.25
    window.close()

    restarted = MainWindow(settings_path)
    assert restarted.global_lp_spin.value() == 10.0
    assert restarted.global_mtf_spin.value() == 30.0
    assert restarted.global_frequency_tolerance_spin.value() == 30.0
    assert restarted.ri_minimum_spin.value() == 82.0
    assert restarted.distortion_limit_spin.value() == 1.25
    restarted.close()
    app.processEvents()


def test_viewer_limits_rois_and_clamps_position() -> None:
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.viewer.set_image(QPixmap(640, 480), 640, 480)

    for number in range(1, 5):
        window.viewer.add_roi(
            RoiData(number, f"ROI {number}", number * 10, number * 10, 100, 80)
        )

    with pytest.raises(ValueError, match="최대 4개"):
        window.viewer.add_roi(RoiData(5, "ROI 5", 0, 0, 20, 20))

    item = window.viewer.selected_roi()
    assert item is not None
    item.setPos(1000, 1000)
    data = item.to_data()
    assert data.x == 540
    assert data.y == 400

    window.close()
    app.processEvents()


def test_drag_creates_roi_and_delete_key_removes_it(tmp_path) -> None:
    app = QApplication.instance() or QApplication([])
    window = MainWindow(tmp_path / "optical_settings.json")
    window.viewer.resize(640, 480)
    window.viewer.set_image(QPixmap(640, 480), 640, 480)
    window.viewer.actual_size()
    window.show()
    app.processEvents()

    start = window.viewer.mapFromScene(100, 80)
    end = window.viewer.mapFromScene(220, 180)
    QTest.mousePress(
        window.viewer.viewport(),
        Qt.MouseButton.LeftButton,
        pos=QPoint(start),
    )
    QTest.mouseMove(window.viewer.viewport(), QPoint(end), delay=10)
    QTest.mouseRelease(
        window.viewer.viewport(),
        Qt.MouseButton.LeftButton,
        pos=QPoint(end),
    )
    app.processEvents()

    assert len(window.viewer.roi_data()) == 1
    assert window.viewer.roi_data()[0].number == 1
    assert window.viewer.roi_data()[0].reference_frequency_lpmm == 15.0
    assert window.viewer.roi_data()[0].target_mtf_at_reference_percent == 30.0

    created = window.viewer.roi_data()[0]
    original_width = created.width
    original_height = created.height
    handle = window.viewer.mapFromScene(
        created.x + created.width,
        created.y + created.height,
    )
    resized = handle + QPoint(30, 20)
    QTest.mousePress(
        window.viewer.viewport(),
        Qt.MouseButton.LeftButton,
        pos=handle,
    )
    QTest.mouseMove(window.viewer.viewport(), resized, delay=10)
    QTest.mouseRelease(
        window.viewer.viewport(),
        Qt.MouseButton.LeftButton,
        pos=resized,
    )
    app.processEvents()
    assert window.viewer.roi_data()[0].width > original_width
    assert window.viewer.roi_data()[0].height > original_height

    window.viewer.setFocus()
    QTest.keyClick(window.viewer, Qt.Key.Key_Delete)
    app.processEvents()
    assert window.viewer.roi_data() == []

    window.close()
    app.processEvents()
