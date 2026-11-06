from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtWidgets import QApplication

from imaging.roi import RoiData
from inspection.mtf_batch import measure_rois_mtf
from ui.usaf_result_dialog import UsafResultDialog


def test_usaf_result_dialog_creates_one_tab_per_roi() -> None:
    app = QApplication.instance() or QApplication([])
    x = np.arange(192)
    bars = np.where((x // 8) % 2 == 0, 50000, 10000).astype(np.uint16)
    image = np.tile(bars, (96, 1))
    rois = [
        RoiData(1, "ROI 1", 0, 0, 96, 96),
        RoiData(2, "ROI 2", 96, 0, 96, 96),
    ]
    result = measure_rois_mtf(
        image,
        rois,
        reference_frequency_lpmm=6.25,
        target_mtf_percent=30,
        pixel_pitch_x_um=10,
        pixel_pitch_y_um=10,
        magnification=1,
        pattern_frequency_tolerance_percent=100,
    )

    dialog = UsafResultDialog(
        image,
        rois,
        result,
        analyzed_at="2026-11-06 12:00:00",
        source_filename="usaf16.tiff",
        reference_frequency_lpmm=6.25,
        target_mtf_percent=30,
        black_level=25000,
    )

    assert dialog.tabs.count() == 2
    assert "ROI 1" in dialog.tabs.tabText(0)
    assert "ROI 2" in dialog.tabs.tabText(1)
    assert len(dialog.figure.axes) == 2
    header = "\n".join(text.get_text() for text in dialog.figure.texts)
    assert "USAF 전체 ROI 밝기 프로파일" in header
    assert "Black Level: 25000" in header
    dialog.close()
    app.processEvents()
