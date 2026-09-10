from __future__ import annotations

import mmap
from pathlib import Path

import numpy as np
from PySide6.QtWidgets import QApplication
import pytest

from camera.camera_link_client import (
    CameraConnectionSettings,
    CameraLinkClient,
    HEADER,
    HEADER_SIZE,
    MAGIC,
    MAP_SIZE,
    default_cam_file,
)
from ui.camera_connection_dialog import CameraConnectionDialog


def test_default_camera_settings_match_flir_640_cam() -> None:
    settings = CameraConnectionSettings(default_cam_file())

    settings.validate()
    assert settings.cam_file.name == "LK_FLIR 640.cam"
    assert settings.connector == "B"
    assert settings.pixel_format == "Mono16"
    assert settings.inspection_interval_ms == 500


@pytest.mark.parametrize("interval", [500, 1000])
def test_connection_dialog_exposes_supported_settings(interval: int) -> None:
    app = QApplication.instance() or QApplication([])
    initial = CameraConnectionSettings(
        Path(default_cam_file()), "A", "Mono8", interval
    )
    dialog = CameraConnectionDialog(initial)

    assert dialog.settings() == initial
    assert [dialog.connector_combo.itemText(i) for i in range(2)] == ["A", "B"]
    assert dialog.connector_combo.isEditable()
    assert {dialog.pixel_format_combo.itemText(i) for i in range(2)} == {
        "Mono8", "Mono16"
    }
    dialog.close()
    app.processEvents()


def test_camera_client_reads_complete_mono16_frame() -> None:
    app = QApplication.instance() or QApplication([])
    client = CameraLinkClient()
    mapping = mmap.mmap(-1, MAP_SIZE)
    client._mapping = mapping
    received = []
    client.frameReady.connect(lambda *args: received.append(args))
    image = np.arange(640 * 512, dtype=np.uint16).reshape(512, 640)
    payload = image.tobytes()
    mapping.seek(0)
    mapping.write(HEADER.pack(MAGIC, 1, 2, 7, 640, 512, 2, len(payload), 30000, 2))
    mapping.seek(HEADER_SIZE)
    mapping.write(payload)

    client._poll_frame()

    assert len(received) == 1
    actual, frame_number, fps, state = received[0]
    assert np.array_equal(actual, image)
    assert (frame_number, fps, state) == (7, 30.0, "ACTIVE")
    client._mapping = None
    mapping.close()
    client.deleteLater()
    app.processEvents()
