from __future__ import annotations

import pytest

from models import OpticalSettings, load_optical_settings, save_optical_settings


def test_optical_settings_json_round_trip(tmp_path) -> None:
    source = OpticalSettings(
        camera_model="FLIR 640",
        lens_model="Lens-A",
        product_name="Sample",
        serial_number="SN-001",
        operator="Tester",
        image_width=640,
        image_height=480,
        pixel_pitch_x_um=15.0,
        pixel_pitch_y_um=15.5,
        magnification=0.5,
        evaluation_frequency_lpmm=12.5,
        target_mtf_percent=35.0,
        pattern_frequency_tolerance_percent=20.0,
        ri_minimum_percent=85.0,
        distortion_limit_percent=1.5,
    )
    path = tmp_path / "optical.json"

    save_optical_settings(path, source)
    loaded = load_optical_settings(path)

    assert loaded == source


def test_optical_settings_rejects_invalid_pitch() -> None:
    settings = OpticalSettings(pixel_pitch_x_um=0)

    with pytest.raises(ValueError, match="Pixel pitch"):
        settings.validate()
