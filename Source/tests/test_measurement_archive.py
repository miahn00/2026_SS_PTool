from __future__ import annotations

import json

import numpy as np
import tifffile

from imaging.measurement_archive import (
    load_measurement_manifest,
    render_result_overlay,
    save_measurement_archive,
)
from imaging.roi import RoiData


def test_archive_preserves_original_pixels_and_links_files(tmp_path) -> None:
    image = np.arange(80 * 64, dtype=np.uint16).reshape(64, 80)
    roi = RoiData(1, "ROI 1", 10, 10, 30, 32)
    rendered = render_result_overlay(
        image,
        mode="Slanted Edge",
        status="PASS",
        rois=[roi],
        roi_labels={1: "ROI 1 PASS MTF 42.0%"},
    )

    archive = save_measurement_archive(
        tmp_path,
        mode="Slanted Edge",
        image=image,
        result_image=rendered,
        result={"overall_status": "PASS"},
        settings={"algorithm_version": "test-v1", "target_mtf_percent": 30.0},
        rois=[roi],
        frame_number=123,
        source_type="camera",
    )

    assert np.array_equal(tifffile.imread(archive / "original.tiff"), image)
    assert tifffile.imread(archive / "result.tiff").shape[:2] == image.shape
    payload, image_path = load_measurement_manifest(archive / "measurement.json")
    assert payload["camera_frame_number"] == 123
    assert payload["measurement_mode"] == "Slanted Edge"
    assert payload["rois"][0]["x"] == 10
    assert image_path == archive / "original.tiff"


def test_archive_uses_unique_measurement_folders(tmp_path) -> None:
    image = np.ones((32, 32), dtype=np.uint8)
    rendered = np.repeat(image[..., None], 3, axis=2)
    values = dict(
        mode="RI",
        image=image,
        result_image=rendered,
        result={"status": "PASS"},
        settings={"algorithm_version": "ri-test"},
    )

    first = save_measurement_archive(tmp_path, **values)
    second = save_measurement_archive(tmp_path, **values)

    assert first != second
    assert first.is_dir() and second.is_dir()
    assert json.loads((second / "measurement.json").read_text(encoding="utf-8"))[
        "measurement_id"
    ] == second.name
