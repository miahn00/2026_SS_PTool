from __future__ import annotations

import json

import numpy as np
import tifffile
from matplotlib.figure import Figure

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


def test_archive_saves_popup_result_and_manifest_hash(tmp_path) -> None:
    image = np.ones((48, 64), dtype=np.uint8)
    rendered = np.repeat(image[..., None], 3, axis=2)
    figure = Figure(figsize=(4, 3))
    figure.add_subplot(111).plot([0, 1], [0, 1])

    archive = save_measurement_archive(
        tmp_path,
        mode="RI",
        image=image,
        result_image=rendered,
        result={"status": "PASS"},
        settings={"algorithm_version": "ri-test"},
        popup_figure=figure,
    )

    popup_path = archive / "popup_result.tiff"
    assert popup_path.is_file()
    assert tifffile.imread(popup_path).size > 0
    payload = json.loads((archive / "measurement.json").read_text(encoding="utf-8"))
    assert payload["image"]["popup_result_filename"] == popup_path.name
    assert len(payload["image"]["popup_result_sha256"]) == 64


def test_archive_saves_black_level_corrected_uint16_image(tmp_path) -> None:
    image = np.arange(48 * 64, dtype=np.uint16).reshape(48, 64) + 25000
    corrected = np.maximum(image.astype(np.int32) - 25000, 0).astype(np.uint16)
    rendered = np.repeat((corrected // 16).astype(np.uint8)[..., None], 3, axis=2)

    archive = save_measurement_archive(
        tmp_path,
        mode="USAF 차트",
        image=image,
        corrected_image=corrected,
        result_image=rendered,
        result={"status": "PASS"},
        settings={
            "algorithm_version": "usaf-test",
            "black_level_correction": {"enabled": True, "applied_level": 25000},
        },
    )

    corrected_path = archive / "black_level_corrected.tiff"
    assert np.array_equal(tifffile.imread(corrected_path), corrected)
    payload = json.loads((archive / "measurement.json").read_text(encoding="utf-8"))
    assert payload["image"]["black_level_corrected_filename"] == corrected_path.name
    assert len(payload["image"]["black_level_corrected_sha256"]) == 64


def test_overlay_osd_stays_visible_for_edge_rois_and_long_labels() -> None:
    image = np.zeros((80, 120), dtype=np.uint8)
    roi = RoiData(1, "ROI 1", 110, 0, 10, 10)
    rendered = render_result_overlay(
        image,
        mode="Slanted Edge",
        status="INVALID",
        rois=[roi],
        roi_labels={1: "ROI 1 INVALID angle and contrast reason is very long"},
    )

    assert rendered.shape == (80, 120, 3)
    assert np.count_nonzero(rendered) > 0
