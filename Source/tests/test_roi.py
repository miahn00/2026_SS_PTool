from __future__ import annotations

import json

import pytest

from imaging.roi import RoiData, load_rois, roi_from_points, save_rois


def test_roi_json_round_trip(tmp_path) -> None:
    source = [
        RoiData(
            1,
            "Center",
            10,
            20,
            100,
            80,
            reference_frequency_lpmm=25.5,
            target_mtf_at_reference_percent=40,
            direction="H",
        ),
        RoiData(2, "Corner", 300, 200, 50, 40, active=False),
    ]
    path = tmp_path / "rois.json"

    save_rois(path, source)
    loaded = load_rois(path, 640, 480)

    assert loaded == source
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["coordinate_system"] == "image_pixels"
    assert payload["rois"][0]["reference_frequency_lpmm"] == 25.5


def test_roi_from_reverse_drag_points() -> None:
    roi = roi_from_points(1, 200, 150, 100, 50, 640, 480)

    assert (roi.x, roi.y, roi.width, roi.height) == (100, 50, 101, 101)


def test_roi_from_points_clamps_to_image() -> None:
    roi = roi_from_points(1, -20, -10, 700, 500, 640, 480)

    assert (roi.x, roi.y, roi.width, roi.height) == (0, 0, 640, 480)


def test_roi_rejects_out_of_bounds(tmp_path) -> None:
    path = tmp_path / "rois.json"
    save_rois(path, [RoiData(1, "Outside", 600, 450, 100, 80)])

    with pytest.raises(ValueError, match="영상 경계"):
        load_rois(path, 640, 480)


def test_roi_rejects_more_than_four(tmp_path) -> None:
    path = tmp_path / "rois.json"
    save_rois(
        path,
        [RoiData(index, f"ROI {index}", 0, 0, 10, 10) for index in range(1, 6)],
    )

    with pytest.raises(ValueError, match="최대 4개"):
        load_rois(path, 640, 480)


def test_roi_rejects_duplicate_numbers(tmp_path) -> None:
    path = tmp_path / "rois.json"
    save_rois(
        path,
        [
            RoiData(1, "First", 0, 0, 10, 10),
            RoiData(1, "Duplicate", 20, 20, 10, 10),
        ],
    )

    with pytest.raises(ValueError, match="중복"):
        load_rois(path, 640, 480)
