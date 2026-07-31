from __future__ import annotations

import numpy as np
from PIL import Image
import pytest
import tifffile

from imaging import ImageLoadError, load_image, to_display_uint8, window_to_uint8


def test_load_raw16_tiff_preserves_pixels(tmp_path) -> None:
    source = np.array(
        [[0, 1, 255], [256, 32768, 65535]],
        dtype=np.uint16,
    )
    path = tmp_path / "raw16.tiff"
    tifffile.imwrite(path, source, photometric="minisblack")

    frame = load_image(path)

    assert frame.width == 3
    assert frame.height == 2
    assert frame.bit_depth == 16
    assert frame.channels == 1
    assert frame.page_count == 1
    assert frame.image.dtype == np.uint16
    assert np.array_equal(frame.image, source)
    assert frame.minimum == 0
    assert frame.maximum == 65535


def test_display_conversion_does_not_modify_raw16() -> None:
    source = np.array([[0, 1000], [40000, 65535]], dtype=np.uint16)
    original = source.copy()

    display = to_display_uint8(source, lower_percentile=0, upper_percentile=100)

    assert display.dtype == np.uint8
    assert display.shape == source.shape
    assert np.array_equal(source, original)
    assert not np.shares_memory(display, source)


def test_display_window_does_not_modify_raw16() -> None:
    source = np.array([[0, 1000], [40000, 65535]], dtype=np.uint16)
    original = source.copy()

    display = window_to_uint8(source, 1000, 40000)

    assert display.tolist() == [[0, 0], [255, 255]]
    assert np.array_equal(source, original)
    assert not np.shares_memory(display, source)


def test_load_rgb_jpeg(tmp_path) -> None:
    source = np.zeros((8, 10, 3), dtype=np.uint8)
    source[:, :, 0] = 200
    path = tmp_path / "sample.jpg"
    Image.fromarray(source, mode="RGB").save(path)

    frame = load_image(path)

    assert frame.width == 10
    assert frame.height == 8
    assert frame.bit_depth == 8
    assert frame.channels == 3
    assert frame.image.dtype == np.uint8


def test_load_first_page_and_report_page_count(tmp_path) -> None:
    pages = np.stack(
        [
            np.full((4, 5), 100, dtype=np.uint16),
            np.full((4, 5), 200, dtype=np.uint16),
        ]
    )
    path = tmp_path / "multipage.tiff"
    with tifffile.TiffWriter(path) as writer:
        for page in pages:
            writer.write(page, photometric="minisblack")

    frame = load_image(path)

    assert frame.page_count == 2
    assert np.array_equal(frame.image, pages[0])


def test_reject_unsupported_extension(tmp_path) -> None:
    path = tmp_path / "sample.bmp"
    path.write_bytes(b"not an image")

    with pytest.raises(ImageLoadError, match="지원하지 않는 확장자"):
        load_image(path)


def test_reject_missing_file(tmp_path) -> None:
    with pytest.raises(ImageLoadError, match="찾을 수 없습니다"):
        load_image(tmp_path / "missing.tiff")
