"""광학 성능 측정 프로그램 실행 진입점."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from imaging import ImageLoadError, load_image, to_display_uint8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SS Optical Performance Tool")
    parser.add_argument(
        "image",
        nargs="?",
        type=Path,
        help="CLI에서 정보를 확인할 TIFF 또는 JPEG 파일",
    )
    return parser.parse_args()


def inspect_image(image_path: Path) -> int:
    try:
        frame = load_image(image_path)
        original = frame.image.copy()
        display = to_display_uint8(frame.image)
    except (ImageLoadError, ValueError) as exc:
        print(f"오류: {exc}", file=sys.stderr)
        return 1

    result = frame.summary()
    result["display_dtype"] = display.dtype.name
    result["original_unchanged"] = bool(np.array_equal(frame.image, original))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def run_gui() -> int:
    from PySide6.QtWidgets import QApplication

    from ui import MainWindow

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


def main() -> int:
    args = parse_args()
    if args.image is not None:
        return inspect_image(args.image)
    return run_gui()


if __name__ == "__main__":
    raise SystemExit(main())
