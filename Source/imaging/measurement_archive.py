"""Lossless measurement archives for repeatable offline verification."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import scipy
import tifffile

from app_version import APP_VERSION
from imaging.roi import RoiData


def _json_value(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _json_value(asdict(value))
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_mode_name(mode: str) -> str:
    return "".join(character for character in mode if character.isalnum()) or "Measurement"


def _unique_directory(root: Path, name: str) -> Path:
    candidate = root / name
    sequence = 2
    while candidate.exists():
        candidate = root / f"{name}_{sequence:03d}"
        sequence += 1
    candidate.mkdir(parents=True)
    return candidate


def save_measurement_archive(
    root: str | Path,
    *,
    mode: str,
    image: np.ndarray,
    result_image: np.ndarray,
    result: Any,
    settings: dict[str, Any],
    rois: list[RoiData] | None = None,
    frame_number: int = 0,
    source_type: str = "file",
    source_filename: str = "",
    measured_at: datetime | None = None,
) -> Path:
    """Write one self-contained archive and verify the TIFF pixel round trip."""
    measured_at = measured_at or datetime.now().astimezone()
    original = np.asarray(image)
    rendered = np.asarray(result_image)
    if original.dtype not in (np.uint8, np.uint16):
        raise ValueError(f"저장할 수 없는 원본 픽셀 형식입니다: {original.dtype}")
    if rendered.dtype != np.uint8:
        raise ValueError("결과 이미지는 uint8 형식이어야 합니다.")
    if rendered.shape[:2] != original.shape[:2]:
        raise ValueError("결과 이미지는 원본과 동일한 해상도여야 합니다.")

    root_path = Path(root)
    day_root = root_path / measured_at.strftime("%Y-%m-%d")
    suffix = f"F{frame_number:08d}" if source_type == "camera" else "FILE"
    measurement_id = (
        f"{measured_at.strftime('%Y%m%d_%H%M%S_%f')[:-3]}_"
        f"{_safe_mode_name(mode)}_{suffix}"
    )
    archive = _unique_directory(day_root, measurement_id)
    original_path = archive / "original.tiff"
    result_path = archive / "result.tiff"
    manifest_path = archive / "measurement.json"

    try:
        tifffile.imwrite(original_path, original, photometric="minisblack" if original.ndim == 2 else "rgb")
        reloaded = tifffile.imread(original_path)
        if not np.array_equal(original, reloaded):
            raise OSError("저장된 원본 TIFF의 픽셀이 측정 프레임과 일치하지 않습니다.")
        tifffile.imwrite(result_path, rendered, photometric="rgb" if rendered.ndim == 3 else "minisblack")
        payload = {
            "archive_version": 1,
            "measurement_id": archive.name,
            "application_version": APP_VERSION,
            "algorithm_version": settings.get("algorithm_version", APP_VERSION),
            "runtime_versions": {
                "numpy": np.__version__,
                "opencv": cv2.__version__,
                "scipy": scipy.__version__,
                "tifffile": tifffile.__version__,
            },
            "measurement_mode": mode,
            "measured_at": measured_at.isoformat(timespec="milliseconds"),
            "input_type": source_type,
            "source_filename": source_filename,
            "camera_frame_number": frame_number if source_type == "camera" else None,
            "image": {
                "filename": original_path.name,
                "result_filename": result_path.name,
                "width": int(original.shape[1]),
                "height": int(original.shape[0]),
                "dtype": original.dtype.name,
                "channels": 1 if original.ndim == 2 else int(original.shape[2]),
                "sha256": _sha256(original_path),
            },
            "settings": _json_value(settings),
            "rois": [roi.to_dict() for roi in (rois or [])],
            "result": _json_value(result),
        }
        manifest_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8",
        )
    except Exception:
        for path in (manifest_path, result_path, original_path):
            path.unlink(missing_ok=True)
        try:
            archive.rmdir()
        except OSError:
            pass
        raise
    return archive


def load_measurement_manifest(path: str | Path) -> tuple[dict[str, Any], Path]:
    """Load and validate an archive manifest and its unmodified source TIFF."""
    manifest_path = Path(path).resolve()
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        image_info = payload["image"]
        image_path = manifest_path.parent / image_info["filename"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("올바른 측정 데이터 JSON 파일이 아닙니다.") from exc
    if not image_path.is_file():
        raise ValueError(f"측정 원본 TIFF를 찾을 수 없습니다: {image_path}")
    expected = str(image_info.get("sha256", ""))
    if expected and _sha256(image_path) != expected:
        raise ValueError("원본 TIFF의 무결성 검증(SHA-256)에 실패했습니다.")
    return payload, image_path


def render_result_overlay(
    image: np.ndarray,
    *,
    mode: str,
    status: str,
    rois: list[RoiData] | None = None,
    roi_labels: dict[int, str] | None = None,
    ri_cells: list[tuple[tuple[int, int, int, int], str, str]] | None = None,
    detected_points: np.ndarray | None = None,
    fitted_points: np.ndarray | None = None,
    rejected_points: np.ndarray | None = None,
    center: tuple[float, float] | None = None,
) -> np.ndarray:
    """Render a review image at exactly the source pixel dimensions."""
    source = np.asarray(image)
    if source.dtype == np.uint8:
        gray = source.copy()
    else:
        low, high = np.percentile(source, (1.0, 99.0))
        if high <= low:
            gray = np.zeros(source.shape[:2], dtype=np.uint8)
        else:
            gray = np.clip((source.astype(np.float32) - low) * 255.0 / (high - low), 0, 255).astype(np.uint8)
    if gray.ndim == 2:
        canvas = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    elif gray.shape[2] == 4:
        canvas = cv2.cvtColor(gray, cv2.COLOR_RGBA2RGB)
    else:
        canvas = gray[..., :3].copy()

    scale = max(0.45, min(canvas.shape[:2]) / 600.0)
    thickness = max(1, int(round(scale * 2)))
    status_color = (0, 220, 0) if status in {"PASS", "VALID"} else (255, 70, 50)
    cv2.putText(canvas, f"{mode}  {status}", (8, max(18, int(24 * scale))), cv2.FONT_HERSHEY_SIMPLEX, scale, status_color, thickness, cv2.LINE_AA)
    for roi in rois or []:
        color = (0, 220, 0) if status == "PASS" else (255, 190, 0)
        cv2.rectangle(canvas, (roi.x, roi.y), (roi.x + roi.width - 1, roi.y + roi.height - 1), color, thickness)
        label = (roi_labels or {}).get(roi.number, roi.name)
        cv2.putText(canvas, label, (roi.x, max(12, roi.y - 4)), cv2.FONT_HERSHEY_SIMPLEX, scale * 0.7, color, thickness, cv2.LINE_AA)
    color_map = {"CENTER": (0, 230, 100), "LOW": (255, 40, 40), "MID": (255, 180, 0), "OK": (40, 190, 255)}
    for (x, y, width, height), label, level in ri_cells or []:
        color = color_map.get(level, (40, 190, 255))
        cv2.rectangle(canvas, (x, y), (x + width - 1, y + height - 1), color, thickness)
        cv2.putText(canvas, label, (x, max(12, y - 3)), cv2.FONT_HERSHEY_SIMPLEX, scale * 0.55, color, thickness, cv2.LINE_AA)
    for points, color in ((fitted_points, (0, 176, 255)), (detected_points, (255, 0, 210))):
        if points is not None:
            for x, y in np.asarray(points):
                cv2.circle(canvas, (int(round(x)), int(round(y))), max(1, thickness), color, -1)
    if rejected_points is not None:
        for x, y in np.asarray(rejected_points):
            point = (int(round(x)), int(round(y)))
            cv2.drawMarker(canvas, point, (255, 70, 0), cv2.MARKER_TILTED_CROSS, max(5, thickness * 4), thickness)
    if center is not None:
        cv2.drawMarker(canvas, (int(round(center[0])), int(round(center[1]))), (255, 0, 0), cv2.MARKER_CROSS, max(8, thickness * 6), thickness)
    return canvas
