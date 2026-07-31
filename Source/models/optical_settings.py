"""카메라와 광학계의 공통 검사 설정."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path


@dataclass(slots=True)
class OpticalSettings:
    camera_model: str = ""
    lens_model: str = ""
    product_name: str = ""
    serial_number: str = ""
    operator: str = ""
    image_width: int = 640
    image_height: int = 480
    pixel_pitch_x_um: float = 15.0
    pixel_pitch_y_um: float = 15.0
    magnification: float = 1.0
    evaluation_frequency_lpmm: float = 15.0
    target_mtf_percent: float = 30.0
    pattern_frequency_tolerance_percent: float = 10.0
    ri_minimum_percent: float = 0.0
    distortion_limit_percent: float = 2.0

    def validate(self) -> None:
        if self.image_width <= 0 or self.image_height <= 0:
            raise ValueError("영상 해상도는 양의 정수여야 합니다.")
        if self.pixel_pitch_x_um <= 0 or self.pixel_pitch_y_um <= 0:
            raise ValueError("Pixel pitch는 0보다 커야 합니다.")
        if self.magnification <= 0:
            raise ValueError("배율은 0보다 커야 합니다.")
        if self.evaluation_frequency_lpmm <= 0:
            raise ValueError("평가 주파수는 0보다 커야 합니다.")
        if not 0 <= self.target_mtf_percent <= 100:
            raise ValueError("목표 MTF는 0~100% 범위여야 합니다.")
        if not 0 <= self.pattern_frequency_tolerance_percent <= 100:
            raise ValueError("패턴 주파수 검증 허용오차는 0~100% 범위여야 합니다.")
        if not 0 <= self.ri_minimum_percent <= 100:
            raise ValueError("최소 RI 판정 기준은 0~100% 범위여야 합니다.")

        if not 0 <= self.distortion_limit_percent <= 100:
            raise ValueError("Distortion 판정 기준은 0~100% 범위여야 합니다.")

    def to_dict(self) -> dict[str, str | int | float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> "OpticalSettings":
        try:
            settings = cls(
                camera_model=str(value.get("camera_model", "")),
                lens_model=str(value.get("lens_model", "")),
                product_name=str(value.get("product_name", "")),
                serial_number=str(value.get("serial_number", "")),
                operator=str(value.get("operator", "")),
                image_width=int(value.get("image_width", 640)),
                image_height=int(value.get("image_height", 480)),
                pixel_pitch_x_um=float(value.get("pixel_pitch_x_um", 15.0)),
                pixel_pitch_y_um=float(value.get("pixel_pitch_y_um", 15.0)),
                magnification=float(value.get("magnification", 1.0)),
                evaluation_frequency_lpmm=float(
                    value.get("evaluation_frequency_lpmm", 15.0)
                ),
                target_mtf_percent=float(
                    value.get("target_mtf_percent", 30.0)
                ),
                pattern_frequency_tolerance_percent=float(
                    value.get("pattern_frequency_tolerance_percent", 10.0)
                ),
                ri_minimum_percent=float(value.get("ri_minimum_percent", 0.0)),
                distortion_limit_percent=float(
                    value.get("distortion_limit_percent", 2.0)
                ),
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("광학 설정 값의 형식이 올바르지 않습니다.") from exc
        settings.validate()
        return settings


def save_optical_settings(path: str | Path, settings: OpticalSettings) -> None:
    settings.validate()
    payload = {"version": 1, "optical_settings": settings.to_dict()}
    Path(path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_optical_settings(path: str | Path) -> OpticalSettings:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("광학 설정 JSON 파일을 읽을 수 없습니다.") from exc
    if not isinstance(payload, dict) or not isinstance(
        payload.get("optical_settings"), dict
    ):
        raise ValueError("올바른 광학 설정 JSON 형식이 아닙니다.")
    return OpticalSettings.from_dict(payload["optical_settings"])
