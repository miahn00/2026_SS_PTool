"""원본 영상 좌표계의 ROI 데이터와 JSON 직렬화."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path


@dataclass(slots=True)
class RoiData:
    """사각형 ROI의 원본 픽셀 좌표."""

    number: int
    name: str
    x: int
    y: int
    width: int
    height: int
    active: bool = True
    reference_frequency_lpmm: float = 0.0
    target_mtf_at_reference_percent: float = 0.0
    usaf_frequency_lpmm: float = 0.0
    usaf_group: int | None = None
    usaf_element: int | None = None
    direction: str = "Auto"
    include_in_judgment: bool = True

    def validate(self, image_width: int, image_height: int) -> None:
        if self.number < 1:
            raise ValueError("ROI 번호는 1 이상이어야 합니다.")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("ROI 크기는 1픽셀 이상이어야 합니다.")
        if self.x < 0 or self.y < 0:
            raise ValueError("ROI 좌표는 음수일 수 없습니다.")
        if self.x + self.width > image_width or self.y + self.height > image_height:
            raise ValueError("ROI가 영상 경계를 벗어납니다.")
        if self.reference_frequency_lpmm < 0:
            raise ValueError("기준 공간주파수는 음수일 수 없습니다.")
        if not 0 <= self.target_mtf_at_reference_percent <= 100:
            raise ValueError("목표 MTF는 0~100% 범위여야 합니다.")
        if self.usaf_frequency_lpmm < 0:
            raise ValueError("USAF 요소 주파수는 음수일 수 없습니다.")
        if (self.usaf_group is None) != (self.usaf_element is None):
            raise ValueError("USAF Group과 Element는 함께 지정해야 합니다.")
        if self.usaf_group is not None and self.usaf_element not in range(1, 7):
            raise ValueError("USAF Element는 1~6 범위여야 합니다.")
        if self.direction not in {"Auto", "H", "V"}:
            raise ValueError("측정 방향은 Auto, H, V 중 하나여야 합니다.")

    def to_dict(self) -> dict[str, int | str | bool]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> "RoiData":
        required = {"number", "name", "x", "y", "width", "height"}
        missing = required.difference(value)
        if missing:
            raise ValueError(f"ROI 필수 항목이 없습니다: {', '.join(sorted(missing))}")
        return cls(
            number=int(value["number"]),
            name=str(value["name"]),
            x=int(value["x"]),
            y=int(value["y"]),
            width=int(value["width"]),
            height=int(value["height"]),
            active=bool(value.get("active", True)),
            reference_frequency_lpmm=float(
                value.get(
                    "reference_frequency_lpmm",
                    value.get("nominal_lp_per_mm", 0.0),
                )
            ),
            target_mtf_at_reference_percent=float(
                value.get(
                    "target_mtf_at_reference_percent",
                    value.get("target_mtf_percent", 0.0),
                )
            ),
            usaf_frequency_lpmm=float(value.get("usaf_frequency_lpmm", 0.0)),
            usaf_group=(
                int(value["usaf_group"])
                if value.get("usaf_group") is not None else None
            ),
            usaf_element=(
                int(value["usaf_element"])
                if value.get("usaf_element") is not None else None
            ),
            direction=str(value.get("direction", "Auto")),
            include_in_judgment=bool(value.get("include_in_judgment", True)),
        )


def roi_from_points(
    number: int,
    start_x: int,
    start_y: int,
    end_x: int,
    end_y: int,
    image_width: int,
    image_height: int,
) -> RoiData:
    """두 원본 픽셀 좌표로 영상 경계 안의 ROI를 생성한다."""
    left = min(max(min(start_x, end_x), 0), image_width - 1)
    top = min(max(min(start_y, end_y), 0), image_height - 1)
    right = min(max(max(start_x, end_x), 0), image_width - 1)
    bottom = min(max(max(start_y, end_y), 0), image_height - 1)
    roi = RoiData(
        number=number,
        name=f"ROI {number}",
        x=left,
        y=top,
        width=right - left + 1,
        height=bottom - top + 1,
    )
    roi.validate(image_width, image_height)
    return roi


def save_rois(path: str | Path, rois: list[RoiData]) -> None:
    """ROI 목록을 UTF-8 JSON 파일로 저장한다."""
    payload = {
        "version": 1,
        "coordinate_system": "image_pixels",
        "rois": [roi.to_dict() for roi in rois],
    }
    Path(path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_rois(
    path: str | Path,
    image_width: int,
    image_height: int,
    *,
    maximum_count: int = 4,
) -> list[RoiData]:
    """JSON에서 ROI를 읽고 개수, 번호, 영상 경계를 검증한다."""
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("ROI JSON 파일을 읽을 수 없습니다.") from exc

    if not isinstance(payload, dict) or not isinstance(payload.get("rois"), list):
        raise ValueError("올바른 ROI JSON 형식이 아닙니다.")
    if len(payload["rois"]) > maximum_count:
        raise ValueError(f"ROI는 최대 {maximum_count}개까지 불러올 수 있습니다.")

    rois = [RoiData.from_dict(value) for value in payload["rois"]]
    numbers = [roi.number for roi in rois]
    if len(numbers) != len(set(numbers)):
        raise ValueError("중복된 ROI 번호가 있습니다.")
    for roi in rois:
        roi.validate(image_width, image_height)
    return rois
