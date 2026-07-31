"""카메라와 파일 입력이 공통으로 사용하는 영상 데이터 모델."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(slots=True)
class ImageFrame:
    """원본 픽셀 배열과 입력 영상의 기본 정보를 보관한다."""

    image: NDArray[np.generic]
    width: int
    height: int
    bit_depth: int
    channels: int
    source_type: str
    timestamp: datetime
    frame_number: int = 0
    source_path: Path | None = None
    page_count: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def dtype_name(self) -> str:
        return self.image.dtype.name

    @property
    def minimum(self) -> int | float:
        return self.image.min().item()

    @property
    def maximum(self) -> int | float:
        return self.image.max().item()

    def summary(self) -> dict[str, Any]:
        """UI와 CLI에서 사용할 JSON 직렬화 가능 정보를 반환한다."""
        return {
            "source_path": str(self.source_path) if self.source_path else None,
            "source_type": self.source_type,
            "width": self.width,
            "height": self.height,
            "bit_depth": self.bit_depth,
            "channels": self.channels,
            "dtype": self.dtype_name,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "page_count": self.page_count,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }

