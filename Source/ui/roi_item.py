"""이동 및 크기 변경이 가능한 ROI 그래픽 항목."""

from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QPainterPath, QPen
from PySide6.QtWidgets import (
    QGraphicsItem,
    QGraphicsRectItem,
    QGraphicsSceneMouseEvent,
    QGraphicsSimpleTextItem,
)

from imaging.roi import RoiData

ROI_COLORS = ("#ff5252", "#40c4ff", "#69f0ae", "#ffd740")


class RoiItem(QGraphicsRectItem):
    """원본 영상 좌표로 움직이는 사각형 ROI."""

    HANDLE_PIXELS = 16.0
    MINIMUM_SIZE = 8.0

    def __init__(
        self,
        data: RoiData,
        image_width: int,
        image_height: int,
        changed: Callable[["RoiItem"], None],
        selected: Callable[["RoiItem"], None],
    ) -> None:
        super().__init__(0, 0, data.width, data.height)
        self.data = data
        self.image_width = image_width
        self.image_height = image_height
        self._changed_callback = changed
        self._selected_callback = selected
        self._resizing = False
        self._active_handle: str | None = None
        self._original_geometry = QRectF()
        self._press_scene_position = QPointF()

        color = QColor(ROI_COLORS[(data.number - 1) % len(ROI_COLORS)])
        self._base_color = color
        pen = QPen(color, 2)
        pen.setCosmetic(True)
        self.setPen(pen)
        self.setBrush(QBrush(QColor(color.red(), color.green(), color.blue(), 25)))
        self.setPos(data.x, data.y)
        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsMovable
            | QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
            | QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges
        )
        self.setAcceptHoverEvents(True)
        self.setZValue(10)

        self._label = QGraphicsSimpleTextItem(data.name, self)
        self._label.setBrush(QBrush(color))
        self._label.setFlag(
            QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations,
            True,
        )
        self._label.setPos(3, 3)
        self.refresh_active()

    def refresh_active(self) -> None:
        """활성 상태를 투명도로 구분한다."""
        self.setOpacity(1.0 if self.data.active else 0.4)

    def apply_data(self) -> None:
        """데이터 모델의 좌표와 이름을 그래픽 항목에 반영한다."""
        self.setPos(self.data.x, self.data.y)
        self.setRect(0, 0, self.data.width, self.data.height)
        self._label.setText(self.data.name)
        self.refresh_active()
        self.update()

    def set_measurement_result(self, status: str | None, text: str = "") -> None:
        """판정 상태를 테두리 색상과 ROI 라벨에 반영한다."""
        color = {
            "PASS": QColor("#00c853"),
            "FAIL": QColor("#ff1744"),
            "INVALID": QColor("#ff9100"),
            "OUT_OF_RANGE": QColor("#ff9100"),
            "MEASURED": QColor("#2979ff"),
            "INACTIVE": QColor("#9e9e9e"),
        }.get(status, self._base_color)
        pen = QPen(color, 3 if status else 2)
        pen.setCosmetic(True)
        self.setPen(pen)
        self._label.setBrush(QBrush(color))
        self._label.setText(
            f"{self.data.name}\n{text}" if status and text else self.data.name
        )
        self.update()

    def _handle_size(self) -> float:
        views = self.scene().views() if self.scene() is not None else []
        scale = views[0].transform().m11() if views else 1.0
        return self.HANDLE_PIXELS / max(scale, 0.01)

    def handle_rects(self) -> dict[str, QRectF]:
        rect = self.rect()
        size = self._handle_size()
        half = size / 2.0
        return {
            "top_left": QRectF(rect.left() - half, rect.top() - half, size, size),
            "top_right": QRectF(rect.right() - half, rect.top() - half, size, size),
            "bottom_left": QRectF(
                rect.left() - half,
                rect.bottom() - half,
                size,
                size,
            ),
            "bottom_right": QRectF(
                rect.right() - half,
                rect.bottom() - half,
                size,
                size,
            ),
        }

    def _handle_at(self, position: QPointF) -> str | None:
        for name, rect in self.handle_rects().items():
            if rect.contains(position):
                return name
        return None

    def boundingRect(self) -> QRectF:  # noqa: N802 - Qt API
        margin = self._handle_size() / 2.0 + 2.0
        return self.rect().adjusted(-margin, -margin, margin, margin)

    def shape(self) -> QPainterPath:
        path = QPainterPath()
        path.addRect(self.rect())
        for handle in self.handle_rects().values():
            path.addRect(handle)
        return path

    def paint(self, painter, option, widget=None) -> None:
        super().paint(painter, option, widget)
        painter.setPen(QPen(Qt.GlobalColor.white, 1))
        painter.setBrush(self.pen().color())
        for handle in self.handle_rects().values():
            painter.drawRect(handle)

    def hoverMoveEvent(self, event) -> None:  # noqa: N802 - Qt API
        handle = self._handle_at(event.pos())
        if handle in {"top_left", "bottom_right"}:
            self.setCursor(Qt.CursorShape.SizeFDiagCursor)
        elif handle in {"top_right", "bottom_left"}:
            self.setCursor(Qt.CursorShape.SizeBDiagCursor)
        else:
            self.setCursor(Qt.CursorShape.SizeAllCursor)
        super().hoverMoveEvent(event)

    def hoverLeaveEvent(self, event) -> None:  # noqa: N802 - Qt API
        self.unsetCursor()
        super().hoverLeaveEvent(event)

    def to_data(self) -> RoiData:
        position = self.pos()
        rect = self.rect()
        self.data.x = int(round(position.x()))
        self.data.y = int(round(position.y()))
        self.data.width = int(round(rect.width()))
        self.data.height = int(round(rect.height()))
        return self.data

    def itemChange(self, change, value):  # noqa: N802 - Qt API
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionChange:
            point = value
            max_x = max(0.0, self.image_width - self.rect().width())
            max_y = max(0.0, self.image_height - self.rect().height())
            return QPointF(
                min(max(point.x(), 0.0), max_x),
                min(max(point.y(), 0.0), max_y),
            )
        if (
            change == QGraphicsItem.GraphicsItemChange.ItemSelectedHasChanged
            and bool(value)
        ):
            self._selected_callback(self)
        return super().itemChange(change, value)

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        handle = self._handle_at(event.pos())
        if handle is not None:
            self._resizing = True
            self._active_handle = handle
            self._press_scene_position = event.scenePos()
            self._original_geometry = QRectF(
                self.pos().x(),
                self.pos().y(),
                self.rect().width(),
                self.rect().height(),
            )
            self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, False)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        if not self._resizing:
            super().mouseMoveEvent(event)
            return

        delta = event.scenePos() - self._press_scene_position
        original = self._original_geometry
        left = original.left()
        top = original.top()
        right = original.right()
        bottom = original.bottom()

        if self._active_handle in {"top_left", "bottom_left"}:
            left = min(
                max(original.left() + delta.x(), 0.0),
                original.right() - self.MINIMUM_SIZE,
            )
        if self._active_handle in {"top_right", "bottom_right"}:
            right = max(
                min(original.right() + delta.x(), float(self.image_width)),
                original.left() + self.MINIMUM_SIZE,
            )
        if self._active_handle in {"top_left", "top_right"}:
            top = min(
                max(original.top() + delta.y(), 0.0),
                original.bottom() - self.MINIMUM_SIZE,
            )
        if self._active_handle in {"bottom_left", "bottom_right"}:
            bottom = max(
                min(original.bottom() + delta.y(), float(self.image_height)),
                original.top() + self.MINIMUM_SIZE,
            )

        width = round(right - left)
        height = round(bottom - top)
        self.setRect(0, 0, width, height)
        self.setPos(round(left), round(top))
        self.update()
        event.accept()

    def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        if self._resizing:
            self._resizing = False
            self._active_handle = None
            self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
            self._changed_callback(self)
            event.accept()
            return
        super().mouseReleaseEvent(event)
        self._changed_callback(self)
