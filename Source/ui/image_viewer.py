"""원본 픽셀 좌표를 유지하는 확대/이동 영상 뷰어."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPoint, QRectF, Qt, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QKeyEvent,
    QMouseEvent,
    QPen,
    QPixmap,
    QWheelEvent,
)
from PySide6.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsLineItem,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
    QGraphicsView,
)

from imaging.roi import MAXIMUM_ROI_COUNT, RoiData, roi_from_points
from ui.roi_item import RoiItem


class ImageViewer(QGraphicsView):
    """QGraphicsView 기반 영상 표시 위젯."""

    pixelHovered = Signal(int, int, bool)
    zoomChanged = Signal(float)
    roiSelected = Signal(object)
    roisChanged = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self._pixmap_item = QGraphicsPixmapItem()
        self._scene.addItem(self._pixmap_item)
        self.setScene(self._scene)

        self._image_width = 0
        self._image_height = 0
        self._fit_mode = True
        self._roi_items: list[RoiItem] = []
        self._ri_grid_items: list[QGraphicsRectItem | QGraphicsSimpleTextItem] = []
        self._roi_guide_enabled = False
        self._roi_guide_root: QGraphicsRectItem | None = None
        self._roi_guide_horizontal: QGraphicsLineItem | None = None
        self._roi_guide_vertical: QGraphicsLineItem | None = None
        self._roi_guide_field: QGraphicsEllipseItem | None = None
        self._roi_drawing_enabled = True
        self._draw_start: tuple[int, int] | None = None
        self._draw_preview: QGraphicsRectItem | None = None
        self._automatic_minimum_roi_size: int | None = None
        self.setBackgroundBrush(Qt.GlobalColor.black)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setMouseTracking(True)
        self.viewport().setMouseTracking(True)

    @property
    def has_image(self) -> bool:
        return not self._pixmap_item.pixmap().isNull()

    @property
    def zoom_percent(self) -> float:
        return self.transform().m11() * 100.0

    def set_image(self, pixmap: QPixmap, width: int, height: int) -> None:
        self.clear_ri_grid()
        self.clear_rois()
        self._pixmap_item.setPixmap(pixmap)
        self._image_width = width
        self._image_height = height
        self._scene.setSceneRect(self._pixmap_item.boundingRect())
        self._update_roi_guide()
        self.fit_to_window()

    def update_pixmap(self, pixmap: QPixmap) -> None:
        """확대·이동 상태를 유지하면서 표시 영상만 교체한다."""
        self._pixmap_item.setPixmap(pixmap)

    def clear_image(self) -> None:
        self._clear_roi_guide()
        self.clear_ri_grid()
        self.clear_rois()
        self._pixmap_item.setPixmap(QPixmap())
        self._image_width = 0
        self._image_height = 0
        self.resetTransform()
        self.pixelHovered.emit(0, 0, False)

    def set_roi_guide_enabled(self, enabled: bool) -> None:
        """Show or hide the display-only center and 0.7-field guide."""
        self._roi_guide_enabled = bool(enabled)
        self._update_roi_guide()

    def _clear_roi_guide(self) -> None:
        if self._roi_guide_root is not None:
            self._scene.removeItem(self._roi_guide_root)
        self._roi_guide_root = None
        self._roi_guide_horizontal = None
        self._roi_guide_vertical = None
        self._roi_guide_field = None

    def _update_roi_guide(self) -> None:
        self._clear_roi_guide()
        if (
            not self._roi_guide_enabled
            or not self.has_image
            or self._image_width <= 0
            or self._image_height <= 0
        ):
            return

        width = float(self._image_width)
        height = float(self._image_height)
        center_x = width / 2.0
        center_y = height / 2.0
        field_radius = 0.7 * 0.5 * float(np.hypot(width, height))

        root = QGraphicsRectItem(0.0, 0.0, width, height)
        root.setPen(QPen(Qt.PenStyle.NoPen))
        root.setBrush(QBrush(Qt.BrushStyle.NoBrush))
        root.setFlag(
            QGraphicsItem.GraphicsItemFlag.ItemClipsChildrenToShape,
            True,
        )
        root.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        root.setZValue(5)

        cross_pen = QPen(QColor(0, 255, 255, 190), 1.0, Qt.PenStyle.DashLine)
        cross_pen.setCosmetic(True)
        field_pen = QPen(QColor(255, 215, 0, 210), 1.0, Qt.PenStyle.DashLine)
        field_pen.setCosmetic(True)

        horizontal = QGraphicsLineItem(0.0, center_y, width, center_y, root)
        vertical = QGraphicsLineItem(center_x, 0.0, center_x, height, root)
        field = QGraphicsEllipseItem(
            QRectF(
                center_x - field_radius,
                center_y - field_radius,
                field_radius * 2.0,
                field_radius * 2.0,
            ),
            root,
        )
        for item in (horizontal, vertical, field):
            item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        horizontal.setPen(cross_pen)
        vertical.setPen(cross_pen)
        field.setPen(field_pen)
        field.setBrush(QBrush(Qt.BrushStyle.NoBrush))

        self._scene.addItem(root)
        self._roi_guide_root = root
        self._roi_guide_horizontal = horizontal
        self._roi_guide_vertical = vertical
        self._roi_guide_field = field

    def add_roi(self, data: RoiData) -> RoiItem:
        if len(self._roi_items) >= MAXIMUM_ROI_COUNT:
            raise ValueError(
                f"ROI는 최대 {MAXIMUM_ROI_COUNT}개까지 생성할 수 있습니다."
            )
        data.validate(self._image_width, self._image_height)
        item = RoiItem(
            data,
            self._image_width,
            self._image_height,
            self._roi_changed,
            self._roi_selected,
            self._automatic_minimum_roi_size or RoiItem.DEFAULT_MINIMUM_SIZE,
        )
        self._scene.addItem(item)
        self._roi_items.append(item)
        item.setSelected(True)
        self.roisChanged.emit()
        return item

    def _next_roi_number(self) -> int:
        used = {item.data.number for item in self._roi_items}
        return next(
            number
            for number in range(1, MAXIMUM_ROI_COUNT + 1)
            if number not in used
        )

    def set_automatic_minimum_roi_size(self, size: int | None) -> None:
        """Enable mode-specific minimum creation/resize size enforcement."""
        self._automatic_minimum_roi_size = (
            max(1, int(size)) if size is not None else None
        )
        minimum = self._automatic_minimum_roi_size or RoiItem.DEFAULT_MINIMUM_SIZE
        for item in self._roi_items:
            item.set_minimum_size(minimum)

    def _drag_geometry(
        self,
        start: tuple[int, int],
        end: tuple[int, int],
    ) -> tuple[int, int, int, int]:
        """Return bounded drag geometry, expanding to the active mode minimum."""
        start_x, start_y = start
        end_x, end_y = end
        minimum = self._automatic_minimum_roi_size
        width = abs(end_x - start_x) + 1
        height = abs(end_y - start_y) + 1
        if minimum is None:
            return min(start_x, end_x), min(start_y, end_y), width, height

        width = min(self._image_width, max(width, minimum))
        height = min(self._image_height, max(height, minimum))
        left = start_x if end_x >= start_x else start_x - width + 1
        top = start_y if end_y >= start_y else start_y - height + 1
        left = min(max(left, 0), self._image_width - width)
        top = min(max(top, 0), self._image_height - height)
        return left, top, width, height

    def selected_roi(self) -> RoiItem | None:
        selected = [item for item in self._roi_items if item.isSelected()]
        return selected[0] if selected else None

    def roi_data(self) -> list[RoiData]:
        return [item.to_data() for item in self._roi_items]

    def delete_selected_roi(self) -> bool:
        item = self.selected_roi()
        if item is None:
            return False
        self._scene.removeItem(item)
        self._roi_items.remove(item)
        self.roisChanged.emit()
        self.roiSelected.emit(None)
        return True

    def clear_rois(self) -> None:
        self._cancel_drawing()
        for item in self._roi_items:
            self._scene.removeItem(item)
        self._roi_items.clear()
        self.roisChanged.emit()
        self.roiSelected.emit(None)

    def replace_rois(self, rois: list[RoiData]) -> None:
        self.clear_rois()
        for data in rois:
            self.add_roi(data)

    def set_roi_measurement_result(
        self,
        roi_number: int,
        status: str | None,
        text: str = "",
    ) -> None:
        for item in self._roi_items:
            if item.data.number == roi_number:
                item.set_measurement_result(status, text)
                return

    def clear_measurement_results(self) -> None:
        for item in self._roi_items:
            item.set_measurement_result(None)

    def set_roi_drawing_enabled(self, enabled: bool) -> None:
        self._roi_drawing_enabled = enabled
        for item in self._roi_items:
            item.setVisible(enabled)

    def show_ri_grid(
        self,
        cells: list[
            tuple[tuple[int, int, int, int], str, QColor]
        ],
    ) -> None:
        self.clear_ri_grid()
        for (x, y, width, height), label, color in cells:
            rectangle = QGraphicsRectItem(x, y, width, height)
            rectangle.setPen(QPen(color, 1.5))
            fill = QColor(color)
            fill.setAlpha(35)
            rectangle.setBrush(QBrush(fill))
            rectangle.setZValue(25)
            self._scene.addItem(rectangle)
            self._ri_grid_items.append(rectangle)

            text = QGraphicsSimpleTextItem(label)
            text.setBrush(QBrush(color))
            text.setPos(x + 2, y + 2)
            text.setZValue(26)
            self._scene.addItem(text)
            self._ri_grid_items.append(text)

    def clear_ri_grid(self) -> None:
        for item in self._ri_grid_items:
            self._scene.removeItem(item)
        self._ri_grid_items.clear()

    def _roi_changed(self, item: RoiItem) -> None:
        item.to_data()
        self.roisChanged.emit()
        self.roiSelected.emit(item.data)

    def _roi_selected(self, item: RoiItem) -> None:
        for other in self._roi_items:
            if other is not item:
                other.setSelected(False)
        self.roiSelected.emit(item.to_data())

    def fit_to_window(self) -> None:
        if not self.has_image:
            return
        self._fit_mode = True
        self.resetTransform()
        self.fitInView(
            self._pixmap_item,
            Qt.AspectRatioMode.KeepAspectRatio,
        )
        self.zoomChanged.emit(self.zoom_percent)

    def actual_size(self) -> None:
        if not self.has_image:
            return
        self._fit_mode = False
        self.resetTransform()
        self.centerOn(self._pixmap_item)
        self.zoomChanged.emit(100.0)

    def image_position(self, viewport_position: QPoint) -> tuple[int, int] | None:
        """뷰포트 좌표를 원본 영상 정수 픽셀 좌표로 변환한다."""
        if not self.has_image:
            return None
        scene_position = self.mapToScene(viewport_position)
        item_position = self._pixmap_item.mapFromScene(scene_position)
        x = int(np.floor(item_position.x()))
        y = int(np.floor(item_position.y()))
        if 0 <= x < self._image_width and 0 <= y < self._image_height:
            return x, y
        return None

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802 - Qt API
        if not self.has_image or event.angleDelta().y() == 0:
            super().wheelEvent(event)
            return

        factor = 1.25 if event.angleDelta().y() > 0 else 0.8
        next_zoom = self.zoom_percent * factor
        if not 5.0 <= next_zoom <= 3200.0:
            return

        self._fit_mode = False
        self.scale(factor, factor)
        self.zoomChanged.emit(self.zoom_percent)
        event.accept()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt API
        position = self.image_position(event.position().toPoint())
        if position is None:
            self.pixelHovered.emit(0, 0, False)
        else:
            self.pixelHovered.emit(position[0], position[1], True)
        if self._draw_start is not None and position is not None:
            left, top, width, height = self._drag_geometry(
                self._draw_start, position
            )
            if self._draw_preview is not None:
                self._draw_preview.setRect(left, top, width, height)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt API
        if (
            event.button() == Qt.MouseButton.LeftButton
            and self.has_image
            and self._roi_drawing_enabled
            and len(self._roi_items) < MAXIMUM_ROI_COUNT
            and not self._roi_at(event.position().toPoint())
        ):
            position = self.image_position(event.position().toPoint())
            if position is not None:
                self._draw_start = position
                self._draw_preview = QGraphicsRectItem(
                    position[0], position[1], 1, 1
                )
                self._draw_preview.setPen(QPen(QColor("#ffffff"), 1))
                self._draw_preview.setZValue(20)
                self._scene.addItem(self._draw_preview)
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt API
        if (
            self._draw_start is not None
            and event.button() == Qt.MouseButton.LeftButton
        ):
            position = self.image_position(event.position().toPoint())
            start = self._draw_start
            if self._draw_preview is not None:
                self._scene.removeItem(self._draw_preview)
                self._draw_preview = None
            self._draw_start = None
            if position is not None:
                if self._automatic_minimum_roi_size is not None:
                    left, top, width, height = self._drag_geometry(start, position)
                    number = self._next_roi_number()
                    roi = RoiData(
                        number,
                        f"ROI {number}",
                        left,
                        top,
                        width,
                        height,
                    )
                else:
                    roi = roi_from_points(
                        self._next_roi_number(),
                        start[0],
                        start[1],
                        position[0],
                        position[1],
                        self._image_width,
                        self._image_height,
                    )
                minimum = self._automatic_minimum_roi_size or 4
                if roi.width >= minimum and roi.height >= minimum:
                    self.add_roi(roi)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802 - Qt API
        if event.key() == Qt.Key.Key_Delete and self.delete_selected_roi():
            event.accept()
            return
        super().keyPressEvent(event)

    def _roi_at(self, viewport_position: QPoint) -> bool:
        item = self.itemAt(viewport_position)
        while item is not None:
            if isinstance(item, RoiItem):
                return True
            item = item.parentItem()
        return False

    def _cancel_drawing(self) -> None:
        self._draw_start = None
        if self._draw_preview is not None:
            self._scene.removeItem(self._draw_preview)
            self._draw_preview = None

    def leaveEvent(self, event) -> None:  # noqa: N802 - Qt API
        self.pixelHovered.emit(0, 0, False)
        super().leaveEvent(event)

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt API
        super().resizeEvent(event)
        if self._fit_mode:
            self.fit_to_window()
