"""Drag four corners onto the seating area."""
import cv2
import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFontMetrics, QImage, QMouseEvent, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QSizePolicy, QWidget

from jacket import config
from ren.plan import CORNER_NAMES, Calibration, PlanLayout, grid_lines
from ren.theme import Fonts, color, rounded
from ren.video_view import fit_rect, to_qimage

HANDLE_PX = 18
GRAB_RADIUS_PX = 22
CORNER_WORDS = ("back left", "back right", "front right", "front left")


class CalibrationView(QWidget):
    changed = Signal(object)  # the new Calibration, once a drag ends

    def __init__(self, fonts: Fonts) -> None:
        super().__init__()
        self._fonts = fonts
        self._image: QImage | None = None
        self._calibration = Calibration()
        self._layout = PlanLayout()
        self._people: list[tuple[float, float]] = []
        self._dragging: int | None = None
        self._hover: int | None = None
        self.setMouseTracking(True)
        self.setMinimumSize(480, 320)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def calibration(self) -> Calibration:
        return self._calibration

    def set_calibration(self, calibration: Calibration) -> None:
        if self._dragging is None:
            self._calibration = calibration
            self.update()

    def set_layout(self, layout: PlanLayout) -> None:
        self._layout = layout
        self.update()

    def show_frame(self, frame_bgr: np.ndarray, people: list[tuple[float, float]]) -> None:
        """People are shoulder points in image pixels."""
        self._image, self._people = to_qimage(frame_bgr), people
        self.update()

    # --- geometry ---------------------------------------------------------------

    def _target(self) -> QRectF:
        return fit_rect(self._image.size(), QRectF(self.rect()))

    def _to_widget(self, fx: float, fy: float, target: QRectF) -> QPointF:
        return QPointF(target.left() + fx * target.width(), target.top() + fy * target.height())

    def _handle_at(self, position: QPointF) -> int | None:
        if self._image is None:
            return None
        target = self._target()
        for index, (fx, fy) in enumerate(self._calibration.corners()):
            point = self._to_widget(fx, fy, target)
            if (point - position).manhattanLength() <= GRAB_RADIUS_PX * 1.4:
                return index
        return None

    # --- mouse ------------------------------------------------------------------

    def mousePressEvent(self, event: QMouseEvent) -> None:
        self._dragging = self._handle_at(event.position())

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._dragging is None:
            hover = self._handle_at(event.position())
            if hover != self._hover:
                self._hover = hover
                self.setCursor(Qt.CursorShape.SizeAllCursor if hover is not None else Qt.CursorShape.ArrowCursor)
                self.update()
            return
        target = self._target()
        fraction = (
            min(max((event.position().x() - target.left()) / target.width(), 0.0), 1.0),
            min(max((event.position().y() - target.top()) / target.height(), 0.0), 1.0),
        )
        moved = self._calibration.with_corner(self._dragging, fraction)
        if moved.is_valid():  # corners may not flip the quad
            self._calibration = moved
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self._dragging is not None:
            self._dragging = None
            self.changed.emit(self._calibration)

    # --- painting ---------------------------------------------------------------

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.TextAntialiasing)
        painter.fillRect(self.rect(), color("background"))
        if self._image is None:
            painter.setFont(self._fonts.text(config.UI_FONT_PX["body"]))
            painter.setPen(color("label"))
            painter.drawText(self.rect().adjusted(8, 8, -8, -8), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop, "waiting for camera...")
            return
        target = self._target()
        painter.save()
        painter.setClipPath(rounded(target, config.UI_RADIUS["viewport"] - 4))
        painter.drawImage(target, self._image)
        painter.restore()
        scale = target.width() / self._image.width()
        size = (self._image.width(), self._image.height())
        self._paint_grid(painter, target, scale, size)
        self._paint_people(painter, target, scale)
        self._paint_quad(painter, target)

    def _paint_grid(self, painter: QPainter, target: QRectF, scale: float, size: tuple[int, int]) -> None:
        for (x1, y1), (x2, y2) in grid_lines(self._layout, self._calibration, size):
            start = QPointF(target.left() + x1 * scale, target.top() + y1 * scale)
            end = QPointF(target.left() + x2 * scale, target.top() + y2 * scale)
            painter.setPen(QPen(QColor(0, 0, 0, 80), 3))
            painter.drawLine(start, end)
            painter.setPen(QPen(QColor(255, 255, 255, 200), 1))
            painter.drawLine(start, end)

    def _paint_people(self, painter: QPainter, target: QRectF, scale: float) -> None:
        painter.setBrush(QColor(config.UI_BRAND_DOTS[0]))
        painter.setPen(QPen(QColor(255, 255, 255), 2))
        for x, y in self._people:
            painter.drawEllipse(QPointF(target.left() + x * scale, target.top() + y * scale), 6, 6)

    def _paint_quad(self, painter: QPainter, target: QRectF) -> None:
        points = [self._to_widget(fx, fy, target) for fx, fy in self._calibration.corners()]
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(QColor(255, 255, 255, 220), 4))
        painter.drawPolygon(QPolygonF(points))
        painter.setPen(QPen(color("accent"), 1.6))
        painter.drawPolygon(QPolygonF(points))
        font = self._fonts.text(config.UI_FONT_PX["small"], "medium")
        painter.setFont(font)
        metrics = QFontMetrics(font)
        for index, point in enumerate(points):
            active = index in (self._dragging, self._hover)
            painter.setPen(QPen(color("accent"), 2))
            painter.setBrush(color("accent_soft") if active else color("background"))
            painter.drawEllipse(point, HANDLE_PX / 2, HANDLE_PX / 2)
            word = CORNER_WORDS[index]
            pill = QRectF(0, 0, metrics.horizontalAdvance(word) + 18, metrics.height() + 6)
            pill.moveTopLeft(QPointF(point.x() + 14, point.y() - pill.height() - 8))
            if index in (1, 2):  # keep labels on the picture at the right edge
                pill.moveRight(point.x() - 14)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(255, 255, 255, 235))
            painter.drawRoundedRect(pill, pill.height() / 2, pill.height() / 2)
            painter.setPen(color("text"))
            painter.drawText(pill, Qt.AlignmentFlag.AlignCenter, word)
