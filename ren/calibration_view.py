"""Drag four corners onto the seating area, then the inner lines onto the seat borders."""
from dataclasses import replace

import cv2
import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFontMetrics, QImage, QMouseEvent, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QSizePolicy, QWidget

from jacket import config
from ren.plan import Calibration, PlanLayout, even_lines, grid_lines, image_to_plan, lines_ordered, plan_to_image
from ren.theme import Fonts, color, rounded
from ren.video_view import fit_rect, to_qimage

HANDLE_PX = 18
GRAB_RADIUS_PX = 22
LINE_HANDLE_PX = 9
LINE_GRAB_PX = 12
LINE_FIELDS = {"col": "col_lines", "row": "row_lines"}
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
        self._dragging: tuple | None = None  # what is held: ("corner", i) or (axis, line, end); end None is the middle
        self._hover: tuple | None = None
        self._grab_from: tuple[float, float, Calibration] | None = None  # plan spot and calibration at the press
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

    def _image_size(self) -> tuple[int, int]:
        return self._image.width(), self._image.height()

    def _plan_at(self, position: QPointF) -> tuple[float, float]:
        """A widget point in even plan units (before the inner lines bend it)."""
        target = self._target()
        scale = target.width() / self._image.width()
        pixel = np.float32([[[(position.x() - target.left()) / scale, (position.y() - target.top()) / scale]]])
        u, v = cv2.perspectiveTransform(pixel, image_to_plan(self._calibration, self._image_size()))[0, 0]
        return float(u), float(v)

    def line_handles(self) -> list[tuple[tuple, QPointF]]:
        """Both ends and the middle of every inner line, in widget pixels."""
        spots = []
        for index, (front, back) in enumerate(self._calibration.cols_for(self._layout)):
            spots += [(("col", index, 0), (front, 0.0)), (("col", index, 1), (back, 1.0)),
                      (("col", index, None), ((front + back) / 2, 0.5))]
        for index, (left, right) in enumerate(self._calibration.rows_for(self._layout)):
            spots += [(("row", index, 0), (0.0, left)), (("row", index, 1), (1.0, right)),
                      (("row", index, None), (0.5, (left + right) / 2))]
        if not spots:
            return []
        target = self._target()
        scale = target.width() / self._image.width()
        points = np.float32([uv for _, uv in spots]).reshape(-1, 1, 2)
        mapped = cv2.perspectiveTransform(points, plan_to_image(self._calibration, self._image_size())).reshape(-1, 2)
        return [(key, QPointF(target.left() + x * scale, target.top() + y * scale)) for (key, _), (x, y) in zip(spots, mapped)]

    def _handle_at(self, position: QPointF) -> tuple | None:
        """Corners win over line handles."""
        if self._image is None:
            return None
        target = self._target()
        for index, (fx, fy) in enumerate(self._calibration.corners()):
            point = self._to_widget(fx, fy, target)
            if (point - position).manhattanLength() <= GRAB_RADIUS_PX * 1.4:
                return ("corner", index)
        for key, point in self.line_handles():
            if (point - position).manhattanLength() <= LINE_GRAB_PX * 1.4:
                return key
        return None

    def _lines(self, calibration: Calibration, axis: str) -> tuple:
        return calibration.cols_for(self._layout) if axis == "col" else calibration.rows_for(self._layout)

    def _with_line(self, calibration: Calibration, axis: str, index: int, line: tuple[float, float]) -> Calibration:
        lines = list(self._lines(calibration, axis))
        lines[index] = line
        return replace(calibration, **{LINE_FIELDS[axis]: tuple(lines)})

    def _line_dragged(self, position: QPointF) -> Calibration:
        axis, index, end = self._dragging
        start_u, start_v, before = self._grab_from
        u, v = self._plan_at(position)
        spot, start = (u, start_u) if axis == "col" else (v, start_v)
        line = self._lines(before, axis)[index]
        if end is None:
            line = (line[0] + spot - start, line[1] + spot - start)
        else:
            line = (spot, line[1]) if end == 0 else (line[0], spot)
        return self._with_line(before, axis, index, line)

    def _reset_line(self, axis: str, index: int) -> None:
        """Put one inner line back at its even spot, if the others leave room."""
        parts = (self._layout.cols if axis == "col" else self._layout.rows)
        reset = self._with_line(self._calibration, axis, index, even_lines(parts)[index])
        if lines_ordered(self._lines(reset, axis)) and reset != self._calibration:
            self._calibration = reset
            self.update()
            self.changed.emit(reset)

    # --- mouse ------------------------------------------------------------------

    def mousePressEvent(self, event: QMouseEvent) -> None:
        held = self._handle_at(event.position())
        if event.button() == Qt.MouseButton.RightButton:
            if held is not None and held[0] != "corner":
                self._reset_line(held[0], held[1])
            return
        self._dragging = held
        if held is not None:
            self._grab_from = (*self._plan_at(event.position()), self._calibration)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._dragging is None:
            hover = self._handle_at(event.position())
            if hover != self._hover:
                self._hover = hover
                self.setCursor(self._cursor_for(hover))
                self.update()
            return
        if self._dragging[0] == "corner":
            target = self._target()
            fraction = (
                min(max((event.position().x() - target.left()) / target.width(), 0.0), 1.0),
                min(max((event.position().y() - target.top()) / target.height(), 0.0), 1.0),
            )
            moved = self._calibration.with_corner(self._dragging[1], fraction)
            if not moved.is_valid():  # corners may not flip the quad
                return
        else:
            moved = self._line_dragged(event.position())
            if not lines_ordered(self._lines(moved, self._dragging[0])):  # lines may not cross or leave the quad
                return
        self._calibration = moved
        self.update()

    @staticmethod
    def _cursor_for(hover: tuple | None) -> Qt.CursorShape:
        if hover is None:
            return Qt.CursorShape.ArrowCursor
        return {"corner": Qt.CursorShape.SizeAllCursor, "col": Qt.CursorShape.SizeHorCursor,
                "row": Qt.CursorShape.SizeVerCursor}[hover[0]]

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self._dragging is not None:
            before = self._grab_from[2]
            self._dragging, self._grab_from = None, None
            if self._calibration != before:
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
        self._paint_line_handles(painter)
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

    def _paint_line_handles(self, painter: QPainter) -> None:
        for key, point in self.line_handles():
            active = key in (self._dragging, self._hover)
            painter.setPen(QPen(color("accent"), 1.6))
            painter.setBrush(color("accent_soft") if active else QColor(255, 255, 255, 235))
            painter.drawEllipse(point, LINE_HANDLE_PX / 2, LINE_HANDLE_PX / 2)

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
            active = ("corner", index) in (self._dragging, self._hover)
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
