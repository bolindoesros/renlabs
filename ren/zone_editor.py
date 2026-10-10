"""Draw your own zones over the seating plan."""
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen

from jacket import config
from ren.plan import PlanLayout, ZoneRect, make_zones, next_zone_name
from ren.seating_view import SeatingPlanView, empty_seat_color
from ren.theme import Fonts, color, with_alpha

HANDLE_PX = 12  # grab distance around a corner
HANDLE_DRAW_PX = 5
ZONE_FILL_ALPHA = 60
CORNERS = ((0, 0), (1, 0), (1, 1), (0, 1))  # which bound each corner uses: (u0|u1, v0|v1)


def zone_color(index: int) -> QColor:
    return QColor(config.UI_BRAND_DOTS[index % len(config.UI_BRAND_DOTS)])


def new_zone(zones: tuple[ZoneRect, ...]) -> ZoneRect:
    """A fresh zone in the middle of the plan."""
    half = config.ZONE_NEW_SIZE / 2
    return ZoneRect(next_zone_name(zones), 0.5 - half, 0.5 - half, 0.5 + half, 0.5 + half)


class ZoneEditorView(SeatingPlanView):
    """Drag a zone to move it, drag a corner to resize, right-click to delete."""

    zones_changed = Signal(object)  # tuple[ZoneRect, ...], on release

    def __init__(self, fonts: Fonts) -> None:
        super().__init__(fonts)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._rects: tuple[ZoneRect, ...] = ()
        self._selected: int | None = None
        self._grab: tuple[str, int, tuple[float, float], ZoneRect] | None = None  # mode, zone, start uv, zone then
        self._corner = (0, 0)

    def set_zones(self, layout: PlanLayout, zones: tuple[ZoneRect, ...]) -> None:
        self._layout, self._rects = layout, zones
        if self._selected is not None and self._selected >= len(zones):
            self._selected = None
        self.update()

    def zones(self) -> tuple[ZoneRect, ...]:
        return self._rects

    def add_zone(self) -> None:
        self._rects = self._rects + (new_zone(self._rects),)
        self._selected = len(self._rects) - 1
        self.zones_changed.emit(self._rects)
        self.update()

    # --- geometry ---------------------------------------------------------------

    def zone_rect(self, zone: ZoneRect) -> QRectF:
        u0, v0, u1, v1 = zone.bounds
        return QRectF(self.point_at(u0, v1), self.point_at(u1, v0))

    def corner_point(self, zone: ZoneRect, corner: tuple[int, int]) -> QPointF:
        u0, v0, u1, v1 = zone.bounds
        return self.point_at((u0, u1)[corner[0]], (v0, v1)[corner[1]])

    def zone_at(self, point: QPointF) -> int | None:
        """The topmost zone under a point; later zones sit on top."""
        for index in reversed(range(len(self._rects))):
            if self.zone_rect(self._rects[index]).contains(point):
                return index
        return None

    def _corner_at(self, point: QPointF) -> tuple[int, int] | None:
        if self._selected is None:
            return None
        for corner in CORNERS:
            handle = self.corner_point(self._rects[self._selected], corner)
            if abs(handle.x() - point.x()) <= HANDLE_PX and abs(handle.y() - point.y()) <= HANDLE_PX:
                return corner
        return None

    # --- mouse ------------------------------------------------------------------

    def mousePressEvent(self, event) -> None:
        point = event.position()
        index = self.zone_at(point)
        if event.button() == Qt.MouseButton.RightButton:
            if index is not None:
                self._rects = self._rects[:index] + self._rects[index + 1:]
                self._selected = None
                self.zones_changed.emit(self._rects)
                self.update()
            return
        corner = self._corner_at(point)
        if corner is not None:
            self._corner = corner
            self._grab = ("corner", self._selected, self.uv_at(point), self._rects[self._selected])
        elif index is not None:
            self._selected = index
            self._grab = ("move", index, self.uv_at(point), self._rects[index])
        else:
            self._selected = None
        self.update()

    def mouseMoveEvent(self, event) -> None:
        if self._grab is None:
            return
        mode, index, (start_u, start_v), before = self._grab
        u, v = self.uv_at(event.position())
        if mode == "move":
            changed = before.moved_by(u - start_u, v - start_v)
        else:
            changed = self._resized(before, min(max(u, 0.0), 1.0), min(max(v, 0.0), 1.0))
        self._rects = self._rects[:index] + (changed,) + self._rects[index + 1:]
        self.update()

    def _resized(self, zone: ZoneRect, u: float, v: float) -> ZoneRect:
        """Move one corner; the opposite one stays put, and the zone never gets too small."""
        u0, v0, u1, v1 = zone.bounds
        least = config.ZONE_MIN_SIZE
        if self._corner[0]:
            u1 = max(u, u0 + least)
        else:
            u0 = min(u, u1 - least)
        if self._corner[1]:
            v1 = max(v, v0 + least)
        else:
            v0 = min(v, v1 - least)
        return ZoneRect(zone.name, u0, v0, u1, v1)

    def mouseReleaseEvent(self, event) -> None:
        grab, self._grab = self._grab, None
        if grab is not None and self._rects[grab[1]] != grab[3]:
            self.zones_changed.emit(self._rects)

    # --- painting ---------------------------------------------------------------

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.TextAntialiasing)
        painter.fillRect(self.rect(), color("background"))
        painter.setPen(QPen(color("hairline").darker(115), 1.2))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(self.plan_rect())
        if self._rects:
            for index, zone in enumerate(self._rects):
                self._paint_zone(painter, zone, index)
        else:
            self._paint_automatic(painter)
        self._paint_seat_dots(painter)
        self._paint_edges(painter)

    def _paint_zone(self, painter: QPainter, zone: ZoneRect, index: int) -> None:
        rect = self.zone_rect(zone)
        tint = zone_color(index)
        selected = index == self._selected
        painter.setPen(QPen(color("accent") if selected else tint.darker(130), 2.0 if selected else 1.4))
        painter.setBrush(with_alpha(tint, ZONE_FILL_ALPHA))
        painter.drawRect(rect)
        painter.setFont(self._fonts.text(config.UI_FONT_PX["small"], "medium"))
        painter.setPen(color("label"))
        painter.drawText(rect.adjusted(8, 4, -4, -4), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop, zone.name)
        if selected:
            painter.setPen(QPen(color("accent"), 1.5))
            painter.setBrush(color("background"))
            for corner in CORNERS:
                painter.drawEllipse(self.corner_point(zone, corner), HANDLE_DRAW_PX, HANDLE_DRAW_PX)

    def _paint_automatic(self, painter: QPainter) -> None:
        """The equal split used until you draw a zone."""
        painter.setFont(self._fonts.text(config.UI_FONT_PX["small"]))
        for zone in make_zones(self._layout):
            u0, v0, u1, v1 = zone.bounds
            rect = QRectF(self.point_at(u0, v1), self.point_at(u1, v0)).adjusted(3, 3, -3, -3)
            painter.setPen(QPen(color("hairline").darker(130), 1.0, Qt.PenStyle.DashLine))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(rect)
            painter.setPen(color("muted"))
            painter.drawText(rect.adjusted(8, 4, -4, -4), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop,
                             f"{zone.name} (auto)")

    def _paint_seat_dots(self, painter: QPainter) -> None:
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(empty_seat_color().darker(115))
        radius = self.seat_size() * 0.12
        for row, col in self._layout.seats():
            painter.drawEllipse(self.seat_point(row, col), radius, radius)
