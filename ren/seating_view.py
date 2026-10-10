"""The seating plan, zones and vent aim."""
import math

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFontMetrics, QLinearGradient, QPainter, QPainterPath, QPen, QPolygonF
from PySide6.QtWidgets import QSizePolicy, QWidget

from jacket import config
from ren.decision import Decision, VentSettings
from ren.engine import ViewSettings
from ren.plan import PlanLayout, SeatingScene, Zone
from ren.theme import Fonts, color, paint_soft_shadow, state_color, with_alpha

MARGIN_PX = 30
BEAM_HALF_ANGLE_DEG = 16.0
BEAM_ALPHA = (45, 130)  # at the vent, at the landing spot
SEAT_DOT = 0.60  # occupied seat diameter, in cells
EMPTY_DOT = 0.14  # empty seats are small grey dots
ZONE_RADIUS_PX = 16
SHUT_TILT_DEG = config.VENT_CLOSED_TILT_DEG - 5  # beyond this the flaps count as shut
BLADE_COUNT = 8  # the grille has eight blades


def empty_seat_color() -> QColor:
    return color("hairline").darker(108)


def air_line() -> QColor:
    """Airflow green, dark enough for thin lines and text."""
    return color("air").darker(150)


class SeatingPlanView(QWidget):
    def __init__(self, fonts: Fonts) -> None:
        super().__init__()
        self._fonts = fonts
        self._layout = PlanLayout()
        self._scene: SeatingScene | None = None
        self._decision: Decision | None = None
        self._zones: list[Zone] = []
        self._vent = VentSettings()
        self._layers = ViewSettings()
        self.setMinimumSize(300, 240)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_state(
        self, layout: PlanLayout, scene: SeatingScene | None, decision: Decision | None,
        zones: list[Zone], vent: VentSettings, layers: ViewSettings = ViewSettings(),
    ) -> None:
        self._layout, self._scene, self._decision, self._zones, self._vent = layout, scene, decision, zones, vent
        self._layers = layers
        self.update()

    # --- geometry ---------------------------------------------------------------

    def plan_rect(self) -> QRectF:
        """The room, centred, with square cells."""
        available = QRectF(self.rect()).adjusted(MARGIN_PX, MARGIN_PX + 8, -MARGIN_PX, -MARGIN_PX - 8)
        cell = min(available.width() / self._layout.cols, available.height() / self._layout.rows)
        width, height = cell * self._layout.cols, cell * self._layout.rows
        return QRectF(
            available.left() + (available.width() - width) / 2,
            available.top() + (available.height() - height) / 2, width, height,
        )

    def cell_rect(self, row: int, col: int) -> QRectF:
        """Row 0 is the front, drawn at the bottom."""
        plan = self.plan_rect()
        cell = plan.width() / self._layout.cols
        return QRectF(plan.left() + col * cell, plan.bottom() - (row + 1) * cell, cell, cell)

    def point_at(self, u: float, v: float) -> QPointF:
        plan = self.plan_rect()
        return QPointF(plan.left() + u * plan.width(), plan.bottom() - v * plan.height())

    # --- painting ---------------------------------------------------------------

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.TextAntialiasing)
        painter.fillRect(self.rect(), color("background"))
        if self._layers.show_zones:
            self._paint_zones(painter)  # tints sit under the seats
        if self._layers.show_seats:
            self._paint_seats(painter)
        if self._layers.show_zones:
            self._paint_shares(painter)
        if self._decision is not None:
            self._paint_vent(painter, self._decision)
        self._paint_edges(painter)

    def _paint_seats(self, painter: QPainter) -> None:
        painter.setPen(Qt.PenStyle.NoPen)
        for row in range(self._layout.rows):
            for col in range(self._layout.cols):
                cell = self.cell_rect(row, col)
                reading = self._scene.seats.get((row, col)) if self._scene else None
                if reading is None:
                    painter.setBrush(empty_seat_color())
                    radius = cell.width() * EMPTY_DOT / 2
                else:
                    painter.setBrush(state_color(reading.state))
                    radius = cell.width() * SEAT_DOT / 2
                painter.drawEllipse(cell.center(), radius, radius)

    def _zone_rect(self, zone: Zone) -> QRectF:
        rects = [self.cell_rect(row, col) for row, col in zone.seats()]
        combined = rects[0]
        for rect in rects[1:]:
            combined = combined.united(rect)
        return combined

    def _targeted(self, zone: Zone) -> bool:
        return self._decision is not None and self._decision.target_zone == zone.name

    def _paint_zones(self, painter: QPainter) -> None:
        for zone in self._zones:
            rect = self._zone_rect(zone).adjusted(3, 3, -3, -3)
            if self._targeted(zone):
                painter.setPen(QPen(air_line(), 1.6))
                painter.setBrush(with_alpha(color("air"), 46))
            else:
                painter.setPen(QPen(color("hairline"), 1.0))
                painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRoundedRect(rect, ZONE_RADIUS_PX, ZONE_RADIUS_PX)

    def _paint_shares(self, painter: QPainter) -> None:
        if self._decision is None:
            return
        for zone in self._zones:
            share = self._decision.zone_shares.get(zone.name, 0.0)
            if share >= 0.005:
                rect = self._zone_rect(zone).adjusted(3, 3, -3, -3)
                self._paint_share(painter, rect, f"{share:.0%}", self._targeted(zone))

    def _paint_share(self, painter: QPainter, zone: QRectF, text: str, targeted: bool) -> None:
        """Zone air share on its top edge, haloed."""
        font = self._fonts.text(config.UI_FONT_PX["small"] + 1, "medium")
        bounds = QFontMetrics(font).tightBoundingRect(text)
        baseline = zone.top() - (bounds.top() + bounds.height() / 2)  # digits centred on the edge
        outline = QPainterPath()
        outline.addText(QPointF(zone.left() + ZONE_RADIUS_PX, baseline), font, text)
        halo = QPen(color("background"), 6, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
        painter.strokePath(outline, halo)
        painter.fillPath(outline, air_line() if targeted else color("muted"))

    def _paint_edges(self, painter: QPainter) -> None:
        plan = self.plan_rect()
        painter.setFont(self._fonts.text(config.UI_FONT_PX["small"], "medium"))
        painter.setPen(color("muted"))
        centre = plan.center().x()
        width = QFontMetrics(painter.font()).horizontalAdvance
        painter.drawText(QPointF(centre - width("back") / 2, plan.top() - 22), "back")
        painter.drawText(QPointF(centre - width("front") / 2, plan.bottom() + 30), "front")

    def _paint_vent(self, painter: QPainter, decision: Decision) -> None:
        center = self.point_at(self._vent.u, self._vent.v)
        plan = self.plan_rect()
        cell = plan.width() / self._layout.cols
        azimuth = decision.plan_azimuth_deg
        open_amount = decision.tilt_deg < SHUT_TILT_DEG
        if open_amount and self._layers.show_beam:
            reach_tilt = min(decision.tilt_deg, self._vent.tilt_max_deg)  # closing flaps fade, not fly out
            landing = self._landing_point(center, azimuth, reach_tilt, cell)
            self._paint_beam(painter, center, landing, cell, decision.reachable, self._fade(decision.tilt_deg))
        if self._layers.show_vent:
            self._paint_glyph(painter, center, azimuth, min(cell * 0.62, 46.0), open_amount)

    def _fade(self, tilt: float) -> float:
        """1 while aimed, falling to 0 as flaps close."""
        span = SHUT_TILT_DEG - self._vent.tilt_max_deg
        return 1.0 if tilt <= self._vent.tilt_max_deg else max(0.0, (SHUT_TILT_DEG - tilt) / span)

    def _landing_point(self, center: QPointF, azimuth: float, tilt: float, cell: float) -> QPointF:
        """Where the slanted air reaches head height, in pixels."""
        reach_m = self._vent.drop_m * math.tan(math.radians(tilt))
        east, north = math.sin(math.radians(azimuth)), math.cos(math.radians(azimuth))
        return QPointF(
            center.x() + east * reach_m * cell / self._vent.seat_width_m,
            center.y() - north * reach_m * cell / self._vent.row_depth_m,
        )

    def _paint_beam(
        self, painter: QPainter, vent: QPointF, landing: QPointF, cell: float, reachable: bool, fade: float = 1.0
    ) -> None:
        dx, dy = landing.x() - vent.x(), landing.y() - vent.y()
        length = math.hypot(dx, dy)
        half_width = max(cell * 0.42, length * math.tan(math.radians(BEAM_HALF_ANGLE_DEG)))
        if length < 1e-6:
            normal = QPointF(1.0, 0.0)
        else:
            normal = QPointF(-dy / length, dx / length)
        wedge = QPolygonF([
            vent,
            QPointF(landing.x() + normal.x() * half_width, landing.y() + normal.y() * half_width),
            QPointF(landing.x() - normal.x() * half_width, landing.y() - normal.y() * half_width),
        ])
        cool = color("air") if reachable else color("error")
        gradient = QLinearGradient(vent, landing)
        gradient.setColorAt(0.0, QColor(cool.red(), cool.green(), cool.blue(), int(BEAM_ALPHA[0] * fade)))
        gradient.setColorAt(1.0, QColor(cool.red(), cool.green(), cool.blue(), int(BEAM_ALPHA[1] * fade)))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(gradient)
        painter.drawPolygon(wedge)
        ring_color = with_alpha(cool.darker(130), int(255 * fade))
        painter.setPen(QPen(ring_color, 1.6, Qt.PenStyle.DashLine))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(landing, half_width * 0.85, half_width * 0.85)

    def _paint_glyph(self, painter: QPainter, center: QPointF, azimuth: float, size: float, open_amount: bool) -> None:
        """The grille from above, turned to the airflow."""
        painter.save()
        painter.translate(center)
        painter.rotate(azimuth)
        half = size / 2
        body = QRectF(-half, -half, size, size)
        corner = size * 0.2
        paint_soft_shadow(painter, body, corner)
        painter.setPen(QPen(color("label"), 1.3))
        painter.setBrush(color("background") if open_amount else color("hairline"))
        painter.drawRoundedRect(body, corner, corner)
        if open_amount:
            painter.setPen(QPen(color("switch"), 1.0))
            inner = half - corner * 0.6
            for index in range(BLADE_COUNT):
                y = -half + (index + 0.5) * size / BLADE_COUNT
                painter.drawLine(QPointF(-inner, y), QPointF(inner, y))
            arrow = QPolygonF([QPointF(0, -half - 10), QPointF(-6, -half - 2), QPointF(6, -half - 2)])
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(air_line())
            painter.drawPolygon(arrow)
        painter.restore()
