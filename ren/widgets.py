"""Small self-painted controls in the soft, pastel style."""
from PySide6.QtCore import QPointF, QRect, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QFontMetrics, QPainter, QPen
from PySide6.QtWidgets import QAbstractButton, QComboBox, QFrame, QHBoxLayout, QLabel, QSizePolicy, QVBoxLayout, QWidget

from jacket import config
from ren.theme import Fonts, color, paint_soft_shadow, state_color


DROPDOWN_WIDTH_PX = 210
STEPPER_VALUE_PX = 118


class Hairline(QFrame):
    """A 1px rule between rows."""

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("hairline")
        self.setFixedHeight(1)


class Card(QWidget):
    """A rounded white card with a soft shadow."""

    def __init__(self) -> None:
        super().__init__()
        margin = config.UI_CARD_MARGIN_PX
        self._frame = QFrame()
        self._frame.setObjectName("card")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(margin, margin - 4, margin, margin + 4)  # shadows fall down
        outer.addWidget(self._frame)

    def body(self) -> QFrame:
        return self._frame

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        paint_soft_shadow(painter, QRectF(self._frame.geometry()), config.UI_RADIUS["card"])


class DotMark(QWidget):
    """The brand mark: four pastel dots in a square."""

    DOT_PX = 9.0
    GAP_PX = 3.0

    def __init__(self) -> None:
        super().__init__()
        side = int(2 * self.DOT_PX + self.GAP_PX) + 2
        self.setFixedSize(side, side)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        step = self.DOT_PX + self.GAP_PX
        for index, hex_color in enumerate(config.UI_BRAND_DOTS):
            row, col = divmod(index, 2)
            painter.setBrush(QColor(hex_color))
            painter.drawEllipse(QRectF(1 + col * step, 1 + row * step, self.DOT_PX, self.DOT_PX))


DOT_COLORS = {"live": "air", "idle": "switch", "error": "error"}


class StatusDot(QWidget):
    """A small dot: green live, grey idle, red error."""

    SIZE_PX = 10

    def __init__(self) -> None:
        super().__init__()
        self._state = "idle"
        self.setFixedSize(self.SIZE_PX + 8, self.SIZE_PX + 8)

    def state(self) -> str:
        return self._state

    def set_state(self, state: str) -> None:
        if state not in DOT_COLORS:
            raise ValueError(f"unknown dot state {state!r}")
        if state != self._state:
            self._state = state
            self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        dot = color(DOT_COLORS[self._state])
        centre = QRectF(self.rect()).center()
        if self._state == "live":  # a soft ring reads as "on air"
            ring = QColor(dot)
            ring.setAlpha(70)
            painter.setBrush(ring)
            painter.drawEllipse(centre, self.SIZE_PX / 2 + 3, self.SIZE_PX / 2 + 3)
        painter.setBrush(dot)
        painter.drawEllipse(centre, self.SIZE_PX / 2, self.SIZE_PX / 2)


class _FlatButton(QAbstractButton):
    """Shared cursor, focus and hover handling."""

    def __init__(self, text: str, fonts: Fonts) -> None:
        super().__init__()
        self._fonts = fonts
        self._hovered = False
        self._keyboard_focus = False  # Tab focus only; nothing underlined at startup
        self.setText(text)
        self.setCheckable(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.TabFocus)

    def enterEvent(self, event) -> None:
        self._hovered = True
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:
        self._hovered = False
        self.update()
        super().leaveEvent(event)

    def focusInEvent(self, event) -> None:
        self._keyboard_focus = event.reason() in (
            Qt.FocusReason.TabFocusReason, Qt.FocusReason.BacktabFocusReason
        )
        self.update()
        super().focusInEvent(event)

    def focusOutEvent(self, event) -> None:
        self._keyboard_focus = False
        self.update()
        super().focusOutEvent(event)


class SwitchRow(_FlatButton):
    """A ruled row: label left, switch right."""

    TRACK = (40.0, 24.0)

    def __init__(self, text: str, fonts: Fonts) -> None:
        super().__init__(text, fonts)
        self.setFixedHeight(config.UI_ROW_HEIGHT_PX)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def sizeHint(self) -> QSize:
        return QSize(200, config.UI_ROW_HEIGHT_PX)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.TextAntialiasing)
        width, height = self.TRACK
        track = QRectF(self.width() - width - 2, (self.height() - height) / 2, width, height)
        on = self.isChecked()
        if not self.isEnabled():
            fill = color("hairline")
        elif on:
            fill = QColor(config.UI_BRAND_DOTS[0])  # pastel blue
        else:
            fill = color("switch") if self._hovered else color("hairline")
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(fill)
        painter.drawRoundedRect(track, height / 2, height / 2)
        knob = height - 6
        x = track.right() - knob - 3 if on else track.left() + 3
        painter.setBrush(color("background"))
        painter.drawEllipse(QRectF(x, track.top() + 3, knob, knob))

        text_font = self._fonts.text(config.UI_FONT_PX["body"])
        painter.setFont(text_font)
        painter.setPen(color("text" if self.isEnabled() else "muted"))
        painter.drawText(
            QRect(4, 0, int(track.left()) - 16, self.height()),
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, self.text(),
        )
        if self._keyboard_focus:  # underline the label
            metrics = QFontMetrics(text_font)
            y = (self.height() + metrics.height()) // 2 + 1
            painter.drawLine(0, y, metrics.horizontalAdvance(self.text()), y)
        painter.setPen(color("hairline"))
        painter.drawLine(0, self.height() - 1, self.width(), self.height() - 1)


class NavLink(_FlatButton):
    """A pill tab, tinted when active."""

    def __init__(self, text: str, fonts: Fonts, align_left: bool = False) -> None:
        super().__init__(text, fonts)
        self.setAutoExclusive(True)
        self._align_left = align_left

    def _font(self):
        return self._fonts.text(config.UI_FONT_PX["body"] - 1, "medium")

    def sizeHint(self) -> QSize:
        return QSize(QFontMetrics(self._font()).horizontalAdvance(self.text()) + 36, 38)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.TextAntialiasing)
        pill = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        if self.isChecked():
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(color("accent_soft"))
            painter.drawRoundedRect(pill, pill.height() / 2, pill.height() / 2)
        elif self.isEnabled() and (self._hovered or self._keyboard_focus):
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(color("hover"))
            painter.drawRoundedRect(pill, pill.height() / 2, pill.height() / 2)
        painter.setFont(self._font())
        painter.setPen(color("text" if self.isChecked() else "label"))
        if self._align_left:
            area = self.rect().adjusted(18, 0, -8, 0)
            painter.drawText(area, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, self.text())
        else:
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self.text())


class NavChoice(QWidget):
    """A caption then exclusive text links."""

    chosen = Signal(str)

    def __init__(self, caption: str, options: list[str], current: str, fonts: Fonts) -> None:
        super().__init__()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        if caption:
            label = QLabel(caption)
            label.setFont(fonts.text(config.UI_FONT_PX["body"] - 1))
            label.setStyleSheet(f"color: {config.UI_COLORS['muted']};")
            layout.addWidget(label)
        self._links: dict[str, NavLink] = {}
        for name in options:
            link = NavLink(name, fonts)
            link.setChecked(name == current)
            link.clicked.connect(lambda _checked=False, choice=name: self.chosen.emit(choice))
            layout.addWidget(link)
            self._links[name] = link

    def current(self) -> str:
        return next(name for name, link in self._links.items() if link.isChecked())

    def select(self, name: str) -> None:
        """Same as the user clicking that link."""
        self._links[name].click()

    def set_current(self, name: str) -> None:
        """Show a choice made elsewhere, without announcing it."""
        self._links[name].setChecked(True)


class Legend(QWidget):
    """A dot per seat state, then its text."""

    GAP_PX = 22
    SQUARE_PX = 12

    def __init__(self, entries: list[tuple[str, str]], fonts: Fonts) -> None:
        super().__init__()
        self._entries = entries
        self._font = fonts.text(config.UI_FONT_PX["small"] + 1)
        self.setFixedHeight(30)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)

    def set_entries(self, entries: list[tuple[str, str]]) -> None:
        self._entries = entries
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.TextAntialiasing)
        painter.setFont(self._font)
        metrics = QFontMetrics(self._font)
        x = 0
        for state, text in self._entries:
            y = (self.height() - self.SQUARE_PX) / 2
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(state_color(state))
            painter.drawEllipse(QRectF(x, y, self.SQUARE_PX, self.SQUARE_PX))
            painter.setPen(color("label"))
            text_x = x + self.SQUARE_PX + 8
            painter.drawText(text_x, (self.height() + metrics.ascent() - metrics.descent()) // 2, text)
            x = text_x + metrics.horizontalAdvance(text) + self.GAP_PX


class TextButton(_FlatButton):
    """A soft pill button; compact ones are round."""

    def __init__(self, text: str, fonts: Fonts, compact: bool = False) -> None:
        super().__init__(text, fonts)
        self.setCheckable(False)
        self._compact = compact

    def _font(self):
        return self._fonts.text(config.UI_FONT_PX["body"] + (3 if self._compact else -1), "medium")

    def sizeHint(self) -> QSize:
        if self._compact:
            return QSize(36, 36)
        metrics = QFontMetrics(self._font())
        return QSize(metrics.horizontalAdvance(self.text()) + 40, 38)

    def minimumSizeHint(self) -> QSize:
        return self.sizeHint()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.TextAntialiasing)
        box = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        active = self.isEnabled() and (self._hovered or self._keyboard_focus)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(color("accent_soft") if active else color("chip"))
        painter.drawRoundedRect(box, box.height() / 2, box.height() / 2)
        painter.setFont(self._font())
        painter.setPen(color("text" if self.isEnabled() else "switch"))
        painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self.text())


class Stepper(QWidget):
    """A − 4 + control for a bounded number."""

    valueChanged = Signal(float)

    def __init__(
        self, value: float, minimum: float, maximum: float, step: float, fonts: Fonts,
        decimals: int = 0, suffix: str = "",
    ) -> None:
        super().__init__()
        self._minimum, self._maximum, self._step = minimum, maximum, step
        self._decimals, self._suffix = decimals, suffix
        self._value = value
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        self._less = TextButton("−", fonts, compact=True)
        self._more = TextButton("+", fonts, compact=True)
        self._text = QLabel()
        self._text.setFont(fonts.text(config.UI_FONT_PX["body"]))
        self._text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._text.setFixedWidth(STEPPER_VALUE_PX)  # keeps every row's buttons aligned
        for widget in (self._less, self._text, self._more):
            layout.addWidget(widget)
        self._less.clicked.connect(lambda: self._nudge(-1))
        self._more.clicked.connect(lambda: self._nudge(+1))
        self.setValue(value)

    def value(self) -> float:
        return self._value

    def setValue(self, value: float) -> None:
        """Set without emitting; used when settings change elsewhere."""
        self._value = round(min(max(value, self._minimum), self._maximum), self._decimals)
        self._text.setText(f"{self._value:.{self._decimals}f}{self._suffix}")
        self._less.setEnabled(self._value > self._minimum)
        self._more.setEnabled(self._value < self._maximum)

    def _nudge(self, direction: int) -> None:
        before = self._value
        self.setValue(self._value + direction * self._step)
        if self._value != before:
            self.valueChanged.emit(self._value)


class SettingRow(QWidget):
    """A ruled row: label left, any control right."""

    def __init__(self, text: str, control: QWidget, fonts: Fonts) -> None:
        super().__init__()
        self.setFixedHeight(config.UI_ROW_HEIGHT_PX)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 0, 0, 1)
        label = QLabel(text)
        label.setFont(fonts.text(config.UI_FONT_PX["body"]))
        layout.addWidget(label)
        layout.addStretch(1)
        layout.addWidget(control)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setPen(color("hairline"))
        painter.drawLine(0, self.height() - 1, self.width(), self.height() - 1)


def make_dropdown(options: list[str], current: str) -> QComboBox:
    """A minimal dropdown, styled to match."""
    box = QComboBox()
    box.addItems(options)
    box.setCurrentText(current)
    box.setCursor(Qt.CursorShape.PointingHandCursor)
    box.setFixedWidth(DROPDOWN_WIDTH_PX)
    round_popup(box.view().window())
    return box


def round_popup(window: QWidget) -> None:
    """Let a popup draw its own rounded corners."""
    flags = Qt.WindowType.FramelessWindowHint | Qt.WindowType.NoDropShadowWindowHint
    window.setWindowFlags(window.windowFlags() | flags)
    window.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)


def _corners(painter: QPainter, centre: QPointF, reach: float, arm: float, inward: bool) -> None:
    """Four L-shaped corners; inward ones mean restore."""
    for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
        if inward:
            corner = QPointF(centre.x() + sx * reach * 0.35, centre.y() + sy * reach * 0.35)
            ends = (QPointF(corner.x() + sx * arm, corner.y()), QPointF(corner.x(), corner.y() + sy * arm))
        else:
            corner = QPointF(centre.x() + sx * reach, centre.y() + sy * reach)
            ends = (QPointF(corner.x() - sx * arm, corner.y()), QPointF(corner.x(), corner.y() - sy * arm))
        for end in ends:
            painter.drawLine(corner, end)


def _cross(painter: QPainter, centre: QPointF, reach: float) -> None:
    span = reach * 0.8
    painter.drawLine(QPointF(centre.x() - span, centre.y() - span), QPointF(centre.x() + span, centre.y() + span))
    painter.drawLine(QPointF(centre.x() - span, centre.y() + span), QPointF(centre.x() + span, centre.y() - span))


ICON_KINDS = ("expand", "restore", "hide", "add")


class IconButton(_FlatButton):
    """A small drawn icon: expand, restore or hide."""

    SIZE_PX = 34
    REACH_PX = 6.0

    def __init__(self, kind: str, tooltip: str, fonts: Fonts) -> None:
        super().__init__("", fonts)
        self.setCheckable(False)
        self.setFixedSize(self.SIZE_PX, self.SIZE_PX)
        self.set_kind(kind, tooltip)

    def kind(self) -> str:
        return self._kind

    def set_kind(self, kind: str, tooltip: str) -> None:
        if kind not in ICON_KINDS:
            raise ValueError(f"unknown icon {kind!r}")
        self._kind = kind
        self.setToolTip(tooltip)
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        if self.isEnabled() and (self._hovered or self._keyboard_focus):
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(color("hover"))
            painter.drawEllipse(QRectF(self.rect()))
        pen = QPen(color("label" if self.isEnabled() else "switch"), 1.5)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        centre = QRectF(self.rect()).center()
        if self._kind == "hide":
            _cross(painter, centre, self.REACH_PX)
        elif self._kind == "add":
            painter.drawLine(QPointF(centre.x() - 6, centre.y()), QPointF(centre.x() + 6, centre.y()))
            painter.drawLine(QPointF(centre.x(), centre.y() - 6), QPointF(centre.x(), centre.y() + 6))
        else:
            _corners(painter, centre, self.REACH_PX, 3.5, inward=self._kind == "restore")

