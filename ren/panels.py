"""Side-by-side card panels you can expand or hide."""
from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QPainter
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QSplitter, QSplitterHandle, QVBoxLayout, QWidget

from jacket import config
from ren.theme import Fonts, color
from ren.widgets import Card, IconButton, StatusDot

HANDLE_PX = 8
EXPAND_TIP = "expand (double-click the title)"
RESTORE_TIP = "restore (esc)"
HIDE_TIP = "hide (bring it back from view)"


class LineHandle(QSplitterHandle):
    """Invisible until hovered, then a small grip pill."""

    def __init__(self, orientation, parent: QSplitter) -> None:
        super().__init__(orientation, parent)
        self._hovered = False

    def enterEvent(self, event) -> None:
        self._hovered = True
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:
        self._hovered = False
        self.update()
        super().leaveEvent(event)

    def paintEvent(self, event) -> None:
        if not self._hovered:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(color("switch"))
        grip = QRectF(self.width() / 2 - 2, self.height() / 2 - 24, 4, 48)
        painter.drawRoundedRect(grip, 2, 2)


class Splitter(QSplitter):
    def createHandle(self) -> QSplitterHandle:
        return LineHandle(self.orientation(), self)


class Panel(Card):
    """A titled card with status dot, expand, hide."""

    expand_clicked = Signal()
    hide_clicked = Signal()

    def __init__(self, title: str, content: QWidget, fonts: Fonts) -> None:
        super().__init__()
        layout = QVBoxLayout(self.body())
        layout.setContentsMargins(0, 0, 0, 12)
        layout.setSpacing(0)

        self._header = QFrame()
        self._header.setFixedHeight(config.UI_PANEL_HEADER_PX)
        row = QHBoxLayout(self._header)
        row.setContentsMargins(14, 4, 10, 0)
        row.setSpacing(2)
        self.dot = StatusDot()
        label = QLabel(title)
        label.setFont(fonts.text(config.UI_FONT_PX["body"], "medium"))
        row.addWidget(self.dot)
        row.addSpacing(4)
        row.addWidget(label)
        row.addStretch(1)
        self._expand = IconButton("expand", EXPAND_TIP, fonts)
        self._hide = IconButton("hide", HIDE_TIP, fonts)
        row.addWidget(self._expand)
        row.addWidget(self._hide)
        layout.addWidget(self._header)
        inset = QHBoxLayout()
        inset.setContentsMargins(12, 0, 12, 0)  # content sits inside the rounded card
        inset.addWidget(content)
        layout.addLayout(inset, 1)

        self._expand.clicked.connect(self.expand_clicked)
        self._hide.clicked.connect(self.hide_clicked)
        self._header.mouseDoubleClickEvent = lambda event: self.expand_clicked.emit()

    def set_expanded(self, expanded: bool) -> None:
        self._expand.set_kind(*(("restore", RESTORE_TIP) if expanded else ("expand", EXPAND_TIP)))

    def is_expanded(self) -> bool:
        return self._expand.kind() == "restore"


class PanelArea(QWidget):
    """Holds the panels and their hidden or expanded state."""

    changed = Signal()

    def __init__(self, fonts: Fonts) -> None:
        super().__init__()
        self._fonts = fonts
        self._panels: dict[str, Panel] = {}
        self._shares: list[int] = []
        self._hidden: set[str] = set()
        self._expanded: str | None = None
        self._splitter = Splitter(Qt.Orientation.Horizontal)
        self._splitter.setHandleWidth(HANDLE_PX)
        self._splitter.setStyleSheet("QSplitter { background: transparent; }")
        self._splitter.setChildrenCollapsible(False)
        self._empty = QLabel("every panel is hidden, show one from view")
        self._empty.setFont(fonts.text(config.UI_FONT_PX["body"]))
        self._empty.setStyleSheet(f"color: {config.UI_COLORS['muted']};")
        self._empty.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self._empty.hide()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._splitter, 1)
        layout.addWidget(self._empty, 1)

    def add_panel(self, name: str, title: str, content: QWidget, share: int = 2) -> Panel:
        """Panels split the width by share."""
        panel = Panel(title, content, self._fonts)
        panel.expand_clicked.connect(lambda: self.toggle_expanded(name))
        panel.hide_clicked.connect(lambda: self.set_visible(name, False))
        self._panels[name] = panel
        self._splitter.addWidget(panel)
        self._splitter.setStretchFactor(self._splitter.count() - 1, share)
        self._shares.append(share)
        self._splitter.setSizes([500 * s for s in self._shares])
        return panel

    def dot(self, name: str):
        return self._panels[name].dot

    # --- state ------------------------------------------------------------------

    def is_visible(self, name: str) -> bool:
        """Not hidden by the user, maybe covered."""
        return name not in self._hidden

    def is_shown(self, name: str) -> bool:
        return self.is_visible(name) and (self._expanded is None or self._expanded == name)

    def expanded(self) -> str | None:
        return self._expanded

    def set_visible(self, name: str, visible: bool) -> None:
        (self._hidden.discard if visible else self._hidden.add)(name)
        if not visible and self._expanded == name:
            self._expanded = None
        self._apply()

    def toggle_expanded(self, name: str) -> None:
        self._expanded = None if self._expanded == name else name
        self._hidden.discard(name)  # expanding a hidden panel brings it back
        self._apply()

    def restore(self) -> bool:
        """Leave the expanded view; returns whether anything changed."""
        if self._expanded is None:
            return False
        self._expanded = None
        self._apply()
        return True

    def _apply(self) -> None:
        shown_any = False
        for name, panel in self._panels.items():
            shown = self.is_shown(name)
            panel.setVisible(shown)
            panel.set_expanded(self._expanded == name)
            shown_any = shown_any or shown
        self._splitter.setVisible(shown_any)
        self._empty.setVisible(not shown_any)
        self.changed.emit()
