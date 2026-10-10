"""Settings section listing venues: use, remove, add."""
from PySide6.QtCore import QTimer, Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QLineEdit, QVBoxLayout, QWidget

from jacket import config
from ren.theme import Fonts
from ren.widgets import Hairline, SettingRow, TextButton

CONFIRM_MS = 3000


class VenueRow(SettingRow):
    """A venue name with use and remove buttons."""

    use_clicked = Signal(str)
    remove_confirmed = Signal(str)

    def __init__(self, name: str, active: bool, removable: bool, fonts: Fonts) -> None:
        self.use = TextButton("in use" if active else "use", fonts)
        self.remove = TextButton("remove", fonts)
        buttons = QWidget()
        row = QHBoxLayout(buttons)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)
        row.addWidget(self.use)
        row.addWidget(self.remove)
        super().__init__(name, buttons, fonts)
        self._name = name
        self._armed = False
        self.use.setEnabled(not active)
        self.remove.setEnabled(removable)
        self.use.clicked.connect(lambda: self.use_clicked.emit(name))
        self.remove.clicked.connect(self._on_remove)

    def _on_remove(self) -> None:
        if not self._armed:  # a stray click must not delete a calibration
            self._armed = True
            self.remove.setText("sure?")
            QTimer.singleShot(CONFIRM_MS, self._disarm)
            return
        self.remove_confirmed.emit(self._name)

    def _disarm(self) -> None:
        self._armed = False
        self.remove.setText("remove")


class VenueList(QWidget):
    chosen = Signal(str)
    added = Signal(str)
    removed = Signal(str)

    def __init__(self, fonts: Fonts) -> None:
        super().__init__()
        self._fonts = fonts
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        hint = QLabel("each venue keeps its own seats, calibration and vent")
        hint.setFont(fonts.text(config.UI_FONT_PX["small"]))
        hint.setStyleSheet(f"color: {config.UI_COLORS['muted']};")
        layout.addWidget(hint)
        layout.addSpacing(12)
        layout.addWidget(Hairline())
        self._rows = QVBoxLayout()
        self._rows.setSpacing(0)
        layout.addLayout(self._rows)

        add_row = QHBoxLayout()
        add_row.setContentsMargins(0, 18, 0, 0)
        self._name = QLineEdit()
        self._name.setPlaceholderText("new venue, e.g. lt3")
        self._name.setFont(fonts.text(config.UI_FONT_PX["body"]))
        self._name.returnPressed.connect(self._on_add)
        add = TextButton("add venue", fonts)
        add.clicked.connect(self._on_add)
        add_row.addWidget(self._name, 1)
        add_row.addWidget(add)
        layout.addLayout(add_row)
        self._error = QLabel()
        self._error.setFont(fonts.text(config.UI_FONT_PX["small"]))
        self._error.setStyleSheet(f"color: {config.UI_COLORS['error']};")
        layout.addWidget(self._error)

    def rows(self) -> list[VenueRow]:
        return [self._rows.itemAt(i).widget() for i in range(self._rows.count())]

    def set_venues(self, names: list[str], active: str) -> None:
        for row in self.rows():
            row.setParent(None)
            row.deleteLater()
        for name in names:
            row = VenueRow(name, name == active, len(names) > 1, self._fonts)
            row.use_clicked.connect(self.chosen)
            row.remove_confirmed.connect(self.removed)
            self._rows.addWidget(row)

    def show_error(self, text: str) -> None:
        self._error.setText(text)

    def clear_input(self) -> None:
        self._name.clear()
        self._error.clear()

    def _on_add(self) -> None:
        self.added.emit(self._name.text())
