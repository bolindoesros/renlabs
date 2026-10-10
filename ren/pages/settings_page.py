"""Settings: sections on the left, controls right."""
from dataclasses import dataclass
from typing import Callable

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QStackedWidget, QVBoxLayout, QWidget

from jacket import config
from ren.settings import AppSettings, add_venue, get_path, remove_venue, replace_path, select_venue
from ren.theme import Fonts
from ren.venue_list import VenueList
from ren.widgets import Card, Hairline, NavLink, SettingRow, Stepper, SwitchRow, TextButton, make_dropdown

RESET_ARMED_MS = 3000
SECTION_WIDTH_PX = 620
NAV_WIDTH_PX = 200
Path = tuple[str, ...]


@dataclass(frozen=True)
class ToggleRow:
    label: str
    path: Path


@dataclass(frozen=True)
class StepperRow:
    label: str
    path: Path
    minimum: float
    maximum: float
    step: float
    decimals: int = 0
    suffix: str = ""


@dataclass(frozen=True)
class DropdownRow:
    label: str
    path: Path
    options: tuple[str, ...]


@dataclass(frozen=True)
class ActionRow:
    label: str
    button: str


VENUES_SECTION = "venues"
ROOM_SECTIONS = ("seating plan", "vent")  # edit the venue in use

SECTIONS: dict[str, list] = {
    VENUES_SECTION: [],
    "vision": [
        DropdownRow("model", ("view", "model_key"), tuple(config.CLIP_MODELS)),
        ToggleRow("human detection", ("view", "detect_people")),
        ToggleRow("clothing detection", ("view", "classify_clothing")),
        ToggleRow("mirror camera", ("view", "mirror")),
    ],
    "seating plan": [
        StepperRow("rows", ("layout", "rows"), 1, 12, 1),
        StepperRow("seats per row", ("layout", "cols"), 1, 16, 1),
        ActionRow("calibration", "edit"),
    ],
    "decision": [
        DropdownRow("aim mode", ("decision", "aim_mode"), ("sweep", "focus")),
        StepperRow("hot seat need", ("decision", "weights", "hot"), 0, 1, 0.05, 2),
        StepperRow("unsure seat need", ("decision", "weights", "unsure"), 0, 1, 0.05, 2),
        StepperRow("cold seat need", ("decision", "weights", "cold"), 0, 1, 0.05, 2),
        StepperRow("time per zone", ("decision", "min_dwell_s"), 0, 60, 1, 0, " s"),
        StepperRow("shut vent below", ("decision", "close_below"), 0, 10, 0.25, 2, " people"),
    ],
    "vent": [
        StepperRow("ceiling to head", ("vent", "drop_m"), 0.5, 10, 0.1, 1, " m"),
        StepperRow("max flap tilt", ("vent", "tilt_max_deg"), 5, 80, 5, 0, "°"),
        StepperRow("rotation zero", ("vent", "rotation_offset_deg"), -180, 180, 5, 0, "°"),
        ToggleRow("clockwise rotation", ("vent", "clockwise")),
        StepperRow("position left to right", ("vent", "u"), 0, 1, 0.05, 2),
        StepperRow("position front to back", ("vent", "v"), 0, 1, 0.05, 2),
        StepperRow("turn speed", ("vent", "slew_deg_per_s"), 5, 360, 5, 0, " °/s"),
        StepperRow("seat width", ("vent", "seat_width_m"), 0.3, 1.2, 0.05, 2, " m"),
        StepperRow("row depth", ("vent", "row_depth_m"), 0.4, 2, 0.05, 2, " m"),
    ],
}


class SettingsPage(QWidget):
    changed = Signal(object)  # AppSettings
    calibrate_requested = Signal()

    def __init__(self, fonts: Fonts, settings: AppSettings) -> None:
        super().__init__()
        self._fonts = fonts
        self._settings = settings
        self._refreshers: list[Callable[[AppSettings], None]] = []
        self._reset_armed = False

        pad = config.UI_PAGE_PADDING_PX
        root = QHBoxLayout(self)
        root.setContentsMargins(pad, pad - 10, pad, pad - 10)
        root.setSpacing(pad)
        root.addLayout(self._build_nav())
        self._stack = QStackedWidget()
        self._room_labels: list[QLabel] = []
        self._venues = VenueList(fonts)
        self._venues.chosen.connect(lambda name: self._apply(select_venue(self._settings, name)))
        self._venues.added.connect(self._on_add_venue)
        self._venues.removed.connect(lambda name: self._apply(remove_venue(self._settings, name)))
        for name, rows in SECTIONS.items():
            self._stack.addWidget(self._build_section(name, rows))
        card = Card()
        inner = QVBoxLayout(card.body())
        inner.setContentsMargins(36, 30, 36, 30)
        inner.addWidget(self._stack)
        root.addWidget(card, 1)
        self._links[0].setChecked(True)
        self.set_settings(settings)

    # --- layout -----------------------------------------------------------------

    def _build_nav(self) -> QVBoxLayout:
        nav = QVBoxLayout()
        nav.setContentsMargins(0, config.UI_CARD_MARGIN_PX, 0, config.UI_CARD_MARGIN_PX)
        nav.setSpacing(4)
        self._links: list[NavLink] = []
        for index, name in enumerate(SECTIONS):
            link = NavLink(name, self._fonts, align_left=True)
            link.setFixedSize(NAV_WIDTH_PX, 42)
            link.clicked.connect(lambda _checked=False, i=index: self._stack.setCurrentIndex(i))
            nav.addWidget(link)
            self._links.append(link)
        nav.addStretch(1)
        self._reset = TextButton("reset to defaults", self._fonts)
        self._reset.setToolTip("resets vision and decision; keeps your venues")
        self._reset.clicked.connect(self._on_reset)
        nav.addWidget(self._reset)
        return nav

    def _build_section(self, name: str, rows: list) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        title = QLabel(name)
        title.setFont(self._fonts.text(config.UI_FONT_PX["title"], "bold"))
        layout.addWidget(title)
        if name in ROOM_SECTIONS:
            room = QLabel()
            room.setFont(self._fonts.text(config.UI_FONT_PX["small"]))
            room.setStyleSheet(f"color: {config.UI_COLORS['muted']};")
            layout.addWidget(room)
            self._room_labels.append(room)
        layout.addSpacing(16)
        if name == VENUES_SECTION:
            layout.addWidget(self._venues)
        else:
            layout.addWidget(Hairline())
        for row in rows:
            layout.addWidget(self._build_row(row))
        layout.addStretch(1)
        page.setFixedWidth(SECTION_WIDTH_PX)
        wrapper = QWidget()
        outer = QHBoxLayout(wrapper)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(page, 1, Qt.AlignmentFlag.AlignLeft)
        outer.addStretch(0)
        return wrapper

    def _build_row(self, row) -> QWidget:
        if isinstance(row, ToggleRow):
            return self._toggle(row)
        if isinstance(row, StepperRow):
            return self._stepper(row)
        if isinstance(row, DropdownRow):
            return self._dropdown(row)
        button = TextButton(row.button, self._fonts)
        button.clicked.connect(self.calibrate_requested)
        return SettingRow(row.label, button, self._fonts)

    def _toggle(self, row: ToggleRow) -> QWidget:
        toggle = SwitchRow(row.label, self._fonts)
        toggle.setChecked(get_path(self._settings, row.path))
        toggle.toggled.connect(lambda value: self._edit(row.path, value))
        self._refreshers.append(lambda s: self._quietly(toggle, lambda: toggle.setChecked(get_path(s, row.path))))
        return toggle

    def _stepper(self, row: StepperRow) -> QWidget:
        stepper = Stepper(get_path(self._settings, row.path), row.minimum, row.maximum, row.step,
                          self._fonts, row.decimals, row.suffix)
        stepper.valueChanged.connect(lambda value: self._edit(row.path, value))
        self._refreshers.append(lambda s: stepper.setValue(get_path(s, row.path)))
        return SettingRow(row.label, stepper, self._fonts)

    def _dropdown(self, row: DropdownRow) -> QWidget:
        box: QComboBox = make_dropdown(list(row.options), get_path(self._settings, row.path))
        box.textActivated.connect(lambda value: self._edit(row.path, value))
        self._refreshers.append(lambda s: self._quietly(box, lambda: box.setCurrentText(get_path(s, row.path))))
        return SettingRow(row.label, box, self._fonts)

    @staticmethod
    def _quietly(widget: QWidget, action: Callable[[], None]) -> None:
        widget.blockSignals(True)
        action()
        widget.blockSignals(False)

    # --- editing ----------------------------------------------------------------

    def _edit(self, path: Path, value) -> None:
        current = get_path(self._settings, path)
        if isinstance(current, int) and not isinstance(current, bool):
            value = int(round(value))
        self._settings = replace_path(self._settings, path, value)
        self.changed.emit(self._settings)

    def set_settings(self, settings: AppSettings) -> None:
        """Show settings changed elsewhere, without announcing them again."""
        self._settings = settings
        for refresh in self._refreshers:
            refresh(settings)
        self._venues.set_venues(settings.venue_names(), settings.venue)
        for label in self._room_labels:
            label.setText(f"for {settings.venue}")

    def _apply(self, settings: AppSettings) -> None:
        self.set_settings(settings)
        self.changed.emit(settings)

    def _on_add_venue(self, text: str) -> None:
        try:
            settings = add_venue(self._settings, text)
        except ValueError as error:
            self._venues.show_error(str(error))
            return
        self._venues.clear_input()
        self._apply(settings)

    def _on_reset(self) -> None:
        if not self._reset_armed:  # first click arms it; a stray click changes nothing
            self._reset_armed = True
            self._reset.setText("click again to confirm")
            QTimer.singleShot(RESET_ARMED_MS, self._disarm)
            return
        self._disarm()
        keep_rooms = AppSettings(venues=self._settings.venues, venue=self._settings.venue)  # calibrations are precious
        self._apply(keep_rooms)

    def _disarm(self) -> None:
        self._reset_armed = False
        self._reset.setText("reset to defaults")
        self._reset.updateGeometry()
        self._reset.update()
