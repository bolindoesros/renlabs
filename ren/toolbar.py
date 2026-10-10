"""Live page toolbar: venue, detector, model and view dropdowns."""
from dataclasses import dataclass

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QHBoxLayout, QLabel, QMenu, QPushButton, QWidget

from jacket import config
from ren.engine import ViewSettings
from ren.theme import Fonts
from ren.widgets import IconButton, make_dropdown, on_key_chosen, round_popup, select_key


MENU_TEXT_INSET_PX = 18  # matches the dropdowns' text inset
MENU_ARROW_PX = 46  # chevron plus breathing room
VIEW_BUTTON_WIDTH_PX = 210
DETECTOR_PARTS = ("body", "head")
NONE = "none"  # no model: the plain stream, or people without clothing
NO_DETECTOR_LABEL = "no detector"


def detector_mode(ticked: tuple[str, ...]) -> str:
    """Ticked parts to a stored mode: both parts are "both", none is NONE."""
    if not ticked:
        return NONE
    return "both" if len(ticked) == len(DETECTOR_PARTS) else ticked[0]


def detector_label(mode: str) -> str:
    return NO_DETECTOR_LABEL if mode == NONE else config.DETECTOR_NAMES[mode]


@dataclass(frozen=True)
class ViewItem:
    key: str  # a ViewSettings field, or a panel name
    label: str
    is_panel: bool = False


VIEW_SECTIONS: dict[str, tuple[ViewItem, ...]] = {
    "panels": (
        ViewItem("camera", "camera", is_panel=True),
        ViewItem("plan", "seating plan", is_panel=True),
        ViewItem("crops", "crops", is_panel=True),
    ),
    "camera": (
        ViewItem("show_image", "picture"),
        ViewItem("show_people", "people"),
        ViewItem("show_labels", "labels"),
        ViewItem("show_torso_line", "shoulder line"),
        ViewItem("show_grid", "seat grid"),
    ),
    "seating plan": (
        ViewItem("show_seats", "seats"),
        ViewItem("show_zones", "zones"),
        ViewItem("show_vent", "vent"),
        ViewItem("show_beam", "airflow"),
    ),
}


class StickyMenu(QMenu):
    """Ticking an item keeps the menu open."""

    def mouseReleaseEvent(self, event) -> None:
        action = self.activeAction()
        if action is not None and action.isEnabled() and action.isCheckable():
            action.trigger()
            return
        super().mouseReleaseEvent(event)


class MenuButton(QPushButton):
    """A pill that opens a menu. Qt ignores left padding on left-aligned
    menu buttons, so an inset label draws the text instead."""

    def __init__(self, text: str) -> None:
        super().__init__()
        self.setObjectName("viewButton")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._label = QLabel(text)
        self._label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        row = QHBoxLayout(self)
        row.setContentsMargins(MENU_TEXT_INSET_PX, 0, MENU_ARROW_PX, 0)
        row.addWidget(self._label)

    def text(self) -> str:
        return self._label.text()

    def setText(self, text: str) -> None:
        self._label.setText(text)

    def fit_text(self) -> None:
        """Just wide enough for the current text."""
        width = self._label.fontMetrics().horizontalAdvance(self._label.text())
        self.setFixedWidth(width + MENU_TEXT_INSET_PX + MENU_ARROW_PX + 4)  # 4: rounding slack


class Toolbar(QWidget):
    venue_chosen = Signal(str)
    venue_add_requested = Signal()
    model_chosen = Signal(str)
    detector_chosen = Signal(str)
    layer_toggled = Signal(str, bool)  # ViewSettings field, shown
    panel_toggled = Signal(str, bool)  # panel name, shown

    def __init__(
        self, fonts: Fonts, model_key: str, view: ViewSettings, venues: list[str], venue: str,
    ) -> None:
        super().__init__()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)
        self._venue = make_dropdown(venues, venue)
        self._venue.textActivated.connect(self.venue_chosen)
        self._add_venue = IconButton("add", "add a venue (copies this one)", fonts)
        self._add_venue.clicked.connect(self.venue_add_requested)
        self._model = make_dropdown([*config.CLIP_MODELS, NONE], model_key, config.CLIP_MODEL_NAMES)
        self._model.setToolTip("clothing model, or none to only find people")
        on_key_chosen(self._model, self.model_chosen.emit)
        self._detector = MenuButton("")  # tick body, head, or both
        self._detector.setToolTip("people detectors: body pose, heads, both so the two agree, or none for the plain stream")
        detector_menu = StickyMenu(self._detector)
        round_popup(detector_menu)
        self._detector.setMenu(detector_menu)
        self._detector_parts: dict[str, QAction] = {}
        for part in DETECTOR_PARTS:
            action = detector_menu.addAction(config.DETECTOR_NAMES[part])
            action.setCheckable(True)
            action.toggled.connect(self._on_detector_toggled)
            self._detector_parts[part] = action
        self.set_detector(view.detector)
        self._view_button = MenuButton("layers")
        self._view_button.setFixedWidth(VIEW_BUTTON_WIDTH_PX)
        self._menu = StickyMenu(self._view_button)
        round_popup(self._menu)
        self._view_button.setMenu(self._menu)
        self._actions: dict[str, QAction] = {}
        self._build_menu(fonts)

        for widget in (self._caption("venue", fonts), self._venue, self._add_venue):
            layout.addWidget(widget)
        layout.addSpacing(20)
        for widget in (self._caption("models", fonts), self._detector, self._model):  # people, then clothing
            layout.addWidget(widget)
        layout.addSpacing(20)
        layout.addWidget(self._caption("view", fonts))
        layout.addWidget(self._view_button)
        layout.addStretch(1)
        self.set_view(view)

    @staticmethod
    def _caption(text: str, fonts: Fonts) -> QLabel:
        label = QLabel(text)
        label.setFont(fonts.text(config.UI_FONT_PX["body"], "medium"))
        label.setStyleSheet(f"color: {config.UI_COLORS['label']};")
        return label

    def _build_menu(self, fonts: Fonts) -> None:
        heading_font = fonts.text(config.UI_FONT_PX["small"] - 1, "medium")
        for index, (section, items) in enumerate(VIEW_SECTIONS.items()):
            if index:
                self._menu.addSeparator()
            heading = self._menu.addAction(section)
            heading.setEnabled(False)
            heading.setFont(heading_font)
            for item in items:
                action = self._menu.addAction(item.label)
                action.setCheckable(True)
                action.setChecked(True)
                action.toggled.connect(lambda shown, i=item: self._on_toggled(i, shown))
                self._actions[item.key] = action

    def _on_toggled(self, item: ViewItem, shown: bool) -> None:
        (self.panel_toggled if item.is_panel else self.layer_toggled).emit(item.key, shown)

    def action(self, key: str) -> QAction:
        return self._actions[key]

    def model(self) -> str:
        return self._model.currentData()

    def set_view(self, view: ViewSettings) -> None:
        """Show settings changed elsewhere, without announcing them."""
        for items in VIEW_SECTIONS.values():
            for item in items:
                if not item.is_panel:
                    self._quietly(self._actions[item.key], getattr(view, item.key))
        self.set_detector(view.detector if view.detect_people else NONE)
        self.set_model(view.model_key if view.classify_clothing else NONE)
        self._model.setEnabled(view.detect_people)  # clothing needs people

    def venue(self) -> str:
        return self._venue.currentText()

    def set_venues(self, names: list[str], active: str) -> None:
        """Show venues changed elsewhere, without announcing them."""
        self._venue.blockSignals(True)
        if [self._venue.itemText(i) for i in range(self._venue.count())] != names:
            self._venue.clear()
            self._venue.addItems(names)
        self._venue.setCurrentText(active)
        self._venue.blockSignals(False)

    def detector(self) -> str:
        return detector_mode(self._ticked_parts())

    def set_detector(self, mode: str) -> None:
        for part, action in self._detector_parts.items():
            self._quietly(action, mode in (part, "both"))
        self._detector_mode = mode
        self._detector.setText(detector_label(mode))
        self._detector.fit_text()

    def _ticked_parts(self) -> tuple[str, ...]:
        return tuple(part for part, action in self._detector_parts.items() if action.isChecked())

    def _on_detector_toggled(self) -> None:
        self.set_detector(self.detector())
        self.detector_chosen.emit(self._detector_mode)

    def set_model(self, model_key: str) -> None:
        select_key(self._model, model_key)

    def set_panel_visible(self, panel: str, visible: bool) -> None:
        self._quietly(self._actions[panel], visible)

    @staticmethod
    def _quietly(action: QAction, checked: bool) -> None:
        action.blockSignals(True)
        action.setChecked(checked)
        action.blockSignals(False)
