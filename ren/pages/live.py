"""The demo page: toolbar, then the panels."""
import logging
from dataclasses import replace
import time
from typing import Callable

from PySide6.QtCore import QTimer, Signal
from PySide6.QtWidgets import QVBoxLayout, QWidget

from jacket import config
from ren.decision import DecisionMaker
from ren.engine import FrameResult
from ren.panels import PanelArea
from ren.plan import grid_lines, make_zones, read_scene
from ren.plan_panel import PlanPanel, status_text
from ren.settings import AppSettings, replace_path, select_venue
from ren.theme import Fonts
from ren.toolbar import NONE, Toolbar
from ren.crops_view import CropsPanel
from ren.video_view import VideoView

logger = logging.getLogger("ren.live")
PANELS = ("camera", "plan", "crops")


class LivePage(QWidget):
    changed = Signal(object)  # AppSettings, from the toolbar
    venue_add_requested = Signal()

    def __init__(self, fonts: Fonts, settings: AppSettings, clock: Callable[[], float] = time.monotonic) -> None:
        super().__init__()
        self._clock = clock
        self._settings: AppSettings | None = None
        self._plan_error: str | None = None
        self._grid_cache: tuple | None = None
        self._grid: list = []
        self._unseated: tuple[int, ...] = ()
        self._scene = self._decision = None
        self._seen_at: dict[str, float] = {}

        self._camera = VideoView(fonts)
        self._plan_panel = PlanPanel(fonts)
        self._crops = CropsPanel(fonts)
        self._area = PanelArea(fonts)
        self._area.add_panel("camera", "camera", self._camera)
        self._area.add_panel("plan", "seating plan", self._plan_panel)
        self._area.add_panel("crops", "torso crops", self._crops, share=1)
        self._toolbar = Toolbar(fonts, settings.view.model_key, settings.view, settings.venue_names(), settings.venue)

        pad = config.UI_PAGE_PADDING_PX
        root = QVBoxLayout(self)
        root.setContentsMargins(pad, pad - 10, pad, pad - 6)
        root.setSpacing(16)
        root.addWidget(self._toolbar)
        root.addWidget(self._area, 1)

        self._toolbar.venue_chosen.connect(lambda name: self.changed.emit(select_venue(self._settings, name)))
        self._toolbar.venue_add_requested.connect(self.venue_add_requested)
        self._toolbar.model_chosen.connect(self._choose_model)
        self._toolbar.detector_chosen.connect(self._choose_detector)
        self._toolbar.preset_chosen.connect(self._choose_preset)
        self._toolbar.layer_toggled.connect(lambda key, shown: self._edit(("view", key), shown))
        self._toolbar.panel_toggled.connect(self._area.set_visible)
        self._area.changed.connect(self._sync_panels)
        self._plan_panel.retry_requested.connect(self._retry_plan)
        self._dot_timer = QTimer(self)
        self._dot_timer.timeout.connect(self.refresh_dots)
        self._dot_timer.start(500)
        self.set_settings(settings)

    # --- public -----------------------------------------------------------------

    def set_panel_visible(self, name: str, visible: bool) -> None:
        self._area.set_visible(name, visible)

    def collapse_expanded(self) -> bool:
        return self._area.restore()

    def show_failure(self, message: str) -> None:
        self._camera.show_message(message)
        self._crops.show_message("")
        self._seen_at.pop("camera", None)
        self._area.dot("camera").set_state("error")

    def refresh_dots(self) -> None:
        """Green while frames arrive, grey once they stop."""
        now = self._clock()
        for name in PANELS:
            dot = self._area.dot(name)
            if name == "plan" and self._plan_error is not None:
                dot.set_state("error")
            elif name in self._seen_at and now - self._seen_at[name] <= config.UI_LIVE_TIMEOUT_S:
                dot.set_state("live")
            elif dot.state() != "error" or name in self._seen_at:
                dot.set_state("idle")

    def set_settings(self, settings: AppSettings) -> None:
        previous, self._settings = self._settings, settings
        self._camera.set_layers(settings.view)
        self._toolbar.set_view(settings.view)
        self._toolbar.set_venues(settings.venue_names(), settings.venue)
        self._area.set_visible("crops", settings.view.show_crops)  # remembered between runs
        self._grid_cache = None
        if self._plan_error is not None:
            self._retry_plan()  # a settings change may have fixed it
        inputs = (settings.layout, settings.decision, settings.vent, settings.view.mirror)
        if previous is None or inputs != (previous.layout, previous.decision, previous.vent, previous.view.mirror):
            self._zones = make_zones(settings.layout)
            self._maker = DecisionMaker(
                settings.layout, settings.decision, settings.vent, self._zones, mirrored=settings.view.mirror
            )
            self._scene = self._decision = None
        self._plan_panel.set_state(settings.layout, self._scene, self._decision, self._zones, settings.vent, settings.view)

    def show_result(self, result: FrameResult) -> None:
        now = self._clock()
        self._unseated, self._grid = (), []
        if self._area.is_shown("plan") and self._plan_error is None:
            try:
                self._update_plan(result, now)
                self._seen_at["plan"] = now
            except Exception as error:  # broad on purpose; camera survives plan failures
                logger.exception("seating plan failed")
                self._fail_plan(f"{type(error).__name__}: {error}")
        if self._area.is_shown("camera"):
            self._camera.set_unseated(self._unseated)
            self._camera.set_grid(self._grid)
            self._camera.show_result(result)
            self._seen_at["camera"] = now
        if self._area.is_shown("crops"):
            self._crops.show_result(result)
            self._seen_at["crops"] = now
        self.refresh_dots()

    # --- the plan stage ---------------------------------------------------------

    def _update_plan(self, result: FrameResult, now: float) -> None:
        settings = self._settings
        height, width = result.frame.shape[:2]
        scene = read_scene(result.boxes, result.results, (width, height), settings.layout, settings.calibration)
        decision = self._maker.update(scene, now)
        self._scene, self._decision = scene, decision
        self._unseated = scene.unseated
        if settings.view.show_grid:
            self._grid = self._grid_for(width, height)
        self._plan_panel.set_state(settings.layout, scene, decision, self._zones, settings.vent, settings.view)
        self._plan_panel.set_status(status_text(decision, scene, len(result.boxes)))

    def _grid_for(self, width: int, height: int) -> list:
        key = (width, height, self._settings.layout, self._settings.calibration)
        if self._grid_cache is None or self._grid_cache[0] != key:
            self._grid_cache = (key, grid_lines(self._settings.layout, self._settings.calibration, (width, height)))
        return self._grid_cache[1]

    def _fail_plan(self, message: str) -> None:
        self._plan_error = message
        self._plan_panel.show_error(message)

    def _retry_plan(self) -> None:
        self._plan_error = None
        self._plan_panel.clear_error()
        self._seen_at.pop("plan", None)
        self._area.dot("plan").set_state("idle")

    # --- toolbar ----------------------------------------------------------------

    def _choose_detector(self, mode: str) -> None:
        """NONE switches detection off but remembers the last detector."""
        on = {"detect_people": False} if mode == NONE else {"detect_people": True, "detector": mode}
        self.changed.emit(replace(self._settings, view=replace(self._settings.view, **on)))

    def _choose_model(self, key: str) -> None:
        on = {"classify_clothing": False} if key == NONE else {"classify_clothing": True, "model_key": key}
        self.changed.emit(replace(self._settings, view=replace(self._settings.view, **on)))

    def _choose_preset(self, name: str) -> None:
        """Detector and clothing model in one change."""
        detector, model = config.MODEL_PRESETS[name]
        view = replace(self._settings.view, detect_people=detector != NONE, classify_clothing=model != NONE)
        if detector != NONE:
            view = replace(view, detector=detector)
        if model != NONE:
            view = replace(view, model_key=model)
        self.changed.emit(replace(self._settings, view=view))

    def _edit(self, path: tuple[str, ...], value) -> None:
        self.changed.emit(replace_path(self._settings, path, value))

    def _sync_panels(self) -> None:
        for name in PANELS:
            self._toolbar.set_panel_visible(name, self._area.is_visible(name))
        crops_open = self._area.is_visible("crops")
        if self._settings is not None and crops_open != self._settings.view.show_crops:
            self._edit(("view", "show_crops"), crops_open)  # the engine only cuts crops while it is open
