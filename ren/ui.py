"""Demo window. Run: python -m ren.ui"""
import argparse
import dataclasses
import logging
import os
import signal
import sys
import time
from pathlib import Path
from typing import Callable, ContextManager

import cv2

# cv2's Qt5 plugin path breaks PySide6
for _name in ("QT_QPA_PLATFORM_PLUGIN_PATH", "QT_QPA_FONTDIR"):
    os.environ.pop(_name, None)

from PySide6.QtCore import QObject, QThread, QTimer, Signal  # noqa: E402
from PySide6.QtGui import QKeySequence, QShortcut  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QApplication, QFrame, QHBoxLayout, QInputDialog, QLabel, QStackedWidget, QVBoxLayout, QWidget,
)

from jacket import config  # noqa: E402
from jacket.camera import Camera, find_camera_index  # noqa: E402
from jacket.classifier import ClothingClassifier  # noqa: E402
from jacket.detectors import build_detector  # noqa: E402
from jacket.pipeline import ClothingPipeline  # noqa: E402
from jacket.sources import FrameSource, ImageSource, VideoFileSource  # noqa: E402
from ren.engine import FrameEngine, FrameResult, ViewSettings  # noqa: E402
from ren.pages.calibration import CalibrationPage  # noqa: E402
from ren.pages.live import LivePage  # noqa: E402
from ren.pages.settings_page import SettingsPage  # noqa: E402
from ren.plan import Calibration  # noqa: E402
from ren.settings import AppSettings, add_venue, load_settings, replace_path, sanitize, save_settings  # noqa: E402
from ren.venues import suggest_name  # noqa: E402
from ren.theme import Fonts, build_stylesheet, load_fonts  # noqa: E402
from ren.widgets import DotMark, NavChoice  # noqa: E402

logger = logging.getLogger("ren.ui")

SourceFactory = Callable[[], ContextManager[FrameSource]]
LIVE, SETTINGS, CALIBRATE = range(3)
NAV_PAGES = ["live", "settings"]


class FrameWorker(QThread):
    """Reads frames and runs the engine off-thread."""

    frame_ready = Signal(object, float)  # FrameResult, instantaneous fps
    status = Signal(str)
    failed = Signal(str)

    def __init__(self, open_source: SourceFactory, engine: FrameEngine, settings: ViewSettings) -> None:
        super().__init__()
        self._open_source = open_source
        self._engine = engine
        self._settings = settings
        self._stop_requested = False

    def set_settings(self, settings: ViewSettings) -> None:
        self._settings = settings  # replacing one reference is atomic

    def stop(self) -> None:
        self._stop_requested = True

    def run(self) -> None:
        # Broad catch on purpose; dead threads look frozen
        try:
            with self._open_source() as source:
                self._load_if_needed(self._settings)
                previous = time.perf_counter()
                warm_up_due = True
                while not self._stop_requested:
                    frame = source.read()
                    settings = self._settings
                    self._load_if_needed(settings)
                    result = self._engine.process(frame, settings)
                    now = time.perf_counter()
                    self.frame_ready.emit(result, 1 / max(now - previous, 1e-6))
                    previous = now
                    if warm_up_due:  # after the first frame, so the picture shows at once
                        warm_up_due = False
                        self._warm_up_detectors()
        except Exception as error:
            logger.exception("worker stopped")
            self.failed.emit(f"{type(error).__name__}: {error}")

    def _load_if_needed(self, settings: ViewSettings) -> None:
        if settings.detect_people and not self._engine.detector_loaded(settings.detector):
            self.status.emit(f"loading {settings.detector} detector...")
            self._engine.load_detector(settings.detector)
            self.status.emit("")
        if settings.detect_people and settings.classify_clothing and not self._engine.is_loaded(settings.model_key):
            self.status.emit(f"loading {settings.model_key} model...")
            self._engine.load(settings.model_key)
            self.status.emit("")

    def _warm_up_detectors(self) -> None:
        """Load every detector now, so swapping later is instant."""
        for mode in config.DETECTOR_MODES:
            try:
                self._engine.load_detector(mode)
            except Exception:  # the mode in use still works; this one loads (and fails loudly) when picked
                logger.warning("could not preload the %s detector", mode, exc_info=True)


class MainWindow(QWidget):
    def __init__(
        self, worker: FrameWorker | QObject, settings: AppSettings, fonts: Fonts,
        save: Callable[[AppSettings], None] = save_settings,
        ask_name: Callable[[str], str | None] | None = None,
    ) -> None:
        super().__init__()
        self._worker, self._settings, self._fonts, self._save = worker, settings, fonts, save
        self._ask_name = ask_name or self._ask_name_dialog
        self.setWindowTitle(config.UI_WINDOW_TITLE)
        self.resize(*config.UI_START_SIZE)
        self.setMinimumSize(*config.UI_MIN_SIZE)
        self.setStyleSheet(build_stylesheet())

        self._live = LivePage(fonts, settings)
        self._settings_page = SettingsPage(fonts, settings)
        self._calibration = CalibrationPage(fonts, settings.calibration, settings.layout)
        self._calibration.set_venue(settings.venue)
        self._pages = QStackedWidget()
        for page in (self._live, self._settings_page, self._calibration):
            self._pages.addWidget(page)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self._build_header())
        root.addWidget(self._pages, 1)

        self._settings_page.changed.connect(self._on_changed)
        self._live.changed.connect(self._on_changed)
        self._live.venue_add_requested.connect(self._add_venue)
        self._settings_page.calibrate_requested.connect(lambda: self._open(CALIBRATE))
        self._calibration.changed.connect(self._on_calibration)
        self._calibration.finished.connect(lambda: self._open(SETTINGS))
        worker.frame_ready.connect(self._show_frame)
        worker.status.connect(self._status.setText)
        worker.failed.connect(self._show_failure)

        QShortcut(QKeySequence("Q"), self, activated=self.close)
        QShortcut(QKeySequence("Esc"), self, activated=self._back)
        QShortcut(QKeySequence("1"), self, activated=lambda: self._open(LIVE))
        QShortcut(QKeySequence("2"), self, activated=lambda: self._open(SETTINGS))
        QShortcut(QKeySequence("F11"), self, activated=self._toggle_fullscreen)

    # --- layout -----------------------------------------------------------------

    def _label(self, text: str, px: int, color_name: str, weight: str = "regular") -> QLabel:
        label = QLabel(text)
        label.setFont(self._fonts.text(px, weight))
        label.setStyleSheet(f"color: {config.UI_COLORS[color_name]};")
        return label

    def _build_brand(self) -> QWidget:
        """Four pastel dots, then a bold ren.ui."""
        brand = QWidget()
        row = QHBoxLayout(brand)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(12)
        name, dot, suffix = config.UI_APP_NAME.partition(".")
        title = self._label(f'{name}<span style="color:{config.UI_BRAND_DOTS[0]}">{dot}</span>{suffix}',
                            config.UI_FONT_PX["brand"], "text", "bold")
        title.setObjectName("brand")
        row.addWidget(DotMark())
        row.addWidget(title)
        return brand

    def _build_header(self) -> QFrame:
        header = QFrame()
        header.setObjectName("header")
        header.setFixedHeight(76)
        layout = QHBoxLayout(header)
        layout.setContentsMargins(config.UI_PAGE_PADDING_PX + 4, 8, config.UI_PAGE_PADDING_PX, 0)
        layout.addWidget(self._build_brand())
        layout.addStretch(1)
        small = config.UI_FONT_PX["small"]
        self._status = self._label("", small, "muted")
        self._error = self._label("", small, "error")
        layout.addWidget(self._status)
        layout.addWidget(self._error)
        layout.addSpacing(28)
        self._nav = NavChoice("", NAV_PAGES, "live", self._fonts)
        self._nav.chosen.connect(lambda name: self._open(NAV_PAGES.index(name)))
        layout.addWidget(self._nav)
        return header

    # --- navigation -------------------------------------------------------------

    def _open(self, index: int) -> None:
        self._pages.setCurrentIndex(index)
        nav_name = NAV_PAGES[min(index, SETTINGS)]  # calibrating counts as settings
        if self._nav.current() != nav_name:
            self._nav.set_current(nav_name)

    def _back(self) -> None:
        index = self._pages.currentIndex()
        if index == LIVE:
            self._live.collapse_expanded()
        elif index == CALIBRATE:
            self._open(SETTINGS)
        elif index == SETTINGS:
            self._open(LIVE)

    # --- settings ---------------------------------------------------------------

    def _on_changed(self, settings: AppSettings) -> None:
        settings = sanitize(settings)
        self._error.setText("")
        if settings == self._settings:
            return
        self._settings = settings
        self._save(settings)
        self._worker.set_settings(settings.view)
        self._live.set_settings(settings)
        self._settings_page.set_settings(settings)
        self._calibration.set_layout(settings.layout)
        self._calibration.set_calibration(settings.calibration)
        self._calibration.set_venue(settings.venue)

    def _on_calibration(self, calibration: Calibration) -> None:
        self._on_changed(replace_path(self._settings, ("calibration",), calibration))

    def _ask_name_dialog(self, suggestion: str) -> str | None:
        text, ok = QInputDialog.getText(self, "new venue", "name (copies the venue in use)", text=suggestion)
        return text if ok else None

    def _add_venue(self) -> None:
        text = self._ask_name(suggest_name(self._settings.venue_names()))
        if text is None:
            return
        try:
            self._on_changed(add_venue(self._settings, text))
        except ValueError as error:
            self._error.setText(str(error))

    # --- frames -----------------------------------------------------------------

    def _show_frame(self, result: FrameResult, fps: float) -> None:
        self._error.setText("")
        page = self._pages.currentIndex()
        if page == LIVE:
            self._live.show_result(result)
        elif page == CALIBRATE:
            self._calibration.show_result(result)

    def _show_failure(self, message: str) -> None:
        hint = "try --camera N, --video FILE or --image FILE"
        self._live.show_failure(f"no video\n\n{message}\n\n{hint}")
        self._error.setText("camera stopped")
        self._error.setToolTip(message)

    def set_vision_only(self) -> None:
        """The fallback: just the camera and the vision model."""
        self._live.set_panel_visible("plan", False)

    def _toggle_fullscreen(self) -> None:
        self.showNormal() if self.isFullScreen() else self.showFullScreen()

    def closeEvent(self, event) -> None:
        self.hide()  # vanish at once, even if the worker is busy
        self._worker.stop()
        if not self._worker.wait(config.UI_SHUTDOWN_WAIT_MS):
            # terminate() mid-load aborts with a core dump
            logger.warning("worker still busy (model loading?); exiting without waiting")
            logging.shutdown()
            os._exit(0)
        super().closeEvent(event)


def make_source_factory(args: argparse.Namespace) -> SourceFactory:
    """Opened in the worker thread, so errors show."""
    if args.image:
        return lambda: ImageSource(args.image)
    if args.video:
        return lambda: VideoFileSource(args.video)
    return lambda: Camera(
        args.camera if args.camera is not None else find_camera_index(config.CAMERA_NAME_HINT),
        config.CAMERA_WIDTH, config.CAMERA_HEIGHT, config.CAMERA_WARMUP_FRAMES,
    )


def close_on_signals(window: QWidget) -> None:
    """Ctrl+C and SIGTERM close the window."""
    for signal_number in (signal.SIGINT, getattr(signal, "SIGTERM", signal.SIGINT)):
        signal.signal(signal_number, lambda *_: window.close())
    tick = QTimer(window)
    tick.timeout.connect(lambda: None)
    tick.start(config.UI_SIGNAL_TICK_MS)


def build_engine() -> FrameEngine:
    """Cheap: models load in the worker."""
    return FrameEngine(
        build_detector, lambda key: ClothingPipeline(ClothingClassifier(key, config.DEVICE).classify)
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(config.CLIP_MODELS), help="override the saved model")
    parser.add_argument("--detector", choices=config.DETECTOR_MODES, help="override the saved detector")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--camera", type=int, help="camera index; default finds the C920 by name")
    source.add_argument("--video", type=Path, help="loop a recorded video instead of the camera")
    source.add_argument("--image", type=Path, help="use one still image instead of the camera")
    parser.add_argument("--fullscreen", action="store_true")
    parser.add_argument("--vision-only", action="store_true", help="start with the seating plan hidden")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    for name in ("jacket", "ren"):
        logging.getLogger(name).setLevel(logging.INFO)  # keep library logs quiet

    app = QApplication(sys.argv)
    fonts = load_fonts()
    app.setFont(fonts.text(config.UI_FONT_PX["body"]))
    settings = load_settings()
    if args.model:
        settings = dataclasses.replace(settings, view=dataclasses.replace(settings.view, model_key=args.model))
    if args.detector:
        settings = dataclasses.replace(settings, view=dataclasses.replace(settings.view, detector=args.detector))
    worker = FrameWorker(make_source_factory(args), build_engine(), settings.view)
    window = MainWindow(worker, settings, fonts)
    close_on_signals(window)
    if args.vision_only:
        window.set_vision_only()
    window.showFullScreen() if args.fullscreen else window.show()
    worker.start()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
