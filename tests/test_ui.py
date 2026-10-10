"""Window shell with a fake worker, offscreen."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from jacket.types import Box, ClothingResult
from ren.engine import FrameResult
from ren.plan import Calibration, PlanLayout
from ren.settings import AppSettings, replace_path
from ren.theme import load_fonts
from ren.ui import CALIBRATE, DATA, LIVE, SETTINGS, MainWindow


class FakeWorker(QObject):
    frame_ready = Signal(object, float)
    status = Signal(str)
    failed = Signal(str)
    finishes_in_time = True

    def __init__(self) -> None:
        super().__init__()
        self.latest = None
        self.stopped = False

    def set_settings(self, settings) -> None:
        self.latest = settings

    def show_still(self, image) -> None:
        self.still = image

    def stop(self) -> None:
        self.stopped = True

    def wait(self, timeout: int = 0) -> bool:
        return self.finishes_in_time


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def worker() -> FakeWorker:
    return FakeWorker()


@pytest.fixture
def saved() -> list:
    return []


@pytest.fixture
def window(app, worker, saved) -> MainWindow:
    return MainWindow(worker, AppSettings(), load_fonts(), save=saved.append, ask_name=lambda suggestion: suggestion)


def frame_result(boxes=None, results=None) -> FrameResult:
    frame = np.full((450, 800, 3), 100, dtype=np.uint8)
    return FrameResult(frame, boxes or [], results or [], (), "0", "")


def test_starts_on_the_live_page(window):
    assert window._pages.currentIndex() == LIVE


def test_nav_links_switch_pages(window):
    window._nav.select("settings")
    assert window._pages.currentIndex() == SETTINGS
    window._nav.select("live")
    assert window._pages.currentIndex() == LIVE


def test_escape_walks_back_one_page(window):
    window._open(CALIBRATE)
    window._back()
    assert window._pages.currentIndex() == SETTINGS
    window._back()
    assert window._pages.currentIndex() == LIVE
    window._back()
    assert window._pages.currentIndex() == LIVE  # never leaves the live page


def test_calibrating_still_counts_as_settings_in_the_nav(window):
    window._open(CALIBRATE)
    assert window._nav.current() == "settings"


def test_the_calibrate_button_opens_calibration_and_done_returns(window):
    window._settings_page.calibrate_requested.emit()
    assert window._pages.currentIndex() == CALIBRATE
    window._calibration.finished.emit()
    assert window._pages.currentIndex() == SETTINGS


def test_a_settings_change_is_saved_and_sent_to_the_worker_and_live_page(window, worker, saved):
    changed = replace(AppSettings(), view=replace(AppSettings().view, model_key="clip", mirror=False))
    window._settings_page.changed.emit(changed)
    assert saved == [changed] and worker.latest == changed.view
    assert window._live._settings == changed


def test_an_unchanged_setting_is_not_saved_again(window, saved):
    window._settings_page.changed.emit(AppSettings())
    assert saved == []


def test_out_of_range_values_are_repaired_before_saving(window, saved):
    window._settings_page.changed.emit(replace_path(AppSettings(), ("layout",), PlanLayout(rows=500, cols=0)))
    assert saved[-1].layout == PlanLayout(rows=12, cols=1)


def test_calibration_changes_are_saved(window, saved):
    corners = Calibration((0.2, 0.2), (0.8, 0.2), (0.9, 0.9), (0.1, 0.9))
    window._calibration.changed.emit(corners)
    assert saved[-1].calibration == corners


def test_layout_change_reaches_the_calibration_page(window):
    window._settings_page.changed.emit(replace_path(AppSettings(), ("layout",), PlanLayout(5, 9)))
    assert window._calibration._view._layout == PlanLayout(5, 9)


def test_frames_reach_the_camera_panel(window, worker):
    worker.frame_ready.emit(frame_result(), 24.0)
    assert window._live._camera._image is not None


def test_frames_reach_the_seating_plan_too(window, worker):
    worker.frame_ready.emit(frame_result(), 24.0)
    assert window._live._plan_panel._status.text() == "no one in view"


def test_frames_go_to_calibration_when_that_page_is_open(window, worker):
    window._open(CALIBRATE)
    worker.frame_ready.emit(frame_result([Box(100, 100, 200, 300, 0.9, "person", torso_top_y=150)]), 20.0)
    assert window._calibration._view._image is not None
    assert window._calibration._view._people == [(150.0, 150.0)]


def test_failure_is_shown_not_swallowed(window, worker):
    worker.failed.emit("CameraError: no camera named like 'C920'")
    assert "C920" in window._live._camera.message() and "--camera" in window._live._camera.message()
    assert window._error.text() == "camera stopped" and "C920" in window._error.toolTip()


def test_status_text_is_shown(window, worker):
    worker.status.emit("loading clip model...")
    assert window._status.text() == "loading clip model..."


def test_closing_stops_the_worker(window, worker):
    window.close()
    assert worker.stopped


def test_stuck_worker_exits_cleanly_instead_of_terminating(window, worker, monkeypatch):
    exits = []
    monkeypatch.setattr("ren.ui.os._exit", exits.append)
    worker.finishes_in_time = False
    window.close()
    assert exits == [0] and window.isHidden()


def test_normal_close_never_force_exits(window, worker, monkeypatch):
    exits = []
    monkeypatch.setattr("ren.ui.os._exit", exits.append)
    window.close()
    assert exits == [] and window.isHidden()


def test_the_whole_window_paints_on_every_page(window):
    for index in (LIVE, SETTINGS, CALIBRATE):
        window._open(index)
        assert not window.grab().isNull()


def test_escape_leaves_an_expanded_panel_before_anything_else(window):
    window._live._area.toggle_expanded("plan")
    window._back()
    assert window._live._area.expanded() is None and window._pages.currentIndex() == LIVE


def test_vision_only_hides_the_plan_but_keeps_the_camera(window, worker):
    window.set_vision_only()
    worker.frame_ready.emit(frame_result(), 20.0)
    assert not window._live._area.is_shown("plan") and window._live._camera._image is not None
    assert not window._live._toolbar.action("plan").isChecked()


def test_a_view_menu_tick_is_saved_like_any_other_setting(window, worker, saved):
    window._live._toolbar.action("show_beam").trigger()
    assert saved[-1].view.show_beam is False and worker.latest.show_beam is False


def test_the_toolbar_model_dropdown_switches_the_model(window, worker, saved):
    model = window._live._toolbar._model
    model.activated.emit(model.findData("clip"))
    assert saved[-1].view.model_key == "clip" and worker.latest.model_key == "clip"


def test_a_model_chosen_in_settings_shows_in_the_toolbar(window):
    from dataclasses import replace as _replace
    window._settings_page.changed.emit(_replace(AppSettings(), view=_replace(AppSettings().view, model_key="clip")))
    assert window._live._toolbar.model() == "clip"


def test_there_is_no_footer_any_more(window):
    assert not hasattr(window, "_fps_label")



def test_the_plus_button_adds_a_venue_with_the_suggested_name(window, saved):
    window._live._toolbar._add_venue.click()
    assert saved[-1].venue_names() == ["lt1", "lt2"] and saved[-1].venue == "lt2"
    assert window._live._toolbar.venue() == "lt2"
    assert window._calibration._title.text() == "calibrate lt2"


def test_cancelling_the_name_prompt_adds_nothing(app, worker, saved):
    window = MainWindow(worker, AppSettings(), load_fonts(), save=saved.append, ask_name=lambda suggestion: None)
    window._live._toolbar._add_venue.click()
    assert saved == []


def test_a_rejected_name_is_shown_in_the_header(app, worker, saved):
    window = MainWindow(worker, AppSettings(), load_fonts(), save=saved.append, ask_name=lambda suggestion: "lt1")
    window._live._toolbar._add_venue.click()
    assert saved == [] and "already exists" in window._error.text()


def test_switching_venue_from_the_toolbar_is_saved(window, saved):
    window._live._toolbar._add_venue.click()
    window._live._toolbar._venue.textActivated.emit("lt1")
    assert saved[-1].venue == "lt1"


def test_a_dropped_photo_replaces_the_camera_until_going_back(window, worker, tmp_path):
    import cv2
    photo = tmp_path / "group.jpg"
    cv2.imwrite(str(photo), np.full((40, 60, 3), 80, dtype=np.uint8))
    window._open(DATA)
    window.show_photo(photo)
    assert worker.still is not None and window._pages.currentIndex() == LIVE
    assert "group.jpg" in window._status.text() and not window._back_to_camera.isHidden()
    window.back_to_camera()
    assert worker.still is None and window._back_to_camera.isHidden()


def test_an_unreadable_photo_is_reported(window, worker, tmp_path):
    bad = tmp_path / "broken.jpg"
    bad.write_bytes(b"nope")
    window.show_photo(bad)
    assert "broken.jpg" in window._error.text()


def test_the_data_page_is_in_the_nav(window):
    window._nav.select("data")
    assert window._pages.currentIndex() == DATA
    window._back()
    assert window._pages.currentIndex() == LIVE


def test_a_photo_still_works_when_the_camera_fails(app):
    import threading
    import time as clock
    from ren.ui import FrameWorker

    class Engine:
        def detector_loaded(self, mode): return True
        def is_loaded(self, key): return True
        def load_detector(self, mode): pass
        def process(self, frame, settings): return frame_result()

    def broken_camera():
        raise RuntimeError("no camera")

    worker = FrameWorker(broken_camera, Engine(), replace(AppSettings().view, detect_people=False))
    failures, frames = [], []
    worker.failed.connect(failures.append)
    worker.frame_ready.connect(lambda result, fps: frames.append(result))
    thread = threading.Thread(target=worker.run)
    thread.start()
    deadline = clock.monotonic() + 5
    while not failures and clock.monotonic() < deadline:
        app.processEvents()  # signals from the worker thread are queued
        clock.sleep(0.01)
    worker.show_still(np.zeros((10, 10, 3), dtype=np.uint8))
    while not frames and clock.monotonic() < deadline:
        app.processEvents()
        clock.sleep(0.01)
    worker.stop()
    thread.join(5)
    assert failures == ["RuntimeError: no camera"] and frames
