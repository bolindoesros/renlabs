import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QComboBox, QLabel

from jacket.types import Box, ClothingResult
from ren.engine import FrameResult
from ren.pages.calibration import CalibrationPage
from ren.pages.live import LivePage
from ren.pages.settings_page import SECTIONS, SettingsPage
from ren.plan import Calibration, PlanLayout
from ren.decision import NeedWeights
from ren.settings import AppSettings, replace_path
from ren.theme import load_fonts
from ren.widgets import TextButton, SwitchRow, Stepper

LIGHT = ClothingResult("light", 0.05, "")
WARM = ClothingResult("warm", 0.95, "")
SQUARE = Calibration((0.1, 0.125), (0.9, 0.125), (0.9, 0.875), (0.1, 0.875))
FRAME = np.full((800, 1000, 3), 90, dtype=np.uint8)


@pytest.fixture(scope="module")
def fonts():
    QApplication.instance() or QApplication([])
    return load_fonts()


def person(x: float, y: float) -> Box:
    return Box(int(x - 30), int(y - 40), int(x + 30), int(y + 160), 0.9, "person", torso_top_y=int(y))


def seat_pixel(row: int, col: int) -> tuple[float, float]:
    return 100 + (col + 0.5) / 6 * 800, 700 - (row + 0.5) / 3 * 600


class Clock:
    now = 0.0

    def __call__(self) -> float:
        return self.now


def live_page(fonts, clock=None, **view_changes):
    settings = replace(replace_path(AppSettings(), ("calibration",), SQUARE), view=replace(AppSettings().view, **view_changes))
    return LivePage(fonts, settings, clock or Clock()), settings


def result(boxes, results) -> FrameResult:
    return FrameResult(FRAME, boxes, results, (), "", "")


# ---- live page --------------------------------------------------------------------

def status(page) -> str:
    return page._plan_panel._status.text()


def settle(page, clock, boxes, results, seconds: float = 4.0, step: float = 0.2):
    for tick in range(int(seconds / step)):
        clock.now = tick * step
        page.show_result(result(boxes, results))


def test_a_hot_person_makes_the_vent_aim_at_their_zone(fonts):
    clock = Clock()
    page, _ = live_page(fonts, clock)
    settle(page, clock, [person(*seat_pixel(2, 5))], [LIGHT])
    assert status(page) == "aiming at back right"


def test_an_empty_view_says_no_one_is_there(fonts):
    page, _ = live_page(fonts)
    page.show_result(result([], []))
    assert status(page) == "no one in view"


def test_only_hooded_people_leave_the_vent_shut_and_say_why(fonts):
    page, _ = live_page(fonts)
    page.show_result(result([person(*seat_pixel(0, 0)), person(*seat_pixel(0, 1))], [WARM, WARM]))
    assert status(page) == "vent shut, everyone seated is in a jacket"


def test_people_outside_the_plan_are_marked_and_explained(fonts):
    page, _ = live_page(fonts)
    page.show_result(result([person(20, 20)], [LIGHT]))  # a hot person nowhere near the seats
    assert status(page) == "everyone is outside the plan, check calibration"
    assert page._camera._unseated == frozenset({0})


def test_one_person_outside_the_plan_is_mentioned_but_the_rest_still_count(fonts):
    clock = Clock()
    page, _ = live_page(fonts, clock)
    settle(page, clock, [person(*seat_pixel(0, 0)), person(20, 20)], [LIGHT, LIGHT])
    assert status(page).startswith("aiming at front left") and status(page).endswith("1 outside the plan")
    assert page._camera._unseated == frozenset({1})


def test_the_seat_grid_follows_its_layer(fonts):
    page, settings = live_page(fonts, show_grid=True)
    page.show_result(result([], []))
    assert len(page._camera._grid) == 6 + 1 + 3 + 1
    page.set_settings(replace(settings, view=replace(settings.view, show_grid=False)))
    page.show_result(result([], []))
    assert page._camera._grid == []


def test_the_legend_shows_the_weights_the_algorithm_uses(fonts):
    page, settings = live_page(fonts)
    assert [text for _, text in page._plan_panel._legend._entries] == ["needs air 1", "maybe 0.5", "fine 0.15"]
    heavier = replace(settings, decision=replace(settings.decision, weights=NeedWeights(hot=0.9, unsure=0.4, cold=0.1)))
    page.set_settings(heavier)
    assert [text for _, text in page._plan_panel._legend._entries] == ["needs air 0.9", "maybe 0.4", "fine 0.1"]


def test_display_only_changes_keep_the_vent_state(fonts):
    page, settings = live_page(fonts)
    maker = page._maker
    page.set_settings(replace(settings, view=replace(settings.view, show_labels=False)))
    assert page._maker is maker


def test_layout_changes_rebuild_the_decision_maker(fonts):
    page, settings = live_page(fonts)
    maker = page._maker
    page.set_settings(replace_path(settings, ("layout",), PlanLayout(4, 8)))
    assert page._maker is not maker


def test_hiding_the_plan_stops_the_plan_work_but_not_the_camera(fonts):
    clock = Clock()
    page, _ = live_page(fonts, clock)
    page.set_panel_visible("plan", False)
    page.show_result(result([person(*seat_pixel(0, 0))], [LIGHT]))
    assert page._maker._last is None  # the decision maker was never called
    assert page._camera._image is not None


def test_hiding_the_camera_still_runs_the_plan(fonts):
    clock = Clock()
    page, _ = live_page(fonts, clock)
    page.set_panel_visible("camera", False)
    page.show_result(result([person(*seat_pixel(0, 0))], [LIGHT]))
    assert page._maker._last is not None and page._camera._image is None


def test_a_plan_failure_leaves_the_camera_running(fonts, monkeypatch):
    page, _ = live_page(fonts)
    calls = []

    def broken(*args, **kwargs):
        calls.append(1)
        raise RuntimeError("calibration exploded")

    monkeypatch.setattr("ren.pages.live.read_scene", broken)
    page.show_result(result([person(*seat_pixel(0, 0))], [LIGHT]))
    assert page._camera._image is not None  # the vision half is unharmed
    assert page._plan_panel.showing_error() and "calibration exploded" in page._plan_panel.error_text()
    page.show_result(result([person(*seat_pixel(0, 0))], [LIGHT]))
    assert len(calls) == 1  # it does not retry (and spam) on every frame


def test_retry_runs_the_plan_again_and_clears_the_error(fonts, monkeypatch):
    page, _ = live_page(fonts)
    monkeypatch.setattr("ren.pages.live.read_scene", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    page.show_result(result([], []))
    assert page._plan_panel.showing_error()
    monkeypatch.undo()
    page._plan_panel.retry_requested.emit()
    page.show_result(result([], []))
    assert not page._plan_panel.showing_error() and status(page) == "no one in view"


def test_changing_a_setting_retries_a_failed_plan(fonts, monkeypatch):
    page, settings = live_page(fonts)
    monkeypatch.setattr("ren.pages.live.read_scene", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    page.show_result(result([], []))
    monkeypatch.undo()
    page.set_settings(replace_path(settings, ("layout",), PlanLayout(4, 8)))
    assert not page._plan_panel.showing_error()







def test_a_camera_failure_shows_in_the_camera_panel(fonts):
    page, _ = live_page(fonts)
    page.show_failure("no video\n\nCameraError: nope")
    assert "nope" in page._camera.message()


def test_expanding_a_panel_hides_the_other_until_restored(fonts):
    page, _ = live_page(fonts)
    page._area.toggle_expanded("camera")
    assert page._area.is_shown("camera") and not page._area.is_shown("plan")
    assert page.collapse_expanded() is True and page._area.is_shown("plan")
    assert page.collapse_expanded() is False



def test_the_live_page_paints_in_every_arrangement(fonts):
    page, _ = live_page(fonts)
    page.resize(1320, 700)
    page.show_result(result([person(*seat_pixel(1, 1))], [LIGHT]))
    assert not page.grab().isNull()
    page._area.toggle_expanded("plan")
    assert not page.grab().isNull()
    page._area.restore()
    page.set_panel_visible("camera", False)
    page.set_panel_visible("plan", False)
    assert not page.grab().isNull()  # both hidden shows a note, not a crash


def test_side_by_side_panels_share_the_width_and_never_overlap(fonts):
    page, _ = live_page(fonts)
    page.resize(1320, 700)
    page.show()
    QApplication.processEvents()
    camera, plan = page._camera.geometry(), page._plan_panel.geometry()
    camera_in_page = page._camera.mapTo(page, camera.topLeft()).x()
    plan_in_page = page._plan_panel.mapTo(page, plan.topLeft()).x()
    assert plan_in_page > camera_in_page + camera.width()  # plan sits to the right of the camera
    assert camera.width() > 250 and plan.width() > 250
    page.hide()


# ---- settings page ----------------------------------------------------------------

def find(page: SettingsPage, section: str, kind):
    return page._stack.widget(list(SECTIONS).index(section)).findChildren(kind)


def test_every_section_has_controls(fonts):
    page = SettingsPage(fonts, AppSettings())
    for name in SECTIONS:
        controls = sum(len(find(page, name, kind)) for kind in (SwitchRow, Stepper, TextButton, QComboBox))
        assert controls >= 1, f"section {name} is empty"
    assert list(SECTIONS) == ["venues", "vision", "seating plan", "decision", "vent"]


def test_nav_links_switch_sections(fonts):
    page = SettingsPage(fonts, AppSettings())
    page._links[3].click()
    assert page._stack.currentIndex() == 3


def test_a_stepper_click_announces_the_new_settings(fonts):
    page = SettingsPage(fonts, AppSettings())
    announced = []
    page.changed.connect(announced.append)
    rows_stepper = find(page, "seating plan", Stepper)[0]
    QTest.mouseClick(rows_stepper._more, Qt.MouseButton.LeftButton)
    assert announced[-1].layout.rows == AppSettings().layout.rows + 1
    assert isinstance(announced[-1].layout.rows, int)  # whole-number fields stay whole


def test_nested_weights_can_be_edited(fonts):
    page = SettingsPage(fonts, AppSettings())
    announced = []
    page.changed.connect(announced.append)
    hot = find(page, "decision", Stepper)[0]
    QTest.mouseClick(hot._less, Qt.MouseButton.LeftButton)
    assert announced[-1].decision.weights.hot == pytest.approx(0.95)
    assert announced[-1].decision.weights.cold == AppSettings().decision.weights.cold


def test_the_model_dropdown_announces_the_choice(fonts):
    page = SettingsPage(fonts, AppSettings())
    announced = []
    page.changed.connect(announced.append)
    dropdown = find(page, "vision", QComboBox)[0]
    dropdown.activated.emit(dropdown.findData("clip"))
    assert announced[-1].view.model_key == "clip"


def test_the_detector_dropdown_announces_the_choice(fonts):
    page = SettingsPage(fonts, AppSettings())
    announced = []
    page.changed.connect(announced.append)
    detector = find(page, "vision", QComboBox)[1]
    assert detector.itemText(detector.findData("both")) == "YOLO11n-pose + YOLOv8n-head"
    detector.activated.emit(detector.findData("both"))
    assert announced[-1].view.detector == "both"


def test_the_toolbar_swaps_the_detector(fonts):
    page, settings = live_page(fonts)
    announced = []
    page.changed.connect(announced.append)
    parts = page._toolbar._detector_parts
    parts["head"].trigger()  # tick head alongside body
    assert announced[-1].view.detector == "both"
    assert page._toolbar._detector.text() == "YOLO11n-pose + YOLOv8n-head"
    parts["body"].trigger()  # untick body
    assert announced[-1].view.detector == "head"
    page.set_settings(replace(settings, view=replace(settings.view, detector="both")))
    assert page._toolbar.detector() == "both"


def test_a_toggle_announces_the_change(fonts):
    page = SettingsPage(fonts, AppSettings())
    announced = []
    page.changed.connect(announced.append)
    toggles = find(page, "vision", SwitchRow)
    toggles[2].setChecked(not toggles[2].isChecked())  # human detection, clothing detection, mirror camera
    assert announced[-1].view.mirror is (not AppSettings().view.mirror)


def test_settings_changed_elsewhere_refresh_without_announcing(fonts):
    page = SettingsPage(fonts, AppSettings())
    announced = []
    page.changed.connect(announced.append)
    page.set_settings(replace(replace_path(AppSettings(), ("layout",), PlanLayout(5, 9)), view=replace(AppSettings().view, mirror=True)))
    assert announced == []
    assert find(page, "seating plan", Stepper)[0].value() == 5
    assert find(page, "vision", SwitchRow)[2].isChecked()


def test_reset_needs_two_clicks(fonts):
    from ren.settings import add_venue
    rooms = add_venue(replace_path(AppSettings(), ("layout",), PlanLayout(5, 9)), "lt2")
    changed = replace(rooms, view=replace(rooms.view, mirror=True), decision=replace(rooms.decision, aim_mode="focus"))
    page = SettingsPage(fonts, changed)
    announced = []
    page.changed.connect(announced.append)
    QTest.mouseClick(page._reset, Qt.MouseButton.LeftButton)
    assert announced == [] and "confirm" in page._reset.text()
    QTest.mouseClick(page._reset, Qt.MouseButton.LeftButton)
    reset = announced[-1]
    assert reset.view == AppSettings().view and reset.decision == AppSettings().decision
    assert reset.venues == changed.venues and reset.venue == "lt2"  # venues survive a reset
    assert page._reset.text() == "reset to defaults"


def test_the_calibration_row_asks_to_open_calibration(fonts):
    page = SettingsPage(fonts, AppSettings())
    requested = []
    page.calibrate_requested.connect(lambda: requested.append(True))
    edit = [b for b in find(page, "seating plan", TextButton) if b.text() == "edit"][0]
    QTest.mouseClick(edit, Qt.MouseButton.LeftButton)
    assert requested == [True]


def test_settings_page_paints_every_section(fonts):
    page = SettingsPage(fonts, AppSettings())
    page.resize(1200, 600)
    for index in range(len(SECTIONS)):
        page._stack.setCurrentIndex(index)
        assert not page.grab().isNull()


# ---- calibration page -------------------------------------------------------------

def test_done_finishes_and_reset_restores_the_default(fonts):
    page = CalibrationPage(fonts, SQUARE, PlanLayout())
    events = []
    page.finished.connect(lambda: events.append("done"))
    page.changed.connect(events.append)
    QTest.mouseClick(page._done, Qt.MouseButton.LeftButton)
    QTest.mouseClick(page._reset, Qt.MouseButton.LeftButton)
    assert events == ["done", Calibration()]


def test_calibration_page_shows_people_where_the_shoulder_line_is(fonts):
    page = CalibrationPage(fonts, SQUARE, PlanLayout())
    page.show_result(FrameResult(FRAME, [person(300, 400)], [], (), "", ""))
    assert page._view._people == [(300.0, 400.0)]


def test_both_panels_start_with_equal_widths(fonts):
    page, _ = live_page(fonts)
    page.resize(1320, 700)
    page.show()
    QApplication.processEvents()
    camera, plan = page._camera.width(), page._plan_panel.width()
    assert abs(camera - plan) <= 0.1 * max(camera, plan)
    page.hide()


def test_both_panels_stay_usable_in_the_smallest_window(fonts):
    page, _ = live_page(fonts)
    page.resize(1100, 600)
    page.show()
    QApplication.processEvents()
    assert page._camera.width() >= 250 and page._plan_panel.width() >= 250
    page.hide()


def test_a_long_status_wraps_instead_of_widening_the_panel(fonts):
    page, _ = live_page(fonts)
    page.resize(1100, 600)
    page.show()
    QApplication.processEvents()
    width = page._plan_panel.width()
    page._plan_panel.set_status("everyone is outside the plan, check calibration and try again with a longer sentence")
    QApplication.processEvents()
    assert page._plan_panel.width() == width
    page.hide()



# ---- toolbar on the live page ------------------------------------------------------

def test_ticking_a_view_item_announces_the_new_settings(fonts):
    page, _ = live_page(fonts)
    announced = []
    page.changed.connect(announced.append)
    page._toolbar.action("show_people").trigger()
    assert announced[-1].view.show_people is False and announced[-1].view.show_image is True


def test_the_view_menu_follows_settings_changed_elsewhere_without_announcing(fonts):
    page, settings = live_page(fonts)
    announced = []
    page.changed.connect(announced.append)
    page.set_settings(replace(settings, view=replace(settings.view, show_beam=False, show_zones=False)))
    assert announced == []
    assert not page._toolbar.action("show_beam").isChecked() and not page._toolbar.action("show_zones").isChecked()
    assert page._toolbar.action("show_seats").isChecked()


def test_the_panel_items_and_the_hide_icon_agree(fonts):
    page, _ = live_page(fonts)
    page._toolbar.action("plan").trigger()  # untick
    assert not page._area.is_visible("plan")
    page._toolbar.action("plan").trigger()  # tick again
    assert page._area.is_visible("plan")
    QTest.mouseClick(page._area._panels["camera"]._hide, Qt.MouseButton.LeftButton)
    assert not page._toolbar.action("camera").isChecked() and page._toolbar.action("plan").isChecked()


def test_no_detector_shows_none_and_greys_out_the_clothing_model(fonts):
    page, settings = live_page(fonts)
    page.set_settings(replace(settings, view=replace(settings.view, detect_people=False)))
    assert page._toolbar._detector.text() == "no detector" and not page._toolbar._model.isEnabled()
    page.set_settings(replace(settings, view=replace(settings.view, classify_clothing=False)))
    assert page._toolbar._model.currentData() == "none" and page._toolbar._model.isEnabled()


def test_unticking_every_detector_gives_the_plain_stream(fonts):
    page, _ = live_page(fonts)  # body ticked
    announced = []
    page.changed.connect(announced.append)
    page._toolbar._detector_parts["body"].trigger()
    assert announced[-1].view.detect_people is False and announced[-1].view.detector == "body"  # remembered
    page.set_settings(announced[-1])
    page._toolbar._detector_parts["head"].trigger()
    assert announced[-1].view.detect_people is True and announced[-1].view.detector == "head"


def test_choosing_none_for_clothing_stops_classifying(fonts):
    page, _ = live_page(fonts)
    announced = []
    page.changed.connect(announced.append)
    model = page._toolbar._model
    model.activated.emit(model.findData("none"))
    assert announced[-1].view.classify_clothing is False
    page.set_settings(announced[-1])
    model.activated.emit(model.findData("clip"))
    assert announced[-1].view.classify_clothing is True and announced[-1].view.model_key == "clip"


def test_stepper_buttons_line_up_across_rows(fonts):
    page = SettingsPage(fonts, AppSettings())
    page.resize(1200, 700)
    page._links[list(SECTIONS).index("decision")].click()
    page.show()
    QApplication.processEvents()
    lefts = {stepper._less.mapTo(page, stepper._less.rect().topLeft()).x() for stepper in find(page, "decision", Stepper)}
    assert len(lefts) == 1 and lefts != {0}  # aligned, and actually laid out
    page.hide()



# ---- venues in settings ------------------------------------------------------------

def venue_page(fonts, settings=None):
    page = SettingsPage(fonts, settings or AppSettings())
    announced = []
    page.changed.connect(announced.append)
    return page, announced


def test_the_venue_list_shows_every_venue_and_marks_the_one_in_use(fonts):
    from ren.settings import add_venue
    page, _ = venue_page(fonts, add_venue(AppSettings(), "lt2"))
    rows = page._venues.rows()
    assert [r.findChild(QLabel).text() for r in rows] == ["lt1", "lt2"]
    assert not rows[1].use.isEnabled() and rows[1].use.text() == "in use" and rows[0].use.isEnabled()


def test_adding_a_venue_from_settings(fonts):
    page, announced = venue_page(fonts)
    page._venues._name.setText("LT 2")
    page._venues._name.returnPressed.emit()
    assert announced[-1].venue == "lt 2" and announced[-1].venue_names() == ["lt1", "lt 2"]
    assert page._venues._name.text() == ""


def test_a_bad_venue_name_shows_an_error_and_changes_nothing(fonts):
    page, announced = venue_page(fonts)
    page._venues._name.setText("lt1")
    page._venues._name.returnPressed.emit()
    assert announced == [] and "already exists" in page._venues._error.text()


def test_use_switches_venue(fonts):
    from ren.settings import add_venue
    page, announced = venue_page(fonts, add_venue(AppSettings(), "lt2"))
    QTest.mouseClick(page._venues.rows()[0].use, Qt.MouseButton.LeftButton)
    assert announced[-1].venue == "lt1"


def test_remove_needs_two_clicks(fonts):
    from ren.settings import add_venue
    page, announced = venue_page(fonts, add_venue(AppSettings(), "lt2"))
    remove = page._venues.rows()[1].remove
    QTest.mouseClick(remove, Qt.MouseButton.LeftButton)
    assert announced == [] and remove.text() == "sure?"
    QTest.mouseClick(remove, Qt.MouseButton.LeftButton)
    assert announced[-1].venue_names() == ["lt1"]


def test_the_only_venue_cannot_be_removed(fonts):
    page, _ = venue_page(fonts)
    assert not page._venues.rows()[0].remove.isEnabled()


def test_room_sections_say_which_venue_they_edit(fonts):
    from ren.settings import add_venue
    page, _ = venue_page(fonts, add_venue(AppSettings(), "lt2"))
    assert [label.text() for label in page._room_labels] == ["for lt2", "for lt2"]
    page.set_settings(AppSettings())
    assert [label.text() for label in page._room_labels] == ["for lt1", "for lt1"]


def test_room_edits_from_settings_go_to_the_venue_in_use(fonts):
    from ren.settings import add_venue, select_venue
    page, announced = venue_page(fonts, add_venue(AppSettings(), "lt2"))
    QTest.mouseClick(find(page, "seating plan", Stepper)[0]._more, Qt.MouseButton.LeftButton)
    edited = announced[-1]
    assert edited.layout.rows == AppSettings().layout.rows + 1
    assert select_venue(edited, "lt1").layout.rows == AppSettings().layout.rows


def test_switching_venue_on_the_live_page_rebuilds_the_plan(fonts):
    from ren.settings import add_venue, select_venue
    page, settings = live_page(fonts)
    two = replace_path(add_venue(settings, "lt2"), ("layout",), PlanLayout(2, 4))
    page.set_settings(two)
    assert page._zones and page._plan_panel._view._layout == PlanLayout(2, 4)
    page.set_settings(select_venue(two, "lt1"))
    assert page._plan_panel._view._layout == settings.layout
    assert page._toolbar.venue() == "lt1"


def test_clicking_the_seat_map_removes_and_restores_a_seat(fonts):
    from ren.seating_view import SeatingPlanView
    page = SettingsPage(fonts, AppSettings())
    page.resize(1200, 900)
    announced = []
    page.changed.connect(announced.append)
    seat_map = next(v for v in page.findChildren(SeatingPlanView) if v._editable)
    seat_map.resize(600, 300)
    target = seat_map.cell_rect(0, 1).center().toPoint()
    QTest.mouseClick(seat_map, Qt.MouseButton.LeftButton, pos=target)
    assert not announced[-1].layout.has_seat(0, 1)
    QTest.mouseClick(seat_map, Qt.MouseButton.LeftButton, pos=target)
    assert announced[-1].layout.has_seat(0, 1)


def test_hiding_the_crops_panel_stops_cutting_crops(fonts):
    page, _ = live_page(fonts, show_crops=True)
    announced = []
    page.changed.connect(announced.append)
    page._toolbar.action("crops").trigger()
    assert announced[-1].view.show_crops is False
    page.set_settings(announced[-1])
    assert not page._area.is_visible("crops")
