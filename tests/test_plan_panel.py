import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication

from ren.decision import Decision
from ren.plan import PlanLayout, SeatReading, SeatingScene, make_zones
from ren.plan_panel import LEGEND, PlanPanel, status_text
from ren.theme import load_fonts
from ren.decision import VentSettings
from ren.engine import ViewSettings

LAYOUT = PlanLayout(3, 6)


def decision(**changes) -> Decision:
    base = dict(
        phase="aiming", closed=False, rotation_deg=0.0, plan_azimuth_deg=0.0, tilt_deg=20.0,
        command_rotation_deg=0.0, command_tilt_deg=20.0, target_zone="front centre", aim=(0.5, 0.2),
        reachable=True, zone_shares={}, off_plan=0,
    )
    base.update(changes)
    return Decision(**base)


def scene(states=("hot",), unseated=()) -> SeatingScene:
    seats = {(0, index): SeatReading(state, None) for index, state in enumerate(states)}
    return SeatingScene(LAYOUT, seats, tuple(unseated))


def test_nobody_in_view():
    assert status_text(decision(closed=True), scene(()), people=0) == "no one in view"


def test_everyone_outside_the_plan_points_at_calibration():
    assert status_text(decision(closed=True), scene((), unseated=(0, 1)), people=2) == "everyone is outside the plan, check calibration"


def test_the_vent_names_the_zone_it_serves():
    assert status_text(decision(), scene(), people=1) == "aiming at front centre"


@pytest.mark.parametrize("phase,words", [("turning", "turning to"), ("opening", "opening towards"), ("aiming", "aiming at")])
def test_the_status_follows_the_phase(phase, words):
    assert status_text(decision(phase=phase), scene(), people=1) == f"{words} front centre"


def test_out_of_reach_is_said_plainly():
    assert status_text(decision(reachable=False), scene(), people=1) == "aiming at front centre, out of reach"


def test_a_shut_vent_never_blames_clothing():
    text = status_text(decision(closed=True, target_zone=None), scene(("cold", "cold")), people=2)
    assert text == "vent shut, nobody seated yet"


def test_people_outside_the_plan_are_mentioned_alongside():
    text = status_text(decision(), scene(("hot",), unseated=(1,)), people=2)
    assert text == "aiming at front centre, 1 outside the plan"


def test_the_legend_names_the_clothing_colours_without_weights():
    assert LEGEND == [("hot", "shirt"), ("unsure", "unsure"), ("cold", "jacket")]


@pytest.fixture(scope="module")
def fonts():
    QApplication.instance() or QApplication([])
    return load_fonts()


def test_errors_replace_the_plan_and_retry_brings_it_back(fonts):
    panel = PlanPanel(fonts)
    retries = []
    panel.retry_requested.connect(lambda: retries.append(1))
    assert not panel.showing_error()
    panel.show_error("ValueError: nope")
    assert panel.showing_error() and "ValueError: nope" in panel.error_text() and "hide this panel" in panel.error_text()
    panel.retry_requested.emit()
    panel.clear_error()
    assert not panel.showing_error() and retries == [1]


def test_the_panel_paints_with_and_without_data(fonts):
    panel = PlanPanel(fonts)
    panel.resize(640, 520)
    assert not panel.grab().isNull()
    panel.set_state(LAYOUT, scene(("hot", "cold")), decision(), make_zones(LAYOUT), VentSettings(), ViewSettings())
    panel.set_status("aiming at front centre")
    assert not panel.grab().isNull()
    panel.show_error("boom")
    assert not panel.grab().isNull()
