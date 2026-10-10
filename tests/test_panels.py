import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QLabel

from ren.panels import PanelArea
from ren.theme import load_fonts


@pytest.fixture(scope="module")
def fonts():
    QApplication.instance() or QApplication([])
    return load_fonts()


@pytest.fixture
def area(fonts):
    widget = PanelArea(fonts)
    widget.add_panel("camera", "camera", QLabel("camera content"))
    widget.add_panel("plan", "seating plan", QLabel("plan content"))
    widget.resize(900, 500)
    widget.show()
    QApplication.processEvents()
    yield widget
    widget.hide()


def test_both_panels_start_shown_side_by_side(area):
    assert area.is_shown("camera") and area.is_shown("plan") and area.expanded() is None
    camera, plan = area._panels["camera"].geometry(), area._panels["plan"].geometry()
    assert plan.left() > camera.right() and camera.width() > 100 and plan.width() > 100


def test_hiding_a_panel_gives_its_room_to_the_other(area):
    before = area._panels["camera"].width()
    area.set_visible("plan", False)
    QApplication.processEvents()
    assert not area.is_shown("plan") and area.is_shown("camera") and not area.is_visible("plan")
    assert area._panels["camera"].width() > before


def test_showing_it_again_brings_it_back(area):
    area.set_visible("plan", False)
    area.set_visible("plan", True)
    assert area.is_shown("plan")


def test_the_hide_button_hides_that_panel(area):
    QTest.mouseClick(area._panels["camera"]._hide, Qt.MouseButton.LeftButton)
    assert not area.is_visible("camera") and area.is_visible("plan")


def test_expanding_covers_the_other_panel_without_hiding_it_for_good(area):
    area.toggle_expanded("camera")
    assert area.expanded() == "camera" and area.is_shown("camera")
    assert not area.is_shown("plan") and area.is_visible("plan")  # covered, not hidden by the user
    assert area._panels["camera"].is_expanded()


def test_the_expand_button_toggles_and_restore_returns(area):
    button = area._panels["plan"]._expand
    QTest.mouseClick(button, Qt.MouseButton.LeftButton)
    assert area.expanded() == "plan" and not area.is_shown("camera")
    QTest.mouseClick(button, Qt.MouseButton.LeftButton)
    assert area.expanded() is None and area.is_shown("camera")
    assert not area._panels["plan"].is_expanded()


def test_restore_reports_whether_it_did_anything(area):
    assert area.restore() is False
    area.toggle_expanded("plan")
    assert area.restore() is True and area.expanded() is None


def test_double_clicking_a_header_expands_the_panel(area):
    header = area._panels["camera"]._header
    QTest.mouseDClick(header, Qt.MouseButton.LeftButton)
    assert area.expanded() == "camera"


def test_hiding_the_expanded_panel_leaves_expanded_mode(area):
    area.toggle_expanded("camera")
    area.set_visible("camera", False)
    assert area.expanded() is None and area.is_shown("plan")


def test_expanding_a_hidden_panel_brings_it_back(area):
    area.set_visible("plan", False)
    area.toggle_expanded("plan")
    assert area.is_visible("plan") and area.expanded() == "plan"


def test_hiding_both_shows_a_note_instead_of_an_empty_hole(area):
    area.set_visible("camera", False)
    area.set_visible("plan", False)
    QApplication.processEvents()
    assert area._empty.isVisible() and not area._splitter.isVisible()
    area.set_visible("camera", True)
    assert not area._empty.isVisible() and area._splitter.isVisible()


def test_every_change_is_announced(area):
    changes = []
    area.changed.connect(lambda: changes.append(1))
    area.set_visible("plan", False)
    area.toggle_expanded("camera")
    area.restore()
    assert len(changes) == 3


def test_the_divider_can_be_dragged_to_resize(area):
    before = area._panels["camera"].width()
    area._splitter.setSizes([before + 200, area._panels["plan"].width() - 200])
    QApplication.processEvents()
    assert area._panels["camera"].width() > before


def test_panels_paint(area):
    assert not area.grab().isNull()
