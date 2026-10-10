import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from ren.plan import PlanLayout, ZoneRect
from ren.theme import load_fonts
from ren.zone_editor import ZoneEditorView

ZONE = ZoneRect("table 1", 0.2, 0.2, 0.5, 0.6)


@pytest.fixture(scope="module")
def fonts():
    QApplication.instance() or QApplication([])
    return load_fonts()


def editor(fonts, zones=(ZONE,)):
    view = ZoneEditorView(fonts)
    view.resize(800, 500)
    view.set_zones(PlanLayout(3, 6), zones)
    changes = []
    view.zones_changed.connect(changes.append)
    return view, changes


def drag(view, start, end):
    QTest.mousePress(view, Qt.MouseButton.LeftButton, pos=start)
    QTest.mouseMove(view, end)
    QTest.mouseRelease(view, Qt.MouseButton.LeftButton, pos=end)


def test_add_zone_puts_a_new_named_zone_in_the_middle(fonts):
    view, changes = editor(fonts)
    view.add_zone()
    assert [z.name for z in changes[-1]] == ["table 1", "zone 1"]
    assert changes[-1][1].contains(0.5, 0.5)


def test_dragging_a_zone_moves_it(fonts):
    view, changes = editor(fonts)
    start = view.zone_rect(ZONE).center().toPoint()
    drag(view, start, start + QPoint(80, 0))
    moved = changes[-1][0]
    assert moved.bounds[0] > ZONE.bounds[0] and moved.bounds[1] == pytest.approx(ZONE.bounds[1])


def test_dragging_a_selected_corner_resizes(fonts):
    view, changes = editor(fonts)
    QTest.mouseClick(view, Qt.MouseButton.LeftButton, pos=view.zone_rect(ZONE).center().toPoint())  # select
    corner = view.corner_point(ZONE, (1, 1)).toPoint()  # back-right
    drag(view, corner, corner + QPoint(60, -40))
    u0, v0, u1, v1 = changes[-1][0].bounds
    assert (u0, v0) == pytest.approx((0.2, 0.2)) and u1 > 0.5 and v1 > 0.6


def test_right_click_deletes_a_zone(fonts):
    view, changes = editor(fonts)
    QTest.mouseClick(view, Qt.MouseButton.RightButton, pos=view.zone_rect(ZONE).center().toPoint())
    assert changes == [()]


def test_a_plain_click_changes_nothing(fonts):
    view, changes = editor(fonts)
    QTest.mouseClick(view, Qt.MouseButton.LeftButton, pos=view.zone_rect(ZONE).center().toPoint())
    assert changes == []
