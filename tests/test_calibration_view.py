import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from ren.calibration_view import CalibrationView
from ren.plan import Calibration, PlanLayout
from ren.theme import load_fonts


@pytest.fixture(scope="module")
def fonts():
    QApplication.instance() or QApplication([])
    return load_fonts()


@pytest.fixture
def view(fonts):
    widget = CalibrationView(fonts)
    widget.resize(800, 450)
    widget.set_layout(PlanLayout(3, 6))
    widget.show_frame(np.full((450, 800, 3), 90, dtype=np.uint8), [(400.0, 250.0)])
    return widget


def handle_pixel(view: CalibrationView, index: int) -> QPoint:
    fx, fy = view.calibration().corners()[index]
    target = view._target()
    return QPoint(int(target.left() + fx * target.width()), int(target.top() + fy * target.height()))


def drag(view: CalibrationView, index: int, to: QPoint) -> None:
    start = handle_pixel(view, index)
    QTest.mousePress(view, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, start)
    view.mouseMoveEvent(_move_event(view, to))
    QTest.mouseRelease(view, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, to)


def _move_event(view, point: QPoint):
    from PySide6.QtCore import QEvent, QPointF
    from PySide6.QtGui import QMouseEvent
    return QMouseEvent(
        QEvent.Type.MouseMove, QPointF(point), QPointF(view.mapToGlobal(point)),
        Qt.MouseButton.NoButton, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
    )


def test_dragging_a_corner_moves_it_and_announces_the_result_once(view):
    announced = []
    view.changed.connect(announced.append)
    drag(view, 0, QPoint(120, 80))
    assert len(announced) == 1
    new_x, new_y = announced[0].back_left
    assert new_x == pytest.approx(120 / 800, abs=0.02) and new_y == pytest.approx(80 / 450, abs=0.02)
    assert announced[0].back_right == Calibration().back_right  # the others stay put


def test_dragging_cannot_flip_the_quad(view):
    before = view.calibration()
    drag(view, 0, QPoint(780, 440))  # back-left dragged past the opposite corner
    assert view.calibration().is_valid()
    assert view.calibration() != Calibration((1.0, 1.0), before.back_right, before.front_right, before.front_left)


def test_corners_stay_inside_the_picture(view):
    drag(view, 1, QPoint(5000, -5000))
    x, y = view.calibration().back_right
    assert 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0


def test_clicking_empty_space_changes_nothing(view):
    announced = []
    view.changed.connect(announced.append)
    QTest.mouseClick(view, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPoint(400, 225))
    assert announced == [] and view.calibration() == Calibration()


def test_external_changes_are_ignored_while_dragging(view):
    QTest.mousePress(view, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, handle_pixel(view, 2))
    view.set_calibration(Calibration((0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8)))
    assert view.calibration() == Calibration()
    QTest.mouseRelease(view, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, handle_pixel(view, 2))


def test_it_paints_with_and_without_a_picture(fonts):
    empty = CalibrationView(fonts)
    empty.resize(640, 360)
    assert not empty.grab().isNull()
    filled = CalibrationView(fonts)
    filled.resize(640, 360)
    filled.show_frame(np.zeros((360, 640, 3), dtype=np.uint8), [])
    assert not filled.grab().isNull()
