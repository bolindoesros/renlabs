import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QFocusEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from ren.theme import load_fonts
from ren.widgets import TextButton, SwitchRow, NavChoice


@pytest.fixture(scope="module")
def fonts():
    QApplication.instance() or QApplication([])
    return load_fonts()


def test_toggle_flips_on_click(fonts):
    toggle = SwitchRow("labels", fonts)
    toggle.resize(200, 40)
    QTest.mouseClick(toggle, Qt.MouseButton.LeftButton)
    assert toggle.isChecked()
    QTest.mouseClick(toggle, Qt.MouseButton.LeftButton)
    assert not toggle.isChecked()


def test_disabled_toggle_ignores_clicks(fonts):
    toggle = SwitchRow("labels", fonts)
    toggle.setEnabled(False)
    QTest.mouseClick(toggle, Qt.MouseButton.LeftButton)
    assert not toggle.isChecked()


def test_toggle_renders_without_error_in_every_state(fonts):
    toggle = SwitchRow("shirt / hoodie detection", fonts)
    toggle.resize(300, 40)
    for checked in (False, True):
        for enabled in (True, False):
            toggle.setChecked(checked)
            toggle.setEnabled(enabled)
            assert not toggle.grab().isNull()


def test_nav_choice_select_emits_and_moves_the_active_link(fonts):
    choice = NavChoice("model", ["clip", "fashion"], "fashion", fonts)
    chosen = []
    choice.chosen.connect(chosen.append)
    assert choice.current() == "fashion"
    choice.select("clip")
    assert chosen == ["clip"] and choice.current() == "clip"


def test_only_tab_focus_counts_as_keyboard_focus(fonts):
    toggle = SwitchRow("labels", fonts)
    toggle.focusInEvent(QFocusEvent(QEvent.Type.FocusIn, Qt.FocusReason.ActiveWindowFocusReason))
    assert not toggle._keyboard_focus  # initial focus must not underline anything
    toggle.focusInEvent(QFocusEvent(QEvent.Type.FocusIn, Qt.FocusReason.TabFocusReason))
    assert toggle._keyboard_focus
    toggle.focusOutEvent(QFocusEvent(QEvent.Type.FocusOut, Qt.FocusReason.TabFocusReason))
    assert not toggle._keyboard_focus




def test_bracket_buttons_click_and_render(fonts):
    clicked = []
    button = TextButton("retry", fonts)
    button.clicked.connect(lambda: clicked.append(1))
    QTest.mouseClick(button, Qt.MouseButton.LeftButton)
    assert clicked == [1] and not button.grab().isNull()
    button.setEnabled(False)
    QTest.mouseClick(button, Qt.MouseButton.LeftButton)
    assert clicked == [1]



def test_icon_buttons_click_switch_kind_and_reject_unknown_icons(fonts):
    from ren.widgets import IconButton
    button = IconButton("expand", "expand", fonts)
    clicked = []
    button.clicked.connect(lambda: clicked.append(1))
    QTest.mouseClick(button, Qt.MouseButton.LeftButton)
    assert clicked == [1] and button.toolTip() == "expand"
    pictures = {}
    for kind in ("expand", "restore", "hide"):
        button.set_kind(kind, kind)
        pictures[kind] = button.grab().toImage()
    assert pictures["expand"] != pictures["restore"] != pictures["hide"] != pictures["expand"]
    with pytest.raises(ValueError):
        button.set_kind("banana", "?")
