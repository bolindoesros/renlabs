import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from dataclasses import fields

import pytest

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication

from jacket import config
from ren.engine import ViewSettings
from ren.theme import load_fonts
from ren.toolbar import VIEW_SECTIONS, Toolbar


@pytest.fixture(scope="module")
def fonts():
    QApplication.instance() or QApplication([])
    return load_fonts()


@pytest.fixture
def toolbar(fonts):
    return Toolbar(fonts, config.DEFAULT_CLIP_MODEL, ViewSettings(), ["lt1", "lt2"], "lt1")


def test_every_view_item_is_a_real_setting_or_panel():
    names = {field.name for field in fields(ViewSettings)}
    for items in VIEW_SECTIONS.values():
        for item in items:
            assert item.is_panel or item.key in names, item.key


def test_panel_items_are_exactly_the_two_panels():
    assert [item.key for item in VIEW_SECTIONS["panels"]] == ["camera", "plan"]


def test_ticks_start_from_the_settings(fonts):
    toolbar = Toolbar(fonts, "clip", ViewSettings(show_labels=False, show_crops=True), ["lt1"], "lt1")
    assert not toolbar.action("show_labels").isChecked() and toolbar.action("show_crops").isChecked()
    assert toolbar.action("camera").isChecked() and toolbar.model() == "clip"


def test_ticking_a_layer_announces_it(toolbar):
    seen = []
    toolbar.layer_toggled.connect(lambda key, shown: seen.append((key, shown)))
    toolbar.action("show_zones").trigger()
    toolbar.action("show_zones").trigger()
    assert seen == [("show_zones", False), ("show_zones", True)]


def test_ticking_a_panel_announces_the_panel_not_a_layer(toolbar):
    layers, panels = [], []
    toolbar.layer_toggled.connect(lambda *a: layers.append(a))
    toolbar.panel_toggled.connect(lambda *a: panels.append(a))
    toolbar.action("plan").trigger()
    assert panels == [("plan", False)] and layers == []


def test_quiet_updates_do_not_announce(toolbar):
    seen = []
    toolbar.layer_toggled.connect(lambda *a: seen.append(a))
    toolbar.panel_toggled.connect(lambda *a: seen.append(a))
    toolbar.model_chosen.connect(lambda *a: seen.append(a))
    toolbar.set_view(ViewSettings(show_vent=False))
    toolbar.set_panel_visible("camera", False)
    toolbar.set_model("clip")
    assert seen == []
    assert not toolbar.action("show_vent").isChecked() and not toolbar.action("camera").isChecked()
    assert toolbar.model() == "clip"


def test_choosing_a_model_announces_it(toolbar):
    chosen = []
    toolbar.model_chosen.connect(chosen.append)
    toolbar._model.textActivated.emit("clip")
    assert chosen == ["clip"]


def test_section_headings_are_not_clickable(toolbar):
    headings = [a for a in toolbar._menu.actions() if not a.isSeparator() and not a.isCheckable()]
    assert [a.text() for a in headings] == list(VIEW_SECTIONS)
    assert not any(a.isEnabled() for a in headings)


def test_the_menu_and_toolbar_paint(toolbar):
    toolbar.resize(800, 50)
    assert not toolbar.grab().isNull()
    toolbar._menu.adjustSize()
    assert not toolbar._menu.grab().isNull()



def test_choosing_a_venue_announces_it(toolbar):
    chosen = []
    toolbar.venue_chosen.connect(chosen.append)
    toolbar._venue.textActivated.emit("lt2")
    assert chosen == ["lt2"]


def test_the_plus_button_asks_for_a_new_venue(toolbar):
    asked = []
    toolbar.venue_add_requested.connect(lambda: asked.append(True))
    toolbar._add_venue.click()
    assert asked == [True]


def test_venue_list_updates_quietly(toolbar):
    chosen = []
    toolbar.venue_chosen.connect(chosen.append)
    toolbar.set_venues(["lt1", "lt2", "lt3"], "lt3")
    assert toolbar.venue() == "lt3" and chosen == []
    assert [toolbar._venue.itemText(i) for i in range(toolbar._venue.count())] == ["lt1", "lt2", "lt3"]
