import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from jacket.types import Box, ClothingResult, CropResult
from ren.crops_view import EMPTY_TEXT, CropsView, tile_side
from ren.engine import FrameResult
from ren.theme import load_fonts

FRAME = np.full((480, 640, 3), 90, dtype=np.uint8)
CROP = np.full((60, 40, 3), (0, 0, 255), dtype=np.uint8)  # BGR red
BOX = Box(100, 100, 200, 300, 0.9, "person", torso_top_y=150)


@pytest.fixture(scope="module")
def fonts():
    QApplication.instance() or QApplication([])
    return load_fonts()


def test_one_tile_per_person_with_its_tag(fonts):
    view = CropsView(fonts)
    view.show_result(FrameResult(FRAME, [BOX, BOX], [ClothingResult("light", 0.1, ""), ClothingResult("warm", 0.9, "")],
                                 (CropResult(CROP, ""), CropResult(CROP, "")), "", ""))
    assert [tile.text for tile in view.tiles()] == ["shirt 90%", "jacket 90%"]


def test_an_uncut_torso_shows_the_reason(fonts):
    view = CropsView(fonts)
    view.show_result(FrameResult(FRAME, [BOX], [], (CropResult(None, "crop too small 20x20"),), "", ""))
    (tile,) = view.tiles()
    assert tile.image is None and tile.text == "crop too small 20x20"


def test_no_people_says_so_and_paints(fonts):
    view = CropsView(fonts)
    view.show_result(FrameResult(FRAME, [], [], (), "", ""))
    view.resize(300, 200)
    assert view.tiles() == [] and view._message == EMPTY_TEXT and not view.grab().isNull()


def test_tiles_spread_to_fill_the_panel():
    assert tile_side(1, 300, 300)[0] == 1
    assert tile_side(4, 600, 200)[0] == 4  # wide panel: one row
    assert tile_side(4, 200, 600)[0] == 1  # tall panel: one column


def test_save_crops_writes_what_is_shown(fonts, tmp_path, monkeypatch):
    from jacket import config
    from ren.crops_view import CropsPanel
    monkeypatch.setattr(config, "RAW_CROPS_DIR", tmp_path / "raw")
    panel = CropsPanel(fonts)
    assert panel.save() == [] and panel.status.text() == "nothing to save"
    panel.show_result(FrameResult(FRAME, [BOX, BOX], [], (CropResult(CROP, ""), CropResult(None, "too small")), "", ""))
    assert len(panel.save()) == 1 and "saved 1" in panel.status.text()
