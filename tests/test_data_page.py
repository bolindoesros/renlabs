import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtCore import QMimeData, QUrl
from PySide6.QtWidgets import QApplication

from jacket import config
from jacket.types import CropResult
from ren import dataset
from ren.pages.data_page import DataPage, image_paths
from ren.theme import load_fonts


@pytest.fixture(scope="module")
def fonts():
    QApplication.instance() or QApplication([])
    return load_fonts()


@pytest.fixture(autouse=True)
def data_dirs(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RAW_CROPS_DIR", tmp_path / "raw")
    monkeypatch.setattr(config, "EVAL_DIR", tmp_path / "eval")
    monkeypatch.setattr(config, "DISCARDED_CROPS_DIR", tmp_path / "discarded")


def collect(count: int) -> list:
    return dataset.save_snapshot([CropResult(np.full((40, 30, 3), 90, dtype=np.uint8), "")] * count)


def test_collected_crops_show_up_to_sort(fonts):
    collect(3)
    page = DataPage(fonts)
    assert page.grid.count() == 3


def test_dropping_on_a_bin_files_the_crops_and_updates_counts(fonts):
    paths = collect(2)
    page = DataPage(fonts)
    page.bins["warm"].dropped.emit(paths)
    assert page.grid.count() == 0 and len(dataset.images_in(config.EVAL_DIR / "warm")) == 2


def test_dragging_a_crop_carries_its_file(fonts):
    (path,) = collect(1)
    page = DataPage(fonts)
    mime = page.grid.mimeData([page.grid.item(0)])
    assert image_paths(mime) == [path]


def test_only_image_files_count_as_a_drop():
    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile("/tmp/a.jpg"), QUrl.fromLocalFile("/tmp/b.txt"), QUrl("https://x.com/c.png")])
    assert [p.name for p in image_paths(mime)] == ["a.jpg"]


def test_a_dropped_or_chosen_photo_is_announced(fonts, tmp_path):
    photo = tmp_path / "group.jpg"
    page = DataPage(fonts, choose_photo=lambda parent: str(photo))
    chosen = []
    page.photo_chosen.connect(chosen.append)
    page.photo_target.dropped.emit([photo])
    page._browse()
    assert chosen == [photo, photo]
