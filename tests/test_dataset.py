import numpy as np
import pytest

from jacket import config
from jacket.types import CropResult
from ren import dataset


@pytest.fixture(autouse=True)
def data_dirs(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RAW_CROPS_DIR", tmp_path / "raw")
    monkeypatch.setattr(config, "EVAL_DIR", tmp_path / "eval")
    monkeypatch.setattr(config, "DISCARDED_CROPS_DIR", tmp_path / "discarded")
    return tmp_path


CROP = np.full((40, 30, 3), 128, dtype=np.uint8)


def test_a_snapshot_saves_only_usable_crops():
    paths = dataset.save_snapshot([CropResult(CROP, ""), CropResult(None, "too small"), CropResult(CROP, "")])
    assert len(paths) == 2 and dataset.images_in(config.RAW_CROPS_DIR) == sorted(paths)


def test_nothing_usable_saves_nothing():
    assert dataset.save_snapshot([CropResult(None, "too small")]) == []
    assert not config.RAW_CROPS_DIR.exists()


def test_collected_crops_move_into_their_bin():
    (path,) = dataset.save_snapshot([CropResult(CROP, "")])
    (filed,) = dataset.file_into([path], "warm")
    assert filed.parent == config.EVAL_DIR / "warm" and not path.exists()


def test_files_from_elsewhere_are_copied_not_moved(tmp_path):
    outside = tmp_path / "mine.png"
    outside.write_bytes(b"not really a png")
    dataset.file_into([outside], "light")
    assert outside.exists() and (config.EVAL_DIR / "light" / "mine.png").exists()


def test_name_clashes_get_a_suffix(tmp_path):
    outside = tmp_path / "a.jpg"
    outside.write_bytes(b"x")
    first, second = dataset.file_into([outside], "discard") + dataset.file_into([outside], "discard")
    assert first.name == "a.jpg" and second.name == "a_1.jpg" and first.parent == config.DISCARDED_CROPS_DIR


def test_non_images_are_skipped(tmp_path):
    text = tmp_path / "notes.txt"
    text.write_text("hi")
    assert dataset.file_into([text], "warm") == []
