import cv2
import numpy as np

from jacket.debug_view import CropSaver
from jacket.types import CropResult


class FakeClock:
    now = 0.0

    def __call__(self) -> float:
        return self.now


def usable() -> CropResult:
    return CropResult(np.full((60, 80, 3), 120, dtype=np.uint8), "")


def rejected() -> CropResult:
    return CropResult(None, "touches left edge")


def test_saves_only_usable_crops_as_readable_images(tmp_path):
    saver = CropSaver(tmp_path / "raw", 2.0, FakeClock())
    paths = saver.maybe_save([usable(), rejected(), usable()])
    assert len(paths) == 2
    assert cv2.imread(str(paths[0])).shape == (60, 80, 3)


def test_second_save_waits_for_interval(tmp_path):
    clock = FakeClock()
    saver = CropSaver(tmp_path, 2.0, clock)
    assert len(saver.maybe_save([usable()])) == 1
    clock.now = 1.0
    assert saver.maybe_save([usable()]) == []
    clock.now = 2.5
    assert len(saver.maybe_save([usable()])) == 1


def test_nothing_is_created_when_nothing_saved(tmp_path):
    CropSaver(tmp_path / "raw", 2.0, FakeClock()).maybe_save([rejected()])
    assert not (tmp_path / "raw").exists()
