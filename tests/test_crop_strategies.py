import numpy as np
import pytest

from jacket import config
from jacket.crop import crop_torso
from jacket.crop_strategies import CropRejected, head_region, person_region
from jacket.types import Box, Region

FRAME = np.zeros((480, 640, 3), dtype=np.uint8)


def test_person_region_starts_at_torso_top_and_is_sized_by_width():
    box = Box(200, 100, 400, 450, 0.9, "person", torso_top_y=180)
    region = person_region(box)
    assert (region.y1, region.height) == (180, int(200 * config.PERSON_CROP_HEIGHT_WIDTHS))
    assert (region.x1, region.x2) == (200, 400)


def test_person_region_never_passes_box_bottom():
    box = Box(200, 100, 400, 250, 0.9, "person", torso_top_y=200)
    assert person_region(box).y2 == 250


def test_person_region_rejects_without_anchor():
    with pytest.raises(CropRejected, match="no shoulders"):
        person_region(Box(200, 100, 400, 400, 0.9, "person"))


def test_head_region_is_below_the_chin_and_centered():
    region = head_region(Box(280, 100, 360, 180, 0.9, "head"))
    assert region.y1 > 180
    assert (region.x1 + region.x2) / 2 == 320


def test_custom_strategy_can_be_injected_without_editing_crop():
    def whole_box(box: Box) -> Region:
        return Region(box.x1, box.y1, box.x2, box.y2)

    box = Box(100, 100, 300, 300, 0.9, "person")  # no torso_top_y needed here
    result = crop_torso(FRAME, box, strategies={"person": whole_box})
    assert result.crop.shape[:2] == (200, 200)


def test_missing_strategy_fails_loudly():
    with pytest.raises(KeyError, match="head"):
        crop_torso(FRAME, Box(100, 100, 200, 200, 0.9, "head"), strategies={})
