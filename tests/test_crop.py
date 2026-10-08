import numpy as np
import pytest

from jacket import config
from jacket.crop import crop_torso
from jacket.types import Box

FRAME_H, FRAME_W = 480, 640


@pytest.fixture
def frame() -> np.ndarray:
    return np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)


def person(x1: int, y1: int, x2: int, y2: int, confidence: float = 0.9) -> Box:
    return Box(x1, y1, x2, y2, confidence, "person", torso_top_y=y1 + (y2 - y1) // 4)


def head(x1: int, y1: int, x2: int, y2: int, confidence: float = 0.9) -> Box:
    return Box(x1, y1, x2, y2, confidence, "head")


def assert_valid_crop(result) -> None:
    assert result.crop is not None, result.reason
    height, width, channels = result.crop.shape
    assert channels == 3 and result.crop.dtype == np.uint8
    assert min(height, width) >= config.MIN_CROP_SIDE_PX


def test_person_crop_is_sized_by_box_width(frame):
    result = crop_torso(frame, person(200, 100, 400, 400))
    assert_valid_crop(result)
    assert result.crop.shape[:2] == (120, 200)  # 0.6 of 200 px width


def test_crop_is_a_copy(frame):
    result = crop_torso(frame, person(200, 100, 400, 400))
    result.crop[:] = 255
    assert frame.max() == 0


def test_low_confidence_is_unknown(frame):
    assert "low confidence" in crop_torso(frame, person(200, 100, 400, 400, 0.2)).reason


@pytest.mark.parametrize(
    "box,side",
    [
        (person(0, 100, 200, 400), "left"),
        (person(440, 100, 640, 400), "right"),
        (person(200, 0, 400, 300), "top"),
    ],
)
def test_person_touching_edge_is_unknown(frame, box, side):
    assert side in crop_torso(frame, box).reason


def test_person_touching_bottom_is_allowed(frame):
    assert_valid_crop(crop_torso(frame, person(200, 100, 400, FRAME_H)))


def test_tiny_box_is_unknown(frame):
    assert "too small" in crop_torso(frame, person(300, 200, 330, 240)).reason


def test_empty_and_inverted_boxes_are_unknown(frame):
    assert crop_torso(frame, person(300, 200, 300, 300)).crop is None
    assert crop_torso(frame, person(400, 300, 300, 200)).crop is None


def test_box_fully_outside_frame_is_unknown(frame):
    assert crop_torso(frame, person(900, 900, 1100, 1300)).crop is None


def test_head_crop_sits_below_chin(frame):
    result = crop_torso(frame, head(280, 100, 360, 180))
    assert_valid_crop(result)
    assert result.crop.shape[:2] == (160, 160)  # 2 head sizes each way


def test_head_near_bottom_is_unknown(frame):
    result = crop_torso(frame, head(280, 400, 360, 470))
    assert result.crop is None and "too small" in result.reason


def test_head_crop_clipped_at_side_edge(frame):
    assert_valid_crop(crop_torso(frame, head(0, 100, 100, 200)))


def test_blank_one_pixel_frame_never_crashes():
    tiny_frame = np.zeros((1, 1, 3), dtype=np.uint8)
    assert crop_torso(tiny_frame, person(0, 0, 1, 1)).crop is None


def test_random_boxes_never_return_invalid_array(frame):
    rng = np.random.default_rng(0)
    for _ in range(2000):
        x1, x2 = sorted(rng.integers(-200, 900, size=2).tolist())
        y1, y2 = sorted(rng.integers(-200, 700, size=2).tolist())
        kind_box = person if rng.random() < 0.5 else head
        result = crop_torso(frame, kind_box(x1, y1, x2, y2))
        if result.crop is None:
            assert result.reason
        else:
            assert_valid_crop(result)


def test_tall_seated_box_skips_the_head(frame):
    result = crop_torso(frame, person(200, 50, 400, 470))  # box reaches frame bottom
    assert_valid_crop(result)
    assert result.crop.shape[:2] == (120, 200)  # height set by width, not box height


def test_torso_top_sets_crop_top(frame):
    box = Box(200, 50, 400, 470, 0.9, "person", torso_top_y=150)
    result = crop_torso(frame, box)
    assert_valid_crop(result)
    assert result.crop.shape[:2] == (120, 200)  # starts at row 150, 0.6 of width tall


def test_person_without_torso_anchor_is_unknown(frame):
    box = Box(200, 100, 400, 400, 0.9, "person")
    assert "no shoulders or face" in crop_torso(frame, box).reason
