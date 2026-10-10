import numpy as np
import pytest

from jacket.detectors import detect, fuse, head_fit, merge, parts_of
from jacket.types import Box

FRAME = np.zeros((480, 640, 3), dtype=np.uint8)
BODY = Box(200, 100, 300, 400, 0.6, "person", torso_top_y=160)
HEAD = Box(230, 95, 270, 145, 0.7, "head", torso_top_y=145)


class Fixed:
    def __init__(self, boxes) -> None:
        self.boxes = boxes

    def detect(self, frame):
        return self.boxes


def test_modes_name_their_models():
    assert parts_of("body") == ("body",) and parts_of("head") == ("head",)
    assert parts_of("both") == ("body", "head")
    with pytest.raises(KeyError):
        parts_of("eyes")


def test_a_head_at_the_top_of_a_body_fits_it():
    assert head_fit(BODY, HEAD) is not None


def test_a_head_at_the_feet_or_beside_the_body_does_not_fit():
    assert head_fit(BODY, Box(230, 330, 270, 380, 0.9, "head")) is None
    assert head_fit(BODY, Box(400, 95, 440, 145, 0.9, "head")) is None


def test_agreement_raises_confidence_and_keeps_the_body_box():
    fused = fuse([BODY], [HEAD])
    assert len(fused) == 1
    assert (fused[0].kind, fused[0].x1, fused[0].y2) == ("person", 200, 400)
    assert fused[0].confidence == pytest.approx(1 - 0.4 * 0.3)
    assert fused[0].torso_top_y == 160  # keypoints win when present


def test_the_chin_fills_in_missing_shoulders():
    shoulderless = Box(200, 100, 300, 400, 0.6, "person", torso_top_y=None)
    assert merge(shoulderless, HEAD).torso_top_y == 145


def test_what_only_one_model_saw_is_kept():
    hidden = Box(500, 300, 540, 350, 0.8, "head", torso_top_y=350)  # body behind a seat
    alone = Box(10, 10, 100, 300, 0.5, "person", torso_top_y=60)  # head turned away
    fused = fuse([BODY, alone], [HEAD, hidden])
    assert [box.kind for box in fused] == ["person", "person", "head"]
    assert fused[1] == alone and fused[2] == hidden


def test_each_head_joins_at_most_one_body():
    twin = Box(205, 100, 305, 400, 0.6, "person", torso_top_y=160)  # overlapping body, same head
    fused = fuse([BODY, twin], [HEAD])
    assert sum(box.confidence > 0.6 for box in fused) == 1 and len(fused) == 2


def test_two_heads_pick_their_nearest_bodies():
    left, right = Box(0, 0, 100, 300, 0.5, "person"), Box(90, 0, 190, 300, 0.5, "person")
    right_head, left_head = Box(120, 5, 160, 50, 0.5, "head"), Box(30, 5, 70, 50, 0.5, "head")
    fused = fuse([left, right], [right_head, left_head])
    assert [box.torso_top_y for box in fused] == [50, 50]
    assert len(fused) == 2


def test_detect_runs_only_what_the_mode_needs():
    detectors = {"body": Fixed([BODY]), "head": Fixed([HEAD])}
    assert detect(FRAME, detectors, "body") == [BODY]
    assert detect(FRAME, detectors, "head") == [HEAD]
    assert len(detect(FRAME, detectors, "both")) == 1
    assert detect(FRAME, {"head": Fixed([HEAD])}, "head") == [HEAD]  # body never touched
