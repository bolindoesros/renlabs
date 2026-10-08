import numpy as np
import pytest

from jacket import config
from jacket.pipeline import ClothingPipeline, count_warm, iou, smooth
from jacket.types import Box, ClothingResult

FRAME = np.zeros((480, 640, 3), dtype=np.uint8)


def person(x1: int = 200, y1: int = 100, x2: int = 400, y2: int = 400, confidence: float = 0.9) -> Box:
    return Box(x1, y1, x2, y2, confidence, "person", torso_top_y=y1 + 75)


class FakeClassifier:
    def __init__(self, warm_prob: float = 0.9) -> None:
        self.warm_prob = warm_prob
        self.calls = 0

    def __call__(self, crop: np.ndarray) -> ClothingResult:
        self.calls += 1
        return ClothingResult("warm", self.warm_prob, "")


class FakeClock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def classifier() -> FakeClassifier:
    return FakeClassifier()


@pytest.fixture
def pipeline(classifier, clock) -> ClothingPipeline:
    return ClothingPipeline(classifier, clock)


def test_iou_identical_disjoint_and_half():
    assert iou(person(), person()) == 1.0
    assert iou(person(0, 0, 10, 10), person(50, 50, 60, 60)) == 0.0
    assert iou(person(0, 0, 10, 10), person(5, 0, 15, 10)) == pytest.approx(1 / 3)


def test_smooth_first_score_is_taken_as_is():
    assert smooth(None, 0.8, 0.4) == 0.8


def test_smooth_blends_old_and_new():
    assert smooth(1.0, 0.0, 0.4) == pytest.approx(0.6)


def test_classifies_once_per_interval_not_per_frame(pipeline, classifier, clock):
    for _ in range(10):
        pipeline.process(FRAME, [person()])
        clock.now += config.CLASSIFY_INTERVAL_SECONDS / 20
    assert classifier.calls == 1
    clock.now += config.CLASSIFY_INTERVAL_SECONDS
    pipeline.process(FRAME, [person()])
    assert classifier.calls == 2


def test_label_is_smoothed_across_scores(pipeline, classifier, clock):
    assert pipeline.process(FRAME, [person()])[0].label == "warm"  # 0.9
    classifier.warm_prob = 0.0
    clock.now += config.CLASSIFY_INTERVAL_SECONDS
    result = pipeline.process(FRAME, [person()])[0]
    expected = config.SMOOTHING_ALPHA * 0.0 + (1 - config.SMOOTHING_ALPHA) * 0.9
    assert result.warm_prob == pytest.approx(expected)  # one bad score does not flip it


def test_moving_box_keeps_its_track(pipeline, classifier, clock):
    pipeline.process(FRAME, [person(200, 100, 400, 400)])
    clock.now += 0.1
    pipeline.process(FRAME, [person(210, 105, 410, 405)])
    assert classifier.calls == 1


def test_new_person_is_scored_immediately(pipeline, classifier, clock):
    pipeline.process(FRAME, [person(100, 100, 250, 400)])
    clock.now += 0.1
    pipeline.process(FRAME, [person(100, 100, 250, 400), person(400, 100, 550, 400)])
    assert classifier.calls == 2


def test_person_forgotten_after_missing_timeout(pipeline, classifier, clock):
    pipeline.process(FRAME, [person()])
    clock.now += config.TRACK_MAX_MISSING_SECONDS + 1
    pipeline.process(FRAME, [])
    pipeline.process(FRAME, [person()])
    assert classifier.calls == 2


def test_unusable_crop_is_unknown_without_classifying(pipeline, classifier):
    result = pipeline.process(FRAME, [person(300, 200, 330, 240)])[0]  # tiny box
    assert result.label == "unknown" and result.warm_prob is None and "too small" in result.reason
    assert classifier.calls == 0


def test_one_result_per_box_in_order(pipeline):
    boxes = [person(100, 100, 250, 400), person(300, 200, 330, 240), person(400, 100, 550, 400)]
    labels = [result.label for result in pipeline.process(FRAME, boxes)]
    assert labels == ["warm", "unknown", "warm"]


def test_count_warm_ignores_unknown():
    results = [
        ClothingResult("warm", 0.9, ""),
        ClothingResult("warm", 0.8, ""),
        ClothingResult("light", 0.1, ""),
        ClothingResult("unknown", None, "crop too small 30x10"),
    ]
    assert count_warm(results) == (2, 3)
