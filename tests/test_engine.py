from dataclasses import replace

import numpy as np
import pytest

from jacket.pipeline import ClothingPipeline
from jacket.types import Box, ClothingResult
from ren.engine import FrameEngine, ViewSettings

FRAME = np.zeros((480, 640, 3), dtype=np.uint8)
BOX = Box(200, 100, 400, 450, 0.9, "person", torso_top_y=180)


class FakeDetector:
    def __init__(self, boxes=None) -> None:
        self.boxes = [BOX] if boxes is None else boxes
        self.calls = 0

    def detect(self, frame):
        self.calls += 1
        return self.boxes


class FakeClassifier:
    def __init__(self, warm_prob: float) -> None:
        self.warm_prob = warm_prob
        self.calls = 0

    def __call__(self, crop):
        self.calls += 1
        return ClothingResult("warm", self.warm_prob, "")


@pytest.fixture
def detector() -> FakeDetector:
    return FakeDetector()


@pytest.fixture
def classifiers() -> dict[str, FakeClassifier]:
    return {"clip": FakeClassifier(0.9), "fashion": FakeClassifier(0.1)}


@pytest.fixture
def engine(detector, classifiers) -> FrameEngine:
    return FrameEngine(detector, lambda key: ClothingPipeline(classifiers[key]))


def settings(**changes) -> ViewSettings:
    return replace(ViewSettings(model_key="clip", mirror=False), **changes)


def test_everything_on_classifies_and_summarizes(engine):
    result = engine.process(FRAME, settings())
    assert len(result.boxes) == 1 and result.results[0].label == "warm"
    assert (result.headline, result.caption) == ("1 of 1", "visible people warm")


def test_detection_off_skips_detector_and_classifier(engine, detector, classifiers):
    result = engine.process(FRAME, settings(detect_people=False))
    assert detector.calls == 0 and classifiers["clip"].calls == 0
    assert result.boxes == [] and (result.headline, result.caption) == ("off", "human detection")


def test_classification_off_detects_but_does_not_classify(engine, detector, classifiers):
    result = engine.process(FRAME, settings(classify_clothing=False))
    assert detector.calls == 1 and classifiers["clip"].calls == 0
    assert result.results == [] and (result.headline, result.caption) == ("1", "person detected")


def test_plural_caption_for_several_people(classifiers):
    boxes = [BOX, replace(BOX, x1=10, x2=150)]
    engine = FrameEngine(FakeDetector(boxes), lambda key: ClothingPipeline(classifiers[key]))
    assert engine.process(FRAME, settings(classify_clothing=False)).caption == "people detected"


def test_nobody_in_view(classifiers):
    engine = FrameEngine(FakeDetector([]), lambda key: ClothingPipeline(classifiers[key]))
    result = engine.process(FRAME, settings())
    assert (result.headline, result.caption) == ("0", "no people in view")


def test_nobody_classifiable_is_not_reported_as_zero_warm(classifiers):
    tiny = Box(300, 200, 330, 240, 0.9, "person", torso_top_y=210)  # crop too small -> unknown
    engine = FrameEngine(FakeDetector([tiny]), lambda key: ClothingPipeline(classifiers[key]))
    result = engine.process(FRAME, settings())
    assert (result.headline, result.caption) == ("-", "no visible people classified")


def test_model_switch_uses_the_other_classifier(engine, classifiers):
    engine.process(FRAME, settings(model_key="clip"))
    result = engine.process(FRAME, settings(model_key="fashion"))
    assert classifiers["fashion"].calls == 1
    assert result.results[0].warm_prob == 0.1


def test_models_load_lazily_and_once(engine):
    assert not engine.is_loaded("fashion")
    engine.process(FRAME, settings(model_key="fashion"))
    engine.process(FRAME, settings(model_key="fashion"))
    assert engine.is_loaded("fashion") and not engine.is_loaded("clip")


def test_crop_strip_only_when_requested(engine):
    assert engine.process(FRAME, settings()).crop_strip is None
    assert engine.process(FRAME, settings(show_crops=True)).crop_strip is not None


def test_no_crop_strip_without_usable_crops(classifiers):
    tiny = Box(300, 200, 330, 240, 0.9, "person", torso_top_y=210)  # crop too small
    for boxes in ([], [tiny]):
        engine = FrameEngine(FakeDetector(boxes), lambda key: ClothingPipeline(classifiers[key]))
        assert engine.process(FRAME, settings(show_crops=True)).crop_strip is None


def test_the_frame_comes_back_undrawn(engine):
    bright = np.full((480, 640, 3), 50, dtype=np.uint8)
    result = engine.process(bright, settings())
    assert np.array_equal(result.frame, bright)  # drawing is the window's job


def test_mirror_flips_the_frame(engine):
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:, :10] = 255  # bright strip on the left
    plain = engine.process(frame, settings(detect_people=False, mirror=False)).frame
    mirrored = engine.process(frame, settings(detect_people=False, mirror=True)).frame
    assert plain[0, 0, 0] == 255 and mirrored[0, -1, 0] == 255
