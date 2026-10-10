"""Turns one frame into data for the screen."""
import logging
from dataclasses import dataclass
from typing import Callable, Protocol

import cv2
import numpy as np

from jacket import config
from jacket.crop import crop_torso
from jacket.overlay import build_crop_strip
from jacket.pipeline import ClothingPipeline, count_warm
from jacket.types import Box, ClothingResult

logger = logging.getLogger("ren.engine")


class BoxDetector(Protocol):
    def detect(self, frame: np.ndarray) -> list[Box]: ...


@dataclass(frozen=True)
class ViewSettings:
    """What the user has switched on."""

    # Engine reads the first five; painting reads the rest.
    # Engine reads the first five; painting reads the rest.
    detect_people: bool = True
    classify_clothing: bool = True
    show_crops: bool = False
    mirror: bool = False
    model_key: str = config.DEFAULT_CLIP_MODEL
    show_image: bool = True
    show_people: bool = True
    show_labels: bool = True
    show_torso_line: bool = False
    show_grid: bool = True
    show_seats: bool = True
    show_zones: bool = True
    show_vent: bool = True
    show_beam: bool = True


@dataclass(frozen=True)
class FrameResult:
    frame: np.ndarray  # BGR, mirrored if asked, never drawn on
    boxes: list[Box]
    results: list[ClothingResult]  # one per box; empty when classification is off
    crop_strip: np.ndarray | None  # None unless show_crops
    headline: str  # big text, e.g. "5 of 7"
    caption: str  # small text under it, e.g. "visible people warm"


class FrameEngine:
    def __init__(
        self, detector: BoxDetector, make_pipeline: Callable[[str], ClothingPipeline]
    ) -> None:
        self._detector = detector
        self._make_pipeline = make_pipeline
        self._pipelines: dict[str, ClothingPipeline] = {}

    def is_loaded(self, model_key: str) -> bool:
        return model_key in self._pipelines

    def load(self, model_key: str) -> None:
        """Build a model's pipeline once; slow."""
        if model_key not in self._pipelines:
            self._pipelines[model_key] = self._make_pipeline(model_key)

    def process(self, frame: np.ndarray, settings: ViewSettings) -> FrameResult:
        if settings.mirror:
            frame = cv2.flip(frame, 1)
        boxes = self._detector.detect(frame) if settings.detect_people else []
        results: list[ClothingResult] = []
        if boxes and settings.classify_clothing:
            self.load(settings.model_key)
            results = self._pipelines[settings.model_key].process(frame, boxes)
        strip = None
        if settings.show_crops and boxes:
            crops = [crop_torso(frame, box) for box in boxes]
            if any(crop.crop is not None for crop in crops):  # never show an empty placeholder
                strip = build_crop_strip(crops, config.UI_CROP_STRIP_HEIGHT_PX)
        headline, caption = self._headline(settings, boxes, results)
        return FrameResult(frame, boxes, results, strip, headline, caption)

    @staticmethod
    def _headline(
        settings: ViewSettings, boxes: list[Box], results: list[ClothingResult]
    ) -> tuple[str, str]:
        if not settings.detect_people:
            return "off", "human detection"
        if not boxes:
            return "0", "no people in view"
        if not settings.classify_clothing:
            return str(len(boxes)), "person detected" if len(boxes) == 1 else "people detected"
        warm_count, known_count = count_warm(results)
        if known_count == 0:
            return "-", "no visible people classified"
        return f"{warm_count} of {known_count}", "visible people warm"
