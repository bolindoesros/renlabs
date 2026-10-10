"""Turns one frame into data for the screen."""
import logging
from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np

from jacket import config
from jacket.crop import crop_torso
from jacket.detectors import BoxDetector, detect, parts_of
from jacket.pipeline import ClothingPipeline, count_warm
from jacket.types import Box, ClothingResult, CropResult

logger = logging.getLogger("ren.engine")


@dataclass(frozen=True)
class ViewSettings:
    """What the user has switched on."""

    # Engine reads the first six; painting reads the rest.
    detect_people: bool = True
    classify_clothing: bool = True
    show_crops: bool = True  # follows the torso crops panel
    mirror: bool = False
    model_key: str = config.DEFAULT_CLIP_MODEL
    detector: str = config.DEFAULT_DETECTOR  # body, head or both
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
    crops: tuple[CropResult, ...]  # one per box when show_crops, else empty
    headline: str  # big text, e.g. "5 of 7"
    caption: str  # small text under it, e.g. "visible people warm"


class FrameEngine:
    def __init__(
        self, make_detector: Callable[[str], BoxDetector], make_pipeline: Callable[[str], ClothingPipeline]
    ) -> None:
        self._make_detector = make_detector  # "body" or "head"
        self._make_pipeline = make_pipeline
        self._detectors: dict[str, BoxDetector] = {}
        self._pipelines: dict[str, ClothingPipeline] = {}

    def detector_loaded(self, mode: str) -> bool:
        return all(part in self._detectors for part in parts_of(mode))

    def load_detector(self, mode: str) -> None:
        """Build each detector a mode needs once; slow."""
        for part in parts_of(mode):
            if part not in self._detectors:
                self._detectors[part] = self._make_detector(part)

    def is_loaded(self, model_key: str) -> bool:
        return model_key in self._pipelines

    def load(self, model_key: str) -> None:
        """Build a model's pipeline once; slow."""
        if model_key not in self._pipelines:
            self._pipelines[model_key] = self._make_pipeline(model_key)

    def process(self, frame: np.ndarray, settings: ViewSettings) -> FrameResult:
        if settings.mirror:
            frame = cv2.flip(frame, 1)
        boxes: list[Box] = []
        if settings.detect_people:
            self.load_detector(settings.detector)
            boxes = detect(frame, self._detectors, settings.detector)
        results: list[ClothingResult] = []
        if boxes and settings.classify_clothing:
            self.load(settings.model_key)
            results = self._pipelines[settings.model_key].process(frame, boxes)
        crops = tuple(crop_torso(frame, box) for box in boxes) if settings.show_crops else ()
        headline, caption = self._headline(settings, boxes, results)
        return FrameResult(frame, boxes, results, crops, headline, caption)

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
