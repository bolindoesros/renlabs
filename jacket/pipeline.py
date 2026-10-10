import logging
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from jacket import config
from jacket.classifier import result_from_warm_probability
from jacket.crop import crop_torso
from jacket.types import Box, ClothingResult

logger = logging.getLogger(__name__)

CropClassifier = Callable[[np.ndarray], ClothingResult]


def iou(first: Box, second: Box) -> float:
    inter_w = min(first.x2, second.x2) - max(first.x1, second.x1)
    inter_h = min(first.y2, second.y2) - max(first.y1, second.y1)
    if inter_w <= 0 or inter_h <= 0:
        return 0.0
    intersection = inter_w * inter_h
    area_first = (first.x2 - first.x1) * (first.y2 - first.y1)
    area_second = (second.x2 - second.x1) * (second.y2 - second.y1)
    return intersection / (area_first + area_second - intersection)


def smooth(previous: float | None, new: float, alpha: float) -> float:
    """Exponential moving average; first score taken as is."""
    return new if previous is None else alpha * new + (1 - alpha) * previous


def count_warm(results: list[ClothingResult]) -> tuple[int, int]:
    """(warm count, known count); unknown people left out."""
    known = [result for result in results if result.label != "unknown"]
    return sum(result.label == "warm" for result in known), len(known)


@dataclass
class Track:
    """One person followed across frames."""

    box: Box
    last_seen_at: float
    smoothed_warm_prob: float | None = None
    last_scored_at: float | None = None


class TrackBook:
    """Matches each frame's boxes to known people by overlap."""

    def __init__(self) -> None:
        self._tracks: list[Track] = []

    def match(self, boxes: list[Box], now: float) -> list[Track]:
        """One track per box, in box order."""
        pairs = sorted(
            (
                (iou(box, track.box), box_index, track_index)
                for box_index, box in enumerate(boxes)
                for track_index, track in enumerate(self._tracks)
                if box.kind == track.box.kind
            ),
            reverse=True,
        )
        matched: dict[int, Track] = {}
        used_tracks: set[int] = set()
        for overlap, box_index, track_index in pairs:
            if overlap < config.TRACK_MIN_IOU:
                break
            if box_index in matched or track_index in used_tracks:
                continue
            matched[box_index] = self._tracks[track_index]
            used_tracks.add(track_index)

        for box_index, box in enumerate(boxes):
            if box_index not in matched:
                matched[box_index] = Track(box, now)
                self._tracks.append(matched[box_index])
                logger.debug("new track for %s", box)
            matched[box_index].box = box
            matched[box_index].last_seen_at = now

        max_missing = config.TRACK_MAX_MISSING_SECONDS
        self._tracks = [t for t in self._tracks if now - t.last_seen_at <= max_missing]
        return [matched[i] for i in range(len(boxes))]


class ClothingPipeline:
    """Crop, score on a timer, smooth."""

    def __init__(
        self, classify_crop: CropClassifier, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self._classify_crop = classify_crop
        self._clock = clock
        self._track_book = TrackBook()

    def process(self, frame: np.ndarray, boxes: list[Box]) -> list[ClothingResult]:
        now = self._clock()
        tracks = self._track_book.match(boxes, now)
        return [self._result_for(frame, box, track, now) for box, track in zip(boxes, tracks)]

    def _result_for(self, frame: np.ndarray, box: Box, track: Track, now: float) -> ClothingResult:
        crop_result = crop_torso(frame, box)
        if crop_result.crop is None:
            logger.debug("unknown: %s", crop_result.reason)
            return ClothingResult("unknown", None, crop_result.reason)
        if self._is_due(track, now):
            raw = self._classify_crop(crop_result.crop)
            track.last_scored_at = now
            if raw.warm_prob is None:  # the model could not tell; keep the running score
                if track.smoothed_warm_prob is None:
                    return raw
            else:
                track.smoothed_warm_prob = smooth(
                    track.smoothed_warm_prob, raw.warm_prob, config.SMOOTHING_ALPHA
                )
                logger.debug("raw %.2f -> smoothed %.2f", raw.warm_prob, track.smoothed_warm_prob)
        if track.smoothed_warm_prob is None:
            return ClothingResult("unknown", None, "no score yet")
        return result_from_warm_probability(
            track.smoothed_warm_prob, config.WARM_THRESHOLD, config.LIGHT_THRESHOLD
        )

    @staticmethod
    def _is_due(track: Track, now: float) -> bool:
        if track.last_scored_at is None:
            return True
        return now - track.last_scored_at >= config.CLASSIFY_INTERVAL_SECONDS
