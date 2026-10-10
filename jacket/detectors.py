"""Body, head, or both deciding together."""
from typing import Protocol

import numpy as np

from jacket import config
from jacket.types import Box


class BoxDetector(Protocol):
    def detect(self, frame: np.ndarray) -> list[Box]: ...


def parts_of(mode: str) -> tuple[str, ...]:
    """The single models a detector mode runs."""
    if mode not in config.DETECTOR_MODES:
        raise KeyError(f"unknown detector mode {mode!r}")
    return ("body", "head") if mode == "both" else (mode,)


def build_detector(part: str) -> BoxDetector:
    """Slow: loads weights, downloading the head model once."""
    if part == "body":
        from jacket.person_detector import PersonDetector

        return PersonDetector(
            config.YOLO_WEIGHTS_PATH, config.YOLO_PERSON_CLASS_ID, config.YOLO_MIN_CONFIDENCE,
            config.YOLO_IMAGE_SIZE, config.DEVICE,
        )
    if part == "head":
        from jacket.head_detector import HeadDetector

        return HeadDetector(
            config.HEAD_WEIGHTS_PATH, config.HEAD_WEIGHTS_URL, config.HEAD_WEIGHTS_SHA256,
            config.HEAD_MIN_CONFIDENCE, config.HEAD_IMAGE_SIZE, config.DEVICE,
        )
    raise KeyError(f"unknown detector {part!r}")


def head_fit(person: Box, head: Box) -> float | None:
    """How far a head sits from this body's head spot; None if it cannot be its head."""
    center_x, center_y = (head.x1 + head.x2) / 2, (head.y1 + head.y2) / 2
    margin = config.FUSE_HEAD_SIDE_MARGIN * person.width
    if not person.x1 - margin <= center_x <= person.x2 + margin:
        return None
    if not person.y1 - head.height <= center_y <= person.y1 + config.FUSE_HEAD_TOP_FRACTION * person.height:
        return None
    return abs(center_x - (person.x1 + person.x2) / 2) / person.width + abs(center_y - person.y1) / person.height


def merge(person: Box, head: Box) -> Box:
    """One person both models saw; the chin stands in for missing shoulders."""
    confidence = 1 - (1 - person.confidence) * (1 - head.confidence)  # either alone could be right
    torso_top_y = person.torso_top_y
    if torso_top_y is None and person.y1 < head.y2 < person.y2:
        torso_top_y = head.y2
    return Box(person.x1, person.y1, person.x2, person.y2, confidence, "person", torso_top_y)


def fuse(people: list[Box], heads: list[Box]) -> list[Box]:
    """Pair heads with bodies; keep what only one model saw."""
    pairs = sorted(
        (fit, p, h)
        for p, person in enumerate(people) if person.width > 0 and person.height > 0
        for h, head in enumerate(heads)
        if (fit := head_fit(person, head)) is not None
    )
    head_of: dict[int, int] = {}
    paired: set[int] = set()
    for _, p, h in pairs:
        if p not in head_of and h not in paired:
            head_of[p] = h
            paired.add(h)
    fused = [merge(person, heads[head_of[p]]) if p in head_of else person for p, person in enumerate(people)]
    return fused + [head for h, head in enumerate(heads) if h not in paired]


def detect(frame: np.ndarray, detectors: dict[str, BoxDetector], mode: str) -> list[Box]:
    """Run the mode's detectors; two means fuse."""
    found = [detectors[part].detect(frame) for part in parts_of(mode)]
    return fuse(*found) if len(found) == 2 else found[0]
