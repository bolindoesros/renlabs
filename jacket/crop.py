import numpy as np

from jacket import config
from jacket.types import Box, CropResult


def _unusable(reason: str) -> CropResult:
    return CropResult(crop=None, reason=reason)


def _touches_checked_edge(box: Box, frame_height: int, frame_width: int) -> str | None:
    """Name the first checked frame edge the box touches, else None."""
    margin = config.EDGE_MARGIN_PX
    distances = {
        "left": box.x1,
        "right": frame_width - box.x2,
        "top": box.y1,
        "bottom": frame_height - box.y2,
    }
    for side in config.EDGE_SIDES_CHECKED:
        if distances[side] <= margin:
            return side
    return None


def _person_region(box: Box, torso_top_y: int) -> tuple[int, int, int, int]:
    """Chest area below the chin or shoulders, sized by box width."""
    width = box.x2 - box.x1
    top = torso_top_y
    bottom = min(top + int(width * config.PERSON_CROP_HEIGHT_WIDTHS), box.y2)
    return box.x1, top, box.x2, bottom


def _head_region(box: Box) -> tuple[int, int, int, int]:
    """Torso estimated from head size, starting under the chin."""
    head_width = box.x2 - box.x1
    head_height = box.y2 - box.y1
    center_x = (box.x1 + box.x2) / 2
    half_width = head_width * config.HEAD_CROP_WIDTH_HEADS / 2
    top = box.y2 + head_height * config.HEAD_CROP_GAP_HEADS
    bottom = top + head_height * config.HEAD_CROP_HEIGHT_HEADS
    return int(center_x - half_width), int(top), int(center_x + half_width), int(bottom)


def crop_torso(frame: np.ndarray, box: Box) -> CropResult:
    """Cut the torso region for a box, or explain why it is unusable."""
    frame_height, frame_width = frame.shape[:2]
    if box.confidence < config.MIN_CLASSIFY_CONFIDENCE:
        return _unusable(f"low confidence {box.confidence:.2f}")
    if box.x2 <= box.x1 or box.y2 <= box.y1:
        return _unusable("empty box")

    if box.kind == "person":
        touched_side = _touches_checked_edge(box, frame_height, frame_width)
        if touched_side is not None:
            return _unusable(f"touches {touched_side} edge")
        if box.torso_top_y is None:
            return _unusable("no shoulders or face found")
        x1, y1, x2, y2 = _person_region(box, box.torso_top_y)
    else:
        x1, y1, x2, y2 = _head_region(box)

    # Clip last so head-based estimates may run off the frame.
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, frame_width), min(y2, frame_height)
    if min(x2 - x1, y2 - y1) < config.MIN_CROP_SIDE_PX:
        return _unusable(f"crop too small {x2 - x1}x{y2 - y1}")
    return CropResult(crop=frame[y1:y2, x1:x2].copy(), reason="")
