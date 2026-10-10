"""One function per way to find the torso."""
from typing import Callable

from jacket import config
from jacket.types import Box, BoxKind, Region


class CropRejected(Exception):
    """A strategy found no usable region."""


RegionStrategy = Callable[[Box], Region]


def person_region(box: Box) -> Region:
    """Chest area below the chin, sized by width."""
    if box.torso_top_y is None:
        raise CropRejected("no shoulders or face found")
    top = box.torso_top_y
    bottom = min(top + int(box.width * config.PERSON_CROP_HEIGHT_WIDTHS), box.y2)
    return Region(box.x1, top, box.x2, bottom)


def head_region(box: Box) -> Region:
    """Torso estimated from head size, starting under the chin."""
    head_width = box.x2 - box.x1
    head_height = box.y2 - box.y1
    center_x = (box.x1 + box.x2) / 2
    half_width = head_width * config.HEAD_CROP_WIDTH_HEADS / 2
    top = box.y2 + head_height * config.HEAD_CROP_GAP_HEADS
    bottom = top + head_height * config.HEAD_CROP_HEIGHT_HEADS
    return Region(int(center_x - half_width), int(top), int(center_x + half_width), int(bottom))


STRATEGIES: dict[BoxKind, RegionStrategy] = {
    "person": person_region,
    "head": head_region,
}
