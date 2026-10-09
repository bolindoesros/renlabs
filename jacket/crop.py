import numpy as np

from jacket import config
from jacket.crop_strategies import STRATEGIES, CropRejected, RegionStrategy
from jacket.types import Box, BoxKind, CropResult, Region


def check_box_usable(box: Box) -> None:
    if box.confidence < config.MIN_CLASSIFY_CONFIDENCE:
        raise CropRejected(f"low confidence {box.confidence:.2f}")
    if box.width <= 0 or box.height <= 0:
        raise CropRejected("empty box")


def clip_to_frame(region: Region, frame_height: int, frame_width: int) -> Region:
    """Clip last, so strategies may estimate regions that run off the frame."""
    return Region(
        max(region.x1, 0), max(region.y1, 0),
        min(region.x2, frame_width), min(region.y2, frame_height),
    )


def check_region_big_enough(region: Region) -> None:
    if min(region.width, region.height) < config.MIN_CROP_SIDE_PX:
        raise CropRejected(f"crop too small {region.width}x{region.height}")


def crop_torso(
    frame: np.ndarray, box: Box, strategies: dict[BoxKind, RegionStrategy] = STRATEGIES
) -> CropResult:
    """Cut the torso for a box using the strategy for its kind, or explain why not."""
    if box.kind not in strategies:
        raise KeyError(f"no crop strategy registered for box kind '{box.kind}'")
    try:
        check_box_usable(box)
        region = clip_to_frame(strategies[box.kind](box), *frame.shape[:2])
        check_region_big_enough(region)
    except CropRejected as rejected:
        return CropResult(crop=None, reason=str(rejected))
    return CropResult(crop=frame[region.y1:region.y2, region.x1:region.x2].copy(), reason="")
