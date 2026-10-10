"""Drawing helpers; none of them run a model."""
import cv2
import numpy as np

from jacket import config
from jacket.pipeline import count_warm
from jacket.types import Box, ClothingResult, CropResult


def label_text(box: Box, result: ClothingResult | None, verbose: bool) -> str:
    """Verbose is for debugging, short for the demo."""
    if result is None:
        return f"{box.kind} {box.confidence:.2f}" if verbose else f"{box.kind} {box.confidence:.0%}"
    parts = [f"{box.kind} {box.confidence:.2f}"] if verbose else []
    parts.append(result.label)
    if result.warm_prob is not None:
        parts.append(f"{result.warm_prob:.2f}" if verbose else f"{result.warm_prob:.0%}")
    if result.reason and verbose:
        parts.append(f"- {result.reason}")
    return " ".join(parts)


def draw_box(
    frame: np.ndarray,
    box: Box,
    result: ClothingResult | None,
    *,
    show_label: bool = True,
    show_torso_line: bool = True,
    verbose: bool = True,
) -> None:
    """Draw one person; no result means a neutral box."""
    color = (
        config.OVERLAY_NEUTRAL_COLOR_BGR if result is None
        else config.DEBUG_LABEL_COLORS_BGR[result.label]
    )
    cv2.rectangle(frame, (box.x1, box.y1), (box.x2, box.y2), color, config.DEBUG_BOX_THICKNESS)
    if show_torso_line and box.torso_top_y is not None:
        cv2.line(frame, (box.x1, box.torso_top_y), (box.x2, box.torso_top_y), color, 1)
    if show_label:
        _draw_label(frame, label_text(box, result, verbose), box.x1, box.y1, color)


def _draw_label(frame: np.ndarray, text: str, x: int, y: int, color: tuple[int, int, int]) -> None:
    """Filled tag above the box, readable anywhere."""
    pad = config.OVERLAY_LABEL_PADDING_PX
    (text_w, text_h), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, config.DEBUG_FONT_SCALE, 1
    )
    top = max(y - text_h - baseline - 2 * pad, 0)  # keep the tag on screen
    cv2.rectangle(frame, (x, top), (x + text_w + 2 * pad, top + text_h + baseline + 2 * pad), color, -1)
    cv2.putText(
        frame, text, (x + pad, top + text_h + pad), cv2.FONT_HERSHEY_SIMPLEX,
        config.DEBUG_FONT_SCALE, config.OVERLAY_LABEL_TEXT_BGR, 1, cv2.LINE_AA,
    )


def build_crop_strip(
    crop_results: list[CropResult], height: int = config.DEBUG_CROP_STRIP_HEIGHT_PX
) -> np.ndarray:
    """Usable crops side by side at one height."""
    crops = [result.crop for result in crop_results if result.crop is not None]
    if not crops:
        placeholder = np.zeros((height, 2 * height, 3), dtype=np.uint8)
        cv2.putText(placeholder, "no usable crops", (10, height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, config.DEBUG_FONT_SCALE,
                    (255, 255, 255), config.DEBUG_BOX_THICKNESS)
        return placeholder
    resized = []
    for crop in crops:
        width = max(1, round(crop.shape[1] * height / crop.shape[0]))
        resized.append(cv2.resize(crop, (width, height)))
    return np.hstack(resized)


def group_text(results: list[ClothingResult]) -> str:
    warm_count, known_count = count_warm(results)
    if known_count == 0:
        return "no visible people classified"
    return f"{warm_count} of {known_count} visible people warm"
