"""Live view. Run: python -m jacket [--model clip|fashion] [--camera N] [--save-crops]"""
import argparse
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from jacket import config
from jacket.camera import Camera, find_camera_index
from jacket.classifier import ClothingClassifier
from jacket.crop import crop_torso
from jacket.person_detector import PersonDetector
from jacket.pipeline import ClothingPipeline, count_warm
from jacket.types import Box, ClothingResult, CropResult

logger = logging.getLogger("jacket.debug_view")  # explicit: __name__ is __main__ under -m


def draw_box(frame: np.ndarray, box: Box, result: ClothingResult) -> None:
    color = config.DEBUG_LABEL_COLORS_BGR[result.label]
    cv2.rectangle(
        frame, (box.x1, box.y1), (box.x2, box.y2), color, config.DEBUG_BOX_THICKNESS,
    )
    if box.torso_top_y is not None:
        cv2.line(frame, (box.x1, box.torso_top_y), (box.x2, box.torso_top_y), color, 1)
    label = f"{box.kind} {box.confidence:.2f} {result.label}"
    if result.warm_prob is not None:
        label += f" {result.warm_prob:.2f}"
    if result.reason:
        label += f": {result.reason}"
    text_y = max(box.y1 - 6, 14)  # keep label on screen
    cv2.putText(
        frame, label, (box.x1, text_y), cv2.FONT_HERSHEY_SIMPLEX,
        config.DEBUG_FONT_SCALE, color, config.DEBUG_BOX_THICKNESS,
    )


def build_crop_strip(crop_results: list[CropResult]) -> np.ndarray:
    """Place all usable crops side by side at one height."""
    height = config.DEBUG_CROP_STRIP_HEIGHT_PX
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


class CropSaver:
    """Writes usable crops to disk, at most once per interval."""

    def __init__(
        self, output_dir: Path, interval_seconds: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._output_dir = output_dir
        self._interval_seconds = interval_seconds
        self._clock = clock
        self._last_saved_at: float | None = None

    def maybe_save(self, crop_results: list[CropResult]) -> list[Path]:
        crops = [result.crop for result in crop_results if result.crop is not None]
        now = self._clock()
        due = self._last_saved_at is None or now - self._last_saved_at >= self._interval_seconds
        if not crops or not due:
            return []
        self._output_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        paths = []
        for index, crop in enumerate(crops):
            path = self._output_dir / f"{stamp}_{index}.jpg"
            if not cv2.imwrite(str(path), crop):
                raise OSError(f"could not write crop to {path}")
            paths.append(path)
        self._last_saved_at = now
        logger.info("saved %d crop(s) to %s", len(paths), self._output_dir)
        return paths


def group_text(results: list[ClothingResult]) -> str:
    warm_count, known_count = count_warm(results)
    if known_count == 0:
        return "no visible people classified"
    return f"{warm_count} of {known_count} visible people warm"


def draw_status(frame: np.ndarray, results: list[ClothingResult], fps: float) -> None:
    lines = [
        f"fps: {fps:.1f}  [{config.DEBUG_CROP_KEY}] crops  [{config.DEBUG_QUIT_KEY}] quit",
        group_text(results),
    ]
    for index, line in enumerate(lines):
        cv2.putText(
            frame, line, (10, 24 + 28 * index), cv2.FONT_HERSHEY_SIMPLEX,
            config.DEBUG_FONT_SCALE, (255, 255, 255), config.DEBUG_BOX_THICKNESS,
        )


def run(model_key: str, save_crops: bool, camera_index: int | None) -> None:
    detector = PersonDetector(
        config.YOLO_WEIGHTS_PATH,
        config.YOLO_PERSON_CLASS_ID,
        config.YOLO_MIN_CONFIDENCE,
        config.YOLO_IMAGE_SIZE,
        config.DEVICE,
    )
    pipeline = ClothingPipeline(ClothingClassifier(model_key, config.DEVICE).classify)
    saver = None
    if save_crops:
        saver = CropSaver(config.RAW_CROPS_DIR, config.SAVE_CROP_INTERVAL_SECONDS)
        logger.info("saving crops to %s (gitignored)", config.RAW_CROPS_DIR)
    quit_key_code = ord(config.DEBUG_QUIT_KEY)
    crop_key_code = ord(config.DEBUG_CROP_KEY)
    show_crops = False
    with Camera(
        camera_index if camera_index is not None else find_camera_index(config.CAMERA_NAME_HINT),
        config.CAMERA_WIDTH, config.CAMERA_HEIGHT,
        config.CAMERA_WARMUP_FRAMES,
    ) as camera:
        previous_time = time.perf_counter()
        while True:
            frame = camera.read()
            boxes = detector.detect(frame)
            results = pipeline.process(frame, boxes)
            now = time.perf_counter()
            fps = 1.0 / (now - previous_time)
            previous_time = now
            # Crop before drawing so overlays never leak into crops.
            if show_crops or saver is not None:
                crop_results = [crop_torso(frame, box) for box in boxes]
            if saver is not None:
                saver.maybe_save(crop_results)
            if show_crops:
                strip = build_crop_strip(crop_results)
            for box, result in zip(boxes, results):
                draw_box(frame, box, result)
            draw_status(frame, results, fps)
            cv2.imshow(config.DEBUG_WINDOW_NAME, frame)
            if show_crops:
                cv2.imshow(config.DEBUG_CROP_WINDOW_NAME, strip)
            key = cv2.waitKey(1) & 0xFF
            if key == quit_key_code:
                break
            if key == crop_key_code:
                show_crops = not show_crops
                if not show_crops:
                    cv2.destroyWindow(config.DEBUG_CROP_WINDOW_NAME)
    cv2.destroyAllWindows()


def main(default_model: str = config.DEFAULT_CLIP_MODEL) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(config.CLIP_MODELS), default=default_model)
    parser.add_argument("--camera", type=int, help="camera index; default finds the C920 by name")
    parser.add_argument("--save-crops", action="store_true", help="write crops to data/raw/")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("jacket").setLevel(logging.INFO)  # keep library logs quiet
    run(args.model, args.save_crops, args.camera)


if __name__ == "__main__":
    main()
