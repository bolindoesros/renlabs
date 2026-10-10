import logging
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.utils.downloads import attempt_download_asset

from jacket import config
from jacket.types import Box

logger = logging.getLogger(__name__)


def _is_confident(keypoints_conf: np.ndarray, *indices: int) -> bool:
    return all(keypoints_conf[i] >= config.KEYPOINT_MIN_CONFIDENCE for i in indices)


def torso_top_row(
    keypoints_xy: np.ndarray, keypoints_conf: np.ndarray, box_top: int, box_bottom: int
) -> int | None:
    """Row where clothing starts: shoulders or chin."""
    candidates: list[float] = []
    left_shoulder, right_shoulder = config.KEYPOINT_LEFT_SHOULDER, config.KEYPOINT_RIGHT_SHOULDER
    shoulder_rows = [
        float(keypoints_xy[i][1])
        for i in (left_shoulder, right_shoulder)
        if _is_confident(keypoints_conf, i)
    ]
    if shoulder_rows:
        candidates.append(sum(shoulder_rows) / len(shoulder_rows))

    nose, left_eye, right_eye = config.KEYPOINT_NOSE, config.KEYPOINT_LEFT_EYE, config.KEYPOINT_RIGHT_EYE
    if _is_confident(keypoints_conf, nose, left_eye, right_eye):
        eye_distance = abs(float(keypoints_xy[left_eye][0] - keypoints_xy[right_eye][0]))
        candidates.append(
            float(keypoints_xy[nose][1]) + config.CHIN_BELOW_NOSE_EYE_DISTANCES * eye_distance
        )

    inside_box = [row for row in candidates if box_top < row < box_bottom]
    return int(max(inside_box)) if inside_box else None


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():  # Apple Silicon GPU
        return "mps"
    return "cpu"


class PersonDetector:
    """YOLO wrapper that returns only person boxes."""

    def __init__(
        self,
        weights_path: Path,
        person_class_id: int,
        min_confidence: float,
        image_size: int,
        device: str,
    ) -> None:
        self._person_class_id = person_class_id
        self._min_confidence = min_confidence
        self._image_size = image_size
        self._device = resolve_device(device)
        weights_path.parent.mkdir(parents=True, exist_ok=True)
        self._model = YOLO(attempt_download_asset(str(weights_path)))
        logger.info("person detector: %s on %s", weights_path.name, self._device)

    def detect(self, frame: np.ndarray) -> list[Box]:
        """Return person boxes found in a BGR frame."""
        results = self._model.predict(
            frame,
            classes=[self._person_class_id],
            conf=self._min_confidence,
            imgsz=self._image_size,
            device=self._device,
            verbose=False,
        )
        result = results[0]
        keypoints_xy = result.keypoints.xy.cpu().numpy()
        keypoints_conf = result.keypoints.conf.cpu().numpy()
        boxes: list[Box] = []
        for i, (xyxy, confidence) in enumerate(zip(result.boxes.xyxy, result.boxes.conf)):
            x1, y1, x2, y2 = (int(value) for value in xyxy.tolist())
            torso_top_y = torso_top_row(keypoints_xy[i], keypoints_conf[i], y1, y2)
            boxes.append(Box(x1, y1, x2, y2, float(confidence), "person", torso_top_y))
        return boxes
