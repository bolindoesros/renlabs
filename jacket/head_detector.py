import logging
from pathlib import Path

import numpy as np
from torch.hub import download_url_to_file
from ultralytics import YOLO

from jacket.person_detector import resolve_device
from jacket.types import Box

logger = logging.getLogger(__name__)


def ensure_weights(path: Path, url: str, sha256: str) -> Path:
    """Download once; a wrong hash raises and leaves nothing behind."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("downloading head weights to %s", path)
        download_url_to_file(url, str(path), hash_prefix=sha256)
    return path


class HeadDetector:
    """YOLO head model; finds people whose bodies are hidden."""

    def __init__(
        self, weights_path: Path, weights_url: str, weights_sha256: str,
        min_confidence: float, image_size: int, device: str,
    ) -> None:
        self._min_confidence = min_confidence
        self._image_size = image_size
        self._device = resolve_device(device)
        self._model = YOLO(str(ensure_weights(weights_path, weights_url, weights_sha256)))
        logger.info("head detector: %s on %s", weights_path.name, self._device)

    def detect(self, frame: np.ndarray) -> list[Box]:
        """Head boxes in a BGR frame; the chin is the torso top."""
        result = self._model.predict(
            frame, conf=self._min_confidence, imgsz=self._image_size, device=self._device, verbose=False,
        )[0]
        boxes: list[Box] = []
        for xyxy, confidence in zip(result.boxes.xyxy, result.boxes.conf):
            x1, y1, x2, y2 = (int(value) for value in xyxy.tolist())
            boxes.append(Box(x1, y1, x2, y2, float(confidence), "head", torso_top_y=y2))
        return boxes
