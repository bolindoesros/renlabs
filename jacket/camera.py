import logging
from pathlib import Path
from types import TracebackType

import cv2
import numpy as np

from jacket import config

logger = logging.getLogger(__name__)


class CameraError(RuntimeError):
    pass


def list_cameras() -> dict[int, str]:
    """Map V4L2 index to device name."""
    cameras: dict[int, str] = {}
    for node in sorted(Path("/sys/class/video4linux").glob("video*")):
        cameras[int(node.name.removeprefix("video"))] = (node / "name").read_text().strip()
    return cameras


def find_camera_index(name_hint: str) -> int:
    """Lowest index whose name matches, else raise."""
    cameras = list_cameras()
    for index, name in cameras.items():
        if name_hint.lower() in name.lower():
            return index
    raise CameraError(f"no camera named like '{name_hint}'; found {cameras}. Use --camera N")


class Camera:
    """Webcam frame source. Use as a context manager."""

    def __init__(self, index: int, width: int, height: int, warmup_frames: int) -> None:
        self._index = index
        self._width = width
        self._height = height
        self._warmup_frames = warmup_frames
        self._capture: cv2.VideoCapture | None = None

    def __enter__(self) -> "Camera":
        # V4L2 explicitly: auto-detect can pick a slower backend.
        capture = cv2.VideoCapture(self._index, cv2.CAP_V4L2)
        if not capture.isOpened():
            raise CameraError(f"cannot open camera index {self._index}")
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*config.CAMERA_FOURCC))  # before size
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        capture.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)
        self._capture = capture
        for _ in range(self._warmup_frames):
            self.read()  # discard dark startup frames
        actual_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fourcc = int(capture.get(cv2.CAP_PROP_FOURCC)).to_bytes(4, "little").decode("ascii", "replace")
        if actual_fourcc != config.CAMERA_FOURCC:
            logger.warning("camera gave %s, not %s; fps may be low", actual_fourcc, config.CAMERA_FOURCC)
        logger.info("camera %d opened at %dx%d %s", self._index, actual_w, actual_h, actual_fourcc)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def read(self) -> np.ndarray:
        """Return one BGR frame, or raise CameraError."""
        if self._capture is None:
            raise CameraError("camera is not open; use 'with Camera(...)'")
        was_read, frame = self._capture.read()
        if not was_read:
            raise CameraError(f"camera {self._index} returned no frame")
        return frame
