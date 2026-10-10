"""Frame sources shaped like Camera."""
import logging
import time
from pathlib import Path
from types import TracebackType
from typing import Protocol

import cv2
import numpy as np

from jacket import config
from jacket.camera import CameraError

logger = logging.getLogger("jacket.sources")


class FrameSource(Protocol):
    def __enter__(self) -> "FrameSource": ...

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None: ...

    def read(self) -> np.ndarray: ...


class ImageSource:
    """One still image, repeated at a steady rate."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._image: np.ndarray | None = None

    def __enter__(self) -> "ImageSource":
        image = cv2.imread(str(self._path))
        if image is None:
            raise CameraError(f"cannot read image {self._path}")
        self._image = image
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self._image = None

    def read(self) -> np.ndarray:
        if self._image is None:
            raise CameraError("image source is not open")
        time.sleep(1 / config.STILL_IMAGE_FPS)
        return self._image.copy()


class VideoFileSource:
    """A looped recorded video, the stage fallback."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._capture: cv2.VideoCapture | None = None
        self._frame_seconds = 1 / 30
        self._last_read_at = 0.0

    def __enter__(self) -> "VideoFileSource":
        capture = cv2.VideoCapture(str(self._path))
        if not capture.isOpened():
            raise CameraError(f"cannot open video {self._path}")
        self._capture = capture
        self._frame_seconds = 1 / (capture.get(cv2.CAP_PROP_FPS) or 30)
        logger.info("video %s at %.1f fps", self._path.name, 1 / self._frame_seconds)
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def read(self) -> np.ndarray:
        if self._capture is None:
            raise CameraError("video source is not open")
        was_read, frame = self._capture.read()
        if not was_read:  # end of file: loop back to the start once
            self._capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            was_read, frame = self._capture.read()
            if not was_read:
                raise CameraError(f"video {self._path} has no readable frames")
        wait = self._frame_seconds - (time.perf_counter() - self._last_read_at)
        if wait > 0:
            time.sleep(wait)
        self._last_read_at = time.perf_counter()
        return frame
