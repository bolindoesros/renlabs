from dataclasses import dataclass
from typing import Literal

import numpy as np

BoxKind = Literal["person", "head"]
ClothingLabel = Literal["warm", "light", "unknown"]


@dataclass(frozen=True)
class Box:
    """Pixel box; x2, y2 are exclusive."""

    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    kind: BoxKind
    torso_top_y: int | None = None  # pixel row where clothing starts, if known

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1


@dataclass(frozen=True)
class ClothingResult:
    label: ClothingLabel
    warm_prob: float | None  # None when label is unknown
    reason: str  # filled only when unknown


@dataclass(frozen=True)
class CropResult:
    crop: np.ndarray | None  # BGR uint8, None when unusable
    reason: str  # filled only when crop is None


@dataclass(frozen=True)
class Region:
    """Pixel rectangle; x2, y2 are exclusive. May extend past the frame."""

    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1
