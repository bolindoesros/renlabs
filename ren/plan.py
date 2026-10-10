"""Where people sit: pixels to plan to seats."""
import math
from dataclasses import dataclass, replace
from typing import Literal

import cv2
import numpy as np

from jacket import config
from jacket.types import Box, ClothingResult

SeatState = Literal["hot", "cold", "unsure"]
LABEL_TO_STATE: dict[str, SeatState] = {"light": "hot", "warm": "cold", "unknown": "unsure"}

Corner = tuple[float, float]
Seat = tuple[int, int]  # (row, col)
CORNER_NAMES = ("back_left", "back_right", "front_right", "front_left")
PLAN_UV = np.float32([[0, 1], [1, 1], [1, 0], [0, 0]])  # corners in plan units


@dataclass(frozen=True)
class PlanLayout:
    """Seats in the room: a grid with some cells switched off. Row 0 is the front."""

    rows: int = config.PLAN_ROWS
    cols: int = config.PLAN_COLS
    off: tuple[Seat, ...] = ()  # cells with no seat: aisles, pillars, gaps

    def has_seat(self, row: int, col: int) -> bool:
        return 0 <= row < self.rows and 0 <= col < self.cols and (row, col) not in self.off

    def seats(self) -> list[Seat]:
        return [(row, col) for row in range(self.rows) for col in range(self.cols) if self.has_seat(row, col)]

    @property
    def seat_count(self) -> int:
        return len(self.seats())

    def with_seat_toggled(self, row: int, col: int) -> "PlanLayout":
        """Remove a seat, or bring a removed one back."""
        return replace(self, off=tuple(sorted(set(self.off) ^ {(row, col)})))


@dataclass(frozen=True)
class Calibration:
    """The seating area in the image, as fractions."""

    back_left: Corner = config.PLAN_CORNERS[0]
    back_right: Corner = config.PLAN_CORNERS[1]
    front_right: Corner = config.PLAN_CORNERS[2]
    front_left: Corner = config.PLAN_CORNERS[3]

    def corners(self) -> list[Corner]:
        return [getattr(self, name) for name in CORNER_NAMES]

    def with_corner(self, index: int, point: Corner) -> "Calibration":
        return replace(self, **{CORNER_NAMES[index]: point})

    def is_valid(self) -> bool:
        """A convex quad with real area, corners in order."""
        points = self.corners()
        signs = []
        for i in range(4):
            ax, ay = points[i]
            bx, by = points[(i + 1) % 4]
            cx, cy = points[(i + 2) % 4]
            signs.append((bx - ax) * (cy - by) - (by - ay) * (cx - bx))
        convex = all(s > 1e-6 for s in signs) or all(s < -1e-6 for s in signs)
        return convex and abs(_area(points)) > 1e-3


def _area(points: list[Corner]) -> float:
    return 0.5 * sum(
        points[i][0] * points[(i + 1) % 4][1] - points[(i + 1) % 4][0] * points[i][1] for i in range(4)
    )


def image_to_plan(calibration: Calibration, frame_size: tuple[int, int]) -> np.ndarray:
    """3x3 matrix from frame pixels to plan units."""
    if not calibration.is_valid():
        raise ValueError("calibration corners do not form a proper quad")
    width, height = frame_size
    source = np.float32(calibration.corners()) * np.float32([width, height])
    return cv2.getPerspectiveTransform(source, PLAN_UV)


def plan_to_image(calibration: Calibration, frame_size: tuple[int, int]) -> np.ndarray:
    return np.linalg.inv(image_to_plan(calibration, frame_size))


def anchor_of(box: Box) -> tuple[float, float]:
    """Middle of the box on the shoulder line."""
    if box.torso_top_y is not None:
        y = float(box.torso_top_y)
    else:
        y = box.y1 + config.PLAN_ANCHOR_FALLBACK * box.height
    return (box.x1 + box.x2) / 2, y


@dataclass(frozen=True)
class SeatReading:
    state: SeatState
    warm_prob: float | None  # None when clothing was not read


@dataclass(frozen=True)
class SeatingScene:
    layout: PlanLayout
    seats: dict[tuple[int, int], SeatReading]  # occupied seats by (row, col)
    unseated: tuple[int, ...] = ()  # box indices outside the seating area

    @property
    def off_plan(self) -> int:
        return len(self.unseated)


def read_scene(
    boxes: list[Box],
    results: list[ClothingResult],
    frame_size: tuple[int, int],
    layout: PlanLayout,
    calibration: Calibration,
    radius: float = config.PLAN_SEAT_MATCH_RADIUS,
) -> SeatingScene:
    """Give each person their nearest free seat."""
    if not boxes:
        return SeatingScene(layout, {}, ())
    matrix = image_to_plan(calibration, frame_size)
    anchors = np.float32([anchor_of(box) for box in boxes]).reshape(-1, 1, 2)
    plan_points = cv2.perspectiveTransform(anchors, matrix).reshape(-1, 2)

    pairs = []
    for person, (u, v) in enumerate(plan_points):
        for row, col in layout.seats():
            distance = math.hypot(u * layout.cols - (col + 0.5), v * layout.rows - (row + 0.5))
            if distance <= radius:
                pairs.append((distance, person, row, col))
    pairs.sort()

    seats: dict[tuple[int, int], SeatReading] = {}
    seated: set[int] = set()
    for _, person, row, col in pairs:
        if person in seated or (row, col) in seats:
            continue
        result = results[person] if results else None
        label = result.label if result is not None else "unknown"
        seats[(row, col)] = SeatReading(LABEL_TO_STATE[label], result.warm_prob if result else None)
        seated.add(person)
    return SeatingScene(layout, seats, tuple(i for i in range(len(boxes)) if i not in seated))


@dataclass(frozen=True)
class Zone:
    name: str
    rows: tuple[int, ...]
    cols: tuple[int, ...]

    def seats(self) -> list[tuple[int, int]]:
        return [(row, col) for row in self.rows for col in self.cols]


DEPTH_NAMES = {1: [""], 2: ["front", "back"], 3: ["front", "middle", "back"]}
SIDE_NAMES = {1: [""], 2: ["left", "right"], 3: ["left", "centre", "right"]}


def _split(count: int, parts: int) -> list[tuple[int, ...]]:
    return [tuple(int(i) for i in chunk) for chunk in np.array_split(range(count), parts)]


def make_zones(
    layout: PlanLayout, zone_rows: int = config.PLAN_ZONE_ROWS, zone_cols: int = config.PLAN_ZONE_COLS
) -> list[Zone]:
    """Split the plan into zones, front to back."""
    zone_rows, zone_cols = min(zone_rows, layout.rows), min(zone_cols, layout.cols)
    depth = DEPTH_NAMES.get(zone_rows, [f"band {i + 1}" for i in range(zone_rows)])
    side = SIDE_NAMES.get(zone_cols, [f"column {i + 1}" for i in range(zone_cols)])
    zones = []
    for depth_name, rows in zip(depth, _split(layout.rows, zone_rows)):
        for side_name, cols in zip(side, _split(layout.cols, zone_cols)):
            zones.append(Zone(" ".join(part for part in (depth_name, side_name) if part) or "all seats", rows, cols))
    return zones


Line = tuple[tuple[float, float], tuple[float, float]]


def grid_lines(layout: PlanLayout, calibration: Calibration, frame_size: tuple[int, int]) -> list[Line]:
    """Seat boundaries as pixel lines in the camera image."""
    matrix = plan_to_image(calibration, frame_size)
    segments = [((i / layout.cols, 0.0), (i / layout.cols, 1.0)) for i in range(layout.cols + 1)]
    segments += [((0.0, j / layout.rows), (1.0, j / layout.rows)) for j in range(layout.rows + 1)]
    points = np.float32([point for segment in segments for point in segment]).reshape(-1, 1, 2)
    mapped = cv2.perspectiveTransform(points, matrix).reshape(-1, 2)
    return [((float(a[0]), float(a[1])), (float(b[0]), float(b[1]))) for a, b in zip(mapped[0::2], mapped[1::2])]
