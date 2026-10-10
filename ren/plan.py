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
SeatSpot = tuple[Seat, tuple[float, float]]  # a seat moved off its grid spot, to (u, v)
GridLine = tuple[float, float]  # an inner grid line, by where it meets the two far edges of the quad
CORNER_NAMES = ("back_left", "back_right", "front_right", "front_left")
PLAN_UV = np.float32([[0, 1], [1, 1], [1, 0], [0, 0]])  # corners in plan units


@dataclass(frozen=True)
class PlanLayout:
    """Seats in the room: a grid with some cells switched off. Row 0 is the front."""

    rows: int = config.PLAN_ROWS
    cols: int = config.PLAN_COLS
    off: tuple[Seat, ...] = ()  # cells with no seat: aisles, pillars, gaps
    moved: tuple[SeatSpot, ...] = ()  # hand-placed seats; the rest sit mid-cell

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

    def seat_uv(self, row: int, col: int) -> tuple[float, float]:
        """Plan position: u left to right, v front to back."""
        return dict(self.moved).get((row, col), ((col + 0.5) / self.cols, (row + 0.5) / self.rows))

    def with_seat_moved(self, row: int, col: int, u: float, v: float) -> "PlanLayout":
        spots = dict(self.moved)
        spots[(row, col)] = (min(max(u, 0.0), 1.0), min(max(v, 0.0), 1.0))
        return replace(self, moved=tuple(sorted(spots.items())))

    def with_seat_reset(self, row: int, col: int) -> "PlanLayout":
        """Put a seat back in the middle of its cell."""
        return replace(self, moved=tuple(spot for spot in self.moved if spot[0] != (row, col)))


@dataclass(frozen=True)
class Calibration:
    """The seating area in the image, as fractions."""

    back_left: Corner = config.PLAN_CORNERS[0]
    back_right: Corner = config.PLAN_CORNERS[1]
    front_right: Corner = config.PLAN_CORNERS[2]
    front_left: Corner = config.PLAN_CORNERS[3]
    col_lines: tuple[GridLine, ...] = ()  # u at the front and at the back of each inner column line; () is even
    row_lines: tuple[GridLine, ...] = ()  # v at the left and at the right of each inner row line; () is even

    def corners(self) -> list[Corner]:
        return [getattr(self, name) for name in CORNER_NAMES]

    def cols_for(self, layout: PlanLayout) -> tuple[GridLine, ...]:
        """Inner column lines; even ones if none are set for this many seats."""
        return self.col_lines if len(self.col_lines) == layout.cols - 1 else even_lines(layout.cols)

    def rows_for(self, layout: PlanLayout) -> tuple[GridLine, ...]:
        return self.row_lines if len(self.row_lines) == layout.rows - 1 else even_lines(layout.rows)

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


def even_lines(parts: int) -> tuple[GridLine, ...]:
    return tuple((i / parts, i / parts) for i in range(1, parts))


def lines_ordered(lines: tuple[GridLine, ...]) -> bool:
    """Both ends of every line inside the edges and apart from their neighbours, so no lines cross."""
    gap = config.PLAN_LINE_MIN_GAP
    for end in (0, 1):
        spots = [0.0] + [line[end] for line in lines] + [1.0]
        if any(b - a < gap for a, b in zip(spots, spots[1:])):
            return False
    return True


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


def _along(value: np.ndarray, across: np.ndarray, lines: tuple[GridLine, ...]) -> np.ndarray:
    """Even plan units to seat units on one axis: each gap between lines is one seat wide."""
    parts = len(lines) + 1
    steps = np.linspace(0.0, 1.0, parts + 1)
    out = value.astype(np.float64).copy()
    for i, (x, w) in enumerate(zip(value, np.clip(across, 0.0, 1.0))):
        if 0.0 <= x <= 1.0:
            edges = [0.0] + [a + (b - a) * w for a, b in lines] + [1.0]
            out[i] = np.interp(x, edges, steps)
    return out


def to_seat_units(layout: PlanLayout, calibration: Calibration, points: np.ndarray) -> np.ndarray:
    """Plan points from the corner mapping, moved so the inner grid lines fall on seat borders."""
    if len(points) == 0:
        return points
    u, v = points[:, 0], points[:, 1]
    seat_u = _along(u, v, calibration.cols_for(layout))
    seat_v = _along(v, u, calibration.rows_for(layout))
    return np.stack([seat_u, seat_v], axis=1)


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
class PlanPerson:
    """Someone inside the plan, where they really are."""

    u: float
    v: float
    reading: SeatReading


@dataclass(frozen=True)
class SeatingScene:
    layout: PlanLayout
    seats: dict[tuple[int, int], SeatReading]  # nearest free seat per person, for reference
    unseated: tuple[int, ...] = ()  # box indices outside the plan
    people: tuple[PlanPerson, ...] | None = None  # None: people are where their seats are

    @property
    def off_plan(self) -> int:
        return len(self.unseated)

    def placed(self) -> list[PlanPerson]:
        """Everyone inside the plan, at their real spot."""
        if self.people is not None:
            return list(self.people)
        return [PlanPerson(*self.layout.seat_uv(*seat), reading) for seat, reading in self.seats.items()]


def read_scene(
    boxes: list[Box],
    results: list[ClothingResult],
    frame_size: tuple[int, int],
    layout: PlanLayout,
    calibration: Calibration,
    radius: float = config.PLAN_SEAT_MATCH_RADIUS,
) -> SeatingScene:
    """Place everyone on the plan; also give each their nearest free seat."""
    if not boxes:
        return SeatingScene(layout, {}, (), ())
    matrix = image_to_plan(calibration, frame_size)
    anchors = np.float32([anchor_of(box) for box in boxes]).reshape(-1, 1, 2)
    plan_points = to_seat_units(layout, calibration, cv2.perspectiveTransform(anchors, matrix).reshape(-1, 2))

    def reading_of(person: int) -> SeatReading:
        result = results[person] if results else None
        label = result.label if result is not None else "unknown"
        return SeatReading(LABEL_TO_STATE[label], result.warm_prob if result else None)

    inside = [i for i, (u, v) in enumerate(plan_points) if 0.0 <= u <= 1.0 and 0.0 <= v <= 1.0]
    people = tuple(PlanPerson(float(plan_points[i][0]), float(plan_points[i][1]), reading_of(i)) for i in inside)

    pairs = []
    for person in inside:
        u, v = plan_points[person]
        for row, col in layout.seats():
            seat_u, seat_v = layout.seat_uv(row, col)
            distance = math.hypot((u - seat_u) * layout.cols, (v - seat_v) * layout.rows)
            if distance <= radius:
                pairs.append((distance, person, row, col))
    pairs.sort()

    seats: dict[tuple[int, int], SeatReading] = {}
    seated: set[int] = set()
    for _, person, row, col in pairs:
        if person in seated or (row, col) in seats:
            continue
        seats[(row, col)] = reading_of(person)
        seated.add(person)
    outside = tuple(i for i in range(len(boxes)) if i not in inside)
    return SeatingScene(layout, seats, outside, people)


@dataclass(frozen=True)
class Zone:
    """An equal slice of the room; holds the seats placed inside it."""

    name: str
    members: tuple[Seat, ...]
    bounds: tuple[float, float, float, float]  # u0, v0, u1, v1 in plan units

    def seats(self) -> list[tuple[int, int]]:
        return list(self.members)

    def contains(self, u: float, v: float) -> bool:
        u0, v0, u1, v1 = self.bounds
        return u0 <= u <= u1 and v0 <= v <= v1

    @property
    def middle(self) -> tuple[float, float]:
        u0, v0, u1, v1 = self.bounds
        return (u0 + u1) / 2, (v0 + v1) / 2


@dataclass(frozen=True)
class ZoneRect:
    """A zone drawn by hand, in plan units (u left to right, v front to back)."""

    name: str
    u0: float
    v0: float
    u1: float
    v1: float

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return min(self.u0, self.u1), min(self.v0, self.v1), max(self.u0, self.u1), max(self.v0, self.v1)

    def contains(self, u: float, v: float) -> bool:
        u0, v0, u1, v1 = self.bounds
        return u0 <= u <= u1 and v0 <= v <= v1

    def moved_by(self, du: float, dv: float) -> "ZoneRect":
        """Shifted, but kept inside the plan."""
        u0, v0, u1, v1 = self.bounds
        du = min(max(du, -u0), 1.0 - u1)
        dv = min(max(dv, -v0), 1.0 - v1)
        return ZoneRect(self.name, u0 + du, v0 + dv, u1 + du, v1 + dv)


def zone_of(zones: list["Zone"], u: float, v: float) -> "Zone | None":
    """The first zone holding a plan point."""
    return next((zone for zone in zones if zone.contains(u, v)), None)


def next_zone_name(zones: tuple[ZoneRect, ...]) -> str:
    taken = {zone.name for zone in zones}
    number = 1
    while f"zone {number}" in taken:
        number += 1
    return f"zone {number}"


def drawn_zones(layout: PlanLayout, rects: tuple[ZoneRect, ...]) -> list[Zone]:
    """Hand-drawn zones; a seat in two zones belongs to the first."""
    claimed: set[Seat] = set()
    zones = []
    for rect in rects:
        members = []
        for seat in layout.seats():
            if seat not in claimed and rect.contains(*layout.seat_uv(*seat)):
                members.append(seat)
                claimed.add(seat)
        zones.append(Zone(rect.name, tuple(members), rect.bounds))
    return zones


DEPTH_NAMES = {1: [""], 2: ["front", "back"], 3: ["front", "middle", "back"]}
SIDE_NAMES = {1: [""], 2: ["left", "right"], 3: ["left", "centre", "right"]}


def _band(value: float, parts: int) -> int:
    """Which of `parts` equal bands a 0..1 value falls in; a seat on a border joins the front or left one."""
    return min(max(math.ceil(value * parts) - 1, 0), parts - 1)


def make_zones(
    layout: PlanLayout, zone_rows: int = config.PLAN_ZONE_ROWS, zone_cols: int = config.PLAN_ZONE_COLS,
    drawn: tuple[ZoneRect, ...] = (),
) -> list[Zone]:
    """The hand-drawn zones if there are any, else equal slices of the room."""
    if drawn:
        return drawn_zones(layout, drawn)
    zone_rows, zone_cols = min(zone_rows, layout.rows), min(zone_cols, layout.cols)
    depth = DEPTH_NAMES.get(zone_rows, [f"band {i + 1}" for i in range(zone_rows)])
    side = SIDE_NAMES.get(zone_cols, [f"column {i + 1}" for i in range(zone_cols)])
    members: dict[tuple[int, int], list[Seat]] = {}
    for row in range(layout.rows):
        for col in range(layout.cols):
            u, v = layout.seat_uv(row, col)
            members.setdefault((_band(v, zone_rows), _band(u, zone_cols)), []).append((row, col))
    zones = []
    for i, depth_name in enumerate(depth):
        for j, side_name in enumerate(side):
            name = " ".join(part for part in (depth_name, side_name) if part) or "all seats"
            bounds = (j / zone_cols, i / zone_rows, (j + 1) / zone_cols, (i + 1) / zone_rows)
            zones.append(Zone(name, tuple(members.get((i, j), [])), bounds))
    return zones


Line = tuple[tuple[float, float], tuple[float, float]]


def grid_lines(layout: PlanLayout, calibration: Calibration, frame_size: tuple[int, int]) -> list[Line]:
    """Seat boundaries as pixel lines in the camera image."""
    matrix = plan_to_image(calibration, frame_size)
    cols = ((0.0, 0.0),) + calibration.cols_for(layout) + ((1.0, 1.0),)
    rows = ((0.0, 0.0),) + calibration.rows_for(layout) + ((1.0, 1.0),)
    segments = [((front, 0.0), (back, 1.0)) for front, back in cols]
    segments += [((0.0, left), (1.0, right)) for left, right in rows]
    points = np.float32([point for segment in segments for point in segment]).reshape(-1, 1, 2)
    mapped = cv2.perspectiveTransform(points, matrix).reshape(-1, 2)
    return [((float(a[0]), float(a[1])), (float(b[0]), float(b[1]))) for a, b in zip(mapped[0::2], mapped[1::2])]
