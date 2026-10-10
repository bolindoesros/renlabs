import numpy as np
import pytest

from jacket.types import Box, ClothingResult
from ren.plan import (
    Calibration, PlanLayout, SeatingScene, anchor_of, image_to_plan, make_zones, plan_to_image, read_scene,
)

FRAME = (1000, 800)
LAYOUT = PlanLayout(rows=3, cols=6)
# Upright rectangle, x 100-900, y 100-700
SQUARE = Calibration((0.1, 0.125), (0.9, 0.125), (0.9, 0.875), (0.1, 0.875))
LIGHT, WARM, UNKNOWN = (ClothingResult(label, p, "") for label, p in (("light", 0.1), ("warm", 0.9), ("unknown", None)))


def person(x: float, y: float, width: int = 60) -> Box:
    """A box whose shoulder anchor sits at (x, y)."""
    return Box(int(x - width / 2), int(y - 40), int(x + width / 2), int(y + 160), 0.9, "person", torso_top_y=int(y))


def seat_pixel(row: int, col: int) -> tuple[float, float]:
    """Pixel centre of a seat."""
    return 100 + (col + 0.5) / 6 * 800, 700 - (row + 0.5) / 3 * 600


def test_calibration_default_is_valid_and_degenerate_is_not():
    assert Calibration().is_valid()
    assert not Calibration((0.5, 0.5), (0.5, 0.5), (0.5, 0.5), (0.5, 0.5)).is_valid()
    assert not Calibration((0.1, 0.1), (0.9, 0.9), (0.9, 0.1), (0.1, 0.9)).is_valid()  # bow-tie


def test_matrix_maps_corners_to_plan_corners():
    matrix = image_to_plan(SQUARE, FRAME)
    corners = np.float32([[100, 100], [900, 100], [900, 700], [100, 700]]).reshape(-1, 1, 2)
    import cv2
    mapped = cv2.perspectiveTransform(corners, matrix).reshape(-1, 2)
    assert np.allclose(mapped, [[0, 1], [1, 1], [1, 0], [0, 0]], atol=1e-4)


def test_inverse_matrix_round_trips():
    import cv2
    point = np.float32([[[400, 300]]])
    plan = cv2.perspectiveTransform(point, image_to_plan(SQUARE, FRAME))
    back = cv2.perspectiveTransform(plan, plan_to_image(SQUARE, FRAME))
    assert np.allclose(back, point, atol=1e-2)


def test_invalid_calibration_raises():
    with pytest.raises(ValueError):
        image_to_plan(Calibration((0.5, 0.5), (0.5, 0.5), (0.5, 0.5), (0.5, 0.5)), FRAME)


def test_anchor_uses_shoulder_line_else_a_fraction_of_the_box():
    assert anchor_of(Box(100, 100, 200, 300, 0.9, "person", torso_top_y=150)) == (150.0, 150.0)
    x, y = anchor_of(Box(100, 100, 200, 300, 0.9, "person"))
    assert x == 150.0 and 150 < y < 180


@pytest.mark.parametrize("row,col", [(0, 0), (0, 5), (2, 0), (2, 5), (1, 3)])
def test_person_lands_in_the_seat_under_them(row, col):
    scene = read_scene([person(*seat_pixel(row, col))], [LIGHT], FRAME, LAYOUT, SQUARE)
    assert list(scene.seats) == [(row, col)] and scene.off_plan == 0


def test_front_is_row_zero_and_left_is_column_zero():
    scene = read_scene([person(150, 680)], [LIGHT], FRAME, LAYOUT, SQUARE)  # bottom-left of the image
    assert list(scene.seats) == [(0, 0)]


def test_states_follow_clothing_labels():
    boxes = [person(*seat_pixel(0, 0)), person(*seat_pixel(0, 2)), person(*seat_pixel(0, 4))]
    scene = read_scene(boxes, [LIGHT, WARM, UNKNOWN], FRAME, LAYOUT, SQUARE)
    assert [scene.seats[(0, c)].state for c in (0, 2, 4)] == ["hot", "cold", "unsure"]


def test_without_classification_everyone_is_unsure():
    scene = read_scene([person(*seat_pixel(1, 1))], [], FRAME, LAYOUT, SQUARE)
    assert scene.seats[(1, 1)].state == "unsure" and scene.seats[(1, 1)].warm_prob is None


def test_two_people_never_share_a_seat():
    x, y = seat_pixel(1, 2)
    scene = read_scene([person(x - 10, y), person(x + 10, y)], [LIGHT, WARM], FRAME, LAYOUT, SQUARE)
    assert len(scene.seats) == 2  # the second takes the next nearest free seat


def test_people_outside_the_plan_are_counted_not_seated():
    scene = read_scene([person(*seat_pixel(0, 0)), person(20, 20)], [LIGHT, LIGHT], FRAME, LAYOUT, SQUARE)
    assert len(scene.seats) == 1 and scene.off_plan == 1
    assert scene.unseated == (1,)  # the second box, marked by the camera view


def test_no_people_gives_an_empty_scene():
    scene = read_scene([], [], FRAME, LAYOUT, SQUARE)
    assert scene.seats == {} and scene.unseated == () and scene.off_plan == 0


def test_zones_cover_every_seat_exactly_once():
    zones = make_zones(LAYOUT)
    covered = [seat for zone in zones for seat in zone.seats()]
    assert sorted(covered) == sorted((r, c) for r in range(3) for c in range(6))
    assert len(covered) == len(set(covered))


def test_zone_names_read_front_to_back_left_to_right():
    names = [zone.name for zone in make_zones(LAYOUT, 2, 3)]
    assert names == ["front left", "front centre", "front right", "back left", "back centre", "back right"]


def test_one_by_one_zone_is_the_whole_room():
    zones = make_zones(PlanLayout(2, 2), 1, 1)
    assert [z.name for z in zones] == ["all seats"] and len(zones[0].seats()) == 4


def test_zone_counts_never_exceed_the_layout():
    assert len(make_zones(PlanLayout(1, 2), 2, 3)) == 2


def test_grid_lines_have_one_more_line_than_seats_each_way():
    from ren.plan import grid_lines
    lines = grid_lines(LAYOUT, SQUARE, FRAME)
    assert len(lines) == (6 + 1) + (3 + 1)


def test_grid_lines_run_between_the_calibration_corners():
    from ren.plan import grid_lines
    lines = grid_lines(LAYOUT, SQUARE, FRAME)
    first_vertical, last_horizontal = lines[0], lines[-1]
    # Line 0 is the left edge, front to back
    assert first_vertical[0] == pytest.approx((100, 700), abs=0.1) and first_vertical[1] == pytest.approx((100, 100), abs=0.1)
    assert last_horizontal[0] == pytest.approx((100, 100), abs=0.1) and last_horizontal[1] == pytest.approx((900, 100), abs=0.1)
