import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication

from jacket import config
from ren.decision import DecisionMaker, DecisionSettings, VentSettings
from ren.engine import ViewSettings
from ren.plan import PlanLayout, SeatReading, SeatingScene, make_zones
from ren.seating_view import SeatingPlanView
from ren.theme import color, load_fonts, state_color

LAYOUT = PlanLayout(rows=3, cols=6)
FAST = VentSettings(slew_deg_per_s=100000.0)
INSTANT = DecisionSettings(smoothing_s=0.0, min_dwell_s=0.0)


@pytest.fixture(scope="module")
def fonts():
    QApplication.instance() or QApplication([])
    return load_fonts()


def build(fonts, seats: dict, vent=FAST, now: float = 0.0):
    scene = SeatingScene(LAYOUT, {key: SeatReading(state, None) for key, state in seats.items()})
    maker = DecisionMaker(LAYOUT, INSTANT, vent)
    maker.update(scene, now)  # first update has no elapsed time
    decision = maker.update(scene, now + 0.5)
    view = SeatingPlanView(fonts)
    view.resize(720, 480)
    view.set_state(LAYOUT, scene, decision, make_zones(LAYOUT), vent)
    return view, decision


def rgb(image, point):
    pixel = image.pixelColor(int(point.x()), int(point.y()))
    return pixel.red(), pixel.green(), pixel.blue()


def centre_of(view, row, col):
    return view.cell_rect(row, col).center()


def test_seat_colours_follow_state(fonts):
    view, _ = build(fonts, {(0, 0): "hot", (0, 1): "cold", (0, 5): "unsure"})
    image = view.grab().toImage()
    for (row, col), state in {(0, 0): "hot", (0, 1): "cold", (0, 5): "unsure"}.items():
        expected = state_color(state)
        assert rgb(image, centre_of(view, row, col)) == (expected.red(), expected.green(), expected.blue())


def test_empty_seats_are_small_grey_dots(fonts):
    from ren.seating_view import empty_seat_color
    view, _ = build(fonts, {(0, 0): "hot"})
    image = view.grab().toImage()
    centre = centre_of(view, 2, 5)
    dot = empty_seat_color()
    assert rgb(image, centre) == (dot.red(), dot.green(), dot.blue())
    beside = type(centre)(centre.x() - view.cell_rect(2, 5).width() * 0.3, centre.y())
    background = color("background")
    assert rgb(image, beside) == (background.red(), background.green(), background.blue())


def test_the_front_row_is_drawn_at_the_bottom_and_left_stays_left(fonts):
    view, _ = build(fonts, {})
    assert view.cell_rect(0, 0).center().y() > view.cell_rect(2, 0).center().y()
    assert view.cell_rect(0, 0).center().x() < view.cell_rect(0, 5).center().x()


def test_cells_are_square_and_fit_inside_the_widget(fonts):
    view, _ = build(fonts, {})
    cell = view.cell_rect(1, 1)
    assert cell.width() == pytest.approx(cell.height())
    assert view.rect().contains(view.plan_rect().toRect())


def test_an_open_vent_paints_a_beam_toward_the_aimed_seat(fonts):
    # One hot person back right; beam should point there
    view, decision = build(fonts, {(2, 5): "hot"})
    assert not decision.closed
    image = view.grab().toImage()
    towards = view.point_at(0.5 + 0.18, 0.5 + 0.12)  # partway from the vent to the back right
    away = view.point_at(0.5 - 0.30, 0.5 - 0.30)
    background = color("background")
    assert rgb(image, towards) != (background.red(), background.green(), background.blue())
    assert rgb(image, away) == (background.red(), background.green(), background.blue())


def test_a_closed_vent_paints_no_beam(fonts):
    view, decision = build(fonts, {})
    assert decision.closed
    image = view.grab().toImage()
    background = color("background")
    sample = view.point_at(0.5 + 0.2, 0.5 + 0.15)
    assert rgb(image, sample) == (background.red(), background.green(), background.blue())


def test_the_targeted_zone_is_tinted_with_airflow_green(fonts):
    view, decision = build(fonts, {(2, 5): "hot"})
    zones = make_zones(LAYOUT)
    target = next(z for z in zones if z.name == decision.target_zone)
    other = next(z for z in zones if z.name != decision.target_zone)
    image = view.grab().toImage()

    def inner_corner(zone):
        rect = view._zone_rect(zone)
        return type(rect.bottomRight())(rect.right() - 12, rect.bottom() - 12)  # clear of seats and labels

    tinted, plain = rgb(image, inner_corner(target)), rgb(image, inner_corner(other))
    background = color("background")
    assert plain == (background.red(), background.green(), background.blue())
    assert tinted != plain and tinted[1] >= tinted[0] and tinted[1] >= tinted[2]  # a green tint


def test_it_paints_every_layout_without_error(fonts):
    for rows, cols in [(1, 1), (2, 3), (4, 8), (12, 16)]:
        layout = PlanLayout(rows, cols)
        view = SeatingPlanView(fonts)
        view.resize(640, 420)
        scene = SeatingScene(layout, {(0, 0): SeatReading("hot", None)})
        decision = DecisionMaker(layout, INSTANT, FAST).update(scene, 0.0)
        view.set_state(layout, scene, decision, make_zones(layout), FAST)
        assert not view.grab().isNull()


def test_it_paints_before_any_data_arrives(fonts):
    view = SeatingPlanView(fonts)
    view.resize(640, 420)
    assert not view.grab().isNull()


def with_tilt(view_decision, tilt: float):
    import dataclasses
    return dataclasses.replace(view_decision, tilt_deg=tilt)


def test_closing_flaps_fade_the_beam(fonts):
    view, decision = build(fonts, {(2, 5): "hot"})
    scene = SeatingScene(LAYOUT, {(2, 5): SeatReading("hot", None)})
    beam_point = view.point_at(0.5 + 0.18, 0.5 + 0.12)
    background = color("background")
    plain = (background.red(), background.green(), background.blue())

    def paint(tilt: float):
        view.set_state(LAYOUT, scene, with_tilt(decision, tilt), make_zones(LAYOUT), FAST)
        return view.grab().toImage()

    strength = lambda image: sum(abs(a - b) for a, b in zip(rgb(image, beam_point), plain))
    assert strength(paint(30.0)) > strength(paint(70.0)) > 0
    assert rgb(paint(88.0), beam_point) == plain  # nearly shut: nothing left


def test_fade_is_full_in_the_working_range_then_falls_to_zero(fonts):
    view = SeatingPlanView(fonts)
    view.set_state(LAYOUT, None, None, [], VentSettings(tilt_max_deg=45.0))
    assert view._fade(0.0) == 1.0 and view._fade(45.0) == 1.0
    assert 0.0 < view._fade(65.0) < 1.0
    assert view._fade(85.0) == pytest.approx(0.0) and view._fade(90.0) == 0.0
    assert view._fade(55.0) > view._fade(75.0)


def painted(view, layers: ViewSettings, decision, scene):
    view.set_state(LAYOUT, scene, decision, make_zones(LAYOUT), FAST, layers)
    return view.grab().toImage()


def test_each_layer_can_be_hidden_on_its_own(fonts):
    view, decision = build(fonts, {(2, 5): "hot"})
    scene = SeatingScene(LAYOUT, {(2, 5): SeatReading("hot", None)})
    background = color("background")
    plain = (background.red(), background.green(), background.blue())
    seat_point = centre_of(view, 2, 5)
    beam_point = view.point_at(0.5 + 0.18, 0.5 + 0.12)
    glyph_point = view.point_at(0.5, 0.5)

    everything = painted(view, ViewSettings(), decision, scene)
    assert rgb(everything, seat_point) != plain and rgb(everything, beam_point) != plain

    no_seats = painted(view, ViewSettings(show_seats=False, show_beam=False, show_zones=False), decision, scene)  # beam and tint cover it
    assert rgb(no_seats, seat_point) == plain

    no_beam = painted(view, ViewSettings(show_beam=False), decision, scene)
    assert rgb(no_beam, beam_point) == plain and rgb(no_beam, seat_point) != plain

    no_vent = painted(view, ViewSettings(show_vent=False), decision, scene)
    around = [type(glyph_point)(glyph_point.x() + dx, glyph_point.y() + dy) for dx in range(-22, 23, 4) for dy in range(-22, 23, 4)]
    assert any(rgb(everything, point) != rgb(no_vent, point) for point in around)  # the glyph is drawn only when shown

    zones_only = painted(view, ViewSettings(show_beam=False), decision, scene)  # the beam lands near this edge
    no_zones = painted(view, ViewSettings(show_zones=False, show_beam=False), decision, scene)
    zone = next(z for z in make_zones(LAYOUT) if z.name == decision.target_zone)
    edge = view._zone_rect(zone).adjusted(2, 2, -2, -2)
    top_middle = edge.topLeft() + type(edge.topLeft())(edge.width() / 2, 0)
    assert rgb(zones_only, top_middle) != plain and rgb(no_zones, top_middle) == plain


def test_zone_share_text_never_covers_a_seat(fonts):
    from ren.seating_view import SEAT_DOT
    view, decision = build(fonts, {(2, 0): "hot"})
    assert decision.zone_shares["back left"] == pytest.approx(1.0)  # so a 100% label sits top-left
    image = view.grab().toImage()
    cell = view.cell_rect(2, 0)
    radius = cell.width() * SEAT_DOT / 2
    expected = state_color("hot")
    top = int(cell.center().y() - radius + 4)  # just inside the dot's top
    for x in range(int(cell.center().x() - 6), int(cell.center().x() + 7)):
        pixel = image.pixelColor(x, top)
        assert (pixel.red(), pixel.green(), pixel.blue()) == (expected.red(), expected.green(), expected.blue()), x


def test_zone_share_text_is_drawn_on_the_zone_edge(fonts):
    view, _ = build(fonts, {(2, 0): "hot"})
    plain = SeatingPlanView(fonts)
    plain.resize(720, 480)
    plain.set_state(LAYOUT, None, None, make_zones(LAYOUT), FAST)  # no decision: no shares
    zone = next(z for z in make_zones(LAYOUT) if z.name == "back left")
    edge = view._zone_rect(zone).adjusted(2, 2, -2, -2)
    box = (int(edge.left() + 8), int(edge.top() - 10), int(edge.left() + 60), int(edge.top() + 10))
    with_text, without = view.grab().toImage(), plain.grab().toImage()
    changed = [(x, y) for x in range(box[0], box[2]) for y in range(box[1], box[3]) if with_text.pixelColor(x, y) != without.pixelColor(x, y)]
    assert len(changed) > 40  # glyph pixels appeared around the edge

