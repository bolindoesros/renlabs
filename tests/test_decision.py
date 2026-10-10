import math

import pytest

from jacket import config
from ren.decision import (
    DecisionMaker, DecisionSettings, VentSettings, ZonePicker,
    aim_at, approach_angle,
)
from ren.plan import PlanLayout, SeatReading, SeatingScene

LAYOUT = PlanLayout(rows=3, cols=6)
# No smoothing or slew, so tests see raw output
INSTANT = DecisionSettings(smoothing_s=0.0, min_dwell_s=0.0)
FAST_VENT = VentSettings(slew_deg_per_s=100000.0)


def scene(**seats) -> SeatingScene:
    """scene(r0c0='hot', r2c5='cold')"""
    parsed = {(int(key[1]), int(key[3])): SeatReading(state, None) for key, state in seats.items()}
    return SeatingScene(LAYOUT, parsed)


def maker(decision=INSTANT, vent=FAST_VENT) -> DecisionMaker:
    return DecisionMaker(LAYOUT, decision, vent)


def test_empty_room_closes_the_vent():
    decision = maker().update(scene(), now=0.0)
    assert decision.closed and decision.target_zone is None and decision.aim is None
    assert decision.command_tilt_deg == config.VENT_CLOSED_TILT_DEG


def test_one_hot_person_opens_the_vent_and_aims_at_their_zone():
    decision = maker().update(scene(r2c5="hot"), now=0.0)
    assert not decision.closed and decision.target_zone == "back right"
    assert decision.aim == pytest.approx((5.5 / 6, 2.5 / 3))


def test_one_person_in_a_jacket_opens_the_vent_too():
    decision = maker().update(scene(r2c5="cold"), now=0.0)
    assert not decision.closed and decision.target_zone == "back right"


def test_clothing_never_changes_where_the_air_goes():
    jackets = maker().update(scene(r0c0="cold", r0c1="cold", r2c5="hot"), now=0.0)
    shirts = maker().update(scene(r0c0="hot", r0c1="hot", r2c5="cold"), now=0.0)
    assert jackets.zone_shares == shirts.zone_shares and jackets.target_zone == shirts.target_zone == "front left"


def test_a_single_unsure_person_opens_the_vent():
    assert not maker().update(scene(r1c1="unsure"), now=0.0).closed


def test_air_goes_where_most_people_are():
    decision = maker().update(scene(r0c0="hot", r2c4="cold", r2c5="cold"), now=0.0)  # 1 shirt vs 2 jackets
    assert decision.target_zone == "back right"
    assert decision.zone_shares["back right"] == pytest.approx(2 / 3)


def test_more_heads_get_more_air():
    decision = maker().update(scene(r0c0="hot", r0c1="hot", r1c0="hot", r2c5="hot"), now=0.0)
    assert decision.zone_shares["front left"] == pytest.approx(0.75)
    assert decision.zone_shares["back right"] == pytest.approx(0.25)


def test_shares_sum_to_one_when_someone_is_there():
    decision = maker().update(scene(r0c0="hot", r1c3="unsure", r2c5="cold"), now=0.0)
    assert sum(decision.zone_shares.values()) == pytest.approx(1.0)


def test_empty_room_has_no_shares():
    assert sum(maker().update(scene(), now=0.0).zone_shares.values()) == 0


@pytest.mark.parametrize("seat,rotation", [("r2c2", 0), ("r1c5", 90), ("r1c0", 270)])
def test_rotation_is_the_compass_direction_from_the_vent(seat, rotation):
    # Vent at centre: back 0, right 90, left 270
    vent = VentSettings(u=(2.5 / 6), v=0.5, slew_deg_per_s=100000.0) if seat == "r2c2" else FAST_VENT
    decision = maker(vent=vent).update(scene(**{seat: "hot"}), now=0.0)
    assert decision.command_rotation_deg == pytest.approx(rotation, abs=1.0)


def test_front_target_points_the_vent_at_180_degrees():
    vent = VentSettings(u=0.5 + 0.5 / 6, v=0.5, slew_deg_per_s=100000.0)  # above column 3, middle row
    decision = maker(vent=vent).update(scene(r0c3="hot"), now=0.0)
    assert decision.command_rotation_deg == pytest.approx(180, abs=1.0)


def test_rotation_offset_realigns_the_vent_zero():
    vent = VentSettings(rotation_offset_deg=90.0, slew_deg_per_s=100000.0)
    decision = maker(vent=vent).update(scene(r1c5="hot"), now=0.0)
    assert decision.command_rotation_deg == pytest.approx(180, abs=1.0)  # was 90, plus the 90 offset


def test_tilt_grows_with_distance_and_matches_geometry():
    near = aim_at((0.5 + 1 / 6, 0.5), LAYOUT, VentSettings(), 0.0)  # one seat width to the right
    far = aim_at((0.5 + 2 / 6, 0.5), LAYOUT, VentSettings(), 0.0)
    assert 0 < near.tilt_deg < far.tilt_deg
    assert near.tilt_deg == pytest.approx(math.degrees(math.atan2(0.55, 2.0)), abs=0.01)


def test_directly_below_the_vent_means_zero_tilt_and_held_rotation():
    aim = aim_at((0.5, 0.5), LAYOUT, VentSettings(), hold_rotation=123.0)
    assert aim.tilt_deg == pytest.approx(0, abs=0.01) and aim.rotation_deg == 123.0


def test_out_of_reach_targets_are_flagged_and_tilt_is_clamped():
    far = aim_at((1.0, 1.0), PlanLayout(3, 6), VentSettings(drop_m=0.5, tilt_max_deg=45), 0.0)
    assert not far.reachable and far.tilt_deg == 45.0


def test_rotation_takes_the_short_way_round():
    assert approach_angle(350, 10, 100) == pytest.approx(10)
    assert approach_angle(10, 350, 15) == pytest.approx(355)
    assert approach_angle(0, 180, 30) == pytest.approx(30) or approach_angle(0, 180, 30) == pytest.approx(330)


def test_vent_cannot_move_faster_than_its_slew_rate():
    slow = VentSettings(slew_deg_per_s=40.0)
    made = DecisionMaker(LAYOUT, INSTANT, slow)
    made.update(scene(r1c5="hot"), now=0.0)
    after = made.update(scene(r1c5="hot"), now=0.5)
    assert after.rotation_deg <= 40.0 * 0.5 + 1e-6 or after.rotation_deg >= 360 - 40.0 * 0.5 - 1e-6
    assert after.tilt_deg == config.VENT_CLOSED_TILT_DEG  # a big turn starts with the flaps shut


def run_to_settled(made: DecisionMaker, seats: dict, start: float = 0.0, step: float = 0.1, limit: int = 600):
    """Update until aiming; returns every decision."""
    history = []
    for tick in range(limit):
        decision = made.update(scene(**seats), start + tick * step)
        history.append(decision)
        if decision.phase == "aiming":
            break
    return history


def test_a_big_turn_shuts_the_flaps_turns_then_reopens():
    made = DecisionMaker(LAYOUT, INSTANT, VentSettings(slew_deg_per_s=60.0))
    history = run_to_settled(made, {"r1c5": "hot"})
    phases = [d.phase for d in history]
    assert phases[-1] == "aiming" and "turning" in phases and "opening" in phases
    assert phases.index("turning") < phases.index("opening")


def test_the_flaps_stay_shut_whenever_the_vent_is_turning():
    made = DecisionMaker(LAYOUT, INSTANT, VentSettings(slew_deg_per_s=60.0))
    history = run_to_settled(made, {"r1c5": "hot"})
    turning = [d for d in history if d.phase == "turning"]
    rotated = [d for d, nxt in zip(history, history[1:]) if abs(nxt.rotation_deg - d.rotation_deg) > 0.5 and nxt.phase == "turning"]
    assert rotated and all(d.tilt_deg == config.VENT_CLOSED_TILT_DEG for d in rotated)
    assert turning


def test_rotation_never_moves_while_the_flaps_are_open_during_a_big_turn():
    made = DecisionMaker(LAYOUT, INSTANT, VentSettings(slew_deg_per_s=60.0))
    run_to_settled(made, {"r1c5": "hot"})  # settle on the right
    previous = None
    for tick in range(1, 200):  # now ask for the opposite side
        decision = made.update(scene(r1c0="hot"), 100.0 + tick * 0.1)
        if previous and decision.tilt_deg < config.VENT_CLOSED_TILT_DEG - 1:
            assert abs(angle_difference_for_test(previous.rotation_deg, decision.rotation_deg)) < 25
        previous = decision


def angle_difference_for_test(a: float, b: float) -> float:
    return (b - a + 180) % 360 - 180


def test_small_target_changes_do_not_shut_the_vent():
    made = DecisionMaker(LAYOUT, INSTANT, VentSettings(slew_deg_per_s=60.0))
    run_to_settled(made, {"r1c5": "hot"})
    nudged = made.update(scene(r1c4="hot"), now=200.0)  # same direction, a seat nearer
    assert nudged.phase in ("aiming", "opening") and nudged.tilt_deg < config.VENT_CLOSED_TILT_DEG


def test_phase_is_shut_with_nobody_there_and_closing_on_the_way():
    made = DecisionMaker(LAYOUT, INSTANT, VentSettings(slew_deg_per_s=20.0))  # slow enough to see it close
    assert made.update(scene(), 0.0).phase == "shut"
    run_to_settled(made, {"r1c5": "hot"}, start=1.0)
    assert made.update(scene(), 100.0).phase == "closing"
    for tick in range(100):
        last = made.update(scene(), 101.0 + tick * 0.1)
    assert last.phase == "shut"


def test_dwell_counts_from_arrival_not_from_the_decision_to_move():
    settings = DecisionSettings(aim_mode="sweep", smoothing_s=0.0, min_dwell_s=6.0, min_share=0.05)
    made = DecisionMaker(LAYOUT, settings, VentSettings(slew_deg_per_s=30.0))  # slow: long travel
    seats = {"r0c0": "hot", "r0c5": "hot"}
    arrivals, switches, last_zone, last_phase = [], [], None, None
    for tick in range(4000):
        now = tick * 0.1
        decision = made.update(scene(**seats), now)
        if decision.target_zone != last_zone:
            switches.append(now)
            last_zone = decision.target_zone
        if decision.phase == "aiming" and last_phase != "aiming":
            arrivals.append(now)
        last_phase = decision.phase
    assert len(switches) >= 3 and arrivals
    # Each stay lasts min_dwell after arrival
    for arrival, next_switch in zip(arrivals, switches[2:]):
        assert next_switch - arrival >= 6.0 - 0.2


def test_demand_fades_instead_of_vanishing():
    smooth = DecisionSettings(smoothing_s=1.5, min_dwell_s=0.0)
    made = maker(decision=smooth)
    made.update(scene(r0c0="hot", r0c1="hot", r0c2="hot"), now=0.0)
    made.update(scene(), now=0.0 + 0.1)
    after_blip = made.update(scene(), now=0.2)
    assert not after_blip.closed  # still remembers them a moment later
    for step in range(1, 40):
        last = made.update(scene(), now=0.2 + step * 0.5)
    assert last.closed


def test_people_outside_the_plan_are_reported():
    outside = SeatingScene(LAYOUT, {}, unseated=(0, 2))
    assert maker().update(outside, now=0.0).off_plan == 2


def test_wrong_layout_fails_loudly():
    with pytest.raises(ValueError):
        maker().update(SeatingScene(PlanLayout(2, 2), {}), now=0.0)


def run_picker(mode: str, shares: dict[str, float], seconds: int = 300) -> dict[str, float]:
    picker = ZonePicker(DecisionSettings(aim_mode=mode, min_dwell_s=5.0, min_share=0.05))
    time_in = {zone: 0.0 for zone in shares}
    for tick in range(seconds * 10):
        zone = picker.pick(shares, tick * 0.1)
        if zone:
            time_in[zone] += 0.1
    return {zone: spent / seconds for zone, spent in time_in.items()}


def test_sweep_shares_time_in_proportion_to_demand():
    time_share = run_picker("sweep", {"a": 0.7, "b": 0.3})
    assert time_share["a"] == pytest.approx(0.7, abs=0.1) and time_share["b"] == pytest.approx(0.3, abs=0.1)


def test_sweep_skips_zones_with_negligible_demand():
    time_share = run_picker("sweep", {"a": 0.97, "b": 0.03})
    assert time_share["b"] == 0


def test_focus_stays_on_the_top_zone():
    time_share = run_picker("focus", {"a": 0.7, "b": 0.3})
    assert time_share["a"] > 0.95


def test_focus_ignores_small_wobbles_but_follows_big_changes():
    picker = ZonePicker(DecisionSettings(aim_mode="focus", min_dwell_s=0.0, focus_margin=0.2, min_share=0.0))
    assert picker.pick({"a": 0.55, "b": 0.45}, 0.0) == "a"
    assert picker.pick({"a": 0.48, "b": 0.52}, 1.0) == "a"  # within the margin
    assert picker.pick({"a": 0.2, "b": 0.8}, 2.0) == "b"


def test_no_demand_means_no_zone():
    assert ZonePicker(DecisionSettings()).pick({"a": 0.0, "b": 0.0}, 0.0) is None


def test_mirroring_flips_the_rotation_so_the_real_vent_points_the_right_way():
    seat_right = scene(r1c5="hot")
    straight = DecisionMaker(LAYOUT, INSTANT, FAST_VENT).update(seat_right, 0.0)
    mirrored = DecisionMaker(LAYOUT, INSTANT, FAST_VENT, mirrored=True).update(seat_right, 0.0)
    assert straight.command_rotation_deg == pytest.approx(90, abs=1.0)
    assert mirrored.command_rotation_deg == pytest.approx(270, abs=1.0)  # the plan's right is the room's left


def test_anticlockwise_hardware_flips_the_rotation_sense():
    vent = VentSettings(clockwise=False, slew_deg_per_s=100000.0)
    decision = DecisionMaker(LAYOUT, INSTANT, vent).update(scene(r1c5="hot"), 0.0)
    assert decision.command_rotation_deg == pytest.approx(270, abs=1.0)


def test_mirrored_and_anticlockwise_cancel_out():
    vent = VentSettings(clockwise=False, slew_deg_per_s=100000.0)
    decision = DecisionMaker(LAYOUT, INSTANT, vent, mirrored=True).update(scene(r1c5="hot"), 0.0)
    assert decision.command_rotation_deg == pytest.approx(90, abs=1.0)


def test_the_plan_azimuth_is_independent_of_hardware_conventions():
    from ren.decision import to_plan_azimuth, to_rotation
    for mirrored in (False, True):
        for clockwise in (True, False):
            for offset in (0.0, 35.0, -90.0):
                vent = VentSettings(clockwise=clockwise, rotation_offset_deg=offset)
                for azimuth in (0.0, 45.0, 90.0, 200.0, 359.0):
                    back = to_plan_azimuth(to_rotation(azimuth, vent, mirrored), vent, mirrored)
                    assert back == pytest.approx(azimuth, abs=1e-6)


def test_the_decision_reports_the_plan_azimuth_for_drawing():
    made = DecisionMaker(LAYOUT, INSTANT, FAST_VENT, mirrored=True)
    made.update(scene(r1c5="hot"), 0.0)
    decision = made.update(scene(r1c5="hot"), 1.0)
    assert decision.plan_azimuth_deg == pytest.approx(90, abs=1.0)  # drawn to the right on the plan
    assert decision.rotation_deg == pytest.approx(270, abs=1.0)  # what the hardware is told
