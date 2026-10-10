"""From seats to a vent command."""
import math
from dataclasses import dataclass, field

from jacket import config
from ren.plan import PlanLayout, SeatingScene, SeatState, Zone, make_zones

MIN_AIM_DISTANCE_M = 0.05  # closer than this, the vent points straight down
MAX_STEP_S = 1.0  # longer gaps between updates count as this


@dataclass(frozen=True)
class NeedWeights:
    """How much a seat wants air, per state."""

    hot: float = config.NEED_HOT
    unsure: float = config.NEED_UNSURE
    cold: float = config.NEED_COLD

    def of(self, state: SeatState) -> float:
        return getattr(self, state)


@dataclass(frozen=True)
class DecisionSettings:
    weights: NeedWeights = field(default_factory=NeedWeights)
    aim_mode: str = config.DECISION_AIM_MODE  # "sweep" or "focus"
    smoothing_s: float = config.DECISION_SMOOTHING_S
    min_share: float = config.DECISION_MIN_SHARE
    min_dwell_s: float = config.DECISION_MIN_DWELL_S
    close_below: float = config.DECISION_CLOSE_BELOW
    focus_margin: float = config.DECISION_FOCUS_MARGIN


@dataclass(frozen=True)
class VentSettings:
    """Where the vent hangs and how it moves."""

    u: float = config.VENT_U
    v: float = config.VENT_V
    drop_m: float = config.VENT_DROP_M
    tilt_max_deg: float = config.VENT_TILT_MAX_DEG
    rotation_offset_deg: float = config.VENT_ROTATION_OFFSET_DEG
    clockwise: bool = config.VENT_CLOCKWISE
    slew_deg_per_s: float = config.VENT_SLEW_DEG_PER_S
    seat_width_m: float = config.SEAT_WIDTH_M
    row_depth_m: float = config.ROW_DEPTH_M


@dataclass(frozen=True)
class Aim:
    rotation_deg: float  # airflow compass direction, 0 is back
    tilt_deg: float  # flap tilt from straight down
    reachable: bool


@dataclass(frozen=True)
class Decision:
    phase: str  # shut, closing, turning, opening or aiming
    closed: bool
    rotation_deg: float  # where the vent is now
    plan_azimuth_deg: float  # the same direction as seen on the plan
    tilt_deg: float
    command_rotation_deg: float  # where it was told to go
    command_tilt_deg: float
    target_zone: str | None
    aim: tuple[float, float] | None  # plan units (u, v)
    reachable: bool
    zone_shares: dict[str, float]  # demand split across zones
    off_plan: int  # people seen outside the seating area


def seat_center(row: int, col: int, layout: PlanLayout) -> tuple[float, float]:
    return (col + 0.5) / layout.cols, (row + 0.5) / layout.rows


def to_meters(u: float, v: float, layout: PlanLayout, vent: VentSettings) -> tuple[float, float]:
    """Plan units to metres."""
    return (u - 0.5) * layout.cols * vent.seat_width_m, v * layout.rows * vent.row_depth_m


def to_rotation(plan_azimuth: float, vent: VentSettings, mirrored: bool) -> float:
    """Plan compass direction to the vent's own rotation reading."""
    flip = mirrored != (not vent.clockwise)  # a mirrored view or turning anticlockwise flips the sense
    return ((-plan_azimuth if flip else plan_azimuth) + vent.rotation_offset_deg) % 360


def to_plan_azimuth(rotation: float, vent: VentSettings, mirrored: bool) -> float:
    """Vent rotation back to a plan compass direction."""
    flip = mirrored != (not vent.clockwise)
    angle = rotation - vent.rotation_offset_deg
    return (-angle if flip else angle) % 360


def aim_at(
    target: tuple[float, float], layout: PlanLayout, vent: VentSettings, hold_rotation: float, mirrored: bool = False
) -> Aim:
    """Rotation and tilt that throw air onto `target`."""
    target_x, target_y = to_meters(*target, layout, vent)
    vent_x, vent_y = to_meters(vent.u, vent.v, layout, vent)
    dx, dy = target_x - vent_x, target_y - vent_y
    distance = math.hypot(dx, dy)
    wanted_tilt = math.degrees(math.atan2(distance, vent.drop_m))
    if distance < MIN_AIM_DISTANCE_M:
        rotation = hold_rotation
    else:
        rotation = to_rotation(math.degrees(math.atan2(dx, dy)), vent, mirrored)
    return Aim(rotation, min(wanted_tilt, vent.tilt_max_deg), wanted_tilt <= vent.tilt_max_deg)


def approach(current: float, target: float, max_step: float) -> float:
    return current + max(-max_step, min(max_step, target - current))


def angle_difference(current: float, target: float) -> float:
    """Signed shortest turn from current to target, in degrees."""
    return (target - current + 180) % 360 - 180


def approach_angle(current: float, target: float, max_step: float) -> float:
    """Move along the shortest way round the circle."""
    difference = angle_difference(current, target)
    return (current + max(-max_step, min(max_step, difference))) % 360


class ZonePicker:
    """Chooses which zone the vent serves now."""

    def __init__(self, settings: DecisionSettings) -> None:
        self._settings = settings
        self.reset()

    def reset(self) -> None:
        self._current: str | None = None
        self._since = 0.0
        self._last: float | None = None
        self._credit: dict[str, float] = {}

    def pick(self, shares: dict[str, float], now: float, settled: bool = True) -> str | None:
        """Pick a zone; dwell counts from arrival."""
        if not settled:
            self._since = now
        active = {zone: share for zone, share in shares.items() if share >= self._settings.min_share}
        if not active:
            self.reset()
            return None
        total = sum(active.values())
        active = {zone: share / total for zone, share in active.items()}
        dt = 0.0 if self._last is None else min(now - self._last, MAX_STEP_S)
        self._last = now

        if self._current not in active:
            self._switch(max(active, key=active.get), now)
            self._credit = {zone: 0.0 for zone in active}
            return self._current
        if self._settings.aim_mode == "focus":
            best = max(active, key=active.get)
            clearly_better = active[best] > active[self._current] * (1 + self._settings.focus_margin)
            if clearly_better and now - self._since >= self._settings.min_dwell_s:
                self._switch(best, now)
            return self._current

        # Sweep: zones earn time by share; dwelling spends it
        self._credit = {zone: self._credit.get(zone, 0.0) + active[zone] * dt for zone in active}
        self._credit[self._current] -= dt
        best = max(active, key=lambda zone: (self._credit[zone], active[zone]))
        if best != self._current and now - self._since >= self._settings.min_dwell_s:
            if self._credit[best] > self._credit[self._current]:
                self._switch(best, now)
        return self._current

    def _switch(self, zone: str, now: float) -> None:
        self._current, self._since = zone, now


class DecisionMaker:
    """Turns each seating scene into a vent command."""

    def __init__(
        self,
        layout: PlanLayout,
        decision: DecisionSettings = DecisionSettings(),
        vent: VentSettings = VentSettings(),
        zones: list[Zone] | None = None,
        mirrored: bool = False,
    ) -> None:
        self._layout, self._decision, self._vent = layout, decision, vent
        self._mirrored = mirrored
        self._zones = zones if zones is not None else make_zones(layout)
        self._picker = ZonePicker(decision)
        self._need = {(r, c): 0.0 for r in range(layout.rows) for c in range(layout.cols)}
        self._wanting = dict(self._need)  # need from seats that are not cold
        self._rotation = 0.0
        self._tilt = config.VENT_CLOSED_TILT_DEG  # starts shut
        self._last: float | None = None
        self._turning = False
        self._settled = True

    def update(self, scene: SeatingScene, now: float) -> Decision:
        if scene.layout != self._layout:
            raise ValueError("scene layout does not match this decision maker")
        dt = 0.0 if self._last is None else max(0.0, min(now - self._last, MAX_STEP_S))
        alpha = 1.0 if self._last is None or self._decision.smoothing_s <= 0 else (
            1 - math.exp(-dt / self._decision.smoothing_s)
        )
        self._last = now
        weights = self._decision.weights
        for seat in self._need:
            reading = scene.seats.get(seat)
            wanted = weights.of(reading.state) if reading else 0.0
            from_warm_bodies = wanted if reading and reading.state != "cold" else 0.0
            self._need[seat] += alpha * (wanted - self._need[seat])
            self._wanting[seat] += alpha * (from_warm_bodies - self._wanting[seat])

        demand = {zone.name: sum(self._need[seat] for seat in zone.seats()) for zone in self._zones}
        total = sum(demand.values())
        shares = {name: (value / total if total > 0 else 0.0) for name, value in demand.items()}
        closed = sum(self._wanting.values()) < self._decision.close_below  # hooded people alone never open it

        zone_name = None if closed else self._picker.pick(shares, now, self._settled)
        if closed:
            self._picker.reset()
        aim_point = self._aim_point(zone_name) if zone_name else None
        aim = aim_at(aim_point, self._layout, self._vent, self._rotation, self._mirrored) if aim_point else None
        command_rotation = aim.rotation_deg if aim else self._rotation
        command_tilt = aim.tilt_deg if aim else config.VENT_CLOSED_TILT_DEG

        phase = self._move(command_rotation, command_tilt, closed, self._vent.slew_deg_per_s * dt)
        self._settled = phase == "aiming"
        return self._decision_for(scene, phase, closed, zone_name, aim_point, aim, shares, command_rotation, command_tilt)

    def _move(self, command_rotation: float, command_tilt: float, closed: bool, budget: float) -> str:
        """One motor: shut, turn, then reopen."""
        shut = config.VENT_CLOSED_TILT_DEG
        if closed:
            self._turning = False
            self._tilt = approach(self._tilt, shut, budget)
            return "shut" if self._tilt >= shut else "closing"
        if not self._turning and abs(angle_difference(self._rotation, command_rotation)) > config.VENT_TURN_START_DEG:
            self._turning = True
        if self._turning:
            budget = self._turn(command_rotation, budget)
            if self._turning:
                return "turning"
        # Small corrections need no shutting; leftover reopens flaps
        self._rotation = approach_angle(self._rotation, command_rotation, budget)
        self._tilt = approach(self._tilt, command_tilt, budget)
        arrived = abs(self._tilt - command_tilt) <= config.VENT_SETTLED_TILT_DEG
        return "aiming" if arrived else "opening"

    def _turn(self, command_rotation: float, budget: float) -> float:
        """Close the flaps, then rotate; returns leftover budget."""
        shut = config.VENT_CLOSED_TILT_DEG
        tilt_before = self._tilt
        self._tilt = approach(self._tilt, shut, budget)
        budget -= abs(self._tilt - tilt_before)
        if self._tilt >= shut:
            rotation_before = self._rotation
            self._rotation = approach_angle(self._rotation, command_rotation, budget)
            budget -= abs(angle_difference(rotation_before, self._rotation))
            if abs(angle_difference(self._rotation, command_rotation)) <= config.VENT_TURN_DONE_DEG:
                self._turning = False
        return budget

    def _aim_point(self, zone_name: str) -> tuple[float, float]:
        """Demand-weighted middle of the zone, else its geometric middle."""
        zone = next(z for z in self._zones if z.name == zone_name)
        seats = zone.seats()
        weights = [self._need[seat] for seat in seats]
        if sum(weights) <= 1e-9:
            weights = [1.0] * len(seats)
        total = sum(weights)
        centers = [seat_center(row, col, self._layout) for row, col in seats]
        return (
            sum(w * c[0] for w, c in zip(weights, centers)) / total,
            sum(w * c[1] for w, c in zip(weights, centers)) / total,
        )

    def _decision_for(self, scene, phase, closed, zone_name, aim_point, aim, shares, command_rotation, command_tilt) -> Decision:
        return Decision(
            phase=phase,
            closed=closed,
            rotation_deg=self._rotation,
            plan_azimuth_deg=to_plan_azimuth(self._rotation, self._vent, self._mirrored),
            tilt_deg=self._tilt,
            command_rotation_deg=command_rotation,
            command_tilt_deg=command_tilt,
            target_zone=zone_name,
            aim=aim_point,
            reachable=aim.reachable if aim else True,
            zone_shares=shares,
            off_plan=scene.off_plan,
        )
