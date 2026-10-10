"""Everything the user can change, saved as JSON."""
import dataclasses
import json
import logging
import typing
from dataclasses import dataclass, field
from pathlib import Path

from jacket import config
from ren.decision import DecisionSettings, NeedWeights, VentSettings
from ren.engine import ViewSettings
from ren.plan import Calibration, PlanLayout
from ren.venues import DEFAULT_VENUE, VENUE_FIELDS, Venue, clean_name

logger = logging.getLogger("ren.settings")


@dataclass(frozen=True)
class AppSettings:
    view: ViewSettings = field(default_factory=ViewSettings)
    decision: DecisionSettings = field(default_factory=DecisionSettings)
    venues: tuple[Venue, ...] = (Venue(),)
    venue: str = DEFAULT_VENUE  # the active one

    @property
    def active(self) -> Venue:
        for venue in self.venues:
            if venue.name == self.venue:
                return venue
        raise KeyError(f"no venue named {self.venue!r}")

    @property
    def layout(self) -> PlanLayout:
        return self.active.layout

    @property
    def calibration(self) -> Calibration:
        return self.active.calibration

    @property
    def vent(self) -> VentSettings:
        return self.active.vent

    def venue_names(self) -> list[str]:
        return [venue.name for venue in self.venues]


def with_active(settings: AppSettings, venue: Venue) -> AppSettings:
    """Swap in a new version of the active venue."""
    venues = tuple(venue if v.name == settings.venue else v for v in settings.venues)
    return dataclasses.replace(settings, venues=venues, venue=venue.name)


def select_venue(settings: AppSettings, name: str) -> AppSettings:
    if name not in settings.venue_names():
        raise KeyError(f"no venue named {name!r}")
    return dataclasses.replace(settings, venue=name)


def add_venue(settings: AppSettings, text: str) -> AppSettings:
    """Copy the venue in use under a new name."""
    name = clean_name(text, settings.venue_names())
    copy = dataclasses.replace(settings.active, name=name)
    return dataclasses.replace(settings, venues=settings.venues + (copy,), venue=name)


def remove_venue(settings: AppSettings, name: str) -> AppSettings:
    if name not in settings.venue_names():
        raise KeyError(f"no venue named {name!r}")
    if len(settings.venues) == 1:
        raise ValueError("keep at least one venue")
    venues = tuple(v for v in settings.venues if v.name != name)
    active = settings.venue if settings.venue != name else venues[0].name
    return dataclasses.replace(settings, venues=venues, venue=active)


def _convert(hint, value):
    """Coerce a JSON value to `hint`, or raise TypeError."""
    if dataclasses.is_dataclass(hint):
        return from_dict(hint, value)
    if typing.get_origin(hint) is tuple and typing.get_args(hint)[-1] is Ellipsis:
        if not isinstance(value, (list, tuple)):
            raise TypeError("expected a list")
        return tuple(_convert(typing.get_args(hint)[0], item) for item in value)
    if typing.get_origin(hint) is tuple:
        item_hints = typing.get_args(hint)
        if not isinstance(value, (list, tuple)) or len(value) != len(item_hints):
            raise TypeError(f"expected {len(item_hints)} items")
        return tuple(_convert(item_hint, item) for item_hint, item in zip(item_hints, value))
    numeric = isinstance(value, (int, float)) and not isinstance(value, bool)
    if hint is bool and isinstance(value, bool):
        return value
    if hint is int and isinstance(value, int) and not isinstance(value, bool):
        return value
    if hint is float and numeric:
        return float(value)
    if hint is str and isinstance(value, str):
        return value
    raise TypeError(f"expected {getattr(hint, '__name__', hint)}, got {type(value).__name__}")


def from_dict(cls, data):
    """Build settings from JSON; bad fields keep defaults."""
    if not isinstance(data, dict):
        raise TypeError(f"expected an object for {cls.__name__}")
    hints = typing.get_type_hints(cls)
    names = {item.name for item in dataclasses.fields(cls)}
    kwargs = {}
    for name in names & set(data):
        try:
            kwargs[name] = _convert(hints[name], data[name])
        except (TypeError, ValueError) as error:
            logger.warning("settings: ignoring %s.%s (%s)", cls.__name__, name, error)
    unknown = sorted(set(data) - names)
    if unknown:
        logger.warning("settings: ignoring unknown keys %s in %s", unknown, cls.__name__)
    return cls(**kwargs)


def _clamp(value: float, low: float, high: float, name: str) -> float:
    clamped = min(max(value, low), high)
    if clamped != value:
        logger.warning("settings: %s=%s is out of range, using %s", name, value, clamped)
    return clamped


def _sanitize_venue(venue: Venue) -> Venue:
    layout = PlanLayout(
        int(_clamp(venue.layout.rows, 1, 12, "rows")), int(_clamp(venue.layout.cols, 1, 16, "seats per row"))
    )
    calibration = venue.calibration
    if not calibration.is_valid():
        logger.warning("settings: %s calibration is not a proper quad, using the default", venue.name)
        calibration = Calibration()
    vent = venue.vent
    vent = dataclasses.replace(
        vent, u=_clamp(vent.u, 0.0, 1.0, "vent u"), v=_clamp(vent.v, 0.0, 1.0, "vent v"),
        drop_m=_clamp(vent.drop_m, 0.5, 10.0, "drop"), tilt_max_deg=_clamp(vent.tilt_max_deg, 5.0, 80.0, "tilt max"),
        rotation_offset_deg=_clamp(vent.rotation_offset_deg, -360.0, 360.0, "rotation offset"),
        slew_deg_per_s=_clamp(vent.slew_deg_per_s, 1.0, 720.0, "slew"),
        seat_width_m=_clamp(vent.seat_width_m, 0.2, 2.0, "seat width"), row_depth_m=_clamp(vent.row_depth_m, 0.3, 3.0, "row depth"),
    )
    return Venue(venue.name, layout, calibration, vent)


def _sanitize_venues(settings: AppSettings) -> tuple[tuple[Venue, ...], str]:
    """At least one venue, unique names, valid active."""
    venues: list[Venue] = []
    for venue in settings.venues:
        if not venue.name.strip() or venue.name in [v.name for v in venues]:
            logger.warning("settings: dropping venue with a blank or repeated name %r", venue.name)
            continue
        venues.append(_sanitize_venue(venue))
    if not venues:
        venues = [Venue()]
    names = [venue.name for venue in venues]
    active = settings.venue if settings.venue in names else names[0]
    return tuple(venues), active


def sanitize(settings: AppSettings) -> AppSettings:
    """Pull every value into a safe range."""
    view, decision = settings.view, settings.decision
    if view.model_key not in config.CLIP_MODELS:
        logger.warning("settings: unknown model %r, using %s", view.model_key, config.DEFAULT_CLIP_MODEL)
        view = dataclasses.replace(view, model_key=config.DEFAULT_CLIP_MODEL)
    weights = NeedWeights(*(_clamp(getattr(decision.weights, n), 0.0, 1.0, f"need {n}") for n in ("hot", "unsure", "cold")))
    aim_mode = decision.aim_mode if decision.aim_mode in ("sweep", "focus") else config.DECISION_AIM_MODE
    decision = dataclasses.replace(
        decision, weights=weights, aim_mode=aim_mode,
        smoothing_s=_clamp(decision.smoothing_s, 0.0, 30.0, "smoothing"),
        min_share=_clamp(decision.min_share, 0.0, 0.5, "min share"),
        min_dwell_s=_clamp(decision.min_dwell_s, 0.0, 120.0, "min dwell"),
        close_below=_clamp(decision.close_below, 0.0, 50.0, "close below"),
        focus_margin=_clamp(decision.focus_margin, 0.0, 1.0, "focus margin"),
    )
    venues, active = _sanitize_venues(settings)
    return AppSettings(view, decision, venues, active)


def migrate(raw: dict) -> dict:
    """Older files kept one room at the top level."""
    raw = dict(raw)
    if raw.pop("impact", None) is not None:
        logger.info("settings: dropped the old impact numbers")
    if "venues" not in raw and any(key in raw for key in VENUE_FIELDS):
        room = {key: raw.pop(key) for key in VENUE_FIELDS if key in raw}
        raw["venues"] = [{"name": DEFAULT_VENUE, **room}]
        raw["venue"] = DEFAULT_VENUE
        logger.info("settings: moved your room into venue %s", DEFAULT_VENUE)
    return raw


def load_settings(path: Path = config.SETTINGS_FILE) -> AppSettings:
    """Read saved settings; bad files give defaults."""
    if not path.exists():
        return AppSettings()
    try:
        raw = json.loads(path.read_text())
        if not isinstance(raw, dict):
            raise TypeError("expected an object at the top level")
        return sanitize(from_dict(AppSettings, migrate(raw)))
    except (json.JSONDecodeError, TypeError) as error:
        logger.error("settings file %s is unreadable (%s); using defaults", path, error)
        return AppSettings()


def save_settings(settings: AppSettings, path: Path = config.SETTINGS_FILE) -> None:
    """Write atomically, so a crash leaves no half file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(dataclasses.asdict(settings), indent=2))
    temporary.replace(path)


def get_path(obj, path: tuple[str, ...]):
    for name in path:
        obj = getattr(obj, name)
    return obj


def replace_path(obj, path: tuple[str, ...], value):
    """A copy with one nested field changed."""
    head, *rest = path
    if isinstance(obj, AppSettings) and head in VENUE_FIELDS:
        return with_active(obj, replace_path(obj.active, path, value))
    inner = replace_path(getattr(obj, head), tuple(rest), value) if rest else value
    return dataclasses.replace(obj, **{head: inner})
