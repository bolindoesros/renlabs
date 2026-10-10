"""A venue: one room's seats, calibration and vent."""
import re
from dataclasses import dataclass, field

from ren.decision import VentSettings
from ren.plan import Calibration, PlanLayout, ZoneRect

DEFAULT_VENUE = "lt1"
VENUE_FIELDS = ("layout", "calibration", "vent", "zones")
NAME_MAX_CHARS = 24


@dataclass(frozen=True)
class Venue:
    name: str = DEFAULT_VENUE
    layout: PlanLayout = field(default_factory=PlanLayout)
    calibration: Calibration = field(default_factory=Calibration)
    vent: VentSettings = field(default_factory=VentSettings)
    zones: tuple[ZoneRect, ...] = ()  # none means equal automatic zones


def clean_name(text: str, taken: list[str]) -> str:
    """Lowercase, single-spaced and unique, or raise ValueError."""
    name = re.sub(r"\s+", " ", text).strip().lower()
    if not name:
        raise ValueError("give the venue a name")
    if len(name) > NAME_MAX_CHARS:
        raise ValueError(f"keep names under {NAME_MAX_CHARS} characters")
    if name in taken:
        raise ValueError(f"{name} already exists")
    return name


def suggest_name(taken: list[str]) -> str:
    """The next free lt name, like lt2."""
    number = 1
    while f"lt{number}" in taken:
        number += 1
    return f"lt{number}"
