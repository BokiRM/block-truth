"""
The shapes this service works in.

Everything is minutes from midnight on the delivery date, as a float. Clock
strings are a presentation concern and live in `fmt.py`; arithmetic that has to
survive a round crossing midnight should never be done on "23:50".
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

MINUTES_PER_DAY = 1440.0


@dataclass(frozen=True)
class Stop:
    """One delivery as it was actually recorded.

    `travel_minutes` is the leg *leading to* this stop, so the first stop's leg
    includes the run out from the depot. `service_minutes` is the time spent at
    the door, which is shorter when the delivery failed.
    """
    sequence: int
    arrival_minutes: float
    service_minutes: float
    travel_minutes: float
    outcome: str | None = None
    delivery_point_id: str | None = None


@dataclass(frozen=True)
class Check:
    """One assertion about the recorded data, and whether it held."""
    name: str
    ok: bool
    detail: str
    gap_minutes: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BlockTruth:
    """What actually happened on one vehicle's round, derived from its stops.

    Nothing here is predicted. Every number is either read from the delivery
    records or derived from them by arithmetic that `checks` re-verifies.
    """
    block_id: str
    delivery_date: str
    stops: int

    start_minutes: float
    completion_minutes: float
    duration_minutes: float

    first_arrival_minutes: float
    last_arrival_minutes: float
    last_service_minutes: float
    door_to_door_minutes: float

    service_minutes: float
    travel_minutes: float
    depot_minutes: float
    return_leg_minutes: float

    delivered: int
    failed: int

    completion_source: str          # "recorded duration" | "last stop + return leg"
    duration_recorded: float | None
    checks: list[Check] = field(default_factory=list)
    attributes: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return all(c.ok for c in self.checks)

    @property
    def crosses_midnight(self) -> bool:
        return self.completion_minutes >= MINUTES_PER_DAY

    def as_dict(self, clock: bool = True) -> dict[str, Any]:
        from .fmt import hhmm
        d = {
            "block_id": self.block_id,
            "delivery_date": self.delivery_date,
            **self.attributes,
            "stops": self.stops,
            "delivered": self.delivered,
            "failed": self.failed,
            "start_minutes": round(self.start_minutes, 2),
            "completion_minutes": round(self.completion_minutes, 2),
            "duration_minutes": round(self.duration_minutes, 2),
            "duration_hours": round(self.duration_minutes / 60.0, 3),
            "service_minutes": round(self.service_minutes, 2),
            "travel_minutes": round(self.travel_minutes, 2),
            "depot_minutes": round(self.depot_minutes, 2),
            "return_leg_minutes": round(self.return_leg_minutes, 2),
            "door_to_door_minutes": round(self.door_to_door_minutes, 2),
            "minutes_per_stop": round(self.duration_minutes / self.stops, 3),
            "completion_source": self.completion_source,
            "ok": self.ok,
            "checks": [c.as_dict() for c in self.checks],
        }
        if clock:
            d["start_time"] = hhmm(self.start_minutes)
            d["completion_time"] = hhmm(self.completion_minutes)
            d["first_arrival_time"] = hhmm(self.first_arrival_minutes)
            d["last_arrival_time"] = hhmm(self.last_arrival_minutes)
            d["crosses_midnight"] = self.crosses_midnight
        return d
