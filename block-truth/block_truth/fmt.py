"""Clock strings in and out. Presentation only - the maths is all in minutes."""
from __future__ import annotations

from .model import MINUTES_PER_DAY


def to_minutes(clock: str) -> float:
    """'07:54' -> 474.0. Accepts 'HH:MM' and 'HH:MM:SS'."""
    parts = clock.strip().split(":")
    if len(parts) not in (2, 3):
        raise ValueError(f"not a clock time: {clock!r}")
    h, m = int(parts[0]), int(parts[1])
    s = int(parts[2]) if len(parts) == 3 else 0
    if not (0 <= m < 60 and 0 <= s < 60):
        raise ValueError(f"not a clock time: {clock!r}")
    return h * 60.0 + m + s / 60.0


def hhmm(minutes: float) -> str:
    """474.0 -> '07:54'. Past midnight keeps counting: 1485 -> '00:45 (+1d)'.

    Rounding happens once, on the total, so a round that ends at 15:59.6 reads
    16:00 rather than 15:60.
    """
    total = int(round(minutes))
    day, rem = divmod(total, int(MINUTES_PER_DAY))
    out = f"{rem // 60:02d}:{rem % 60:02d}"
    return f"{out} (+{day}d)" if day else out


def hours(minutes: float) -> str:
    h, m = divmod(int(round(minutes)), 60)
    return f"{h}h {m:02d}m"
