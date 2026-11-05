"""
Traffic, measured - and road closures, declared.

Two different kinds of knowledge live here and the difference matters enough to
keep saying out loud:

* **Traffic** is *measured*. Nobody tells this app that the morning peak is bad.
  Every recorded leg has a distance implied by its two ends and a time recorded
  against it, and it was driven at a known hour on a known weekday. Divide what
  it took by what the city's average pace says it should have taken and you have
  a ratio; average those ratios by hour and by weekday and the shape of the day
  falls out of the records. The curve is *relative*: 1.00 is this city's own
  all-day average, because the per-kilometre figure the estimator already uses
  has the average traffic baked into it.

* **Disruptions** are *declared*. A closed road is not in the delivery records,
  and no amount of history will produce one. They are rows somebody enters - by
  hand, from a workbook, or one day from a live feed - and the app says so
  wherever one changes an answer. `source` on each row keeps the two apart.

The shrinkage matters. An hour with eleven legs behind it should not swing a
forecast, so every bucket is pulled toward 1.00 in proportion to how little
evidence stands behind it:

    factor = (n * observed + K * 1.0) / (n + K)

With K = 60 legs, a bucket needs real weight before it moves the answer, and a
bucket with nothing in it simply says 1.00 and reports `n = 0`.
"""
from __future__ import annotations

import datetime as dt
import math
import statistics as st
from dataclasses import dataclass, field

from .fmt import to_minutes
from .geo import ROAD_FACTOR, haversine_km

SHRINK_K = 60.0             # legs needed before a bucket carries its own weight
MIN_EXPECTED_MIN = 1.5      # ignore legs too short to measure a ratio on
FACTOR_FLOOR, FACTOR_CEIL = 0.6, 2.2

# A closed road is gone around. For a straight line crossing a blocked disc, the
# way round it is the arc rather than the chord, and for a chord of length c the
# detour adds about (pi/2 - 1) * c. For a line through the centre of a disc of
# radius r that gives 1.14 r against the 2 r it replaces, which is the exact
# half-circumference answer - so the approximation is right where it matters
# most and conservative elsewhere.
DETOUR_PER_CHORD_KM = math.pi / 2 - 1

# The detour is a guess at a route nobody has driven, so it carries more doubt
# than ordinary driving: 40% of whatever it adds.
DISRUPTION_SD_SHARE = 0.40

KINDS = ["closure", "works", "congestion", "event"]


@dataclass
class Traffic:
    """How much slower than average the roads are, by hour and by weekday."""
    hour: dict[int, float] = field(default_factory=dict)        # 0..23
    hour_n: dict[int, int] = field(default_factory=dict)
    weekday: dict[int, float] = field(default_factory=dict)     # 0=Mon
    weekday_n: dict[int, int] = field(default_factory=dict)
    city_amplitude: dict[str, float] = field(default_factory=dict)
    legs_used: int = 0
    notes: list[str] = field(default_factory=list)

    def factor(self, minutes: float, date: str | None = None,
               city: str | None = None) -> float:
        """Multiplier on travel time for a leg driven at this time of day."""
        h = int((minutes / 60.0) % 24)
        f = self.hour.get(h, 1.0)
        if city is not None:
            # a city whose peaks are sharper than average gets a sharper curve:
            # the amplitude scales the departure from 1, never the level
            f = 1.0 + (f - 1.0) * self.city_amplitude.get(city, 1.0)
        if date:
            try:
                f *= self.weekday.get(dt.date.fromisoformat(date).weekday(), 1.0)
            except ValueError:
                pass
        return min(max(f, FACTOR_FLOOR), FACTOR_CEIL)

    def busiest(self) -> tuple[int, float]:
        if not self.hour:
            return (0, 1.0)
        h = max(self.hour, key=lambda k: self.hour[k])
        return h, self.hour[h]

    def as_dict(self) -> dict:
        return {
            "hour": [{"hour": h, "factor": round(self.hour.get(h, 1.0), 3),
                      "legs": self.hour_n.get(h, 0)} for h in range(24)],
            "weekday": [{"weekday": d,
                         "name": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][d],
                         "factor": round(self.weekday.get(d, 1.0), 3),
                         "legs": self.weekday_n.get(d, 0)} for d in range(7)],
            "city_amplitude": {k: round(v, 3) for k, v in self.city_amplitude.items()},
            "legs_used": self.legs_used,
            "measured": True,
            "notes": self.notes,
        }


def _shrunk(values: list[float], k: float = SHRINK_K) -> float:
    if not values:
        return 1.0
    n = len(values)
    return (n * st.fmean(values) + k * 1.0) / (n + k)


def fit_traffic(con, cal, dates: set[str] | None = None) -> Traffic:
    """Recover the shape of the day from the recorded legs.

    Two passes, because the hour of the day and the day of the week are mixed
    together in every single leg. Fit the hourly shape first across all days,
    then ask what is left over for each weekday once the hourly shape is taken
    out. Doing it the other way round gives the same answer to within a
    rounding error; doing only one of them blames the hour for the Friday.
    """
    t = Traffic()
    rows = con.execute(
        "SELECT b.delivery_date, b.depot_id, d.city, d.lat dlat, d.lon dlon, "
        "       s.block_id, s.sequence, s.lat, s.lon, s.arrival_time, s.travel_minutes "
        "FROM stops s JOIN blocks b USING(block_id) JOIN depots d USING(depot_id) "
        "WHERE b.source='recorded' AND s.travel_minutes IS NOT NULL "
        "  AND s.arrival_time IS NOT NULL ORDER BY s.block_id, s.sequence").fetchall()
    if dates is not None:
        # a fold: fit on these days only, so the days left out can judge the
        # result without having voted on it
        rows = [r for r in rows if r["delivery_date"] in dates]
    if not rows:
        t.notes.append("no recorded legs: the day is assumed flat")
        return t

    by_hour: dict[int, list[float]] = {}
    by_city_hour: dict[str, list[tuple[int, float]]] = {}
    samples = []        # (date, hour, city, ratio)

    prev_block = None
    prev_xy = None
    for r in rows:
        if r["block_id"] != prev_block:
            prev_block, prev_xy = r["block_id"], (r["dlat"], r["dlon"])
        km = haversine_km(prev_xy[0], prev_xy[1], r["lat"], r["lon"]) * ROAD_FACTOR
        prev_xy = (r["lat"], r["lon"])

        # Measure the ratio on the driving only. A leg is "minutes per kilometre
        # times kilometres, plus a fixed cost for parking and walking", and
        # traffic slows the first part while leaving the second alone. Taking the
        # ratio of the whole leg would mix the two and read every peak as
        # shallower than it is - which is exactly what it did before this line.
        per_km, fixed, _ = cal.travel_for(r["city"])
        expected = per_km.mean * km
        actual = float(r["travel_minutes"]) - fixed
        if expected < MIN_EXPECTED_MIN or actual <= 0.2:
            continue
        # the middle of the leg, not its end: a 20-minute leg arriving at 09:00
        # was mostly driven at 08:50
        mid = to_minutes(r["arrival_time"]) - actual / 2.0
        hour = int((mid / 60.0) % 24)
        ratio = actual / expected
        if not (0.2 < ratio < 5.0):
            continue
        by_hour.setdefault(hour, []).append(ratio)
        by_city_hour.setdefault(r["city"], []).append((hour, ratio))
        samples.append((r["delivery_date"], hour, r["city"], ratio))

    t.legs_used = len(samples)
    if t.legs_used < 200:
        t.notes.append(f"only {t.legs_used} legs: the day is assumed flat")
        return t

    # --- pass 1: the shape of the day ----------------------------------------
    for h in range(24):
        vals = by_hour.get(h, [])
        t.hour[h] = _shrunk(vals)
        t.hour_n[h] = len(vals)

    # the curve is relative to the city's own average pace, so centre it on the
    # legs actually driven - otherwise a fleet that works mornings would read as
    # "the whole day is slow"
    weighted = sum(t.hour[h] * t.hour_n[h] for h in range(24))
    total_n = sum(t.hour_n.values())
    if total_n and weighted > 0:
        level = weighted / total_n
        for h in range(24):
            t.hour[h] /= level

    # --- pass 2: what each weekday has left over ------------------------------
    by_wd: dict[int, list[float]] = {}
    for date, hour, _city, ratio in samples:
        try:
            wd = dt.date.fromisoformat(date).weekday()
        except ValueError:
            continue
        shape = t.hour.get(hour, 1.0)
        if shape > 0:
            by_wd.setdefault(wd, []).append(ratio / shape)
    for d in range(7):
        vals = by_wd.get(d, [])
        t.weekday[d] = _shrunk(vals)
        t.weekday_n[d] = len(vals)
    lvl = st.fmean([t.weekday[d] for d in range(7) if t.weekday_n.get(d)]) or 1.0
    for d in range(7):
        t.weekday[d] /= lvl

    # --- how sharply each city's peaks bite -----------------------------------
    # one number per city: how much of the national curve's departure from 1 that
    # city actually shows. Fitted by least squares through the origin, which is
    # the honest way to say "this city is 1.5 times as peaky" with one parameter.
    for city, pairs in by_city_hour.items():
        num = den = 0.0
        for hour, ratio in pairs:
            x = t.hour.get(hour, 1.0) - 1.0
            num += x * (ratio - 1.0)
            den += x * x
        if den > 1e-9 and len(pairs) >= 200:
            t.city_amplitude[city] = min(max(num / den, 0.0), 3.0)

    h, f = t.busiest()
    t.notes.append(f"measured from {t.legs_used:,} recorded legs; worst hour is "
                   f"{h:02d}:00 at {f:.2f}x the day's average pace")
    return t


# ------------------------------------------------------------------ disruptions

@dataclass
class Disruption:
    """A declared obstruction. Not measured - somebody entered this."""
    disruption_id: str
    kind: str                       # closure | works | congestion | event
    description: str
    lat: float
    lon: float
    radius_km: float
    from_date: str
    to_date: str
    from_time: str | None = None    # time of day, optional
    to_time: str | None = None
    severity: float = 1.5           # multiplier for anything but a closure
    source: str = "manual"

    def active_on(self, date: str) -> bool:
        return self.from_date <= date <= self.to_date

    def active_at(self, minutes: float) -> bool:
        if not self.from_time or not self.to_time:
            return True
        a, b = to_minutes(self.from_time), to_minutes(self.to_time)
        m = minutes % 1440.0
        return a <= m <= b if a <= b else (m >= a or m <= b)

    def contains(self, lat: float, lon: float) -> bool:
        return haversine_km(lat, lon, self.lat, self.lon) <= self.radius_km

    def as_dict(self) -> dict:
        return {"disruption_id": self.disruption_id, "kind": self.kind,
                "description": self.description, "lat": self.lat, "lon": self.lon,
                "radius_km": self.radius_km, "from_date": self.from_date,
                "to_date": self.to_date, "from_time": self.from_time,
                "to_time": self.to_time, "severity": self.severity,
                "source": self.source, "declared": True}


def chord_km(ax: float, ay: float, bx: float, by: float,
             cx: float, cy: float, r_km: float) -> float:
    """How much of the straight leg a->b lies inside the disc at c.

    Plane geometry on a local equirectangular projection, which is accurate
    enough over the few kilometres a leg covers and keeps this readable.
    """
    kx = 111.32 * math.cos(math.radians((ay + by) / 2))
    ky = 110.57
    ax_, ay_ = ax * kx, ay * ky           # note: x is longitude
    bx_, by_ = bx * kx, by * ky
    cx_, cy_ = cx * kx, cy * ky
    dx, dy = bx_ - ax_, by_ - ay_
    seg2 = dx * dx + dy * dy
    if seg2 <= 1e-12:
        return 0.0
    fx, fy = ax_ - cx_, ay_ - cy_
    a = seg2
    b = 2 * (fx * dx + fy * dy)
    c = fx * fx + fy * fy - r_km * r_km
    disc = b * b - 4 * a * c
    if disc <= 0:
        return 0.0
    sq = math.sqrt(disc)
    t1 = max((-b - sq) / (2 * a), 0.0)
    t2 = min((-b + sq) / (2 * a), 1.0)
    if t2 <= t1:
        return 0.0
    return (t2 - t1) * math.sqrt(seg2)


@dataclass
class LegImpact:
    extra_km: float = 0.0
    factor: float = 1.0
    sd_minutes: float = 0.0
    hit: list[str] = field(default_factory=list)


def leg_impact(a: tuple[float, float], b: tuple[float, float],
               minutes: float, date: str,
               disruptions: list[Disruption]) -> LegImpact:
    """What the declared disruptions do to one leg driven at this time."""
    out = LegImpact()
    for d in disruptions:
        if not d.active_on(date) or not d.active_at(minutes):
            continue
        # a[0]/b[0] are latitudes; chord_km wants (lon, lat) as (x, y)
        c = chord_km(a[1], a[0], b[1], b[0], d.lon, d.lat, d.radius_km)
        if c <= 0.0:
            continue
        out.hit.append(d.disruption_id)
        if d.kind == "closure":
            out.extra_km += DETOUR_PER_CHORD_KM * c
        else:
            # only the part inside the area is slowed, so the leg's factor is a
            # blend weighted by how much of it is in there
            share = min(c / max(_km(a, b), 1e-6), 1.0)
            out.factor *= 1.0 + share * (max(d.severity, 1.0) - 1.0)
    return out


def _km(a: tuple[float, float], b: tuple[float, float]) -> float:
    return haversine_km(a[0], a[1], b[0], b[1]) * ROAD_FACTOR


def stops_inside_closures(stops, date: str, disruptions: list[Disruption]) -> list[int]:
    """Stops that sit inside something closed. These need a human, not a route."""
    out = []
    for s in stops:
        for d in disruptions:
            if d.kind == "closure" and d.active_on(date) and d.contains(s.lat, s.lon):
                out.append(s.sequence)
                break
    return out


@dataclass
class Conditions:
    """Everything outside the vehicle that changes how long the round takes."""
    traffic: Traffic = field(default_factory=Traffic)
    disruptions: list[Disruption] = field(default_factory=list)
    use_traffic: bool = True

    def for_date(self, date: str) -> "Conditions":
        return Conditions(self.traffic,
                          [d for d in self.disruptions if d.active_on(date)],
                          self.use_traffic)

    def as_dict(self) -> dict:
        return {"traffic": self.traffic.as_dict(),
                "disruptions": [d.as_dict() for d in self.disruptions],
                "use_traffic": self.use_traffic}


FLAT = Conditions(use_traffic=False)        # the old behaviour, for comparison
