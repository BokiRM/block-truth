"""
Geometry: distances along a round, and how to draw it.

Stops carry coordinates, so the path a vehicle takes is the sequence of straight
legs between them, out from the depot and back. That is a real measurement of
the order of visits - it is not a road route, and nothing here pretends it is:
there is no road network and no GPS trace in the data. Road distance runs longer
than straight-line distance by a fairly stable factor, so `ROAD_FACTOR` turns one
into an estimate of the other and is named rather than buried.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

EARTH_KM = 6371.0088
# Straight line to driven distance. 1.3 is the usual figure for dense urban road
# networks; it is a constant here because the data has no road network to measure.
ROAD_FACTOR = 1.30


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    # A missing coordinate used to arrive here and come back out as "must be
    # real number, not NoneType" from inside math.radians - a stack trace that
    # names a trigonometric function and not the row that is wrong. Callers are
    # supposed to filter those out before they get this far; when one does not,
    # this says what actually happened.
    if lat1 is None or lon1 is None or lat2 is None or lon2 is None:
        raise ValueError(f"a point is missing a coordinate: "
                         f"({lat1}, {lon1}) to ({lat2}, {lon2})")
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_KM * math.asin(math.sqrt(a))


def leg_km(points: list[tuple[float, float]]) -> list[float]:
    """Straight-line length of each leg of a path. First entry is depot -> stop 1."""
    return [haversine_km(a[0], a[1], b[0], b[1]) for a, b in zip(points, points[1:])]


def route_km(points: list[tuple[float, float]], road_factor: float = ROAD_FACTOR) -> float:
    """Driven distance of the whole path, depot out and depot back included."""
    return sum(leg_km(points)) * road_factor


@dataclass(frozen=True)
class Projection:
    """Equirectangular projection onto an SVG box. Fine at the size of one round."""
    lat0: float
    lon0: float
    scale: float          # pixels per kilometre
    width: float
    height: float
    cx: float
    cy: float

    def xy(self, lat: float, lon: float) -> tuple[float, float]:
        kx = math.cos(math.radians(self.lat0)) * 111.32
        x = self.cx + (lon - self.lon0) * kx * self.scale
        y = self.cy - (lat - self.lat0) * 111.32 * self.scale
        return round(x, 2), round(y, 2)


def fit(points: list[tuple[float, float]], width: float = 760, height: float = 520,
        pad: float = 42) -> Projection:
    """A projection that puts every point inside the box with room for labels."""
    lats = [p[0] for p in points] or [0.0]
    lons = [p[1] for p in points] or [0.0]
    lat0 = (min(lats) + max(lats)) / 2
    lon0 = (min(lons) + max(lons)) / 2
    kx = math.cos(math.radians(lat0)) * 111.32
    span_x = max((max(lons) - min(lons)) * kx, 0.20)      # never divide by nothing
    span_y = max((max(lats) - min(lats)) * 111.32, 0.20)
    scale = min((width - 2 * pad) / span_x, (height - 2 * pad) / span_y)
    return Projection(lat0, lon0, scale, width, height, width / 2, height / 2)


def nice_scale_bar(proj: Projection, max_px: float = 150) -> tuple[float, float]:
    """A round number of kilometres that fits in `max_px`, and its width."""
    for km in (0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100):
        px = km * proj.scale
        if px > max_px:
            break
        best, best_px = km, px
    return best, round(best_px, 1)
