"""
Seed the database with recorded rounds.

The app has to stand on its own, so this builds its own history rather than
borrowing one. It is synthetic, and the point of the construction is that it is
*coherent*: stops are placed on the map first and the times follow from the
geometry, so the distance between two stops and the minutes recorded between
them agree. An estimator calibrated on this recovers a speed because there is a
real speed in there to recover - not because a speed was written down.

    python3 -m block_truth.seed                 # 6 depots, 21 days
    python3 -m block_truth.seed --days 60
    python3 -m block_truth.seed --from-delivery-db ../delivery-agent/data/delivery.db
"""
from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

from .db import PARCEL_SIZES, POINT_TYPES, VEHICLE_TYPES, connect, counts
from .fmt import hhmm, to_minutes
from .geo import ROAD_FACTOR, haversine_km

# Real city centres, so the map looks like somewhere.
DEPOTS = [
    ("D01", "Leeds Central",      "Leeds",      "Yorkshire",    53.7997, -1.5492),
    ("D02", "Manchester East",    "Manchester", "North West",   53.4808, -2.2426),
    ("D03", "Birmingham South",   "Birmingham", "West Midlands", 52.4862, -1.8904),
    ("D04", "Bristol North",      "Bristol",    "South West",   51.4545, -2.5879),
    ("D05", "London South East",  "London",     "London",       51.4934, -0.0098),
    ("D06", "Newcastle West",     "Newcastle",  "North East",   54.9783, -1.6178),
]

STREETS = ["High St", "Church Ln", "Mill Rd", "Station Rd", "Victoria Rd", "Park Ave",
           "Queens Rd", "Albert St", "Kings Way", "Bridge St", "Oak Dr", "Elm Gr",
           "Chapel St", "Market St", "Grange Rd", "Meadow Ln", "Riverside", "Castle St"]
STREETS = [s for s in STREETS if s.isascii()]

# Minutes at the door. The estimator has to find these back out of the records.
POINT_SERVICE = {"House": 2.1, "Flat": 3.4, "Office": 2.7,
                 "Business": 2.5, "Parcel Locker": 0.9}
PARCEL_SERVICE = {"Letter": 0.2, "Small": 0.5, "Medium": 1.0, "Large": 2.2, "Bulky": 5.0}
FAILED_SERVICE = 1.4                    # knock, wait, card, back to the van

PARK_MIN = {"London": 1.6, "Manchester": 1.1, "Birmingham": 1.1}   # per stop
PARK_DEFAULT = 0.9
SPEED_KMH = {"London": 13.0, "Manchester": 17.0, "Birmingham": 17.5}
SPEED_DEFAULT = 21.0
DEPOT_LOAD = {"Small Van": 13.0, "Large Van": 17.0, "Box Truck": 23.0}
VEHICLE_SPEED = {"Small Van": 1.0, "Large Van": 0.94, "Box Truck": 0.82}

PROMISED = [("Before 12:00", "08:00", "12:00", 0.26),
            ("Afternoon", "12:00", "18:00", 0.34),
            ("Evening", "18:00", "21:00", 0.10),
            ("Anytime", "07:30", "21:00", 0.30)]

# ---------------------------------------------------------------- the traffic
# The truth about traffic, as this synthetic history has it. It is deliberately
# NOT shared with `traffic.py`, which has to find this shape back out of the
# recorded legs without ever being told it. If the two shared a function the
# recovery test would be circular and would prove nothing.
#
# A multiplier on travel *time*: above 1 is slower than this city's own average
# pace, below 1 is faster. Two peaks, a quiet middle, an easy Saturday.
TRUE_PEAKS = [(8.25, 1.38, 1.25),      # centre hour, height, width in hours
              (16.75, 1.32, 1.40)]
TRUE_WEEKDAY = {0: 1.05, 1: 1.00, 2: 0.99, 3: 1.02, 4: 1.11, 5: 0.86, 6: 0.80}
TRUE_CITY_AMPLITUDE = {"London": 1.5, "Manchester": 1.15, "Birmingham": 1.1}


def _true_traffic(minutes: float, weekday: int, city: str) -> float:
    """What the seeded world does to a leg driven at this time. Not the model."""
    hour = (minutes / 60.0) % 24.0
    amp = TRUE_CITY_AMPLITUDE.get(city, 0.85)
    bump = 0.0
    for centre, height, width in TRUE_PEAKS:
        bump += (height - 1.0) * math.exp(-((hour - centre) / width) ** 2)
    return (1.0 + amp * bump) * TRUE_WEEKDAY.get(weekday, 1.0)


def _wedge(rng, lat0, lon0, n, spread_km):
    """Stops in one angular wedge out from the depot, visited nearest-first.

    A driver does not zigzag across the patch, and neither does this: the order
    is a nearest-neighbour tour from the depot, which is both what rounds look
    like and what makes the drawn path on the map worth looking at.
    """
    bearing = rng.uniform(0, 2 * math.pi)
    width = rng.uniform(0.4, 0.9)
    pts = []
    for _ in range(n):
        ang = bearing + rng.uniform(-width, width)
        dist = spread_km * math.sqrt(rng.random())
        dlat = dist * math.cos(ang) / 111.32
        dlon = dist * math.sin(ang) / (111.32 * math.cos(math.radians(lat0)))
        pts.append((round(lat0 + dlat, 6), round(lon0 + dlon, 6)))

    tour, cur, left = [], (lat0, lon0), pts[:]
    while left:
        nxt = min(left, key=lambda p: haversine_km(cur[0], cur[1], p[0], p[1]))
        left.remove(nxt)
        tour.append(nxt)
        cur = nxt
    return tour


def seed(db_path: str | Path = None, days: int = 21, seed_value: int = 7,
         end_date: str | None = None) -> dict:
    """Build a history ending yesterday, so "the month ahead" really is ahead."""
    rng = random.Random(seed_value)
    con = connect(db_path)
    con.execute("DELETE FROM stops")
    con.execute("DELETE FROM blocks")
    con.execute("DELETE FROM depots")
    con.executemany("INSERT INTO depots VALUES (?,?,?,?,?,?)", DEPOTS)

    import datetime as dt
    end = (dt.date.fromisoformat(end_date) if end_date
           else dt.date.today() - dt.timedelta(days=1))
    dates = [(end - dt.timedelta(days=i)).isoformat() for i in range(days)][::-1]

    nb = ns = 0
    for date in dates:
        weekday = dt.date.fromisoformat(date).weekday()
        if weekday == 6:                       # no Sunday rounds
            continue
        for depot_id, _, city, _, dlat, dlon in DEPOTS:
            speed_base = SPEED_KMH.get(city, SPEED_DEFAULT)
            park = PARK_MIN.get(city, PARK_DEFAULT)
            # the whole day runs fast or slow together: traffic, weather, who is on
            day_factor = rng.gauss(1.0, 0.10)
            day_factor = min(max(day_factor, 0.75), 1.35)
            for v in range(rng.randint(2, 4)):
                vt = rng.choices(VEHICLE_TYPES, weights=[0.45, 0.40, 0.15])[0]
                n = rng.randint(18, 42) if vt != "Box Truck" else rng.randint(10, 20)
                # a patch big enough to hold the stops at a plausible density
                density = rng.uniform(1.4, 3.6) if city != "London" else rng.uniform(0.7, 1.6)
                spread = math.sqrt(n / math.pi) * density
                pts = _wedge(rng, dlat, dlon, n, spread)

                vehicle_id = f"{depot_id}-V{v:02d}"
                block_id = f"{date.replace('-', '')}-{vehicle_id}"
                # most vehicles are out in the morning, a quarter of them run a
                # late shift - which is also what puts legs on the road through
                # the evening peak, so there is evidence there to fit
                start_min = (7 * 60 + rng.randint(0, 50) if rng.random() > 0.25
                             else 11 * 60 + rng.randint(30, 150))
                speed = speed_base * VEHICLE_SPEED[vt] * (2.0 - day_factor)

                rows, t = [], start_min + DEPOT_LOAD[vt] * rng.gauss(1.0, 0.08)
                prev = (dlat, dlon)
                for i, (lat, lon) in enumerate(pts, start=1):
                    km = haversine_km(*prev, lat, lon) * ROAD_FACTOR
                    # the clock matters: a leg driven through the morning peak
                    # takes longer than the same leg at eleven
                    tf = _true_traffic(t, weekday, city)
                    travel = km / speed * 60.0 * tf + park * rng.gauss(1.0, 0.25)
                    travel = max(travel, 0.2)
                    t += travel
                    arrival = t
                    pt = rng.choices(POINT_TYPES, weights=[.52, .24, .09, .10, .05])[0]
                    ps = rng.choices(PARCEL_SIZES, weights=[.14, .26, .36, .18, .06])[0]
                    ok = rng.random() > (0.06 if pt != "Flat" else 0.13)
                    svc = (POINT_SERVICE[pt] + PARCEL_SERVICE[ps]) * rng.gauss(1.0, 0.18) \
                        if ok else FAILED_SERVICE * rng.gauss(1.0, 0.2)
                    svc = max(svc, 0.2)
                    t += svc
                    label, pfrom, pto, _ = rng.choices(PROMISED,
                                                       weights=[p[3] for p in PROMISED])[0]
                    rows.append((block_id, i,
                                 f"{rng.randint(1, 180)} {rng.choice(STREETS)}",
                                 f"{city[:2].upper()}{rng.randint(1, 30)} "
                                 f"{rng.randint(1, 9)}{rng.choice('ABDEFGHJLNPQRSTUWXYZ')}"
                                 f"{rng.choice('ABDEFGHJLNPQRSTUWXYZ')}",
                                 lat, lon, pt, ps, pfrom, pto,
                                 hhmm(arrival), round(svc, 2), round(travel, 2),
                                 "Delivered" if ok else "Unsuccessful"))
                    prev = (lat, lon)

                back_km = haversine_km(*prev, dlat, dlon) * ROAD_FACTOR
                return_leg = back_km / speed * 60.0 * _true_traffic(t, weekday, city)
                duration = (t - start_min) + return_leg

                con.execute(
                    "INSERT INTO blocks (block_id, delivery_date, vehicle_id, vehicle_type,"
                    " depot_id, start_time, duration_minutes, source) "
                    "VALUES (?,?,?,?,?,?,?,'recorded')",
                    (block_id, date, vehicle_id, vt, depot_id,
                     hhmm(start_min), round(duration, 1)))
                con.executemany(
                    "INSERT INTO stops (block_id, sequence, address, postcode, lat, lon,"
                    " point_type, parcel_size, promised_from, promised_to, arrival_time,"
                    " service_minutes, travel_minutes, outcome) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
                nb += 1
                ns += len(rows)

    con.commit()
    out = counts(con)
    con.close()
    return out


def import_delivery_db(src: str | Path, db_path: str | Path = None,
                       days: int = 21) -> dict:
    """Optional: pull recorded rounds out of the delivery pipeline's database."""
    import sqlite3
    src = Path(src)
    if not src.exists():
        raise FileNotFoundError(f"no database at {src}")
    s = sqlite3.connect(f"file:{src}?mode=ro", uri=True)
    s.row_factory = sqlite3.Row
    con = connect(db_path)
    con.execute("DELETE FROM stops"); con.execute("DELETE FROM blocks")
    con.execute("DELETE FROM depots")

    dates = [r[0] for r in s.execute(
        "SELECT DISTINCT delivery_date FROM deliveries ORDER BY 1 DESC LIMIT ?",
        (days,))]
    marks = ",".join("?" * len(dates))
    depots = s.execute(
        f"SELECT depot_id, city, region, avg(lat) la, avg(lon) lo FROM deliveries "
        f"WHERE delivery_date IN ({marks}) GROUP BY depot_id", dates).fetchall()
    con.executemany("INSERT INTO depots VALUES (?,?,?,?,?,?)",
                    [(d["depot_id"], f"{d['city']} depot", d["city"], d["region"],
                      round(d["la"], 6), round(d["lo"], 6)) for d in depots])

    size_of = {1: "Letter", 2: "Letter", 3: "Small", 4: "Small", 5: "Small", 6: "Small",
               7: "Small", 8: "Medium", 9: "Medium", 10: "Medium", 11: "Medium",
               12: "Medium", 13: "Large", 14: "Large", 15: "Large", 16: "Bulky",
               17: "Bulky", 18: "Bulky", 19: "Bulky", 20: "Bulky"}
    rows = s.execute(
        f"SELECT block_id, delivery_date, vehicle_id, vehicle_type, depot_id, "
        f"block_start_time, block_duration_minutes, stop_sequence, address_line, "
        f"postcode, lat, lon, point_type, parcel_type, arrival_time, service_minutes, "
        f"travel_minutes, outcome, delivery_window FROM deliveries "
        f"WHERE delivery_date IN ({marks}) ORDER BY block_id, stop_sequence",
        dates).fetchall()
    seen = set()
    for r in rows:
        if r["block_id"] not in seen:
            seen.add(r["block_id"])
            con.execute(
                "INSERT INTO blocks (block_id, delivery_date, vehicle_id, vehicle_type,"
                " depot_id, start_time, duration_minutes, source) "
                "VALUES (?,?,?,?,?,?,?,'recorded')",
                (r["block_id"], r["delivery_date"], r["vehicle_id"], r["vehicle_type"],
                 r["depot_id"], r["block_start_time"], r["block_duration_minutes"]))
        con.execute(
            "INSERT INTO stops (block_id, sequence, address, postcode, lat, lon,"
            " point_type, parcel_size, arrival_time, service_minutes, travel_minutes,"
            " outcome) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (r["block_id"], r["stop_sequence"], r["address_line"], r["postcode"],
             r["lat"], r["lon"], r["point_type"], size_of.get(r["parcel_type"], "Medium"),
             r["arrival_time"], r["service_minutes"], r["travel_minutes"], r["outcome"]))
    con.commit()
    out = counts(con)
    con.close(); s.close()
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=None)
    ap.add_argument("--days", type=int, default=21)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--from-delivery-db", dest="src", default=None,
                    help="import recorded rounds from the delivery pipeline instead")
    a = ap.parse_args()
    if a.src:
        print("imported:", import_delivery_db(a.src, a.db, a.days))
    else:
        print("seeded:", seed(a.db, a.days, a.seed))
