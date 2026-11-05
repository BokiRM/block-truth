"""
The month ahead: planned rounds, one per vehicle per working day.

What is honest about this and what is not
----------------------------------------

The *shape* of the schedule is read out of the recorded history, not invented:
which depots exist, how many vehicles each one runs and of what type, how many
stops a vehicle of that type actually carries, when it leaves, and how a Monday
differs from a Saturday. All of that is measured.

The *demand* is not. There is no order book in this database, and no amount of
history will produce next Tuesday's parcels. So the stops themselves are drawn
the same way the seeded history draws them - a wedge of the depot's patch at a
plausible density, in a sensible visiting order. A planned block is therefore a
realistic shape to plan against, not a promise about which doors will be on it.

Every row this writes carries `source='planned'`, so nothing downstream can
mistake it for work that happened. Planned blocks never feed the calibration.

    python3 -m block_truth.schedule --days 30
"""
from __future__ import annotations

import argparse
import datetime as dt
import random
import statistics as st
from pathlib import Path

from .db import VEHICLE_TYPES, connect, counts
from .estimate import StopPlan
from .fmt import hhmm, to_minutes
from .itinerary import _nearest_neighbour, _two_opt
from .seed import _wedge

WORKDAYS = {0, 1, 2, 3, 4, 5}           # Monday to Saturday; no Sunday rounds


def _fleet(con) -> list[dict]:
    """Who is on the road, from the rounds that have actually run.

    A vehicle counts as part of the fleet if it has worked recently. Its type is
    whatever it most often was, and its stop count and start time are its own
    history - a Box Truck does not get a van's day.
    """
    rows = con.execute(
        "SELECT b.depot_id, b.vehicle_id, b.vehicle_type, b.start_time, "
        "       d.city, d.region, d.lat, d.lon, count(s.sequence) stops "
        "FROM blocks b JOIN depots d USING(depot_id) "
        "LEFT JOIN stops s USING(block_id) "
        "WHERE b.source='recorded' AND b.vehicle_id IS NOT NULL "
        "GROUP BY b.block_id").fetchall()
    agg: dict[tuple[str, str], dict] = {}
    for r in rows:
        key = (r["depot_id"], r["vehicle_id"])
        a = agg.setdefault(key, {
            "depot_id": r["depot_id"], "vehicle_id": r["vehicle_id"],
            "city": r["city"], "region": r["region"],
            "lat": r["lat"], "lon": r["lon"],
            "types": [], "stops": [], "starts": [], "blocks": 0})
        a["types"].append(r["vehicle_type"] or "Large Van")
        a["stops"].append(r["stops"])
        a["starts"].append(to_minutes(r["start_time"]))
        a["blocks"] += 1
    out = []
    for a in agg.values():
        if a["blocks"] < 2:                     # one day is not a fleet member
            continue
        a["vehicle_type"] = st.mode(a["types"]) if a["types"] else "Large Van"
        a["stops_mean"] = st.fmean(a["stops"])
        a["stops_sd"] = st.stdev(a["stops"]) if len(a["stops"]) > 1 else 3.0
        # The median, not the mean, and a tight spread. A vehicle that works
        # mornings most of the time and the late shift occasionally has a
        # two-humped start time, and the average of two humps is the middle,
        # which is the one time it never starts. The median picks its usual
        # shift; the cap stops the other hump widening it into nonsense.
        a["start_mean"] = st.median(a["starts"])
        a["start_sd"] = min(max(st.stdev(a["starts"]), 5.0), 20.0) \
            if len(a["starts"]) > 1 else 15.0
        out.append(a)
    out.sort(key=lambda a: (a["depot_id"], a["vehicle_id"]))
    return out


def _weekday_volume(con) -> dict[int, float]:
    """How busy each weekday is, relative to the average, from the history."""
    rows = con.execute(
        "SELECT b.delivery_date, count(s.sequence) n FROM blocks b "
        "JOIN stops s USING(block_id) WHERE b.source='recorded' "
        "GROUP BY b.block_id, b.delivery_date").fetchall()
    by: dict[int, list[int]] = {}
    for r in rows:
        try:
            by.setdefault(dt.date.fromisoformat(r["delivery_date"]).weekday(),
                          []).append(r["n"])
        except ValueError:
            continue
    if not by:
        return {d: 1.0 for d in range(7)}
    means = {d: st.fmean(v) for d, v in by.items() if v}
    level = st.fmean(list(means.values()))
    return {d: (means.get(d, level) / level if level else 1.0) for d in range(7)}


def generate(db_path: str | Path = None, days: int = 30,
             from_date: str | None = None, replace: bool = True) -> dict:
    """Write `days` of planned rounds, starting tomorrow unless told otherwise."""
    con = connect(db_path)
    fleet = _fleet(con)
    if not fleet:
        con.close()
        raise RuntimeError("no recorded history to take a fleet from: seed first")
    volume = _weekday_volume(con)

    start = (dt.date.fromisoformat(from_date) if from_date
             else dt.date.today() + dt.timedelta(days=1))
    dates = [(start + dt.timedelta(days=i)).isoformat() for i in range(days)]

    if replace:
        con.execute("DELETE FROM stops WHERE block_id IN "
                    "(SELECT block_id FROM blocks WHERE source='planned' "
                    " AND delivery_date >= ?)", (dates[0],))
        con.execute("DELETE FROM blocks WHERE source='planned' AND delivery_date >= ?",
                    (dates[0],))

    from .db import PARCEL_SIZES, POINT_TYPES
    from .seed import PROMISED

    made = stops_made = 0
    for date in dates:
        weekday = dt.date.fromisoformat(date).weekday()
        if weekday not in WORKDAYS:
            continue
        vol = volume.get(weekday, 1.0)
        for v in fleet:
            # deterministic per (date, vehicle): regenerating the month gives the
            # same month back, so a plan someone looked at yesterday is still
            # there today
            rng = random.Random(f"{date}|{v['vehicle_id']}")
            n = max(5, round(rng.gauss(v["stops_mean"] * vol, v["stops_sd"] * 0.6)))
            spread = (n / 3.1416) ** 0.5 * (rng.uniform(1.4, 3.6)
                                            if v["city"] != "London"
                                            else rng.uniform(0.7, 1.6))
            pts = _wedge(rng, v["lat"], v["lon"], n, spread)
            start_min = max(5 * 60, round(rng.gauss(v["start_mean"], v["start_sd"])))
            block_id = f"{date.replace('-', '')}-{v['vehicle_id']}-P"

            # Windows have to fit the shift, at both ends. A van leaving at two
            # cannot be given a morning promise; a van that is back by noon
            # cannot be given an evening one. Allowing either would make every
            # window figure meaningless - the round would miss promises no
            # routing could ever have kept, and the suggestions would all look
            # equally bad because they are all equally powerless.
            span = 30 + n * 9.0                      # a rough hour count for the day
            on_road = (start_min + 30, start_min + span)
            windows = [p for p in PROMISED
                       if to_minutes(p[2]) >= on_road[0] and to_minutes(p[1]) <= on_road[1]]
            windows = windows or [PROMISED[-1]]

            manifest = []
            for lat, lon in pts:
                pt = rng.choices(POINT_TYPES, weights=[.52, .24, .09, .10, .05])[0]
                ps = rng.choices(PARCEL_SIZES, weights=[.14, .26, .36, .18, .06])[0]
                _, pfrom, pto, _ = rng.choices(windows,
                                               weights=[p[3] for p in windows])[0]
                manifest.append((lat, lon, pt, ps, pfrom, pto))

            # A manifest is not a route. Work arrives grouped by promise and in
            # whatever order the system spat it out, so that is how the manifest
            # order is recorded.
            manifest.sort(key=lambda m: (to_minutes(m[5]), to_minutes(m[4]),
                                         rng.random()))
            manifest_pos = {m: i for i, m in enumerate(manifest, start=1)}

            # ...and then it is routed, because a planned round is a plan. The
            # stored order is a sensible drive - nearest neighbour with the
            # crossings taken out - and the manifest position is kept beside it,
            # so the suggestions on the day can be measured against both the
            # plan and the raw manifest rather than against nothing.
            plans = [StopPlan(i, m[0], m[1], m[2], m[3], None, None, m[4], m[5])
                     for i, m in enumerate(manifest, start=1)]
            routed = _two_opt(_nearest_neighbour(plans, (v["lat"], v["lon"])),
                              (v["lat"], v["lon"]))
            rows = []
            for i, p in enumerate(routed, start=1):
                m = manifest[p.sequence - 1]
                rows.append((block_id, i, manifest_pos[m], f"stop {manifest_pos[m]}",
                             None, p.lat, p.lon, p.point_type, p.parcel_size,
                             p.promised_from, p.promised_to))
            con.execute(
                "INSERT INTO blocks (block_id, delivery_date, vehicle_id, vehicle_type,"
                " depot_id, start_time, duration_minutes, source, note) "
                "VALUES (?,?,?,?,?,?,NULL,'planned',?)",
                (block_id, date, v["vehicle_id"], v["vehicle_type"], v["depot_id"],
                 hhmm(start_min), "planned from the fleet and volumes in the history"))
            con.executemany(
                "INSERT INTO stops (block_id, sequence, manifest_sequence, address,"
                " postcode, lat, lon, point_type, parcel_size, promised_from,"
                " promised_to) VALUES (?,?,?,?,?,?,?,?,?,?,?)", rows)
            made += 1
            stops_made += len(rows)

    con.commit()
    out = counts(con)
    con.close()
    out.update({"planned_written": made, "planned_stops": stops_made,
                "from": dates[0], "to": dates[-1],
                "vehicles": len(fleet),
                "weekday_volume": {d: round(volume.get(d, 1.0), 3) for d in range(7)}})
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=None)
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--from-date", dest="from_date", default=None,
                    help="default tomorrow")
    a = ap.parse_args()
    r = generate(a.db, a.days, a.from_date)
    print(f"planned {r['planned_written']} rounds ({r['planned_stops']:,} stops) "
          f"for {r['vehicles']} vehicles, {r['from']} to {r['to']}")
    print("weekday volume, measured:", r["weekday_volume"])
