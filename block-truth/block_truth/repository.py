"""
Reading and writing blocks.

The one rule this module exists to enforce: a block that was **recorded** gets a
ground truth, a block that was **entered** gets an estimate, and the two are
never mixed. `block_view` returns one or the other and says which in `kind`, so
nothing downstream has to guess.
"""
from __future__ import annotations

import datetime as dt
import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from .calibrate import Calibration, fit, fit_full
from .db import DEFAULT_DB, connect
from .itinerary import (MINUTES_PER_MISSED_WINDOW, Limits, compare, recommend, suggest)
from .traffic import KINDS, Conditions, Disruption, Traffic
from .estimate import StopPlan, estimate
from .fmt import to_minutes
from .geo import (ROAD_FACTOR, Projection, fit as fit_projection, haversine_km, leg_km,
                  nice_scale_bar)
from .model import Check, Stop
from .truth import TruthError, compute

# The drive home is the one piece of a round that no stop row records: it falls
# out as the residual, duration minus the work up to the last doorstep. That
# makes it the one place a wrong recorded duration can hide, because "parts sum
# to the duration" is an identity once the residual is defined that way, and an
# identity cannot fail.
#
# What the stop rows cannot say, the map can. We know where the last stop was and
# where the depot is, and - better than any assumed speed - we know how fast this
# vehicle drove all day, because its own legs are recorded against its own
# coordinates. So the leftover is judged against the round's own driving: a
# generous multiple of it, plus a fixed allowance for getting back onto the yard.
#
# The band is not a guess. Measured across the recorded history, the leg home
# takes 0.4 to 0.9 times the day's own minutes per kilometre - under 1, because
# the run home is one straight shot rather than a crawl between doors - with the
# middle at 0.8. The limits below sit far outside that, which is the point: this
# is meant to catch a duration wrong by an hour and never to fire on a slow day.
HOME_SLOWEST_FACTOR = 2.0   # observed worst is 0.91
HOME_FASTEST_FACTOR = 0.2   # observed best is 0.42
HOME_FIXED_MIN = 12.0       # gate, yard, keys, paperwork


def _homeward_check(return_leg_minutes: float, km_home: float,
                    km_driven: float, minutes_driven: float) -> Check:
    if km_home <= 0.05 or km_driven <= 0.5 or minutes_driven <= 1.0:
        return Check("the drive home is possible", True,
                     "too little driving recorded to judge the leg home", None)
    per_km = minutes_driven / km_driven
    expect = km_home * per_km
    lo = expect * HOME_FASTEST_FACTOR
    hi = expect * HOME_SLOWEST_FACTOR + HOME_FIXED_MIN
    ok = lo - 1.0 <= return_leg_minutes <= hi
    return Check(
        "the drive home is possible",
        ok,
        (f"{return_leg_minutes:.1f} min for the {km_home:.1f} km back to the depot; "
         f"this round drove {per_km:.2f} min/km all day, so {lo:.0f}-{hi:.0f} min"
         + ("" if ok else ". The recorded duration is the suspect: this leg is only "
                          "ever what is left over after the last doorstep")),
        round(return_leg_minutes - hi, 3) if return_leg_minutes > hi else None)


@dataclass
class Repository:
    path: Path = None
    _con: sqlite3.Connection = None
    _cal: Calibration = None
    _traffic: Traffic = None
    # what the last call had to leave out, so a caller can show it rather than
    # wonder why the hours do not add up
    last_problems: list = None

    def __post_init__(self):
        self.path = Path(self.path or DEFAULT_DB)
        self._con = connect(self.path)
        self.last_problems = []

    @property
    def con(self) -> sqlite3.Connection:
        return self._con

    def close(self):
        self._con.close()

    # ------------------------------------------------------------- calibration
    def calibration(self, refresh: bool = False) -> Calibration:
        if self._cal is None or refresh:
            self._cal, self._traffic = fit_full(con=self._con)
        return self._cal

    def traffic(self, refresh: bool = False) -> Traffic:
        self.calibration(refresh)
        return self._traffic

    def conditions(self, date: str | None = None, use_traffic: bool = True) -> Conditions:
        """The measured traffic and the declared disruptions, together."""
        c = Conditions(traffic=self.traffic(), disruptions=self.disruptions(),
                       use_traffic=use_traffic)
        return c.for_date(date) if date else c

    # ----------------------------------------------------------- disruptions
    def disruptions(self, date: str | None = None,
                    depot_id: str | None = None) -> list[Disruption]:
        try:
            rows = self._con.execute(
                "SELECT * FROM disruptions ORDER BY from_date, disruption_id").fetchall()
        except sqlite3.OperationalError:          # a database made before these
            return []
        out = [Disruption(
            disruption_id=r["disruption_id"], kind=r["kind"],
            description=r["description"], lat=r["lat"], lon=r["lon"],
            radius_km=r["radius_km"], from_date=r["from_date"], to_date=r["to_date"],
            from_time=r["from_time"], to_time=r["to_time"], severity=r["severity"],
            source=r["source"]) for r in rows
            if depot_id is None or r["depot_id"] in (None, depot_id)]
        return [d for d in out if date is None or d.active_on(date)]

    def add_disruption(self, kind: str, description: str, lat: float, lon: float,
                       radius_km: float, from_date: str, to_date: str,
                       from_time: str | None = None, to_time: str | None = None,
                       severity: float = 1.5, depot_id: str | None = None,
                       source: str = "manual",
                       disruption_id: str | None = None) -> str:
        if kind not in KINDS:
            raise TruthError(f"kind must be one of {', '.join(KINDS)}")
        if not description.strip():
            raise TruthError("a disruption needs a description: somebody has to be "
                             "able to tell what was declared and why")
        if not (0.05 <= radius_km <= 50):
            raise TruthError("radius must be between 0.05 and 50 km")
        if to_date < from_date:
            raise TruthError("the disruption ends before it starts")
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            raise TruthError("that is not a place on Earth")
        did = disruption_id or self._next_disruption_id()
        self._con.execute(
            "INSERT INTO disruptions (disruption_id, kind, description, lat, lon,"
            " radius_km, from_date, to_date, from_time, to_time, severity, depot_id,"
            " source) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (did, kind, description.strip(), lat, lon, radius_km, from_date, to_date,
             from_time or None, to_time or None, float(severity), depot_id, source))
        self._con.commit()
        return did

    def delete_disruption(self, disruption_id: str) -> None:
        cur = self._con.execute("DELETE FROM disruptions WHERE disruption_id=?",
                                (disruption_id,))
        if not cur.rowcount:
            raise TruthError(f"no disruption {disruption_id}")
        self._con.commit()

    def _next_disruption_id(self) -> str:
        n = self._con.execute("SELECT count(*) FROM disruptions").fetchone()[0]
        while True:
            did = f"DIS-{n + 1:04d}"
            if not self._con.execute("SELECT 1 FROM disruptions WHERE disruption_id=?",
                                     (did,)).fetchone():
                return did
            n += 1

    # ------------------------------------------------------------------ browse
    def depots(self) -> list[dict]:
        return [dict(r) for r in self._con.execute(
            "SELECT * FROM depots ORDER BY depot_id")]

    def dates(self, limit: int = 60) -> list[str]:
        return [r[0] for r in self._con.execute(
            "SELECT DISTINCT delivery_date FROM blocks ORDER BY 1 DESC LIMIT ?", (limit,))]

    def blocks(self, date: str | None = None, source: str | None = None,
               depot_id: str | None = None, limit: int = 500) -> list[dict]:
        where, args = [], []
        if date:
            where.append("b.delivery_date = ?"); args.append(date)
        if source:
            where.append("b.source = ?"); args.append(source)
        if depot_id:
            where.append("b.depot_id = ?"); args.append(depot_id)
        w = ("WHERE " + " AND ".join(where)) if where else ""
        rows = self._con.execute(
            f"SELECT b.*, d.city, d.region, "
            f"(SELECT count(*) FROM stops s WHERE s.block_id = b.block_id) AS stops "
            f"FROM blocks b JOIN depots d USING(depot_id) {w} "
            f"ORDER BY b.delivery_date DESC, b.block_id LIMIT ?", [*args, limit]).fetchall()
        return [dict(r) for r in rows]

    def block_row(self, block_id: str) -> dict:
        r = self._con.execute(
            "SELECT b.*, d.city, d.region, d.lat AS depot_lat, d.lon AS depot_lon, "
            "d.name AS depot_name FROM blocks b JOIN depots d USING(depot_id) "
            "WHERE b.block_id = ?", (block_id,)).fetchone()
        if not r:
            raise TruthError(f"no block {block_id}")
        return dict(r)

    def stop_rows(self, block_id: str) -> list[dict]:
        return [dict(r) for r in self._con.execute(
            "SELECT * FROM stops WHERE block_id = ? ORDER BY sequence", (block_id,))]

    # --------------------------------------------------- stops into a plan
    @staticmethod
    def plan_from(stops: list[dict], block_id: str = "") -> tuple[list, list[str]]:
        """Stored stops as a plan for the estimator, plus what had to be left out.

        A stop with no coordinates cannot be routed, measured or forecast: the
        distance arithmetic takes it and returns a TypeError from inside a
        trigonometric function, which surfaces as a 500 with a stack trace about
        radians and tells nobody which block is broken. That happened on a real
        database, so the check lives here, once, in front of every caller.

        Bad rows are skipped and named rather than silently dropped. A forecast
        made from eighty of a block's hundred stops is wrong by a fifth, and
        quietly - which is worse than an error.
        """
        plan, problems = [], []
        for s in stops:
            if s.get("lat") is None or s.get("lon") is None:
                problems.append(f"stop {s.get('sequence')} has no coordinates")
                continue
            try:
                lat, lon = float(s["lat"]), float(s["lon"])
            except (TypeError, ValueError):
                problems.append(f"stop {s.get('sequence')}: "
                                f"{s.get('lat')!r}, {s.get('lon')!r} are not coordinates")
                continue
            if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                problems.append(f"stop {s.get('sequence')}: "
                                f"{lat}, {lon} is not a place on Earth")
                continue
            plan.append(StopPlan(s["sequence"], lat, lon,
                                 s.get("point_type") or "House",
                                 s.get("parcel_size") or "Medium",
                                 s.get("address"), s.get("postcode"),
                                 s.get("promised_from"), s.get("promised_to"),
                                 s.get("manifest_sequence")))
        if problems and block_id:
            problems = [f"{block_id}: {p}" for p in problems]
        return plan, problems

    # ------------------------------------------------------- truth or estimate
    def block_view(self, block_id: str, confidence: int = 90,
                   use_traffic: bool = True) -> dict:
        """The one call the UI makes. Ground truth if it ran, an estimate if not."""
        b = self.block_row(block_id)
        stops = self.stop_rows(block_id)
        if not stops:
            raise TruthError(f"block {block_id} has no stops")

        recorded = (b["source"] == "recorded"
                    and all(s["arrival_time"] and s["service_minutes"] is not None
                            and s["travel_minutes"] is not None for s in stops))
        if recorded:
            t = compute(
                block_id=b["block_id"], delivery_date=b["delivery_date"],
                start_minutes=to_minutes(b["start_time"]),
                stops=[Stop(s["sequence"], to_minutes(s["arrival_time"]),
                            float(s["service_minutes"]), float(s["travel_minutes"]),
                            s["outcome"], s["address"]) for s in stops],
                duration_recorded=b["duration_minutes"],
                attributes={k: b[k] for k in ("vehicle_id", "vehicle_type", "depot_id",
                                              "city", "region", "source")})
            out = t.as_dict()
            out["kind"] = "truth"
            # judge the leftover against this round's own recorded driving
            path = [(b["depot_lat"], b["depot_lon"])] + [(s["lat"], s["lon"]) for s in stops]
            out["checks"].append(
                _homeward_check(
                    out["return_leg_minutes"],
                    haversine_km(stops[-1]["lat"], stops[-1]["lon"],
                                 b["depot_lat"], b["depot_lon"]) * ROAD_FACTOR,
                    sum(leg_km(path)) * ROAD_FACTOR,
                    out["travel_minutes"]).as_dict())
            out["ok"] = all(c["ok"] for c in out["checks"])
        else:
            cal = self.calibration()
            plan, bad = self.plan_from(stops, block_id)
            if not plan:
                raise TruthError(f"block {block_id} cannot be estimated: "
                                 + "; ".join(bad[:3]))
            e = estimate(
                block_id=b["block_id"], delivery_date=b["delivery_date"],
                start=b["start_time"], depot=(b["depot_lat"], b["depot_lon"]),
                stops=plan,
                cal=cal, city=b["city"], vehicle_type=b["vehicle_type"] or "Large Van",
                confidence=confidence,
                conditions=self.conditions(b["delivery_date"], use_traffic=use_traffic),
                wait_for_windows=self._waits(b),
                attributes={k: b[k] for k in ("vehicle_id", "vehicle_type", "depot_id",
                                              "city", "region", "source")})
            out = e.as_dict()

        out["block"] = b
        out["stops_detail"] = stops
        out["route"] = self.route(b, stops, out)
        return out

    # ---------------------------------------------------------------- the map
    def route(self, b: dict, stops: list[dict], view: dict) -> dict:
        """Everything the map needs: the path, projected, with each leg measured.

        The path is the order of visits drawn depot - stop - stop - ... - depot.
        It is a real measurement of where the vehicle went and in what order; it
        is not a road route, because there is no road network in the data.
        """
        depot = (b["depot_lat"], b["depot_lon"])
        if depot[0] is None or depot[1] is None:
            raise TruthError(f"depot {b['depot_id']} has no coordinates, so this round "
                             f"cannot be drawn")
        # a stop with no coordinates cannot be put on a map; it is dropped from
        # the drawing and counted, never quietly folded into the line
        skipped = [s["sequence"] for s in stops
                   if s["lat"] is None or s["lon"] is None]
        stops = [s for s in stops if s["lat"] is not None and s["lon"] is not None]
        if not stops:
            raise TruthError(f"block {b['block_id']} has no stop that can be placed "
                             f"on a map")
        pts = [depot] + [(s["lat"], s["lon"]) for s in stops] + [depot]
        proj = fit_projection(pts)
        bar_km, bar_px = nice_scale_bar(proj)

        est = {e["sequence"]: e for e in view.get("stop_estimates", [])}
        nodes, total_km = [], 0.0
        for i, s in enumerate(stops):
            km = haversine_km(*pts[i], s["lat"], s["lon"]) * ROAD_FACTOR
            total_km += km
            x, y = proj.xy(s["lat"], s["lon"])
            e = est.get(s["sequence"], {})
            nodes.append({
                "sequence": s["sequence"], "x": x, "y": y,
                "lat": s["lat"], "lon": s["lon"],
                "address": s["address"], "postcode": s["postcode"],
                "point_type": s["point_type"], "parcel_size": s["parcel_size"],
                "leg_km": round(km, 3),
                "arrival": s["arrival_time"] or e.get("arrival_time"),
                "arrival_min": e.get("arrival_min"), "arrival_max": e.get("arrival_max"),
                "service_minutes": (round(float(s["service_minutes"]), 2)
                                    if s["service_minutes"] is not None
                                    else e.get("service_minutes")),
                "travel_minutes": (round(float(s["travel_minutes"]), 2)
                                   if s["travel_minutes"] is not None
                                   else e.get("travel_minutes")),
                "outcome": s["outcome"],
                "window_from": s["promised_from"], "window_to": s["promised_to"],
                "on_time": e.get("on_time"), "at_risk": e.get("at_risk"),
            })
        home_km = haversine_km(*pts[-2], *depot) * ROAD_FACTOR
        total_km += home_km
        dx, dy = proj.xy(*depot)
        return {
            "depot": {"x": dx, "y": dy, "id": b["depot_id"], "name": b["depot_name"],
                      "lat": depot[0], "lon": depot[1]},
            "nodes": nodes,
            "return_km": round(home_km, 3),
            "total_km": round(total_km, 2),
            "width": proj.width, "height": proj.height,
            "scale_bar": {"km": bar_km, "px": bar_px},
            "road_factor": ROAD_FACTOR,
            "stops_without_coordinates": skipped,
        }

    # --------------------------------------------------------- the month ahead
    def schedule_days(self, days: int = 35, from_date: str | None = None) -> list[dict]:
        """One row per planned day: how much work, and how long it is predicted to take.

        The per-day totals are predictions, every one of them, and they are added
        up as predictions: expected hours sum, but the uncertainties do not -
        they combine in quadrature, because twenty rounds all overrunning
        together is far less likely than any one of them doing so.
        """
        start = from_date or dt.date.today().isoformat()
        rows = self._con.execute(
            "SELECT b.delivery_date, b.depot_id, b.block_id, b.vehicle_id, "
            "       b.vehicle_type, b.start_time, b.chosen_key, "
            "       d.city, d.lat dlat, d.lon dlon, "
            "       (SELECT count(*) FROM stops s WHERE s.block_id=b.block_id) stops "
            "FROM blocks b JOIN depots d USING(depot_id) "
            "WHERE b.source='planned' AND b.delivery_date >= ? "
            "ORDER BY b.delivery_date, b.block_id", (start,)).fetchall()
        problems: list[str] = []
        by_day: dict[str, dict] = {}
        for r in rows:
            d = by_day.setdefault(r["delivery_date"], {
                "date": r["delivery_date"],
                "weekday": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][
                    dt.date.fromisoformat(r["delivery_date"]).weekday()],
                "blocks": 0, "stops": 0, "vehicles": set(), "depots": set(),
                "expected_hours": 0.0, "var": 0.0, "disruptions": 0})
            d["blocks"] += 1
            d["stops"] += r["stops"]
            d["vehicles"].add(r["vehicle_id"])
            d["depots"].add(r["depot_id"])
        # predicting every block of every day is too slow to do on a page load,
        # so the day totals use the quick path and the detail view does it properly
        cal = self.calibration()
        for r in rows:
            day = by_day[r["delivery_date"]]
            if r["dlat"] is None or r["dlon"] is None:
                problems.append(f"{r['depot_id']}: the depot has no coordinates, so "
                                f"nothing starting from it can be measured")
                day["unusable"] = day.get("unusable", 0) + 1
                continue
            cond = self.conditions(r["delivery_date"])
            plan, bad = self.plan_from(self.stop_rows(r["block_id"]), r["block_id"])
            if bad:
                problems += bad
            if not plan:
                day["unusable"] = day.get("unusable", 0) + 1
                continue
            e = estimate(r["block_id"], r["delivery_date"], r["start_time"],
                         (r["dlat"], r["dlon"]), plan, cal, city=r["city"],
                         vehicle_type=r["vehicle_type"] or "Large Van",
                         conditions=cond, wait_for_windows=self._waits(r))
            day["expected_hours"] += e.expected_minutes / 60.0
            day["var"] += (e.sd_minutes / 60.0) ** 2
            day["disruptions"] += 1 if e.conditions["disruptions_hit"] else 0
        out = []
        for d in sorted(by_day.values(), key=lambda x: x["date"])[:days]:
            sd = math.sqrt(d["var"])
            out.append({**d, "unusable": d.get("unusable", 0),
                        "vehicles": len(d["vehicles"]), "depots": len(d["depots"]),
                        "expected_hours": round(d["expected_hours"], 1),
                        "sd_hours": round(sd, 2),
                        "hours_low": round(max(d["expected_hours"] - 1.645 * sd, 0), 1),
                        "hours_high": round(d["expected_hours"] + 1.645 * sd, 1),
                        "var": None})
        self.last_problems = problems
        return out

    def schedule(self, date: str | None = None, depot_id: str | None = None,
                 vehicle_id: str | None = None, city: str | None = None,
                 region: str | None = None, confidence: int = 90,
                 use_traffic: bool = True, limit: int = 300,
                 limits: Limits | None = None) -> list[dict]:
        """Planned rounds with their predictions, filtered the way a planner filters."""
        where = ["b.source='planned'"]
        args: list = []
        if date:
            where.append("b.delivery_date = ?"); args.append(date)
        else:
            where.append("b.delivery_date >= ?"); args.append(dt.date.today().isoformat())
        for col, val in (("b.depot_id", depot_id), ("b.vehicle_id", vehicle_id),
                         ("d.city", city), ("d.region", region)):
            if val:
                where.append(f"{col} = ?"); args.append(val)
        rows = self._con.execute(
            f"SELECT b.*, d.city, d.region, d.lat depot_lat, d.lon depot_lon, "
            f"(SELECT count(*) FROM stops s WHERE s.block_id=b.block_id) stops "
            f"FROM blocks b JOIN depots d USING(depot_id) "
            f"WHERE {' AND '.join(where)} "
            f"ORDER BY b.delivery_date, b.block_id LIMIT ?", [*args, limit]).fetchall()
        cal = self.calibration()
        limits = limits or Limits()
        self.last_problems = []
        out = []
        for b in rows:
            if b["depot_lat"] is None or b["depot_lon"] is None:
                self.last_problems.append(f"{b['depot_id']}: the depot has no "
                                          f"coordinates")
                continue
            plan, bad = self.plan_from(self.stop_rows(b["block_id"]), b["block_id"])
            if bad:
                self.last_problems += bad
            if not plan:
                continue
            e = estimate(b["block_id"], b["delivery_date"], b["start_time"],
                         (b["depot_lat"], b["depot_lon"]), plan, cal,
                         city=b["city"], vehicle_type=b["vehicle_type"] or "Large Van",
                         confidence=confidence,
                         conditions=self.conditions(b["delivery_date"], use_traffic),
                         wait_for_windows=self._waits(b),
                         attributes={k: b[k] for k in ("vehicle_id", "vehicle_type",
                                                       "depot_id", "city", "region",
                                                       "source")})
            d = e.as_dict()
            d["breaches"] = limits.breaches(e)
            d["chosen_key"] = b["chosen_key"]
            d["chosen_by"] = b["chosen_by"]
            d["stops"] = len(plan)
            d["weekday"] = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][
                dt.date.fromisoformat(b["delivery_date"]).weekday()]
            out.append(d)
        return out

    def choose_itinerary(self, block_id: str, key: str, by: str = "operator",
                         confidence: int = 90, use_traffic: bool = True,
                         limits: Limits | None = None) -> dict:
        """Drive this one. Applies the order to the round and records who said so.

        The choice is applied by rewriting the stops' visiting order rather than
        stored alongside it, so there is one answer to "what order is this round
        in" instead of two that can drift apart. `manifest_sequence` is left
        alone, which is what makes the choice reversible and keeps the
        alternatives offerable afterwards.
        """
        b = self.block_row(block_id)
        if b["source"] == "recorded":
            raise TruthError("this round has already run: its order is a record of "
                             "what happened, not a plan to be changed")
        d = self.itineraries(block_id, confidence=confidence, use_traffic=use_traffic,
                             limits=limits)
        opt = next((o for o in d["options"] if o["key"] == key), None)
        if opt is None:
            offered = ", ".join(o["key"] for o in d["options"])
            raise TruthError(f"no option {key!r} for this round; it offers: {offered}")

        stops = {s["sequence"]: s for s in self.stop_rows(block_id)}
        # two passes through a unique index: park the rows out of the way first,
        # or renumbering walks into a stop that is already using the number
        self._con.execute("UPDATE stops SET sequence = -sequence WHERE block_id=?",
                          (block_id,))
        for position, was in enumerate(opt["order"], start=1):
            if was not in stops:
                self._con.rollback()
                raise TruthError(f"the chosen order names a stop {was} that is not "
                                 f"on this round")
            self._con.execute(
                "UPDATE stops SET sequence=? WHERE block_id=? AND sequence=?",
                (position, block_id, -was))
        left = self._con.execute(
            "SELECT count(*) FROM stops WHERE block_id=? AND sequence < 0",
            (block_id,)).fetchone()[0]
        if left:
            self._con.rollback()
            raise TruthError(f"the chosen order left {left} stops unplaced")
        self._con.execute(
            "UPDATE blocks SET chosen_key=?, chosen_by=?, chosen_at=datetime('now') "
            "WHERE block_id=?", (key, by, block_id))
        self._con.commit()
        return self.itineraries(block_id, confidence=confidence,
                                use_traffic=use_traffic, limits=limits)

    def clear_choice(self, block_id: str) -> dict:
        """Put the round back in the order the work arrived in."""
        b = self.block_row(block_id)
        if b["source"] == "recorded":
            raise TruthError("a recorded round has no manifest to go back to")
        stops = self.stop_rows(block_id)
        if any(s["manifest_sequence"] is None for s in stops):
            raise TruthError("this round has no manifest order on record")
        self._con.execute("UPDATE stops SET sequence = -sequence WHERE block_id=?",
                          (block_id,))
        for s in stops:
            self._con.execute(
                "UPDATE stops SET sequence=? WHERE block_id=? AND sequence=?",
                (s["manifest_sequence"], block_id, -s["sequence"]))
        self._con.execute(
            "UPDATE blocks SET chosen_key='as_received', chosen_by='reset', "
            "chosen_at=datetime('now') WHERE block_id=?", (block_id,))
        self._con.commit()
        return {"block_id": block_id, "restored": "the manifest order"}

    @staticmethod
    def _waits(block: dict) -> bool:
        """Does this round wait at a door for a window to open?

        It does if that is what the operator chose. The key is the record of the
        policy as well as of the order, so every later forecast of this round -
        on the schedule, on the day totals, in its own detail - charges for the
        waiting the operator decided to spend.
        """
        try:
            key = block["chosen_key"]          # a dict or a sqlite3.Row
        except (KeyError, IndexError):
            key = None
        return key == "windows"

    def itineraries(self, block_id: str, confidence: int = 90,
                    use_traffic: bool = True,
                    limits: Limits | None = None,
                    minutes_per_missed_window: float | None = None) -> dict:
        """Several ways to drive one planned round, each costed and drawn."""
        b = self.block_row(block_id)
        stops = self.stop_rows(block_id)
        if not stops:
            raise TruthError(f"block {block_id} has no stops")
        if b["source"] == "recorded":
            raise TruthError("this round has already run: there is nothing to route. "
                             "Its ground truth is on the Blocks tab.")
        if b["depot_lat"] is None or b["depot_lon"] is None:
            raise TruthError(f"depot {b['depot_id']} has no coordinates, so this round "
                             f"cannot be routed from it")
        plan, bad = self.plan_from(stops, block_id)
        if not plan:
            raise TruthError(f"block {block_id} has no stop with usable coordinates, "
                             f"so there is nothing to route: " + "; ".join(bad[:3]))
        cond = self.conditions(b["delivery_date"], use_traffic)
        items = suggest(block_id, b["delivery_date"], b["start_time"],
                        (b["depot_lat"], b["depot_lon"]), plan, self.calibration(),
                        city=b["city"], vehicle_type=b["vehicle_type"] or "Large Van",
                        confidence=confidence, conditions=cond,
                        limits=limits or Limits(), plan_waits=self._waits(b))
        by_seq = {s["sequence"]: s for s in stops}
        options = []
        for it in items:
            ordered = []
            for pos, seq in enumerate(it.order, start=1):
                s = dict(by_seq[seq])
                s["plan_sequence"] = seq       # where it sits in the stored round
                s["sequence"] = pos            # where this option visits it
                ordered.append(s)              # manifest_sequence is left alone
            d = it.as_dict()
            d["route"] = self.route(b, ordered, d)
            d["stops_detail"] = ordered
            options.append(d)
        rec = recommend(items, minutes_per_missed_window
                        if minutes_per_missed_window is not None
                        else MINUTES_PER_MISSED_WINDOW)
        chosen = b.get("chosen_key")
        for o in options:
            o["recommended"] = o["key"] == rec.get("key")
            # A choice is recorded against the *order*, and "as planned" is
            # whatever order the round is in now - so that is the option the
            # operator's choice currently points at, whatever it was called when
            # they made it.
            o["chosen"] = (o["key"] == "as_planned" and chosen is not None)
        # Is the choice the recommended one? Compare the *drives*, not the names:
        # once a choice is applied it is the round's own order, so it comes back
        # as "as planned" whatever it was called when it was picked, and
        # comparing the stored key with the recommendation's would call a
        # recommended choice unrecommended.
        chosen_opt = next((o for o in options if o["chosen"]), None)
        return {"block": b, "options": options, "compare": compare(items),
                "recommendation": rec,
                "chosen": {"key": chosen, "by": b.get("chosen_by"),
                           "at": b.get("chosen_at"),
                           "is_recommended": bool(chosen_opt
                                                  and chosen_opt["recommended"])},
                "conditions": cond.as_dict(),
                "limits": (limits or Limits()).as_dict()}

    # ------------------------------------------------------------------ writes
    def create_block(self, block_id: str, delivery_date: str, depot_id: str,
                     stops: list[dict], start_time: str = "07:30",
                     vehicle_id: str | None = None, vehicle_type: str = "Large Van",
                     note: str | None = None, source: str = "manual") -> str:
        if not stops:
            raise TruthError("a block needs at least one stop")
        if not self._con.execute("SELECT 1 FROM depots WHERE depot_id=?",
                                 (depot_id,)).fetchone():
            raise TruthError(f"no depot {depot_id}")
        if self._con.execute("SELECT 1 FROM blocks WHERE block_id=?",
                             (block_id,)).fetchone():
            raise TruthError(f"block {block_id} already exists")
        to_minutes(start_time)                       # raises if it is not a time

        with self._con:
            self._con.execute(
                "INSERT INTO blocks (block_id, delivery_date, vehicle_id, vehicle_type,"
                " depot_id, start_time, source, note) VALUES (?,?,?,?,?,?,?,?)",
                (block_id, delivery_date, vehicle_id, vehicle_type, depot_id,
                 start_time, source, note))
            self._con.executemany(
                "INSERT INTO stops (block_id, sequence, address, postcode, lat, lon,"
                " point_type, parcel_size, promised_from, promised_to) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                [(block_id, i, s.get("address"), s.get("postcode"), s["lat"], s["lon"],
                  s.get("point_type", "House"), s.get("parcel_size", "Medium"),
                  s.get("promised_from"), s.get("promised_to"))
                 for i, s in enumerate(sorted(stops, key=lambda x: x["sequence"]), 1)])
        return block_id

    def delete_block(self, block_id: str) -> None:
        b = self.block_row(block_id)
        if b["source"] == "recorded":
            raise TruthError("recorded blocks are history and are not deleted here")
        with self._con:
            self._con.execute("DELETE FROM stops WHERE block_id=?", (block_id,))
            self._con.execute("DELETE FROM blocks WHERE block_id=?", (block_id,))

    def suggest_block_id(self, date: str, depot_id: str) -> str:
        base = f"{date.replace('-', '')}-{depot_id}-N"
        n = 1
        while self._con.execute("SELECT 1 FROM blocks WHERE block_id=?",
                                (f"{base}{n:02d}",)).fetchone():
            n += 1
        return f"{base}{n:02d}"
