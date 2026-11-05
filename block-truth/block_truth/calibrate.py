"""
Fit the estimator to the rounds that have already happened.

Nothing here is a constant someone chose. Every number is read back out of the
recorded blocks, which is why the seed builds its history geometry-first: there
is a real speed in those records, so a regression can find it.

What gets fitted
----------------

* **service time** per (point type, parcel size), as a mean and a variance, with
  failed stops kept separate because a failed stop is a short stop.
* **travel**, by regressing each leg's recorded minutes on the road distance
  implied by the coordinates of its two ends. The slope is minutes per kilometre
  - the reciprocal of speed - and the intercept is what parking and walking cost
  regardless of distance.
* **depot time** per vehicle type and the **return leg** per kilometre.
* **common mode**: the part of a day that misses together. Traffic, weather, who
  turned up. It is the reason min and max cannot be built by adding up per-stop
  extremes, and it is measured, not assumed - see `_common_mode`.
"""
from __future__ import annotations

import json
import math
import statistics as st
from dataclasses import dataclass, field, asdict

from .db import connect
from .fmt import to_minutes
from .geo import ROAD_FACTOR, haversine_km

MIN_SAMPLES = 25            # below this a cell falls back to its margin
Z = {80: 1.2816, 90: 1.6449, 95: 1.9600, 99: 2.5758}


@dataclass
class Dist:
    """A fitted mean and spread, and how much evidence is behind it."""
    mean: float
    sd: float
    n: int

    def as_dict(self):
        return {"mean": round(self.mean, 4), "sd": round(self.sd, 4), "n": self.n}


@dataclass
class Calibration:
    service: dict[str, Dist] = field(default_factory=dict)     # "House|Medium"
    # a prior, so a database with no recorded rounds yet still estimates rather
    # than crashing; `blocks_used = 0` on the result says the answer rests on it
    service_failed: Dist = field(default_factory=lambda: Dist(1.4, 0.4, 0))
    failure_rate: dict[str, float] = field(default_factory=dict)
    travel_per_km: dict[str, Dist] = field(default_factory=dict)   # per city
    travel_fixed: dict[str, float] = field(default_factory=dict)   # per city, minutes
    travel_resid_sd: dict[str, float] = field(default_factory=dict)
    depot: dict[str, Dist] = field(default_factory=dict)           # per vehicle type
    return_per_km: float = 2.5
    # The regression that gives the per-kilometre figure has an intercept too,
    # and it is not zero: coming off the road, through the gate, across the yard
    # costs the same few minutes whether the last stop was near or far. Dropping
    # it - which this did until it was measured - makes every single round come
    # out about three minutes short.
    return_fixed: float = 0.0
    return_sd: float = 2.0
    common_mode: float = 0.08
    common_mode_flat: float | None = None   # the same figure before traffic was fitted
    traffic_aware: bool = False
    blocks_used: int = 0
    stops_used: int = 0
    notes: list[str] = field(default_factory=list)
    # the per-block material the common-mode fit works on, kept so it can be
    # redone once traffic is known; not part of the published calibration
    _parts: list = field(default_factory=list, repr=False)

    # ------------------------------------------------------------------ lookups
    def service_for(self, point_type: str, parcel_size: str) -> Dist:
        for key in (f"{point_type}|{parcel_size}", f"{point_type}|*", "*|*"):
            if key in self.service:
                return self.service[key]
        return Dist(2.5, 1.0, 0)

    def travel_for(self, city: str) -> tuple[Dist, float, float]:
        d = self.travel_per_km.get(city) or self.travel_per_km.get("*") \
            or Dist(3.0, 0.6, 0)
        fixed = self.travel_fixed.get(city, self.travel_fixed.get("*", 1.0))
        sd = self.travel_resid_sd.get(city, self.travel_resid_sd.get("*", 1.2))
        return d, fixed, sd

    def depot_for(self, vehicle_type: str) -> Dist:
        return self.depot.get(vehicle_type) or self.depot.get("*") or Dist(15.0, 3.0, 0)

    def as_dict(self):
        return {
            "service": {k: v.as_dict() for k, v in self.service.items()},
            "service_failed": self.service_failed.as_dict(),
            "failure_rate": {k: round(v, 4) for k, v in self.failure_rate.items()},
            "travel_per_km": {k: v.as_dict() for k, v in self.travel_per_km.items()},
            "travel_fixed": {k: round(v, 3) for k, v in self.travel_fixed.items()},
            "travel_resid_sd": {k: round(v, 3) for k, v in self.travel_resid_sd.items()},
            "depot": {k: v.as_dict() for k, v in self.depot.items()},
            "return_per_km": round(self.return_per_km, 3),
            "return_fixed": round(self.return_fixed, 3),
            "return_sd": round(self.return_sd, 3),
            "common_mode": round(self.common_mode, 4),
            "common_mode_flat": (round(self.common_mode_flat, 4)
                                 if self.common_mode_flat is not None else None),
            "traffic_aware": self.traffic_aware,
            "blocks_used": self.blocks_used, "stops_used": self.stops_used,
            "road_factor": ROAD_FACTOR,
            "notes": self.notes,
        }


def _sd(values, floor=0.05):
    return max(st.stdev(values), floor) if len(values) > 1 else floor


def _ols(xs, ys):
    """Slope and intercept, plus the residual spread around the line."""
    n = len(xs)
    if n < 3:
        return 0.0, (st.fmean(ys) if ys else 0.0), 0.0
    mx, my = st.fmean(xs), st.fmean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 1e-12:
        return 0.0, my, _sd(ys)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    inter = my - slope * mx
    resid = [y - (slope * x + inter) for x, y in zip(xs, ys)]
    return slope, inter, _sd(resid)


def fit(db_path=None, con=None) -> Calibration:
    own = con is None
    con = con or connect(db_path, read_only=True)
    cal = Calibration()

    blocks = con.execute(
        "SELECT b.*, d.city FROM blocks b JOIN depots d USING(depot_id) "
        "WHERE b.source='recorded' AND b.duration_minutes IS NOT NULL").fetchall()
    if not blocks:
        cal.notes.append("no recorded blocks: falling back to priors throughout")
        if own:
            con.close()
        return cal

    depot_xy = {r["depot_id"]: (r["lat"], r["lon"])
                for r in con.execute("SELECT * FROM depots")}

    svc_cells, svc_pt, svc_fail = {}, {}, []
    fail_n, fail_d = {}, {}
    legs = {}                       # city -> ([km], [minutes])
    depot_by_vt = {}
    ret_km, ret_min = [], []
    block_mean_parts = []           # for the common-mode fit

    for b in blocks:
        stops = con.execute(
            "SELECT * FROM stops WHERE block_id=? ORDER BY sequence",
            (b["block_id"],)).fetchall()
        if len(stops) < 2:
            continue
        city = b["city"]
        prev = depot_xy[b["depot_id"]]
        svc_total = trv_total = 0.0
        for s in stops:
            km = haversine_km(prev[0], prev[1], s["lat"], s["lon"]) * ROAD_FACTOR
            if s["travel_minutes"] is not None:
                legs.setdefault(city, ([], []))
                legs[city][0].append(km)
                legs[city][1].append(float(s["travel_minutes"]))
                trv_total += float(s["travel_minutes"])
            if s["service_minutes"] is not None:
                v = float(s["service_minutes"])
                svc_total += v
                pt, ps = s["point_type"], s["parcel_size"]
                fail_d[pt] = fail_d.get(pt, 0) + 1
                if s["outcome"] == "Unsuccessful":
                    svc_fail.append(v)
                    fail_n[pt] = fail_n.get(pt, 0) + 1
                else:
                    svc_cells.setdefault(f"{pt}|{ps}", []).append(v)
                    svc_pt.setdefault(pt, []).append(v)
            prev = (s["lat"], s["lon"])

        # the depot time is what the first arrival has that its own leg does not
        first = stops[0]
        if first["arrival_time"] and first["travel_minutes"] is not None:
            load = (to_minutes(first["arrival_time"]) - to_minutes(b["start_time"])
                    - float(first["travel_minutes"]))
            if -2 < load < 90:
                depot_by_vt.setdefault(b["vehicle_type"] or "*", []).append(load)

        # the return leg is the residual, and it should scale with the way home
        last = stops[-1]
        if last["arrival_time"] and last["service_minutes"] is not None:
            work = (to_minutes(last["arrival_time"]) + float(last["service_minutes"])
                    - to_minutes(b["start_time"]))
            leg = float(b["duration_minutes"]) - work
            home = haversine_km(last["lat"], last["lon"], *depot_xy[b["depot_id"]]) \
                * ROAD_FACTOR
            if 0 <= leg < 120 and home > 0.05:
                ret_km.append(home); ret_min.append(leg)

        block_mean_parts.append({
            "city": city, "vehicle_type": b["vehicle_type"] or "*",
            "depot": depot_xy[b["depot_id"]], "n": len(stops),
            "duration": float(b["duration_minutes"]),
            "date": b["delivery_date"], "start": b["start_time"],
            "stops": [(r["lat"], r["lon"], r["point_type"], r["parcel_size"])
                      for r in stops],
        })

    # --- service ---------------------------------------------------------------
    for key, vals in svc_cells.items():
        if len(vals) >= MIN_SAMPLES:
            cal.service[key] = Dist(st.fmean(vals), _sd(vals), len(vals))
    for pt, vals in svc_pt.items():
        cal.service[f"{pt}|*"] = Dist(st.fmean(vals), _sd(vals), len(vals))
    allv = [v for vals in svc_pt.values() for v in vals]
    if allv:
        cal.service["*|*"] = Dist(st.fmean(allv), _sd(allv), len(allv))
    cal.service_failed = Dist(st.fmean(svc_fail), _sd(svc_fail), len(svc_fail)) \
        if svc_fail else Dist(1.4, 0.4, 0)
    for pt, d in fail_d.items():
        cal.failure_rate[pt] = fail_n.get(pt, 0) / d
    if fail_d:
        cal.failure_rate["*"] = sum(fail_n.values()) / sum(fail_d.values())

    # --- travel: minutes against kilometres ------------------------------------
    all_km, all_min = [], []
    for city, (km, mins) in legs.items():
        all_km += km; all_min += mins
        if len(km) >= MIN_SAMPLES:
            slope, inter, sd = _ols(km, mins)
            cal.travel_per_km[city] = Dist(slope, sd, len(km))
            cal.travel_fixed[city] = max(inter, 0.0)
            cal.travel_resid_sd[city] = sd
    if all_km:
        slope, inter, sd = _ols(all_km, all_min)
        cal.travel_per_km["*"] = Dist(slope, sd, len(all_km))
        cal.travel_fixed["*"] = max(inter, 0.0)
        cal.travel_resid_sd["*"] = sd

    # --- depot and the way home ------------------------------------------------
    for vt, vals in depot_by_vt.items():
        cal.depot[vt] = Dist(st.fmean(vals), _sd(vals), len(vals))
    flat = [v for vals in depot_by_vt.values() for v in vals]
    if flat:
        cal.depot["*"] = Dist(st.fmean(flat), _sd(flat), len(flat))
    if ret_km:
        slope, inter, sd = _ols(ret_km, ret_min)
        cal.return_per_km = max(slope, 0.1)
        cal.return_fixed = max(inter, 0.0)
        cal.return_sd = sd

    cal.blocks_used = len(blocks)
    cal.stops_used = sum(p["n"] for p in block_mean_parts)
    cal.common_mode = _common_mode(cal, block_mean_parts)
    cal._parts = block_mean_parts          # kept so fit_full can redo it with traffic
    if own:
        con.close()
    return cal


def fit_full(db_path=None, con=None):
    """The whole calibration, in the order the stages depend on each other.

    1. the components, from the recorded legs and doors
    2. the traffic curve, which needs the components to know what a leg "should"
       have taken before it can say the morning is slow
    3. the common mode again, now that traffic is explained - because whatever
       the clock accounts for is no longer "something that happened to the day",
       and leaving it in would quote every round with a wider interval than its
       own evidence supports

    Returns the calibration and the traffic model together: they are fitted on
    the same rounds and they are only honest as a pair.
    """
    from .traffic import Conditions, fit_traffic

    own = con is None
    con = con or connect(db_path, read_only=True)
    try:
        cal = fit(con=con)
        traffic = fit_traffic(con, cal)
        parts = cal._parts
        if traffic.legs_used:
            refit_travel_under_traffic(con, cal, traffic)
        if traffic.legs_used and len(parts) >= 40:
            # Cross-fitted, and it matters. Measuring the leftover with a traffic
            # curve fitted on the very same days flatters the curve: it has
            # already chased some of those days' noise, so the leftover looks
            # smaller than it is and every interval comes out too narrow. Two
            # folds by date - fit the curve on one half, measure the leftover on
            # the other, then swap - and the curve never grades its own homework.
            dates = sorted({p.get("date") for p in parts if p.get("date")})
            half = len(dates) // 2
            folds = [(set(dates[:half]), set(dates[half:])),
                     (set(dates[half:]), set(dates[:half]))]
            shares = []
            for train, test in folds:
                if not train or not test:
                    continue
                t_fold = fit_traffic(con, cal, dates=train)
                held = [p for p in parts if p.get("date") in test]
                if not held or not t_fold.legs_used:
                    continue
                _, sh = _shares(cal, held,
                                Conditions(traffic=t_fold, use_traffic=True))
                shares += sh
            if len(shares) >= 20:
                cal.common_mode_flat = cal.common_mode
                cal.common_mode = _cm_from(shares)
                cal.traffic_aware = True
                cal.notes.append(
                    f"with the measured traffic in the forecast, the part common to "
                    f"the whole day falls from {100 * cal.common_mode_flat:.1f}% to "
                    f"{100 * cal.common_mode:.1f}% - measured on days the traffic "
                    f"curve was not fitted on")
        return cal, traffic
    finally:
        if own:
            con.close()


def refit_travel_under_traffic(con, cal: Calibration, traffic) -> None:
    """Refit the leg regression with the clock in it, in place.

    Every stage here has to avoid re-counting what another stage explains, and
    this is where that bites hardest. The first pass regressed minutes on
    kilometres across the whole day, so the spread left around that line
    includes the morning peak and the evening one - the very thing the traffic
    curve now accounts for. Leave it and a city with sharp peaks is charged for
    them twice: once in its curve and again in its per-leg spread, and its
    rounds get quoted far wider than their own evidence supports.

    So regress the same legs on *kilometres times the factor for the hour they
    were driven*. The slope is unchanged in meaning, and what is left over is
    what the clock could not explain.
    """
    from .traffic import MIN_EXPECTED_MIN

    rows = con.execute(
        "SELECT b.delivery_date, d.city, d.lat dlat, d.lon dlon, s.block_id, "
        "       s.sequence, s.lat, s.lon, s.arrival_time, s.travel_minutes "
        "FROM stops s JOIN blocks b USING(block_id) JOIN depots d USING(depot_id) "
        "WHERE b.source='recorded' AND s.travel_minutes IS NOT NULL "
        "  AND s.arrival_time IS NOT NULL ORDER BY s.block_id, s.sequence").fetchall()
    legs: dict[str, tuple[list, list]] = {}
    prev_block = prev_xy = None
    for r in rows:
        if r["block_id"] != prev_block:
            prev_block, prev_xy = r["block_id"], (r["dlat"], r["dlon"])
        km = haversine_km(prev_xy[0], prev_xy[1], r["lat"], r["lon"]) * ROAD_FACTOR
        prev_xy = (r["lat"], r["lon"])
        actual = float(r["travel_minutes"])
        mid = to_minutes(r["arrival_time"]) - actual / 2.0
        f = traffic.factor(mid, r["delivery_date"], r["city"])
        legs.setdefault(r["city"], ([], []))
        legs[r["city"]][0].append(km * f)
        legs[r["city"]][1].append(actual)

    all_x, all_y = [], []
    for city, (xs, ys) in legs.items():
        all_x += xs; all_y += ys
        if len(xs) >= MIN_SAMPLES:
            slope, inter, sd = _ols(xs, ys)
            if slope > 0:
                cal.travel_per_km[city] = Dist(slope, sd, len(xs))
                cal.travel_fixed[city] = max(inter, 0.0)
                cal.travel_resid_sd[city] = sd
    if all_x:
        slope, inter, sd = _ols(all_x, all_y)
        if slope > 0:
            cal.travel_per_km["*"] = Dist(slope, sd, len(all_x))
            cal.travel_fixed["*"] = max(inter, 0.0)
            cal.travel_resid_sd["*"] = sd
    _ = MIN_EXPECTED_MIN            # documented above; the filter lives in the fit


def _shares(cal: Calibration, parts, conditions=None):
    """Per block: the relative error, and what it implies for the common mode.

    If a block's error has variance indep^2 + (cm * mean)^2, then
    (err^2 - indep^2) / mean^2 is an unbiased single-block estimate of cm^2.
    Averaging those over the blocks solves for cm with no approximation of the
    independent part and no fudge factor.
    """
    from .estimate import StopPlan, predict_parts

    rel, sh = [], []
    for p in parts:
        plan = [StopPlan(i, lat, lon, pt, ps)
                for i, (lat, lon, pt, ps) in enumerate(p["stops"], start=1)]
        mean, indep = predict_parts(plan, p["depot"], cal, p["city"], p["vehicle_type"],
                                    conditions=conditions,
                                    start=p.get("start", 450.0),
                                    date=p.get("date", "2000-01-01"))
        if mean <= 0:
            continue
        err = p["duration"] - mean
        rel.append(err / mean)
        sh.append((err ** 2 - indep ** 2) / (mean ** 2))
    return rel, sh


def _cm_from(shares) -> float:
    cm2 = st.fmean(shares) if shares else 0.0
    return min(math.sqrt(cm2), 0.35) if cm2 > 0 else 0.02


def _common_mode(cal: Calibration, parts, conditions=None) -> float:
    """How much of a round's error is shared by the whole round.

    A second pass: predict every recorded block from its geometry alone - its
    coordinates, its door types, its vehicle - with the components just fitted,
    and look at the errors. Independent per-stop noise shrinks as sqrt(N) and
    the predicted spread says by how much. Whatever the errors have left over
    does not shrink, because it is one thing happening to the whole day: a slow
    morning, a wet Tuesday, a driver who does not know the patch.

    That leftover, as a fraction of the duration, is what keeps a forty-stop
    round's interval from collapsing to nothing.
    """
    from .estimate import StopPlan, predict_parts        # late: estimate needs us

    if len(parts) < 10:
        cal.notes.append("too few blocks to measure common mode: using 8%")
        return 0.08

    # Method of moments against the model's own variance. For each block the
    # error should have variance indep_sd^2 + (cm * mean)^2, so averaging
    # (err^2 - indep_sd^2) / mean^2 over the blocks solves for cm^2 directly -
    # no approximation of the independent part, and no fudge factor.
    rel_err, shares = [], []
    for p in parts:
        plan = [StopPlan(i, lat, lon, pt, ps)
                for i, (lat, lon, pt, ps) in enumerate(p["stops"], start=1)]
        mean, indep = predict_parts(plan, p["depot"], cal, p["city"], p["vehicle_type"],
                                    conditions=conditions,
                                    start=p.get("start", 450.0),
                                    date=p.get("date", "2000-01-01"))
        if mean <= 0:
            continue
        err = p["duration"] - mean
        rel_err.append(err / mean)
        shares.append((err ** 2 - indep ** 2) / (mean ** 2))

    if len(shares) < 10:
        return 0.08
    bias = st.fmean(rel_err)
    if abs(bias) > 0.02:
        cal.notes.append(f"the fitted components run {100 * bias:+.1f}% against "
                         f"recorded durations on average")
    cm2 = st.fmean(shares)
    total = _sd(rel_err)
    if cm2 <= 0:
        cal.notes.append("per-stop noise alone already covers the spread between "
                         "blocks, so almost nothing is common to the whole round")
        return 0.02
    cm = min(math.sqrt(cm2), 0.35)
    cal.notes.append(f"errors spread {100 * total:.1f}% of duration; after the part "
                     f"that cancels across stops, {100 * cm:.1f}% is common to the "
                     f"whole round and does not shrink with its length")
    return cm


def save(cal: Calibration, path) -> None:
    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(cal.as_dict(), indent=2))


if __name__ == "__main__":
    c = fit()
    print(json.dumps(c.as_dict(), indent=2)[:2600])
