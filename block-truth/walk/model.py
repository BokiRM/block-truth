"""
"Should take": how long a block ought to take, fitted to how long it has taken.

The shape of the fit mirrors the shape of the job, because a walk is not one
number with a trend on it - it is a pace, a set of doors, and whatever the day
is doing:

    traversal = length / pace  +  path walking / pace  +  time at the doors

so the model is estimated in that structure rather than as a regression on
whatever columns happen to exist. Three things come out of that choice:

**Pace is identified separately.** A street with nothing to deliver on it is
walked, not worked, so its traversal time is length over pace and nothing else.
Those dead passes - a quarter of all passes - measure pace directly, with no
service time confounded into it. Everything the serviced passes have *above*
that is time at the doors.

**The door model is per kind of door, not per block.** A block of flats and a
terrace are different jobs; so are a parcel and a letter. Those coefficients are
what let the forecast answer "what if this round gets another fifty parcels",
which is the question resourcing actually asks in November.

**What the day does is multiplicative and measured.** Weather, darkness, a
relief walker on somebody else's block, a walker in their first month: each is a
factor fitted from the ratio of what happened to what the structure predicted,
shrunk toward 1 by how much evidence stands behind it. Same device as the
traffic curve in the driven model, for the same reason - an effect with eleven
observations behind it should not move a forecast.

Intervals come out of the same variance argument as everywhere else in this
app: per-segment noise that cancels across a block, plus a per-day common mode
that does not, measured rather than assumed, and checked by its coverage on a
year the fit never saw.
"""
from __future__ import annotations

import datetime as dt
import json
import math
import statistics as st
from dataclasses import dataclass, field

import numpy as np

from .schema import connect

SHRINK_K = 40.0          # observations before a factor carries its own weight
MIN_CELL = 30
Z = {80: 1.2816, 90: 1.6449, 95: 1.9600, 99: 2.5758}

ITEM_KINDS = ["letters", "packets", "parcels", "signed", "special"]
DP_KINDS = ["terrace", "house", "flat", "business", "farm", "locker"]


@dataclass
class WalkModel:
    """Everything fitted, with the evidence behind each piece."""
    pace_mps: dict[str, float] = field(default_factory=dict)        # by area
    pace_n: dict[str, int] = field(default_factory=dict)
    deadhead_bonus: float = 1.2
    offset_share: dict[str, float] = field(default_factory=dict)    # by area
    service_s: dict[str, float] = field(default_factory=dict)       # by dp kind
    item_s: dict[str, float] = field(default_factory=dict)          # by item kind
    weather: dict[str, float] = field(default_factory=dict)
    dark: float = 1.0
    relief: float = 1.0
    experience: dict[str, float] = field(default_factory=dict)      # bucket -> factor
    walker: dict[str, float] = field(default_factory=dict)          # per walker
    # How much of the time at a door moves with the weather. Not all of it: a
    # letterbox is a letterbox in the rain. But the walk up the path is inside
    # the door figure - it cannot be separated from it - and that part does slow
    # down, so a weather factor applied to the street metres alone comes out too
    # big, having to explain the path's slowdown with the street's seconds.
    weather_on_doors: float = 0.0
    common_mode: float = 0.07
    segment_sd_share: float = 0.08
    blocks_used: int = 0
    passes_used: int = 0
    trained_to: str | None = None
    notes: list[str] = field(default_factory=list)

    def pace_for(self, area: str) -> float:
        return self.pace_mps.get(area) or self.pace_mps.get("*") or 1.15

    def day_factor(self, weather: str | None, dark: bool, relief: bool,
                   months: float | None, walker_id: str | None) -> float:
        f = self.weather.get(weather or "clear", 1.0)
        if dark:
            f *= self.dark
        if relief:
            f *= self.relief
        if months is not None:
            f *= self.experience.get(_exp_bucket(months), 1.0)
        if walker_id:
            f *= self.walker.get(walker_id, 1.0)
        return f

    def as_dict(self) -> dict:
        return {
            "pace_mps": {k: round(v, 4) for k, v in self.pace_mps.items()},
            "pace_kmh": {k: round(v * 3.6, 2) for k, v in self.pace_mps.items()},
            "pace_n": self.pace_n,
            "deadhead_bonus": round(self.deadhead_bonus, 3),
            "offset_share": {k: round(v, 3) for k, v in self.offset_share.items()},
            "service_s": {k: round(v, 2) for k, v in self.service_s.items()},
            "item_s": {k: round(v, 3) for k, v in self.item_s.items()},
            "weather": {k: round(v, 3) for k, v in self.weather.items()},
            "dark": round(self.dark, 3), "relief": round(self.relief, 3),
            "experience": {k: round(v, 3) for k, v in self.experience.items()},
            "walkers_fitted": len(self.walker),
            "walker_spread": (round(max(self.walker.values()) - min(self.walker.values()), 3)
                              if self.walker else 0.0),
            "weather_on_doors": round(self.weather_on_doors, 3),
            "common_mode": round(self.common_mode, 4),
            "segment_sd_share": round(self.segment_sd_share, 4),
            "blocks_used": self.blocks_used, "passes_used": self.passes_used,
            "trained_to": self.trained_to, "notes": self.notes,
        }


def _exp_bucket(months: float) -> str:
    if months < 1:
        return "first month"
    if months < 3:
        return "1-3 months"
    if months < 12:
        return "3-12 months"
    return "a year or more"


def _shrunk(values: list[float], k: float = SHRINK_K) -> float:
    if not values:
        return 1.0
    n = len(values)
    return (n * st.fmean(values) + k * 1.0) / (n + k)


def fit(db_path=None, con=None, to_date: str | None = None,
        from_date: str | None = None) -> WalkModel:
    """Fit on the walks in a date range, so a later range can test it honestly.

    Five stages, in the order the things they measure depend on each other.
    The order matters: an earlier attempt measured pace on the streets with
    nothing to deliver, which is a walker in transit rather than at work, and
    every block then came out a fifth short with the shortfall quietly pushed
    into the doors.
    """
    own = con is None
    con = con or connect(db_path, read_only=True)
    m = WalkModel(trained_to=to_date)
    try:
        dw, args = ["1=1"], []
        if to_date:
            dw.append("d.date < ?"); args.append(to_date)
        if from_date:
            dw.append("d.date >= ?"); args.append(from_date)
        w = " AND ".join(dw)

        # --- 1. the pace of a walker in transit, from the streets with
        #        nothing on them -------------------------------------------
        dead = con.execute(
            f"SELECT s.area_type, s.length_m, t.traversal_s FROM segment_traversals t "
            f"JOIN osm_segments s USING(segment_id) JOIN block_days d USING(block_day_id) "
            f"WHERE {w} AND t.serviced=0 AND t.traversal_s > 1 AND s.length_m > 3",
            args).fetchall()
        transit: dict[str, float] = {}
        by_area: dict[str, list[float]] = {}
        for r in dead:
            v = r["length_m"] / r["traversal_s"]
            if 0.2 < v < 3.0:
                by_area.setdefault(r["area_type"], []).append(v)
        for area, vals in by_area.items():
            transit[area] = st.median(vals)
            m.pace_n[area] = len(vals)
        allv = [v for vals in by_area.values() for v in vals]
        transit["*"] = st.median(allv) if allv else 1.3

        # --- 2. the working pace, the doors and the mail, from whole blocks --
        #
        # All of it comes out of one regression, and it has to be run on whole
        # blocks rather than on segments. The reason is where the information
        # is. Within an area, a segment's length and its number of doors are
        # nearly the same column - doors sit four to nine metres apart in a
        # town - so a segment-level fit cannot say which of the two the time
        # belongs to, and it guesses: an earlier version put posties at 0.54 m/s
        # and charged twelve seconds for a letter.
        #
        # Between blocks the picture is completely different: five metres of
        # street per door in a town, seventy-two in the country, a fourteen-fold
        # spread. That is what separates "seconds per metre" from "seconds per
        # door", and it is why this is fitted where it is.
        #
        # The mail needs a third kind of variation again - the same doors
        # getting three times the parcels in December - which two years of
        # history supplies and six months would not.
        statics = block_statics(con)
        day_rows = con.execute(
            f"SELECT d.block_day_id, d.date, d.weather, d.daylight, d.walker_id, "
            f"       d.dps_served, b.block_id, b.area_type, w.started_on, "
            f"       d.items_letters, d.items_packets, d.items_parcels, "
            f"       d.items_signed, d.items_special, "
            f"       (SELECT t.does_take_s FROM block_traversal_totals t "
            f"        WHERE t.block_day_id=d.block_day_id "
            f"        ORDER BY (t.source='ping') DESC LIMIT 1) actual_s "
            f"FROM block_days d JOIN blocks b USING(block_id) "
            f"LEFT JOIN walkers w USING(walker_id) WHERE {w}", args).fetchall()
        regular = regular_walkers(con)
        work = []
        for r in day_rows:
            s = statics.get(r["block_id"])
            if not s or not r["actual_s"] or not r["dps_served"] or s["dps"] == 0:
                continue
            usual = regular.get((r["block_id"], r["date"][:7]))
            work.append({
                "s": s, "actual": r["actual_s"], "area": r["area_type"],
                "weather": r["weather"], "daylight": r["daylight"],
                "walker_id": r["walker_id"],
                "relief": bool(usual and r["walker_id"] and r["walker_id"] != usual),
                "months": _months(r["started_on"], r["date"]),
                "per_dp": {k: r[f"items_{k}"] / r["dps_served"] for k in ITEM_KINDS},
            })
        if len(work) < 300:
            m.notes.append("too few whole blocks to fit anything")
            m.pace_mps = dict(transit)
            return m
        m.blocks_used = len(work)
        areas = sorted({d["area"] for d in work})

        def solve(day_factor) -> None:
            """One pass of the block regression, given what the days were doing."""
            X, y = [], []
            for d in work:
                s, f = d["s"], day_factor(d)
                row = [(s["serviced_m"] + s["deadhead_m"] / m.deadhead_bonus) * f
                       if a == d["area"] else 0.0 for a in areas]
                for a in areas:
                    row += [s["dp_kinds"].get(k, 0) if a == d["area"] else 0
                            for k in DP_KINDS]
                row += [d["per_dp"][k] * s["dps"] for k in ITEM_KINDS]
                X.append(row)
                y.append(d["actual"])
            coef, *_ = np.linalg.lstsq(np.array(X, dtype=float),
                                       np.array(y, dtype=float), rcond=None)
            for i, a in enumerate(areas):
                if coef[i] > 0.05:
                    m.pace_mps[a] = 1.0 / float(coef[i])
            base = len(areas)
            for ai, a in enumerate(areas):
                seen = sum(d["s"]["dp_kinds"].get(k, 0) for d in work
                           if d["area"] == a for k in DP_KINDS)
                for j, k in enumerate(DP_KINDS):
                    n = sum(d["s"]["dp_kinds"].get(k, 0) for d in work if d["area"] == a)
                    if n >= MIN_CELL and seen:
                        m.service_s[f"{a}|{k}"] = float(max(coef[base + ai * len(DP_KINDS) + j], 0.0))
            for i, k in enumerate(ITEM_KINDS):
                m.item_s[k] = float(max(coef[base + len(areas) * len(DP_KINDS) + i], 0.0))
            m.pace_mps.setdefault("*", st.fmean([m.pace_mps[a] for a in areas
                                                 if a in m.pace_mps]) if areas else 1.2)

        def hold_pace_physical() -> None:
            """A walker cannot work a street faster than they cross an empty one.

            The regression that splits metres from doors is only as well
            identified as the spread of block densities in the sample, and on a
            thin one it wanders: it has put urban posties at 0.54 m/s and at
            1.41 m/s on two samples of the same world. Neither is a pace; both
            are the split going wrong. The transit pace is measured directly and
            is a hard ceiling, and nobody works a street at less than two thirds
            of it, so the fitted figure is held inside that band and the doors
            are refitted around whatever that costs.
            """
            moved = []
            for a in list(m.pace_mps):
                if a == "*" or a not in transit:
                    continue
                # at least a little slower when stopping at doors, and never
                # less than two thirds the speed: a band, not an answer
                ceiling, floor = transit[a] / 1.05, transit[a] / 1.6
                if not (floor <= m.pace_mps[a] <= ceiling):
                    moved.append(f"{a} {m.pace_mps[a]:.2f}->"
                                 f"{min(max(m.pace_mps[a], floor), ceiling):.2f}")
                    m.pace_mps[a] = min(max(m.pace_mps[a], floor), ceiling)
            if moved:
                m.notes.append("the pace/door split wandered outside what a walker "
                               "can physically do and was held at the limit: "
                               + ", ".join(moved))

        # first pass: assume every day was an ordinary one
        solve(lambda d: 1.0)
        hold_pace_physical()

        # --- 3. how much faster a walker moves with nothing to deliver ------
        shares = [transit[a] / m.pace_mps[a] for a in areas
                  if a in transit and a in m.pace_mps and m.pace_mps[a] > 0]
        if shares:
            m.deadhead_bonus = min(max(st.fmean(shares), 1.0), 1.8)
            m.notes.append(
                f"a walker covers ground {100 * (m.deadhead_bonus - 1):.0f}% faster on a "
                f"street with nothing to deliver; measuring pace there and using it at "
                f"work would make every block come out short by that much")

        # --- 4. what the day does, applied where it applies -----------------
        #
        # Rain slows a walker. It does not slow the letterbox. Fitted as a
        # multiplier on the walking only: fitted on the whole block instead,
        # every factor comes out damped - heavy rain at 1.08 against a real
        # 1.18, because three quarters of an urban block is time at doors - and
        # that damped figure is then over-applied to a rural block, which is
        # mostly walking.
        for _pass in range(2):
            by_weather: dict[str, list[float]] = {}
            dark_r, relief_r, ratios = [], [], []
            by_exp: dict[str, list[float]] = {}
            by_walker: dict[str, list[float]] = {}
            for d in work:
                s = d["s"]
                walk = walking_seconds(m, s)
                door = door_seconds_total(m, s, d["per_dp"])
                if walk <= 0:
                    continue
                a = m.weather_on_doors
                ratio = (d["actual"] - door * (1 - a)) / (walk + a * door)
                if not (0.3 < ratio < 3.0):
                    continue
                ratios.append(ratio)
                by_weather.setdefault(d["weather"] or "clear", []).append(ratio)
                if not d["daylight"]:
                    dark_r.append(ratio)
                if d["relief"]:
                    relief_r.append(ratio)
                if d["months"] is not None:
                    by_exp.setdefault(_exp_bucket(d["months"]), []).append(ratio)
                if d["walker_id"]:
                    by_walker.setdefault(d["walker_id"], []).append(ratio)
            level = st.fmean(ratios) if ratios else 1.0
            m.weather = {k: _shrunk([x / level for x in v])
                         for k, v in by_weather.items()}
            m.dark = _shrunk([x / level for x in dark_r]) if dark_r else 1.0
            m.relief = _shrunk([x / level for x in relief_r]) if relief_r else 1.0
            m.experience = {k: _shrunk([x / level for x in v])
                            for k, v in by_exp.items()}
            m.walker = {k: _shrunk([x / level for x in v], k=60.0)
                        for k, v in by_walker.items() if len(v) >= 20}
            # how much of the door time moves with the day, fitted as one
            # number: the areas differ enough in their walk-to-door split to
            # identify it, and without it every factor comes out inflated
            num = den = 0.0
            for d in work:
                s = d["s"]
                walk = walking_seconds(m, s)
                door = door_seconds_total(m, s, d["per_dp"])
                f = m.day_factor(d["weather"], not d["daylight"], d["relief"],
                                 d["months"], d["walker_id"])
                x = door * (f - 1.0)
                num += (d["actual"] - walk * f - door) * x
                den += x * x
            if den > 1e-9:
                m.weather_on_doors = min(max(num / den, 0.0), 1.0)

            # ...and now refit knowing what the days were doing, so the weather
            # does not end up inside the pace
            solve(lambda d: m.day_factor(d["weather"], not d["daylight"], d["relief"],
                                         d["months"], d["walker_id"]))
            hold_pace_physical()
        # the doors pay for whatever holding the pace cost, so that whole blocks
        # still come out the right length
        num = den = 0.0
        for d in work:
            f = m.day_factor(d["weather"], not d["daylight"], d["relief"],
                             d["months"], d["walker_id"])
            walk = walking_seconds(m, d["s"]) * f
            door = door_seconds_total(m, d["s"], d["per_dp"])
            num += (d["actual"] - walk) * door
            den += door * door
        if den > 0:
            scale = num / den
            if 0.5 < scale < 2.0:
                for k in list(m.service_s):
                    m.service_s[k] *= scale
                for k in list(m.item_s):
                    m.item_s[k] *= scale
                if abs(scale - 1) > 0.02:
                    m.notes.append(f"the doors were scaled by {scale:.2f} so that whole "
                                   f"blocks come out right with the pace held")
        m.notes.append("the day factors are relative to an average day, so a clear one "
                       "sits a little under 1")
        thin = [k for k in ITEM_KINDS if m.item_s.get(k, 0) <= 0]
        if thin:
            m.notes.append("too rare on this data to put a time on: " + ", ".join(thin))
        m.passes_used = len(work)

        # --- 6. what is left over, which is the interval --------------------
        rel = []
        for d in work:
            f = m.day_factor(d["weather"], not d["daylight"], d["relief"],
                             d["months"], d["walker_id"])
            p2 = predict_block(m, d["s"], d["per_dp"], day_factor=f)
            if p2 > 0:
                rel.append((d["actual"] - p2) / p2)
        if len(rel) > 50:
            m.common_mode = min(max(st.stdev(rel), 0.02), 0.35)
            bias = st.fmean(rel)
            if abs(bias) > 0.02:
                m.notes.append(f"the finished model still runs {100 * bias:+.1f}% against "
                               f"the records on average")
            m.notes.append(f"a block lands {100 * m.common_mode:.1f}% either side of its "
                           f"forecast; that is what the interval is made of")
        return m
    finally:
        if own:
            con.close()


def regular_walkers(con) -> dict[tuple[str, str], str]:
    """Whose block is it? The walker who did it most often that month.

    Needed because covering somebody else's block is slower - you do not know
    which gate sticks - and that is a real effect worth a forecast knowing
    about. Nothing in the records says "relief"; it has to be worked out.
    """
    out: dict[tuple[str, str], str] = {}
    counts: dict[tuple[str, str], dict[str, int]] = {}
    for r in con.execute(
            "SELECT block_id, substr(date,1,7) ym, walker_id, count(*) n "
            "FROM block_days WHERE walker_id IS NOT NULL "
            "GROUP BY block_id, ym, walker_id"):
        counts.setdefault((r["block_id"], r["ym"]), {})[r["walker_id"]] = r["n"]
    for key, who in counts.items():
        out[key] = max(who, key=who.get)
    return out


def _structural_prediction(con, m: WalkModel, w: str, args) -> list[dict]:
    """Predict every walk from its geometry and its mail, with no day factors.

    Uses the block totals, which go back the full two years, rather than only
    the window where per-segment detail is retained: the weather, the seasons
    and a walker's first months need years to show up, and the totals carry
    them. Where both exist the ping-derived total is preferred.
    """
    rows = con.execute(
        f"SELECT d.block_day_id, d.date, d.weather, d.daylight, d.walker_id, "
        f"       d.dps_served, d.items_letters, d.items_packets, d.items_parcels, "
        f"       d.items_signed, d.items_special, b.block_id, b.area_type, "
        f"       w.started_on, "
        f"       (SELECT t.does_take_s FROM block_traversal_totals t "
        f"        WHERE t.block_day_id = d.block_day_id "
        f"        ORDER BY (t.source='ping') DESC, t.segments_covered DESC LIMIT 1) actual_s "
        f"FROM block_days d JOIN blocks b USING(block_id) "
        f"LEFT JOIN walkers w USING(walker_id) WHERE {w}", args).fetchall()
    static = block_statics(con)
    regular = regular_walkers(con)
    out = []
    for r in rows:
        s = static.get(r["block_id"])
        if not s or not r["actual_s"] or not r["dps_served"]:
            continue
        per_dp = {k: r[f"items_{k}"] / r["dps_served"] for k in ITEM_KINDS}
        usual = regular.get((r["block_id"], r["date"][:7]))
        out.append({
            "block_day_id": r["block_day_id"], "date": r["date"],
            "weather": r["weather"], "daylight": r["daylight"],
            "walker_id": r["walker_id"],
            "relief": bool(usual and r["walker_id"] and r["walker_id"] != usual),
            "months": _months(r["started_on"], r["date"]),
            "actual_s": r["actual_s"],
            "pred_s": predict_block(m, s, per_dp),
            "walk_s": walking_seconds(m, s),
            "door_s": door_seconds_total(m, s, per_dp),
        })
    return out


def _months(started_on: str | None, date: str) -> float | None:
    if not started_on:
        return None
    try:
        return (dt.date.fromisoformat(date)
                - dt.date.fromisoformat(started_on)).days / 30.44
    except ValueError:
        return None


def block_statics(con) -> dict[str, dict]:
    """Per block: what the route covers, and what is on it. Fixed until it is replanned."""
    out: dict[str, dict] = {}
    for r in con.execute(
            "SELECT b.block_id, b.area_type, b.do_id, "
            "       sum(s.length_m) street_m, count(*) segments "
            "FROM blocks b JOIN osm_segments s USING(block_id) GROUP BY b.block_id"):
        out[r["block_id"]] = {"block_id": r["block_id"], "area_type": r["area_type"],
                              "do_id": r["do_id"], "street_m": r["street_m"],
                              "segments": r["segments"], "dp_kinds": {}, "offset_m": 0.0,
                              "serviced_m": r["street_m"], "deadhead_m": 0.0}
    for r in con.execute(
            "SELECT s.block_id, p.dp_type, count(*) n, sum(j.distance_m) off_m "
            "FROM dp_segment_join j JOIN osm_segments s USING(segment_id) "
            "JOIN delivery_points p USING(dp_id) GROUP BY s.block_id, p.dp_type"):
        b = out.get(r["block_id"])
        if b:
            b["dp_kinds"][r["dp_type"]] = r["n"]
            b["offset_m"] += r["off_m"]
    # how much of the walk is a second pass, from the route actually walked
    for r in con.execute(
            "SELECT s.block_id, sum(CASE WHEN t.serviced=0 THEN s.length_m ELSE 0 END) dead, "
            "       sum(CASE WHEN t.serviced=1 THEN s.length_m ELSE 0 END) serv, "
            "       count(DISTINCT t.block_day_id) days "
            "FROM segment_traversals t JOIN osm_segments s USING(segment_id) "
            "GROUP BY s.block_id"):
        b = out.get(r["block_id"])
        if b and r["days"]:
            b["deadhead_m"] = r["dead"] / r["days"]
            b["serviced_m"] = r["serv"] / r["days"]
    for b in out.values():
        b["dps"] = sum(b["dp_kinds"].values())
    return out


def walking_seconds(m: WalkModel, s: dict) -> float:
    """The part of the block that is walking: streets, and second passes."""
    pace = m.pace_for(s["area_type"])
    return s["serviced_m"] / pace + s["deadhead_m"] / (pace * m.deadhead_bonus)


def door_seconds_total(m: WalkModel, s: dict, per_dp: dict[str, float]) -> float:
    """The part that is doors and what goes through them."""
    return (sum(door_seconds(m, s["area_type"], k) * n
                for k, n in s["dp_kinds"].items())
            + sum(m.item_s.get(k, 0.0) * per_dp.get(k, 0.0) * s["dps"]
                  for k in ITEM_KINDS))


def door_seconds(m: WalkModel, area: str, kind: str) -> float:
    """Seconds at a door of this kind in this sort of place, path included."""
    for key in (f"{area}|{kind}", kind):
        if key in m.service_s:
            return m.service_s[key]
    return 10.0


def predict_block(m: WalkModel, s: dict, per_dp: dict[str, float],
                  day_factor: float = 1.0) -> float:
    """Seconds, from the streets, the doors and today's mail.

    The day factor multiplies the walking, and the share of the door time that
    is really the walk up the path to it. See the fit for why both.
    """
    walk = walking_seconds(m, s)
    door = door_seconds_total(m, s, per_dp)
    a = m.weather_on_doors
    return (walk + a * door) * day_factor + (1 - a) * door


@dataclass
class Forecast:
    block_id: str
    date: str
    should_take_s: float
    low_s: float
    high_s: float
    confidence: int
    parts: dict
    basis: dict

    def as_dict(self) -> dict:
        return {"block_id": self.block_id, "date": self.date,
                "should_take_s": round(self.should_take_s, 1),
                "should_take_h": round(self.should_take_s / 3600, 3),
                "low_h": round(self.low_s / 3600, 3),
                "high_h": round(self.high_s / 3600, 3),
                "confidence": self.confidence,
                "parts": {k: round(v, 1) for k, v in self.parts.items()},
                "basis": self.basis}


def forecast(m: WalkModel, s: dict, per_dp: dict[str, float], date: str,
             weather: str = "clear", dark: bool = False, relief: bool = False,
             months: float | None = 24.0, walker_id: str | None = None,
             confidence: int = 90) -> Forecast:
    """What this block should take on this day, and how sure that is."""
    f = m.day_factor(weather, dark, relief, months, walker_id)
    total = predict_block(m, s, per_dp, day_factor=f)
    z = Z.get(confidence, Z[90])
    sd = total * m.common_mode
    pace = m.pace_for(s["area_type"])
    return Forecast(
        s["block_id"], date, total, max(total - z * sd, 0.0), total + z * sd,
        confidence,
        {"walking_s": walking_seconds(m, s) * f,
         "doors_s": sum(door_seconds(m, s["area_type"], k) * n
                        for k, n in s["dp_kinds"].items()),
         "items_s": sum(m.item_s.get(k, 0.0) * per_dp.get(k, 0.0) * s["dps"]
                        for k in ITEM_KINDS)},
        {"pace_kmh": round(pace * 3.6, 2), "day_factor": round(f, 3),
         "weather": weather, "relief": relief,
         "walker": walker_id, "common_mode": round(m.common_mode, 4),
         "blocks_behind_it": m.blocks_used})


def save(m: WalkModel, path) -> None:
    from pathlib import Path
    Path(path).write_text(json.dumps(m.as_dict(), indent=2))
