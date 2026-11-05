"""
Resourcing: how many walkers the work needs, and when it will not fit.

A forecast per block is interesting. What a delivery office actually asks is
"how many people do I need on the fifteenth of December, and what happens if I
have four short" - which is the same forecast, added up, with the uncertainty
carried rather than dropped.

Two things this is careful about, because both are easy to get wrong in a way
that looks fine on a spreadsheet:

**Uncertainty does not add.** Twenty blocks each with a half-hour of doubt do
not make a day with ten hours of doubt: they make a day with about two, because
they do not all overrun together. The expected hours add; the spreads combine in
quadrature, plus whatever part is common to the whole office that day - the
weather, the volume - which does not shrink at all. An office planned on summed
worst cases is an office with people standing around.

**A walker is not a pool of hours.** Five blocks of six hours is not thirty
hours of work to be divided by a shift: it is five people out for six hours.
Rounding to whole people at the end, and only at the end, is how a plan that
looks balanced turns into three hours of overtime on four walks.
"""
from __future__ import annotations

import datetime as dt
import math
import statistics as st
from dataclasses import dataclass, field

from .model import (ITEM_KINDS, WalkModel, block_statics, forecast,
                    regular_walkers, _months)
from .schema import connect

SHIFT_HOURS = 7.5           # what a walker is rostered for
OVERTIME_AFTER = 7.5
MAX_SHIFT_HOURS = 9.5       # beyond this a block has to be split, not stretched


@dataclass
class DayPlan:
    date: str
    do_id: str
    region: str
    blocks: int
    dps: int
    expected_h: float
    low_h: float
    high_h: float
    walkers_needed: float
    walkers_available: int
    overtime_h: float = 0.0
    must_split: list[str] = field(default_factory=list)
    over_shift: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def shortfall(self) -> float:
        return max(0.0, self.walkers_needed - self.walkers_available)

    @property
    def overtime_as_people(self) -> float:
        """Overtime is not people, but it is paid like them, so say both.

        A block at eight and a half hours does not need a second walker; it
        needs an hour of overtime. Counting it as a person overstates what the
        office needs and hides the thing that actually costs money.
        """
        return self.overtime_h / SHIFT_HOURS

    def as_dict(self) -> dict:
        return {"date": self.date, "do_id": self.do_id, "region": self.region,
                "blocks": self.blocks, "dps": self.dps,
                "expected_h": round(self.expected_h, 1),
                "low_h": round(self.low_h, 1), "high_h": round(self.high_h, 1),
                "walkers_needed": round(self.walkers_needed, 1),
                "walkers_available": self.walkers_available,
                "shortfall": round(self.shortfall, 1),
                "overtime_h": round(self.overtime_h, 1),
                "overtime_as_people": round(self.overtime_as_people, 1),
                "must_split": self.must_split,
                "blocks_over_shift": len(self.over_shift),
                "over_shift": self.over_shift[:10], "notes": self.notes}


def expected_mail(con, date: str, block_id: str,
                  lookback_years: int = 2) -> dict[str, float]:
    """How much mail a block gets on a date like this one.

    Taken from the same day of the year in previous years rather than from last
    week, because the thing that decides December is December. Where there is no
    history for that date - the first year of a new block - it falls back to the
    same weekday recently, and says so by returning fewer days behind it.
    """
    d = dt.date.fromisoformat(date)
    wanted = []
    for back in range(1, lookback_years + 1):
        for slack in (-3, -2, -1, 0, 1, 2, 3):
            wanted.append((d - dt.timedelta(days=365 * back - slack)).isoformat())
    rows = con.execute(
        f"SELECT dps_served, items_letters, items_packets, items_parcels, "
        f"       items_signed, items_special FROM block_days "
        f"WHERE block_id=? AND date IN ({','.join('?' * len(wanted))})",
        [block_id, *wanted]).fetchall()
    if not rows:
        rows = con.execute(
            "SELECT dps_served, items_letters, items_packets, items_parcels, "
            "       items_signed, items_special FROM block_days "
            "WHERE block_id=? AND date < ? ORDER BY date DESC LIMIT 20",
            (block_id, date)).fetchall()
    if not rows:
        return {k: 0.0 for k in ITEM_KINDS} | {"days_behind_it": 0}
    out = {}
    for k in ITEM_KINDS:
        vals = [r[f"items_{k}"] / max(r["dps_served"], 1) for r in rows]
        out[k] = st.fmean(vals)
    out["days_behind_it"] = len(rows)
    return out


def plan_day(con, m: WalkModel, date: str, do_id: str | None = None,
             confidence: int = 90, statics: dict | None = None) -> list[DayPlan]:
    """What each office needs on one date."""
    statics = statics or block_statics(con)
    where = "WHERE 1=1" + (" AND b.do_id = ?" if do_id else "")
    args = [do_id] if do_id else []
    blocks = con.execute(
        f"SELECT b.*, o.region FROM blocks b JOIN delivery_offices o USING(do_id) "
        f"{where} ORDER BY b.do_id, b.block_id", args).fetchall()
    regular = regular_walkers(con)
    ym = date[:7]

    by_office: dict[str, dict] = {}
    for b in blocks:
        s = statics.get(b["block_id"])
        if not s:
            continue
        mail = expected_mail(con, date, b["block_id"])
        walker = regular.get((b["block_id"], ym))
        months = None
        if walker:
            r = con.execute("SELECT started_on FROM walkers WHERE walker_id=?",
                            (walker,)).fetchone()
            months = _months(r["started_on"], date) if r else None
        f = forecast(m, s, mail, date, weather="clear", dark=False, relief=False,
                     months=months, walker_id=walker, confidence=confidence)
        hours = f.should_take_s / 3600.0
        drive = (b["drive_minutes"] or 0) / 60.0
        # the frame has to be sorted before anybody walks anywhere, and no
        # geofence covers the sorting office, so this is not in "does take"
        prep = (b["prep_minutes"] or 0) / 60.0
        total = hours + drive + prep
        o = by_office.setdefault(b["do_id"], {
            "region": b["region"], "blocks": 0, "dps": 0, "hours": 0.0,
            "var": 0.0, "over": [], "walkers": 0.0})
        o["blocks"] += 1
        o["dps"] += s["dps"]
        o["hours"] += total
        sd = (f.high_s - f.should_take_s) / 3600.0
        o["var"] += sd ** 2
        # a walker is a person, not a pool of hours
        o["walkers"] += 1.0 if total <= MAX_SHIFT_HOURS else math.ceil(
            total / MAX_SHIFT_HOURS)
        o["overtime"] = o.get("overtime", 0.0) + max(0.0, min(total, MAX_SHIFT_HOURS)
                                                     - OVERTIME_AFTER)
        if total > MAX_SHIFT_HOURS:
            o.setdefault("split", []).append(b["block_id"])
        if total > OVERTIME_AFTER:
            o["over"].append({"block_id": b["block_id"], "hours": round(total, 2),
                              "over_by_h": round(total - OVERTIME_AFTER, 2),
                              "walk_h": round(hours, 2), "prep_h": round(prep, 2),
                              "drive_h": round(drive, 2),
                              "dps": s["dps"], "area_type": b["area_type"]})

    out = []
    for do, o in by_office.items():
        have = con.execute(
            "SELECT count(*) n FROM walkers WHERE do_id=? AND started_on <= ? "
            "AND (left_on IS NULL OR left_on > ?)", (do, date, date)).fetchone()["n"]
        sd = math.sqrt(o["var"])
        notes = []
        if o["over"]:
            notes.append(f"{len(o['over'])} blocks run past the {OVERTIME_AFTER:.1f} "
                         f"hour shift; the longest by "
                         f"{max(x['over_by_h'] for x in o['over']):.1f} h")
        dp = DayPlan(date, do, o["region"], o["blocks"], o["dps"], o["hours"],
                     o["hours"] - 1.6449 * sd, o["hours"] + 1.6449 * sd,
                     o["walkers"], have, round(o.get("overtime", 0.0), 2),
                     o.get("split", []), o["over"], notes)
        if dp.shortfall > 0:
            dp.notes.append(f"{dp.shortfall:.0f} walkers short of what the work needs")
        if dp.overtime_h > 0.5:
            dp.notes.append(f"{dp.overtime_h:.1f} hours of overtime, which is "
                            f"{dp.overtime_as_people:.1f} walkers' worth of pay")
        if dp.must_split:
            dp.notes.append(f"{len(dp.must_split)} blocks cannot be done inside "
                            f"{MAX_SHIFT_HOURS} hours at all and have to be split")
        out.append(dp)
    out.sort(key=lambda d: d.do_id)
    return out


def plan_range(con, m: WalkModel, start: str, days: int = 60,
               do_id: str | None = None, confidence: int = 90) -> dict:
    """The next so many days, office by office, with the pinch points named."""
    statics = block_statics(con)
    d0 = dt.date.fromisoformat(start)
    rows: list[DayPlan] = []
    for i in range(days):
        d = d0 + dt.timedelta(days=i)
        if d.weekday() == 6:
            continue
        rows += plan_day(con, m, d.isoformat(), do_id, confidence, statics)
    by_date: dict[str, dict] = {}
    for r in rows:
        x = by_date.setdefault(r.date, {"date": r.date, "hours": 0.0, "walkers": 0.0,
                                        "available": 0, "blocks": 0, "over": 0,
                                        "low": 0.0, "high": 0.0, "overtime": 0.0,
                                        "split": 0})
        x["hours"] += r.expected_h
        x["walkers"] += r.walkers_needed
        x["available"] += r.walkers_available
        x["blocks"] += r.blocks
        x["over"] += len(r.over_shift)
        x["low"] += r.low_h
        x["high"] += r.high_h
        x["overtime"] += r.overtime_h
        x["split"] += len(r.must_split)
    days_out = sorted(by_date.values(), key=lambda x: x["date"])
    for x in days_out:
        x["shortfall"] = max(0.0, x["walkers"] - x["available"])
        for k in ("hours", "walkers", "low", "high", "shortfall", "overtime"):
            x[k] = round(x[k], 1)
    peak = max(days_out, key=lambda x: x["hours"]) if days_out else None
    worst = sorted(days_out, key=lambda x: -x["shortfall"])[:10]
    return {
        "from": start, "days": len(days_out),
        "mean_hours_a_day": round(st.fmean([x["hours"] for x in days_out]), 1)
        if days_out else 0,
        "peak_day": peak,
        "days_short": sum(1 for x in days_out if x["shortfall"] > 0),
        "worst_days": [x for x in worst if x["shortfall"] > 0],
        "rows": days_out,
        "offices": sorted({r.do_id for r in rows}),
    }


def peak_report(con, m: WalkModel, year: int | None = None) -> dict:
    """What December does to the work, measured rather than assumed.

    The number that matters to a delivery office is not "Christmas is busy" but
    "the week of the fifteenth needs eleven more people than the week of the
    first, for nine days". This is that, from the same forecast everything else
    uses.
    """
    year = year or dt.date.today().year
    statics = block_statics(con)
    weeks: dict[str, dict] = {}
    d = dt.date(year, 11, 1)
    end = dt.date(year, 12, 31)
    while d <= end:
        if d.weekday() != 6:
            for p in plan_day(con, m, d.isoformat(), None, 90, statics):
                wk = (d - dt.timedelta(days=d.weekday())).isoformat()
                x = weeks.setdefault(wk, {"week": wk, "hours": 0.0, "walkers": 0.0,
                                          "available": 0, "days": 0,
                                          "overtime": 0.0, "split": 0})
                x["hours"] += p.expected_h
                x["walkers"] += p.walkers_needed
                x["available"] += p.walkers_available
                x["overtime"] += p.overtime_h
                x["split"] += len(p.must_split)
            weeks[(d - dt.timedelta(days=d.weekday())).isoformat()]["days"] += 1
        d += dt.timedelta(days=1)
    rows = sorted(weeks.values(), key=lambda x: x["week"])
    for x in rows:
        x["hours_a_day"] = round(x["hours"] / max(x["days"], 1), 1)
        x["walkers_a_day"] = round(x["walkers"] / max(x["days"], 1), 1)
        x["available_a_day"] = round(x["available"] / max(x["days"], 1), 1)
        x["overtime_a_day"] = round(x["overtime"] / max(x["days"], 1), 1)
        x["splits_a_day"] = round(x["split"] / max(x["days"], 1), 1)
        # what the office actually has to find: people it is short, plus the
        # overtime it will otherwise pay, expressed in the same unit
        x["extra_people"] = round(max(0.0, x["walkers_a_day"] - x["available_a_day"])
                                  + x["overtime_a_day"] / SHIFT_HOURS, 1)
    base = min((x["hours_a_day"] for x in rows), default=0)
    top = max(rows, key=lambda x: x["hours_a_day"]) if rows else None
    return {
        "year": year, "weeks": rows,
        "quietest_hours_a_day": base,
        "busiest_week": top,
        "peak_over_base": round(top["hours_a_day"] / base, 3) if top and base else None,
        "extra_people_at_peak": top["extra_people"] if top else None,
    }
