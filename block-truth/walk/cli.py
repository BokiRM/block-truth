"""
Walked blocks: what they do take, what they should take, and what that needs.

    walk --build                      build two years of mock data
    walk --block DO01-B00 2026-11-05  one block: does take, should take, route
    walk --audit 2026-10-30           the measurement itself, checked
    walk --fences DO01-B00            how wide the geofences should be
    walk --model                      what the model learned, and from how much
    walk --routes                     where the walking is being wasted
    walk --day 2026-12-14             resourcing for one day
    walk --ahead 60                   the next sixty days
    walk --peak 2026                  what December does

Everything here is about blocks that are *walked*. The driven half of this
application - vehicle rounds, depots, delivery windows - is `block-truth`, and
the two are kept apart on purpose: a van visits points and a postie covers
streets, which are different problems with different right answers.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys

from .model import ITEM_KINDS, block_statics, fit, forecast
from .resource import expected_mail, peak_report, plan_day, plan_range
from .route import fleet_saving, options, saving
from .schema import DEFAULT_WALK_DB, connect, counts
from .truth import audit_day, block_time, buffer_sensitivity, overlap_report


def hm(seconds: float) -> str:
    s = int(round(seconds))
    return f"{s // 3600}h {s % 3600 // 60:02d}m"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="walk", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                allow_abbrev=False)
    p.add_argument("--db", default=str(DEFAULT_WALK_DB))
    p.add_argument("--build", action="store_true", help="generate the mock warehouse")
    p.add_argument("--years", type=float, default=2.0)
    p.add_argument("--block", help="one block, with a date")
    p.add_argument("date", nargs="?", help="a date, where one is wanted")
    p.add_argument("--audit", metavar="DATE", help="check the measurement on a date")
    p.add_argument("--fences", metavar="BLOCK", help="how wide the geofences should be")
    p.add_argument("--model", action="store_true", help="what the model learned")
    p.add_argument("--routes", action="store_true", help="where walking is wasted")
    p.add_argument("--day", metavar="DATE", help="resourcing for one day")
    p.add_argument("--ahead", type=int, metavar="DAYS", help="resourcing, next N days")
    p.add_argument("--peak", type=int, metavar="YEAR", help="what December does")
    p.add_argument("--to-date", help="fit only on walks before this date")
    p.add_argument("--json", action="store_true")
    a = p.parse_args(argv)

    if a.build:
        from .generate import generate
        r = generate(a.db, years=a.years)
        print(f"built in {r['seconds']}s: {r['blocks']} blocks, "
              f"{r['osm_segments']:,} segments, {r['delivery_points']:,} delivery points")
        print(f"  {r['block_days']:,} walks from {r['from']} to {r['to']}")
        print(f"  {r['segment_traversals']:,} traversals, {r['pings']:,} pings")
        return 0

    try:
        con = connect(a.db, read_only=True)
    except FileNotFoundError:
        print(f"no database at {a.db}: run  walk --build", file=sys.stderr)
        return 2

    if a.audit:
        r = audit_day(con, a.audit)
        if a.json:
            print(json.dumps(r, indent=2, default=float))
            return 0
        print(f"{a.audit}: {r['walks']} walks, {r['usable']} usable, "
              f"{r['flagged']} with something to say about them")
        if r["mean_hours"]:
            print(f"  mean does-take {r['mean_hours']:.2f} h")
        for row in r["rows"][:12]:
            print(f"  {row['block_id']}  {row['does_take_h']:.2f}h  "
                  f"segments {row['segments_covered']}/{row['segments_in_block']}  "
                  f"time coverage {row['time_coverage']}")
            for n in row["notes"]:
                print(f"      - {n}")
        return 0

    if a.fences:
        bd = con.execute(
            "SELECT d.block_day_id FROM block_days d WHERE d.block_id=? AND EXISTS "
            "(SELECT 1 FROM pings p WHERE p.block_day_id=d.block_day_id) "
            "ORDER BY d.date DESC LIMIT 1", (a.fences,)).fetchone()
        if not bd:
            print(f"no raw pings retained for {a.fences}", file=sys.stderr)
            return 1
        r = buffer_sensitivity(con, bd["block_day_id"])
        o = overlap_report(con, bd["block_day_id"])
        if a.json:
            print(json.dumps({"width": r, "overlap": o}, indent=2, default=float))
            return 0
        print(f"{a.fences}: {r['pings']} pings, median accuracy {r['median_accuracy_m']} m, "
              f"elapsed {r['elapsed_s'] / 3600:.2f} h")
        print(f"  {'buffer':>9} {'does-take':>10} {'coverage':>9} {'pings outside':>14}")
        for row in r["rows"]:
            mark = ("  <- as built" if abs(row["scale"] - 1) < 1e-9 else
                    "  <- suggested" if row["scale"] == r["suggested_scale"] else "")
            print(f"  {row['mean_buffer_m']:7.1f} m {row['does_take_h']:9.2f}h "
                  f"{100 * row['coverage']:8.0f}% {100 * row['pings_outside_share']:13.0f}%"
                  f"{mark}")
        print(f"  {r['note']}")
        print(f"\n  a ping sits inside {o['mean_fences_per_ping']} fences on average "
              f"(worst {o['max_fences_per_ping']})")
        print(f"  {o['verdict']}")
        return 0

    m = fit(con=con, to_date=a.to_date)

    if a.model:
        if a.json:
            print(json.dumps(m.as_dict(), indent=2))
            return 0
        print(f"fitted on {m.blocks_used:,} walks and {m.passes_used:,} segment passes")
        print("\n  pace at work, by area")
        for k, v in sorted(m.pace_mps.items()):
            if k != "*":
                print(f"    {k:9s} {v:.2f} m/s  ({v * 3.6:.1f} km/h)")
        print(f"    a walker covers ground {100 * (m.deadhead_bonus - 1):.0f}% faster "
              f"on a street with nothing to deliver")
        print("\n  seconds at a door, by area and kind (the walk up the path is in here)")
        for k in sorted(x for x in m.service_s if "|" in x):
            print(f"    {k:22s} {m.service_s[k]:6.1f}")
        print("\n  seconds per item")
        for k in ITEM_KINDS:
            print(f"    {k:10s} {m.item_s.get(k, 0):6.2f}")
        clear = m.weather.get("clear", 1.0)
        print("\n  what the day does to the walking (against a clear day)")
        for k, v in sorted(m.weather.items(), key=lambda kv: -kv[1]):
            print(f"    {k:12s} {v / clear:.2f}x")
        print(f"    {'after dark':12s} {m.dark / clear:.2f}x")
        print(f"    {'relief cover':12s} {m.relief / clear:.2f}x")
        for k in ("first month", "1-3 months", "3-12 months", "a year or more"):
            if k in m.experience:
                print(f"    {k:14s} {m.experience[k] / clear:.2f}x")
        print(f"\n  {len(m.walker)} walkers have a figure of their own")
        print(f"  a block lands {100 * m.common_mode:.1f}% either side of its forecast")
        for n in m.notes:
            print(f"  - {n}")
        return 0

    if a.routes:
        r = fleet_saving(con, m)
        if a.json:
            print(json.dumps(r, indent=2, default=float))
            return 0
        print(f"{r['blocks']} blocks: a better route helps on "
              f"{r['blocks_where_it_helps']}, {r['blocks_already_good']} are already "
              f"as good as it gets")
        print(f"  {r['minutes_a_day_total']:.0f} minutes a day across all of them "
              f"= {r['hours_a_year_total']:.0f} walker-hours a year")
        print(f"\n  {'block':12s}{'area':10s}{'min/day':>9}{'m/day':>8}{'h/year':>9}")
        for row in r["worst_first"][:15]:
            print(f"  {row['block_id']:12s}{row['area_type']:10s}"
                  f"{row['minutes_a_day']:9.1f}{row['metres_a_day']:8.0f}"
                  f"{row['hours_a_year']:9.0f}")
        return 0

    if a.block:
        date = a.date or con.execute(
            "SELECT max(date) d FROM block_days WHERE block_id=?",
            (a.block,)).fetchone()["d"]
        bd = con.execute("SELECT block_day_id FROM block_days WHERE block_id=? AND date=?",
                         (a.block, date)).fetchone()
        statics = block_statics(con)
        s = statics.get(a.block)
        if not s:
            print(f"no block {a.block}", file=sys.stderr)
            return 2
        mail = expected_mail(con, date, a.block)
        f = forecast(m, s, mail, date)
        print(f"{a.block}   {date}   {s['area_type']}, {s['dps']} delivery points, "
              f"{s['serviced_m'] / 1000:.2f} km of street")
        print(f"\n  SHOULD TAKE   {hm(f.should_take_s)}   "
              f"({hm(f.low_s)} to {hm(f.high_s)}, {f.confidence}% of the time)")
        for k, v in f.parts.items():
            print(f"      {k.replace('_s', ''):10s} {hm(v)}")
        if bd:
            try:
                bt = block_time(con, bd["block_day_id"])
                print(f"\n  DOES TAKE     {hm(bt.does_take_s)}   from {bt.passes} passes "
                      f"over {bt.segments_covered}/{bt.segments_in_block} segments "
                      f"({bt.source}-derived)")
                if bt.coverage:
                    print(f"      {100 * bt.coverage:.0f}% of the elapsed time was inside "
                          f"a geofence")
                for n in bt.notes:
                    print(f"      - {n}")
                gap = bt.does_take_s - f.should_take_s
                print(f"      against the forecast: {gap / 60:+.0f} min "
                      f"({100 * gap / f.should_take_s:+.1f}%)")
            except LookupError as e:
                print(f"\n  DOES TAKE     not available: {e}")
        print(f"\n  THE ROUTE")
        for o in options(con, a.block, m):
            print(f"      {o.name[:38]:38s}{o.total_m:7.0f} m  "
                  f"{o.deadhead_m:6.0f} m of it deadhead  {o.minutes:6.1f} min")
        sv = saving(con, a.block, m)
        if sv.get("minutes_a_day", 0) > 0.5:
            print(f"      -> {sv['minutes_a_day']} min a day, "
                  f"{sv['hours_a_year']} walker-hours a year")
        return 0

    if a.day:
        rows = plan_day(con, m, a.day)
        if a.json:
            print(json.dumps([r.as_dict() for r in rows], indent=2, default=float))
            return 0
        print(f"{a.day}")
        print(f"  {'office':8s}{'region':15s}{'blocks':>7}{'hours':>8}{'90% range':>16}"
              f"{'need':>6}{'have':>6}{'overtime':>10}")
        for r in rows:
            print(f"  {r.do_id:8s}{r.region[:14]:15s}{r.blocks:7d}{r.expected_h:8.1f}"
                  f"{r.low_h:7.1f}-{r.high_h:5.1f}{r.walkers_needed:6.0f}"
                  f"{r.walkers_available:6d}{r.overtime_h:9.1f}h")
            for n in r.notes:
                print(f"      - {n}")
        return 0

    if a.ahead:
        start = a.date or (dt.date.today()).isoformat()
        r = plan_range(con, m, start, days=a.ahead)
        if a.json:
            print(json.dumps(r, indent=2, default=float))
            return 0
        print(f"{r['days']} working days from {start}: mean {r['mean_hours_a_day']} "
              f"hours a day across {len(r['offices'])} offices")
        p2 = r["peak_day"]
        print(f"  busiest {p2['date']}: {p2['hours']}h ({p2['low']}-{p2['high']}), "
              f"{p2['overtime']}h overtime, {p2['split']} blocks too long for a shift")
        print(f"\n  {'date':12s}{'hours':>8}{'overtime':>10}{'over shift':>12}{'short':>7}")
        for x in r["rows"]:
            if x["overtime"] > 0.5 or x["shortfall"] > 0:
                print(f"  {x['date']:12s}{x['hours']:8.1f}{x['overtime']:9.1f}h"
                      f"{x['over']:12d}{x['shortfall']:7.1f}")
        return 0

    if a.peak:
        r = peak_report(con, m, a.peak)
        if a.json:
            print(json.dumps(r, indent=2, default=float))
            return 0
        print(f"November and December {r['year']}, by week")
        print(f"  {'week from':12s}{'hours/day':>11}{'overtime/day':>14}"
              f"{'to split':>10}{'extra people':>14}")
        for wk in r["weeks"]:
            print(f"  {wk['week']:12s}{wk['hours_a_day']:11.1f}{wk['overtime_a_day']:13.1f}h"
                  f"{wk['splits_a_day']:10.1f}{wk['extra_people']:14.1f}")
        print(f"\n  the peak week is {r['peak_over_base']}x the quietest, and needs "
              f"{r['extra_people_at_peak']} people's worth of extra work")
        return 0

    print(json.dumps(counts(con), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
