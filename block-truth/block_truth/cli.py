"""
Command line for block completion times.

    block-truth 2026-05-30 20260530-D01-V00    one block, with its working
    block-truth 2026-05-30                      every block that day
    block-truth --audit 2026-05-30              only blocks whose records disagree
    block-truth --dates                         what dates are on file
    block-truth --import round.xlsx --depot D01 --date 2026-06-02
    block-truth --calibration                   what the estimator learned
    block-truth 2026-05-30 --csv > labels.csv   for scoring a forecast against

A block that was recorded gets a ground truth; a block that was entered gets an
estimate with an earliest and a latest. The output always says which it is.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys

from .db import DEFAULT_DB
from .itinerary import Limits
from .fmt import hhmm, hours
from .repository import Repository
from .truth import TruthError

FIELDS = ["block_id", "delivery_date", "kind", "vehicle_id", "vehicle_type", "depot_id",
          "city", "stops", "start_time", "completion_time", "completion_expected",
          "completion_min", "completion_max", "duration_minutes", "expected_minutes",
          "min_minutes", "max_minutes", "service_minutes", "travel_minutes",
          "depot_minutes", "ok"]


def explain(v: dict) -> str:
    est = v["kind"] == "estimate"
    who = " / ".join(str(v.get(k)) for k in ("vehicle_id", "vehicle_type", "depot_id", "city")
                     if v.get(k))
    L = [f"{v['block_id']}   {v['delivery_date']}" + (f"   {who}" if who else ""),
         f"   [{'ESTIMATE - this has not happened yet' if est else 'GROUND TRUTH'}]", ""]
    if est:
        L += [
            f"  expected completion  {v['completion_expected']}       <- the answer",
            f"  earliest / latest    {v['completion_min']}  to  {v['completion_max']}"
            f"   ({v['confidence']}% of the time)",
            f"  duration             {hours(v['expected_minutes'])}"
            f"   ({hours(v['min_minutes'])} to {hours(v['max_minutes'])})"
            f" over {v['stops']} stops",
            "",
            "  built from",
            f"    depot, load/brief  {v['depot_minutes']:8.1f} min",
            f"  + travel, all legs   {v['travel_minutes']:8.1f} min"
            f"   ({v['route_km']:.1f} km driven)",
            f"  + service at doors   {v['service_minutes']:8.1f} min"
            f"   ({v['expected_failures']:.1f} expected failures)",
            f"  + return to depot    {v['return_minutes']:8.1f} min",
            f"  = expected           {hours(v['expected_minutes'])}",
            "",
            f"  spread  {v['independent_sd']:.1f} min of per-stop noise that cancels,"
            f"  {v['common_mode_sd']:.1f} min common to the whole day",
            f"  basis   {v['basis']['implied_speed_kmh']} km/h in {v['basis']['city']},"
            f" {v['basis']['fixed_per_stop']} min parking per stop,"
            f" fitted on {v['basis']['blocks_behind_it']} recorded rounds",
        ]
    else:
        L += [
            f"  completion time      {v['completion_time']}       <- the answer",
            f"  duration             {hours(v['duration_minutes'])}"
            f"   ({v['duration_minutes']:.1f} min over {v['stops']} stops,"
            f" {v['minutes_per_stop']:.2f} min each)",
            "",
            "  how it is made up",
            f"    block start        {v['start_time']}",
            f"  + depot, load/brief  {v['depot_minutes']:8.1f} min",
            f"  + travel, all legs   {v['travel_minutes']:8.1f} min"
            f"   (includes the run out to stop 1)",
            f"  + service at doors   {v['service_minutes']:8.1f} min"
            f"   ({v['delivered']} delivered, {v['failed']} not)",
            f"  + return to depot    {v['return_leg_minutes']:8.1f} min"
            f"   (residual: nobody is served on the way home)",
            f"  = completion         {v['completion_time']}",
            "",
            f"  first door {v['first_arrival_time']}   last door {v['last_arrival_time']}"
            f"   door to door {hours(v['door_to_door_minutes'])}",
            f"  source: {v['completion_source']}",
            "",
            "  checks",
        ]
        for c in v["checks"]:
            L.append(f"    [{'ok ' if c['ok'] else 'FAIL'}] {c['name']}: {c['detail']}")
        if not v["ok"]:
            L += ["", "  The records disagree with themselves. The completion time above",
                  "  is what they say; the failing check says why it should not be trusted."]
    r = v.get("route")
    if r:
        L += ["", f"  path: depot {r['depot']['id']} -> {len(r['nodes'])} stops -> depot,"
                  f" {r['total_km']:.1f} km driven"]
    return "\n".join(L)


def itinerary_table(d: dict) -> str:
    """The options for one planned round, side by side, with the trade-off spelled out."""
    b, c = d["block"], d["compare"]
    rec, chosen = d.get("recommendation", {}), d.get("chosen", {})
    L = [f"{b['block_id']}   {b['delivery_date']}   {b['vehicle_id']} / "
         f"{b['vehicle_type']} / {b['depot_id']} / {b['city']}",
         "   [PLANNED - NOT YET DRIVEN: every number below is a forecast]", ""]
    head = (f"  {'option':<22}{'duration':>10}{'earliest-latest':>18}{'km':>8}"
            f"{'windows':>10}{'wait':>7}  notes")
    L += [head, "  " + "-" * (len(head) - 2)]
    for o in d["options"]:
        tags = []
        if o.get("chosen"):
            tags.append("CHOSEN")
        if o.get("recommended"):
            tags.append("recommended")
        if o["key"] == c["fastest"]:
            tags.append("fastest")
        if o["key"] == c["shortest"]:
            tags.append("least driving")
        if o["key"] == c["most_promises_kept"]:
            tags.append("most promises")
        if o["key"] == c["most_predictable"]:
            tags.append("most predictable")
        L.append(
            f"  {o['name'][:22]:<22}{hours(o['expected_minutes']):>10}"
            f"{o['completion_min'] + '-' + o['completion_max']:>18}"
            f"{o['route_km']:>8.1f}"
            f"{str(o['windows_met']) + '/' + str(o['windows_total']):>10}"
            f"{o['wait_minutes']:>6.0f}m  " + ", ".join(tags))
        if o["breaches"]:
            L.append(f"  {'':22}  ! {'; '.join(o['breaches'])}")
    if rec:
        L += ["", f"  RECOMMENDED: {rec['name']}", f"    {rec['why']}.",
              f"    The price of a broken promise is a policy, not a measurement: "
              f"{rec['minutes_per_missed_window']:.0f} min here, and "
              f"--per-missed-window changes it.",
              "",
              f"    {'option':<24}{'hours':>7}{'broken':>8}{'penalty':>9}{'score':>8}"]
        for s in rec["scores"]:
            L.append(f"    {s['name'][:24]:<24}{s['minutes'] / 60:6.2f}h"
                     f"{s['broken_windows']:8.1f}{s['penalty_minutes']:8.0f}m"
                     f"{s['score_minutes']:8.0f}"
                     + ("   set aside: over a limit" if s["set_aside"] else ""))
        if rec.get("all_over_the_limits"):
            L.append("    Every option breaks an operating limit, so this is the "
                     "least bad one rather than a legal one.")
    if chosen.get("key"):
        L += ["", f"  CHOSEN: '{chosen['key']}' by {chosen['by']} at {chosen['at']}"
                  + ("" if chosen.get("is_recommended")
                     else " - against the recommendation, which is allowed and recorded")]
    else:
        L += ["", "  Nobody has chosen yet: the round is in the order it was planned. "
                  "--choose OPTION sets it."]
    L += ["", f"  Between best and worst: {c['minutes_between_best_and_worst']:.0f} min, "
              f"{c['km_between_best_and_worst']:.0f} km, "
              f"{c['windows_between_best_and_worst']} delivery windows."]
    if c.get("agree"):
        L.append("  Every option drives the same order: the stops are close enough "
                 "together that it does not matter.")
    L.append("")
    for o in d["options"]:
        L.append(f"  {o['name']}: {o['why']}")
        L.append(f"      gives up: {o['gives_up']}")
    cond = d["conditions"]
    if cond["disruptions"]:
        L += ["", "  declared on this date:"]
        for x in cond["disruptions"]:
            L.append(f"    {x['disruption_id']}  {x['kind']}  {x['description']}")
    return "\n".join(L)


def table(rows: list[dict]) -> str:
    head = (f"{'block':<24}{'veh':<11}{'city':<13}{'stops':>6}{'start':>7}"
            f"{'done':>8}{'dur':>9}  kind")
    out = [head, "-" * len(head)]
    for v in rows:
        est = v["kind"] == "estimate"
        out.append(
            f"{v['block_id']:<24}{str(v.get('vehicle_id') or ''):<11}"
            f"{str(v.get('city') or '')[:12]:<13}{v['stops']:>6}"
            f"{v['start_time']:>7}"
            f"{(v['completion_expected'] if est else v['completion_time']):>8}"
            f"{hours(v['expected_minutes'] if est else v['duration_minutes']):>9}"
            f"  {'estimate' if est else ('truth' if v.get('ok') else 'truth FAIL')}")
    if rows:
        mins = [v["expected_minutes"] if v["kind"] == "estimate" else v["duration_minutes"]
                for v in rows]
        bad = sum(1 for v in rows if v["kind"] == "truth" and not v.get("ok", True))
        out += ["-" * len(head),
                f"{len(rows)} blocks, {sum(v['stops'] for v in rows):,} stops, "
                f"{sum(mins) / 60:,.1f} vehicle hours, mean {hours(sum(mins) / len(mins))}"
                + (f", {bad} failing checks" if bad else "")]
    return "\n".join(out)


def _views(repo, date, block, conf):
    if block:
        return [repo.block_view(block, confidence=conf)]
    out = []
    for b in repo.blocks(date=date):
        try:
            out.append(repo.block_view(b["block_id"], confidence=conf))
        except TruthError:
            pass
    return out


def main(argv: list[str] | None = None) -> int:
    try:
        return _main(argv)
    except BrokenPipeError:
        try:
            sys.stdout.close()
        finally:
            return 0


def _main(argv: list[str] | None = None) -> int:
    # allow_abbrev=False on purpose: with it on, "--date 2026-06-02" is a unique
    # prefix of "--dates" and silently turns an import into a date listing.
    p = argparse.ArgumentParser(prog="block-truth", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                allow_abbrev=False)
    p.add_argument("date", nargs="?", help="delivery date, YYYY-MM-DD")
    p.add_argument("--date", dest="date_opt", help="the date, spelled as a flag")
    p.add_argument("block", nargs="?", help="block id; omit for every block that day")
    p.add_argument("--db", default=str(DEFAULT_DB), help=f"default {DEFAULT_DB}")
    p.add_argument("--confidence", type=int, default=90, choices=[80, 90, 95, 99])
    p.add_argument("--json", action="store_true")
    p.add_argument("--csv", action="store_true")
    p.add_argument("--audit", action="store_true", help="only blocks that fail a check")
    p.add_argument("--dates", action="store_true")
    p.add_argument("--calibration", action="store_true",
                   help="what the estimator learned from the recorded rounds")
    p.add_argument("--schedule", action="store_true",
                   help="planned rounds for a date, with their predictions")
    p.add_argument("--plan-month", type=int, metavar="DAYS", nargs="?", const=30,
                   help="(re)build the plan for the next DAYS days, default 30")
    p.add_argument("--itineraries", metavar="BLOCK",
                   help="ways to drive one planned round, each costed")
    p.add_argument("--choose", metavar="OPTION",
                   help="with --itineraries: drive this one from now on "
                        "(as_planned, shortest, windows, steady, as_received)")
    p.add_argument("--by", default="operator", help="who is choosing, for the record")
    p.add_argument("--reset-order", action="store_true",
                   help="with --itineraries: back to the order the work arrived in")
    p.add_argument("--per-missed-window", type=float, metavar="MIN",
                   help="what a broken promise is worth when recommending (default 20)")
    p.add_argument("--datacheck", action="store_true",
                   help="rows that will break something later, found before they do")
    p.add_argument("--conditions", action="store_true",
                   help="the measured traffic curve and the declared disruptions")
    p.add_argument("--no-traffic", dest="traffic", action="store_false",
                   help="forecast a flat day, ignoring the measured curve")
    p.add_argument("--max-minutes", type=float), p.add_argument("--min-minutes", type=float)
    p.add_argument("--max-km", type=float), p.add_argument("--max-stops", type=int)
    p.add_argument("--import", dest="xlsx", help="load a round from a workbook")
    p.add_argument("--depot", help="depot for --import")
    p.add_argument("--start", default="07:30")
    p.add_argument("--vehicle-type", dest="vt", default="Large Van")
    p.add_argument("--block-id", dest="bid")
    a = p.parse_args(argv)
    if a.date_opt:
        if a.date and a.block:
            p.error("the date was given twice")
        if a.date:              # "block-truth --date D B" style: shift along
            a.block = a.block or a.date
        a.date = a.date_opt

    try:
        repo = Repository(a.db)
    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    try:
        if a.dates:
            for d in repo.dates():
                print(d)
            return 0

        if a.calibration:
            print(json.dumps(repo.calibration().as_dict(), indent=2))
            return 0

        if a.plan_month is not None:
            from .schedule import generate
            out = generate(repo.path, days=a.plan_month, from_date=a.date)
            print(f"planned {out['planned_written']} rounds "
                  f"({out['planned_stops']:,} stops) for {out['vehicles']} vehicles, "
                  f"{out['from']} to {out['to']}")
            return 0

        if a.datacheck:
            con = repo.con
            bad = 0
            checks = [
                ("stops with no coordinates",
                 "SELECT block_id, sequence, address FROM stops "
                 "WHERE lat IS NULL OR lon IS NULL LIMIT 20"),
                ("stops that are not on Earth",
                 "SELECT block_id, sequence, lat, lon FROM stops "
                 "WHERE lat NOT BETWEEN -90 AND 90 OR lon NOT BETWEEN -180 AND 180 "
                 "LIMIT 20"),
                ("blocks with no stops",
                 "SELECT block_id, delivery_date, source FROM blocks b WHERE NOT EXISTS "
                 "(SELECT 1 FROM stops s WHERE s.block_id=b.block_id) LIMIT 20"),
                ("blocks with a gap in the stop numbering",
                 "SELECT block_id, count(*) stops, min(sequence) lo, max(sequence) hi "
                 "FROM stops GROUP BY block_id HAVING hi - lo + 1 <> stops LIMIT 20"),
                ("depots with no coordinates",
                 "SELECT depot_id, name FROM depots "
                 "WHERE lat IS NULL OR lon IS NULL LIMIT 20"),
            ]
            for title, sql in checks:
                rows = con.execute(sql).fetchall()
                if rows:
                    bad += len(rows)
                    print(f"{title}: {len(rows)}")
                    for r in rows[:10]:
                        print("   " + "  ".join(f"{k}={r[k]}" for k in r.keys()))
            print("nothing wrong with the data" if not bad
                  else f"\n{bad} rows will break a forecast; fix or delete them")
            return 0 if not bad else 1

        if a.conditions:
            t = repo.traffic()
            print(f"TRAFFIC - measured from {t.legs_used:,} recorded legs")
            print("  relative to each city's own average pace; 1.00 is typical\n")
            for h in range(5, 22):
                f, n = t.hour.get(h, 1.0), t.hour_n.get(h, 0)
                bar = "#" * int(round(abs(f - 1) * 120))
                side = "slower" if f >= 1 else "faster"
                print(f"   {h:02d}:00  {f:5.3f}  {bar:<30s} {side if bar else '':6s}"
                      f"  {n:>6,} legs" + ("   (thin evidence)" if n < 80 else ""))
            print("\n  by weekday: " + "  ".join(
                f"{['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][d]} {t.weekday.get(d, 1):.3f}"
                for d in range(7) if t.weekday_n.get(d)))
            if t.city_amplitude:
                print("  how sharply each city peaks: " + "  ".join(
                    f"{c} {v:.2f}x" for c, v in sorted(t.city_amplitude.items(),
                                                       key=lambda kv: -kv[1])))
            ds = repo.disruptions(a.date)
            print(f"\nDISRUPTIONS - declared, not measured ({len(ds)}"
                  + (f" active on {a.date}" if a.date else " on file") + ")")
            for d in ds:
                when = (f"{d.from_date}..{d.to_date}"
                        + (f" {d.from_time}-{d.to_time}" if d.from_time else " all day"))
                effect = ("driven around" if d.kind == "closure"
                          else f"{d.severity:.1f}x slower")
                print(f"   {d.disruption_id}  {d.kind:<10s} {d.description[:38]:<38s} "
                      f"{d.radius_km:4.1f} km  {when:<28s} {effect}")
            if not ds:
                print("   nothing declared: the forecasts assume clear roads apart "
                      "from the usual traffic")
            return 0

        lim = Limits(a.min_minutes, a.max_minutes, a.max_km, a.max_stops)

        if a.itineraries:
            if a.reset_order:
                print(repo.clear_choice(a.itineraries)["restored"]
                      + f" for {a.itineraries}")
            if a.choose:
                d = repo.choose_itinerary(a.itineraries, a.choose, by=a.by,
                                          confidence=a.confidence,
                                          use_traffic=a.traffic, limits=lim)
                ch = d["chosen"]
                print(f"{a.itineraries} is now driving '{ch['key']}', chosen by "
                      f"{ch['by']} at {ch['at']}.")
                print()
            else:
                d = repo.itineraries(a.itineraries, confidence=a.confidence,
                                     use_traffic=a.traffic, limits=lim,
                                     minutes_per_missed_window=a.per_missed_window)
            if a.json:
                print(json.dumps(d, indent=2, default=float))
                return 0
            print(itinerary_table(d))
            return 0

        if a.schedule:
            rows = repo.schedule(date=a.date, depot_id=a.depot, confidence=a.confidence,
                                 use_traffic=a.traffic, limits=lim)
            if a.json:
                print(json.dumps(rows, indent=2, default=float))
            elif not rows:
                print("nothing planned for that date. Build a month with --plan-month")
            else:
                print(table(rows))
                over = [r for r in rows if r["breaches"]]
                for r in over:
                    print(f"  ! {r['block_id']}: {'; '.join(r['breaches'])}")
            return 0

        if a.xlsx:
            from .excel import parse
            if not a.depot or not a.date:
                print("error: --import needs --depot and a date", file=sys.stderr)
                return 2
            with open(a.xlsx, "rb") as fh:
                parsed = parse(fh.read())
            if not parsed.ok:
                print(f"{len(parsed.errors)} problem(s); nothing saved:", file=sys.stderr)
                for e in parsed.errors:
                    print(f"  row {e.row}  {e.column}: {e.problem}", file=sys.stderr)
                return 1
            bid = a.bid or parsed.meta.get("blockid") \
                or repo.suggest_block_id(a.date, a.depot)
            repo.create_block(bid, a.date, a.depot, parsed.stops,
                              start_time=parsed.meta.get("starttime", a.start),
                              vehicle_type=parsed.meta.get("vehicletype", a.vt),
                              source="excel")
            print(explain(repo.block_view(bid, confidence=a.confidence)))
            return 0

        if not a.date:
            p.print_help()
            return 1

        rows = _views(repo, a.date, a.block, a.confidence)
        looked_at = len(rows)
        if a.audit:
            rows = [v for v in rows if v["kind"] == "truth" and not v.get("ok", True)]

        if a.json:
            print(json.dumps(rows, indent=2, default=float))
        elif a.csv:
            w = csv.DictWriter(sys.stdout, fieldnames=FIELDS, extrasaction="ignore")
            w.writeheader()
            for v in rows:
                w.writerow(v)
        elif len(rows) == 1 and a.block:
            print(explain(rows[0]))
        elif not rows:
            if a.audit:
                print(f"{looked_at} recorded block(s) checked, none disagree with themselves"
                      if looked_at else "no blocks on that date")
            else:
                print("no blocks matched")
        else:
            print(table(rows))
        return 0 if all(v.get("ok", True) for v in rows) else 1
    except TruthError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    finally:
        repo.close()


if __name__ == "__main__":
    raise SystemExit(main())
