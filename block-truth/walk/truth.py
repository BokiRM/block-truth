"""
"Does take": their number, computed their way, and then audited.

The sum of a device's segment traversal times is what their second slide
produces, and this module reproduces it exactly. What it adds is the part the
slides leave open, because every one of these can change the answer by more
than the thing being measured:

**Double counting.** A buffer around a 17 m segment with a 10 m width overlaps
its neighbours, so a ping sits inside several fences at once. Reading "entry and
exit per segment" as "every fence the ping is in" inflates the total two- or
three-fold. Reading it as "the one segment the ping was matched to" does not.
`overlap_report` measures how far apart those two readings are on real pings.

**Coverage.** Traversal time only counts while the device is inside a fence. The
walk from the van, a crossing, a break, a doorstep down a 40 m drive - none of
it is. If the fences only cover 70% of the elapsed time, "does take" is not how
long the block took, and planning against it under-resources the day.

**Passes.** A street walked twice yields two traversals. Summed over a block
that is right; divided by segment it is not, and a per-segment pace fitted
without separating the delivering pass from the dead one is fitted on a mixture.

**Who walked it.** Their output is per device. A block is not. Which device, or
what function of them, becomes "the" time for the block is a decision, and it is
made here rather than left to whoever writes the next query.
"""
from __future__ import annotations

import statistics as st
from dataclasses import dataclass, field

from .graph import point_segment_distance_m
from .passes import naive_total_seconds, passes_from, total_seconds
from .schema import WALKING_SPEED_LIMIT_MPS, connect

# A device whose fences covered less of the block than this did not walk it all,
# and its total is not the block's time.
MIN_COVERAGE = 0.80
GAP_S = 45.0


@dataclass
class BlockTime:
    """How long one walk took, and how much that figure can be trusted."""
    block_day_id: int
    block_id: str
    date: str
    does_take_s: float
    source: str                       # ping | simulated
    devices: int
    device_spread_s: float
    segments_in_block: int
    segments_covered: int
    passes: int
    elapsed_s: float | None
    coverage: float | None            # does_take / elapsed
    serviced_s: float = 0.0
    deadhead_s: float = 0.0
    notes: list[str] = field(default_factory=list)

    @property
    def hours(self) -> float:
        return self.does_take_s / 3600.0

    @property
    def usable(self) -> bool:
        return (self.segments_covered / max(self.segments_in_block, 1) >= MIN_COVERAGE
                and self.does_take_s > 0)

    def as_dict(self) -> dict:
        return {
            "block_day_id": self.block_day_id, "block_id": self.block_id,
            "date": self.date, "does_take_s": round(self.does_take_s, 1),
            "does_take_h": round(self.hours, 3), "source": self.source,
            "devices": self.devices,
            "device_spread_s": round(self.device_spread_s, 1),
            "segments_in_block": self.segments_in_block,
            "segments_covered": self.segments_covered,
            "segment_coverage": round(self.segments_covered
                                      / max(self.segments_in_block, 1), 3),
            "passes": self.passes,
            "elapsed_s": round(self.elapsed_s, 1) if self.elapsed_s else None,
            "time_coverage": round(self.coverage, 3) if self.coverage else None,
            "serviced_s": round(self.serviced_s, 1),
            "deadhead_s": round(self.deadhead_s, 1),
            "usable": self.usable, "notes": self.notes,
        }


def block_time(con, block_day_id: int) -> BlockTime:
    """Their sum, with the devices reconciled and the gaps named."""
    bd = con.execute(
        "SELECT d.*, b.block_id, b.area_type, "
        "(SELECT count(*) FROM osm_segments s WHERE s.block_id=d.block_id) segs "
        "FROM block_days d JOIN blocks b USING(block_id) WHERE d.block_day_id=?",
        (block_day_id,)).fetchone()
    if not bd:
        raise LookupError(f"no block day {block_day_id}")

    rows = con.execute(
        "SELECT * FROM segment_traversals WHERE block_day_id=? ORDER BY entry_ts",
        (block_day_id,)).fetchall()
    notes: list[str] = []

    if not rows:
        tot = con.execute(
            "SELECT * FROM block_traversal_totals WHERE block_day_id=?",
            (block_day_id,)).fetchall()
        if not tot:
            raise LookupError(f"block day {block_day_id} has no traversals at all")
        best = max(tot, key=lambda r: r["segments_covered"])
        notes.append("no per-segment detail retained for this day: the block total is "
                     "all there is, so coverage and passes cannot be checked")
        return BlockTime(
            block_day_id, bd["block_id"], bd["date"], best["does_take_s"],
            best["source"], len(tot), 0.0, bd["segs"], best["segments_covered"],
            best["passes"],
            (best["last_exit_ts"] - best["first_entry_ts"])
            if best["first_entry_ts"] is not None else None,
            None, notes=notes)

    by_device: dict[str, list] = {}
    for r in rows:
        by_device.setdefault(r["device_id"], []).append(r)

    totals = {}
    for dev, rs in by_device.items():
        covered = len({r["segment_id"] for r in rs})
        totals[dev] = {
            "sum": sum(r["traversal_s"] for r in rs),
            "covered": covered,
            "passes": len(rs),
            "first": min(r["entry_ts"] for r in rs),
            "last": max(r["exit_ts"] for r in rs),
            "serviced": sum(r["traversal_s"] for r in rs if r["serviced"]),
            "dead": sum(r["traversal_s"] for r in rs if not r["serviced"]),
        }

    # one device becomes the block's time: the one that covered the most of it,
    # and among those the median rather than the quickest. A block is not the
    # best anybody has ever done it in.
    ranked = sorted(totals.items(),
                    key=lambda kv: (-kv[1]["covered"], kv[1]["sum"]))
    best_cover = ranked[0][1]["covered"]
    contenders = [kv for kv in ranked if kv[1]["covered"] == best_cover]
    if len(contenders) > 1:
        notes.append(f"{len(contenders)} devices walked this block; the median of "
                     f"them is reported, not the quickest")
    pick = contenders[len(contenders) // 2]
    dev, t = pick

    spread = (max(x["sum"] for x in totals.values())
              - min(x["sum"] for x in totals.values())) if len(totals) > 1 else 0.0
    elapsed = t["last"] - t["first"]
    coverage = t["sum"] / elapsed if elapsed > 0 else None

    if coverage is not None and coverage > 1.02:
        notes.append(f"the segment times add up to {100 * coverage:.0f}% of the time "
                     f"between the first entry and the last exit, which is impossible "
                     f"unless fences overlap and a ping is being counted more than once")
    elif coverage is not None and coverage < 0.75:
        notes.append(f"only {100 * coverage:.0f}% of the walk happened inside a "
                     f"geofence: the rest - the walk from the van, a break, doorsteps "
                     f"off the road - is not in this figure")
    if t["covered"] < bd["segs"] * MIN_COVERAGE:
        notes.append(f"this device covered {t['covered']} of {bd['segs']} segments, so "
                     f"its total is part of a block rather than a block")
    repeats = t["passes"] - t["covered"]
    if repeats > 0:
        notes.append(f"{repeats} segments were walked more than once; every pass is in "
                     f"the total, which is right for the block and wrong per segment")

    return BlockTime(block_day_id, bd["block_id"], bd["date"], t["sum"],
                     rows[0]["source"], len(totals), spread, bd["segs"], t["covered"],
                     t["passes"], elapsed, coverage, t["serviced"], t["dead"], notes)


def overlap_report(con, block_day_id: int) -> dict:
    """How far apart the two readings of their method are, on this day's pings.

    Recomputed from the raw pings rather than taken on trust, because the whole
    question is which rule the entry and exit times were produced under.
    """
    pings = con.execute(
        "SELECT ping_id, ts, lat, lon, speed_mps FROM pings WHERE block_day_id=? "
        "ORDER BY ts", (block_day_id,)).fetchall()
    if not pings:
        return {"pings": 0, "note": "no raw pings retained for this day"}

    bd = con.execute("SELECT block_id FROM block_days WHERE block_day_id=?",
                     (block_day_id,)).fetchone()
    segs = con.execute(
        "SELECT s.segment_id, s.from_lat, s.from_lon, s.to_lat, s.to_lon, "
        "       g.buffer_m FROM osm_segments s JOIN segment_geofence g USING(segment_id) "
        "WHERE s.block_id=?", (bd["block_id"],)).fetchall()

    inside_counts = []
    any_fence: dict[str, list[float]] = {}
    timeline: list[tuple[float, str | None]] = []
    walking = 0
    for p in pings:
        hits = []
        best, best_d = None, 1e9
        for s in segs:
            d, _ = point_segment_distance_m(p["lat"], p["lon"], s["from_lat"],
                                            s["from_lon"], s["to_lat"], s["to_lon"])
            if d < best_d:
                best, best_d = s["segment_id"], d
            if d <= s["buffer_m"]:
                hits.append(s["segment_id"])
        inside_counts.append(len(hits))
        walking += 1 if p["speed_mps"] < WALKING_SPEED_LIMIT_MPS else 0
        for sid in hits:
            any_fence.setdefault(sid, []).append(p["ts"])
        timeline.append((p["ts"], best if best in hits else None))

    naive = naive_total_seconds(any_fence, GAP_S)
    honest = total_seconds(passes_from(timeline))
    elapsed = pings[-1]["ts"] - pings[0]["ts"]
    return {
        "pings": len(pings),
        "walking_pings": walking,
        "walking_share": round(walking / len(pings), 3),
        "mean_fences_per_ping": round(st.fmean(inside_counts), 2),
        "max_fences_per_ping": max(inside_counts),
        "pings_in_no_fence": sum(1 for c in inside_counts if c == 0),
        "elapsed_s": round(elapsed, 1),
        "geofence_any_s": round(naive, 1),
        "nearest_match_s": round(honest, 1),
        "inflation": round(naive / honest, 3) if honest else None,
        "any_vs_elapsed": round(naive / elapsed, 3) if elapsed else None,
        "nearest_vs_elapsed": round(honest / elapsed, 3) if elapsed else None,
        "verdict": (
            "Counting a ping for every fence it is inside inflates this walk by "
            f"{naive / honest:.1f}x. Whichever rule their production job uses decides "
            "whether every does-take number downstream is right or multiplied."
            if honest and naive / honest > 1.15 else
            "The two readings agree on this block: its fences barely overlap."),
    }


def audit_day(con, date: str, limit: int = 400) -> dict:
    """Every walk on one date, with the ones whose figures cannot be trusted named."""
    rows = con.execute(
        "SELECT block_day_id FROM block_days WHERE date=? ORDER BY block_day_id "
        "LIMIT ?", (date, limit)).fetchall()
    out, bad = [], []
    for r in rows:
        try:
            bt = block_time(con, r["block_day_id"])
        except LookupError:
            continue
        out.append(bt)
        if not bt.usable or bt.notes:
            bad.append(bt.as_dict())
    return {
        "date": date, "walks": len(out),
        "usable": sum(1 for b in out if b.usable),
        "mean_hours": round(st.fmean([b.hours for b in out]), 3) if out else None,
        "flagged": len(bad), "rows": bad[:40],
    }


def buffer_sensitivity(con, block_day_id: int,
                       scales=(0.6, 0.8, 1.0, 1.4, 2.0, 3.0, 4.0)) -> dict:
    """How much of the walk their geofences catch, at different widths.

    Their width comes from one thing: how far the delivery points sit from the
    road. That is the right instinct and it leaves out the other thing that
    decides whether a ping lands inside - how badly the device is lying about
    where it is. In a terraced street the doors are five metres off the road and
    the fence is eight metres wide, while the handset is routinely ten metres
    out, so a quarter of the pings fall outside every fence and the time they
    represent is lost.

    This re-derives the same walk at several widths so the loss can be seen
    rather than argued about. The useful width is where the curve flattens:
    wider than that and the fences start swallowing the next street instead.
    """
    pings = con.execute(
        "SELECT ts, lat, lon, accuracy_m FROM pings WHERE block_day_id=? ORDER BY ts",
        (block_day_id,)).fetchall()
    if not pings:
        return {"pings": 0, "note": "no raw pings retained for this day"}
    bd = con.execute("SELECT block_id FROM block_days WHERE block_day_id=?",
                     (block_day_id,)).fetchone()
    segs = con.execute(
        "SELECT s.segment_id, s.from_lat, s.from_lon, s.to_lat, s.to_lon, g.buffer_m "
        "FROM osm_segments s JOIN segment_geofence g USING(segment_id) "
        "WHERE s.block_id=?", (bd["block_id"],)).fetchall()

    # distance from every ping to every segment, once
    near: list[list[tuple[float, str, float]]] = []
    for p in pings:
        ds = []
        for s in segs:
            d, _ = point_segment_distance_m(p["lat"], p["lon"], s["from_lat"],
                                            s["from_lon"], s["to_lat"], s["to_lon"])
            if d < 120:
                ds.append((d, s["segment_id"], s["buffer_m"]))
        ds.sort()
        near.append(ds)

    elapsed = pings[-1]["ts"] - pings[0]["ts"]
    rows = []
    for k in scales:
        timeline = []
        outside = 0
        for p, ds in zip(pings, near):
            owner = None
            for d, sid, buf in ds:
                if d <= buf * k:
                    owner = sid
                    break
            if owner is None:
                outside += 1
            timeline.append((p["ts"], owner))
        secs = total_seconds(passes_from(timeline))
        rows.append({
            "scale": k,
            "mean_buffer_m": round(st.fmean([s["buffer_m"] for s in segs]) * k, 1),
            "does_take_s": round(secs, 1),
            "does_take_h": round(secs / 3600, 2),
            "coverage": round(secs / elapsed, 3) if elapsed else None,
            "pings_outside": outside,
            "pings_outside_share": round(outside / len(pings), 3),
        })
    best = None
    for a, b in zip(rows, rows[1:]):
        if a["coverage"] and b["coverage"] and b["coverage"] - a["coverage"] < 0.02:
            best = a
            break
    return {
        "block_day_id": block_day_id, "pings": len(pings),
        "median_accuracy_m": round(st.median([p["accuracy_m"] for p in pings]), 1),
        "elapsed_s": round(elapsed, 1),
        "rows": rows,
        "suggested_scale": best["scale"] if best else rows[-1]["scale"],
        "suggested_buffer_m": best["mean_buffer_m"] if best else rows[-1]["mean_buffer_m"],
        "note": ("The width that stops losing time is about "
                 f"{(best or rows[-1])['mean_buffer_m']:.0f} m here, against a median "
                 f"device accuracy of "
                 f"{st.median([p['accuracy_m'] for p in pings]):.0f} m. Sizing the "
                 "fence from the delivery-point distance alone ignores that."),
    }
