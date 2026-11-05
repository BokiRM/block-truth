"""
Turning a sequence of pings into segment traversals, without losing or
double-counting a second.

This is the one piece of their method that has to be got exactly right, because
every number downstream is a sum of what it produces. Three rules, and each one
was arrived at by watching the previous one fail on real tracks:

1. **One segment owns each ping.** Their ping-to-OSM table already matched the
   ping to a street; the geofence decides only whether the ping counts at all.
   Letting a ping count for every fence it falls inside inflates a walk two- to
   five-fold, because buffers around consecutive 17 m segments overlap heavily.

2. **A single stray fix is not a street.** GPS noise flips which segment is
   nearest for one ping at a time. A median filter over three pings removes
   those without touching a genuine change of street.

3. **Time is attributed by interval, not by run.** A pass measured from its own
   first ping to its own last ping silently loses every gap *between* runs - on
   an urban block with short segments that is more than half the walk, and the
   coverage figure collapses to nonsense. Instead each interval between two
   consecutive pings belongs to the segment that owned the *earlier* one. The
   intervals tile the walk exactly, so the traversals sum to the elapsed time
   minus only the time spent outside every fence, which is a real quantity and
   not an artefact.

The third rule is what makes "does take" and "coverage" mean anything at all.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Pass:
    segment_id: str
    pass_no: int
    entry_ts: float
    exit_ts: float
    ping_count: int

    @property
    def seconds(self) -> float:
        return self.exit_ts - self.entry_ts


def smooth(owners: list[str | None]) -> list[str | None]:
    """Remove single-ping flips: a median filter of width three."""
    if len(owners) < 3:
        return list(owners)
    out = list(owners)
    for i in range(1, len(owners) - 1):
        if owners[i] != owners[i - 1] and owners[i - 1] == owners[i + 1]:
            out[i] = owners[i - 1]
    return out


def passes_from(timeline: list[tuple[float, str | None]]) -> list[Pass]:
    """Pings in time order, each with the segment that owns it, into traversals.

    `None` means the ping was inside no geofence - off the end of a long drive,
    crossing a park, back at the van. That time is deliberately attributed to
    nobody, which is exactly what makes the coverage figure able to show it.
    """
    if not timeline:
        return []
    timeline = sorted(timeline)
    owners = smooth([sid for _ts, sid in timeline])
    times = [ts for ts, _sid in timeline]

    out: list[Pass] = []
    counts: dict[str, int] = {}
    i = 0
    n = len(owners)
    while i < n:
        sid = owners[i]
        j = i
        while j + 1 < n and owners[j + 1] == sid:
            j += 1
        if sid is not None:
            # the run owns every interval from its first ping up to the ping
            # that starts the next run: no gap is lost, and none is counted twice
            end = times[j + 1] if j + 1 < n else times[j]
            counts[sid] = counts.get(sid, 0) + 1
            out.append(Pass(sid, counts[sid], times[i], end, j - i + 1))
        i = j + 1
    return out


def total_seconds(ps: list[Pass]) -> float:
    return sum(p.seconds for p in ps)


def naive_total_seconds(by_segment: dict[str, list[float]], gap_s: float = 45.0) -> float:
    """What the other reading gives: per segment, every ping in every fence.

    Kept so the two can be compared on the same pings. The runs it builds can
    overlap in time, which is why its total can exceed the length of the walk.
    """
    out = 0.0
    for ts in by_segment.values():
        ts = sorted(ts)
        a, b = ts[0], ts[0]
        for t in ts[1:]:
            if t - b > gap_s:
                out += b - a
                a = t
            b = t
        out += b - a
    return out
