"""
The calculation: when did this block actually finish?

Pure arithmetic over recorded stops. No database, no model, no prediction - this
is the label a forecast would later be scored against, so it is derived from
primary records and it shows its working.

How a round is laid out
-----------------------

    block start --[depot: load and brief]--> depart
        `-[leg 1]-> stop 1 -[service]-> -[leg 2]-> stop 2 -[service]-> ... ->
                                                  stop N -[service]-> -[return]-> back

Each stop records the leg that *led to it* and the time spent at its door. So
everything from leaving the depot to finishing the last doorstep is the sum of
those two columns, and two pieces sit outside them: the depot time before the
first leg, and the return leg after the last door.

What is derivable, and from what
--------------------------------

    door-to-door      = last arrival + its service - first arrival     (stops alone)
    depot + leg 1     = first arrival - block start                    (stops alone)
    depot minutes     = that, minus the recorded first leg             (stops alone)
    work to last door = last arrival + its service - block start       (stops alone)
    return leg        = recorded duration - work to last door          (needs the duration)
    completion        = block start + recorded duration

The return leg is the one quantity no stop row carries, because nobody is served
on the way home. When the recorded duration is present it falls out as the
residual; when it is absent, completion is reported as of the last doorstep and
`completion_source` says so, rather than inventing a drive home.

Two independent routes to the depot time
----------------------------------------

The depot time can be had either from the first arrival (minus the first leg) or,
once the return leg is known, from the duration. Both come from the same records
by different paths, so disagreement means the records disagree with themselves -
which is worth knowing and is one of the checks below.
"""
from __future__ import annotations

from .model import MINUTES_PER_DAY, BlockTruth, Check, Stop

# Arrival times are recorded to the minute, so a chain of N stops accumulates up
# to half a minute of rounding per link. The tolerances below are generous enough
# not to cry over that and tight enough to catch a real break.
ROUNDING_PER_STOP = 0.5
CHAIN_TOLERANCE = 2.0
RESIDUAL_TOLERANCE = 2.0


class TruthError(ValueError):
    """The records cannot support a completion time at all."""


def _unwrap(stops: list[Stop]) -> list[Stop]:
    """Let a round that runs past midnight keep counting instead of resetting.

    A recorded 23:50 followed by 00:20 is forty minutes later, not twenty-three
    hours earlier, so each arrival that goes backwards gains a day.
    """
    out, offset, prev = [], 0.0, None
    for s in stops:
        a = s.arrival_minutes + offset
        if prev is not None and a < prev - 1e-9:
            offset += MINUTES_PER_DAY
            a += MINUTES_PER_DAY
        out.append(Stop(s.sequence, a, s.service_minutes, s.travel_minutes,
                        s.outcome, s.delivery_point_id))
        prev = a
    return out


def compute(block_id: str, delivery_date: str, start_minutes: float,
            stops: list[Stop], duration_recorded: float | None = None,
            attributes: dict | None = None,
            success_outcome: str = "Delivered") -> BlockTruth:
    """Ground-truth completion time for one block, with its working shown."""
    if not stops:
        raise TruthError(f"{block_id}: no stops recorded, so nothing was completed")

    stops = _unwrap(sorted(stops, key=lambda s: s.sequence))
    n = len(stops)
    first, last = stops[0], stops[-1]

    service_total = sum(s.service_minutes for s in stops)
    travel_total = sum(s.travel_minutes for s in stops)

    # everything from the depot gate to the last doorstep, straight off the stops
    work_to_last_door = last.arrival_minutes + last.service_minutes - start_minutes
    door_to_door = last.arrival_minutes + last.service_minutes - first.arrival_minutes
    depot_from_arrival = first.arrival_minutes - start_minutes - first.travel_minutes

    if duration_recorded is not None:
        duration = float(duration_recorded)
        return_leg = duration - work_to_last_door
        source = "recorded duration"
    else:
        # no duration on file: the records end at the last doorstep and so do we
        duration = work_to_last_door
        return_leg = 0.0
        source = "last stop (no recorded duration, return leg unknown)"

    completion = start_minutes + duration

    checks = _checks(stops, start_minutes, service_total, travel_total,
                     work_to_last_door, depot_from_arrival, return_leg,
                     duration_recorded)

    delivered = sum(1 for s in stops if s.outcome == success_outcome)
    failed = sum(1 for s in stops if s.outcome is not None and s.outcome != success_outcome)

    return BlockTruth(
        block_id=block_id, delivery_date=delivery_date, stops=n,
        start_minutes=start_minutes, completion_minutes=completion,
        duration_minutes=duration,
        first_arrival_minutes=first.arrival_minutes,
        last_arrival_minutes=last.arrival_minutes,
        last_service_minutes=last.service_minutes,
        door_to_door_minutes=door_to_door,
        service_minutes=service_total, travel_minutes=travel_total,
        depot_minutes=depot_from_arrival, return_leg_minutes=return_leg,
        delivered=delivered, failed=failed,
        completion_source=source, duration_recorded=duration_recorded,
        checks=checks, attributes=attributes or {},
    )


def _checks(stops, start, service_total, travel_total, work_to_last_door,
            depot_from_arrival, return_leg, duration_recorded) -> list[Check]:
    """Re-derive what we can a second way, and say so when the two disagree."""
    out: list[Check] = []
    n = len(stops)

    # 1. the arrival chain: each stop should follow the last one by its service
    #    plus the leg that led here
    worst, worst_at = 0.0, None
    for prev, cur in zip(stops, stops[1:]):
        expected = prev.arrival_minutes + prev.service_minutes + cur.travel_minutes
        gap = abs(cur.arrival_minutes - expected)
        if gap > worst:
            worst, worst_at = gap, cur.sequence
    out.append(Check(
        "arrival chain",
        worst <= CHAIN_TOLERANCE,
        (f"every arrival follows the one before it by its service plus the next leg "
         f"(worst gap {worst:.2f} min at stop {worst_at})") if worst_at else
        "single stop, nothing to chain",
        round(worst, 3)))

    # 2. the two routes to the depot time should land in the same place: once from
    #    the first arrival minus the leg that got there, once from the totals
    depot_alt = work_to_last_door - service_total - travel_total
    gap = abs(depot_from_arrival - depot_alt)
    tol = max(RESIDUAL_TOLERANCE, ROUNDING_PER_STOP * 2)
    out.append(Check(
        "depot time agrees two ways",
        gap <= tol,
        (f"from the first arrival {depot_from_arrival:.2f} min, from the totals "
         f"{depot_alt:.2f} min"),
        round(gap, 3)))

    # 3. the parts add back up to the whole. Be honest about what this is: with
    #    the return leg defined as the leftover, it is an identity, and an
    #    identity cannot fail. It is here because the arithmetic is worth showing
    #    and because a float error would show up in it - not because it audits
    #    anything. The thing it cannot catch, a recorded duration that is simply
    #    too long, is caught by the geometry check in `repository`, which knows
    #    how far the depot is from the last stop.
    if duration_recorded is not None:
        rebuilt = depot_alt + service_total + travel_total + return_leg
        gap = abs(rebuilt - duration_recorded)
        out.append(Check(
            "parts sum to the recorded duration (identity)",
            gap <= 1e-6,
            (f"depot {depot_alt:.2f} + service {service_total:.2f} + travel "
             f"{travel_total:.2f} + return {return_leg:.2f} = {rebuilt:.2f} against "
             f"{duration_recorded:.2f} on file - it must, the return leg is the "
             f"leftover"),
            round(gap, 6)))

        # 4. a negative return leg means the round is recorded as ending before its
        #    last doorstep, which cannot be
        out.append(Check(
            "return leg is not negative",
            return_leg >= -ROUNDING_PER_STOP,
            f"return leg {return_leg:.2f} min",
            round(return_leg, 3)))

    # 5. nothing takes negative time
    bad = [s.sequence for s in stops if s.service_minutes < 0 or s.travel_minutes < 0]
    out.append(Check(
        "no negative service or travel",
        not bad,
        "all stops record non-negative time" if not bad
        else f"negative at stop(s) {bad[:5]}",
        None))

    # 6. the sequence should be 1..n with nothing missing
    seqs = [s.sequence for s in stops]
    complete = seqs == list(range(seqs[0], seqs[0] + n))
    out.append(Check(
        "stop sequence is unbroken",
        complete,
        f"stops {seqs[0]}..{seqs[-1]}, {n} rows" if complete
        else f"gaps in the sequence: {seqs[0]}..{seqs[-1]} but only {n} rows", None))

    return out
