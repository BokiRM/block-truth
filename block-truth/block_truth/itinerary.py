"""
Several ways to drive the same round, and what each one costs.

A planner does not want the optimum of a function somebody else chose. The same
stops can be driven to use the least fuel, to keep the most promises, or to be
the least likely to overrun, and those are three different routes - so the job
here is to produce them all, price each one honestly, and say plainly what each
gives up. The planner picks.

The four on offer
-----------------

* **As received** - the order the stops arrived in. The baseline: without it
  nobody can tell whether the clever routes are worth anything.
* **Least driving** - nearest neighbour, then 2-opt until it stops improving.
  Shortest wheels on road. Ignores the clock and the promises entirely.
* **Promises first** - earliest deadline next, with distance breaking ties. Keeps
  windows at the cost of criss-crossing the patch.
* **Steadiest** - greedy on *predicted minutes under the conditions of the day*
  rather than on distance, so it declines to drive a long leg through the peak
  and goes around what is closed. Usually a little longer than "least driving"
  in kilometres and shorter in hours.

Each is then costed by the estimator itself, under the same traffic and the same
disruptions, so the comparison is like for like. Nothing here reorders anything
without saying so, and the stored order is always among the options.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

from .calibrate import Calibration
from .estimate import BlockEstimate, StopPlan, estimate
from .fmt import hhmm, to_minutes
from .geo import ROAD_FACTOR, haversine_km
from .traffic import Conditions, leg_impact

TWO_OPT_PASSES = 40

# What a broken promise is worth, in minutes of driving.
#
# This is the one number here that is NOT measured, and it has to be said out
# loud: there is no fact in the delivery records about how a depot trades an hour
# of vehicle time against a customer who was promised a morning. That is a
# business decision. So it is a single, visible, adjustable figure rather than a
# weighting buried in a sort, and the page shows the arithmetic it produces:
# twenty minutes a missed window, half that for one that could go either way.
# Move it and the recommendation moves, which is the point - an operator who
# disagrees with the default can see exactly what to change.
MINUTES_PER_MISSED_WINDOW = 20.0
AT_RISK_SHARE = 0.5


@dataclass
class Limits:
    """What a round is allowed to be. Nothing is rejected - breaches are shown."""
    min_minutes: float | None = None
    max_minutes: float | None = None
    max_km: float | None = None
    max_stops: int | None = None

    def breaches(self, e: BlockEstimate) -> list[str]:
        out = []
        if self.max_minutes and e.expected_minutes > self.max_minutes:
            out.append(f"over the {self.max_minutes / 60:.1f}h limit by "
                       f"{(e.expected_minutes - self.max_minutes):.0f} min")
        if self.min_minutes and e.expected_minutes < self.min_minutes:
            out.append(f"under the {self.min_minutes / 60:.1f}h minimum by "
                       f"{(self.min_minutes - e.expected_minutes):.0f} min")
        if self.max_km and e.route_km > self.max_km:
            out.append(f"over the {self.max_km:.0f} km limit by "
                       f"{e.route_km - self.max_km:.0f} km")
        if self.max_stops and e.stops > self.max_stops:
            out.append(f"over the {self.max_stops} stop limit by "
                       f"{e.stops - self.max_stops}")
        return out

    def as_dict(self):
        return {"min_minutes": self.min_minutes, "max_minutes": self.max_minutes,
                "max_km": self.max_km, "max_stops": self.max_stops}


@dataclass
class Itinerary:
    key: str
    name: str
    why: str
    gives_up: str
    order: list[int]
    est: BlockEstimate
    windows_met: int = 0
    windows_at_risk: int = 0
    windows_missed: int = 0
    windows_total: int = 0
    breaches: list[str] = field(default_factory=list)

    @property
    def km(self) -> float:
        return self.est.route_km

    def as_dict(self) -> dict:
        d = self.est.as_dict()
        d.update({
            "key": self.key, "name": self.name, "why": self.why,
            "gives_up": self.gives_up, "order": self.order,
            "windows_met": self.windows_met, "windows_at_risk": self.windows_at_risk,
            "windows_missed": self.windows_missed, "windows_total": self.windows_total,
            "breaches": self.breaches,
        })
        return d


# --------------------------------------------------------------- the orderings

def _km(a, b) -> float:
    return haversine_km(a[0], a[1], b[0], b[1]) * ROAD_FACTOR


def _nearest_neighbour(stops: list[StopPlan], depot) -> list[StopPlan]:
    left, cur, out = stops[:], depot, []
    while left:
        nxt = min(left, key=lambda s: _km(cur, (s.lat, s.lon)))
        left.remove(nxt)
        out.append(nxt)
        cur = (nxt.lat, nxt.lon)
    return out


def _tour_km(order: list[StopPlan], depot) -> float:
    total, cur = 0.0, depot
    for s in order:
        total += _km(cur, (s.lat, s.lon))
        cur = (s.lat, s.lon)
    return total + _km(cur, depot)


def _two_opt(order: list[StopPlan], depot) -> list[StopPlan]:
    """Untangle crossings. Nearest neighbour leaves them; 2-opt takes them out."""
    best = order[:]
    best_km = _tour_km(best, depot)
    for _ in range(TWO_OPT_PASSES):
        improved = False
        for i in range(len(best) - 1):
            for j in range(i + 2, len(best)):
                cand = best[:i + 1] + best[i + 1:j + 1][::-1] + best[j + 1:]
                km = _tour_km(cand, depot)
                if km < best_km - 1e-9:
                    best, best_km, improved = cand, km, True
        if not improved:
            break
    return best


def _deadline_first(stops: list[StopPlan], depot, start: float,
                    per_km: float, fixed: float, service: float) -> list[StopPlan]:
    """Keep the most promises, without throwing the day away to do it.

    The obvious version of this - always drive to whatever is due soonest - is a
    disaster, and measurably so: on these rounds it drove four times the
    kilometres of the shortest route to save a handful of windows. Deadlines
    alone are not a route.

    So: among the stops that can still be reached inside their window, take the
    nearest. Only when nothing is reachable in time does the deadline decide, and
    then the one that is closest to being lost goes first. Waiting at a door for
    a window to open is counted as the cost it is.
    """
    left, cur, t, out = stops[:], depot, start, []
    while left:
        reachable = []
        for s in left:
            km = _km(cur, (s.lat, s.lon))
            drive = per_km * km + fixed
            arrive = t + drive
            opens = to_minutes(s.promised_from) if s.promised_from else None
            shuts = to_minutes(s.promised_to) if s.promised_to else None
            wait = max(0.0, opens - arrive) if opens is not None else 0.0
            in_time = shuts is None or arrive <= shuts
            reachable.append((s, drive + wait, in_time,
                              shuts if shuts is not None else 10_000.0))
        live = [r for r in reachable if r[2]]
        if live:
            nxt, cost, _, _ = min(live, key=lambda r: r[1])
        else:
            # everything left is already late; do the least-late one first so
            # the damage stops spreading
            nxt, cost, _, _ = min(reachable, key=lambda r: (r[3], r[1]))
        left.remove(nxt)
        t += cost + service
        cur = (nxt.lat, nxt.lon)
        out.append(nxt)
    return out


def _tour_minutes(order: list[StopPlan], depot, start: float, date: str, city: str,
                  per_km: float, fixed: float, service: float,
                  cond: Conditions) -> float:
    """How long this order takes, driven forward through the day's conditions.

    Not a distance in disguise: each leg is charged at the factor for the hour it
    is actually driven, and a leg through a closure carries its detour, so the
    order changes the answer through more than just which legs are in it.
    """
    t, cur = start, depot
    for s in order:
        km = _km(cur, (s.lat, s.lon))
        f = cond.traffic.factor(t, date, city) if cond.use_traffic else 1.0
        imp = leg_impact(cur, (s.lat, s.lon), t, date, cond.disruptions)
        t += per_km * (km + imp.extra_km) * f * imp.factor + fixed + service
        cur = (s.lat, s.lon)
    f = cond.traffic.factor(t, date, city) if cond.use_traffic else 1.0
    imp = leg_impact(cur, depot, t, date, cond.disruptions)
    t += per_km * (_km(cur, depot) + imp.extra_km) * f * imp.factor
    return t - start


def _time_opt(order: list[StopPlan], depot, start: float, date: str, city: str,
              per_km: float, fixed: float, service: float,
              cond: Conditions) -> list[StopPlan]:
    """2-opt again, but against the clock instead of the odometer.

    Without this the time-greedy route was simply a worse version of the
    distance route - beaten on hours, which are its own objective. A suggestion
    that loses at the thing it is for is not a suggestion, it is noise.
    """
    best = order[:]
    best_t = _tour_minutes(best, depot, start, date, city, per_km, fixed, service, cond)
    for _ in range(TWO_OPT_PASSES):
        improved = False
        for i in range(len(best) - 1):
            for j in range(i + 2, len(best)):
                cand = best[:i + 1] + best[i + 1:j + 1][::-1] + best[j + 1:]
                t = _tour_minutes(cand, depot, start, date, city, per_km, fixed,
                                  service, cond)
                if t < best_t - 1e-6:
                    best, best_t, improved = cand, t, True
        if not improved:
            break
    return best


def _steadiest(stops: list[StopPlan], depot, start: float, date: str, city: str,
               per_km: float, fixed: float, service: float,
               cond: Conditions) -> list[StopPlan]:
    """Greedy on minutes, not kilometres, with the day's conditions applied.

    The difference from "least driving" is the clock: the cost of a leg here is
    what it is predicted to take at the time it would actually be driven, so a
    long hop during the peak looks as expensive as it is, and a leg through a
    closure carries its detour.
    """
    left, cur, t, out = stops[:], depot, start, []
    while left:
        def minutes(s: StopPlan):
            km = _km(cur, (s.lat, s.lon))
            f = cond.traffic.factor(t, date, city) if cond.use_traffic else 1.0
            imp = leg_impact(cur, (s.lat, s.lon), t, date, cond.disruptions)
            return per_km * (km + imp.extra_km) * f * imp.factor + fixed
        nxt = min(left, key=minutes)
        left.remove(nxt)
        t += minutes(nxt) + service
        cur = (nxt.lat, nxt.lon)
        out.append(nxt)
    return out


# ------------------------------------------------------------------ the scoring

def _score(key: str, name: str, why: str, gives_up: str, order: list[StopPlan],
           block_id: str, date: str, start, depot, cal: Calibration, city: str,
           vehicle_type: str, confidence: int, cond: Conditions,
           limits: Limits, wait: bool = False) -> Itinerary:
    plan = [replace(s, sequence=i) for i, s in enumerate(order, start=1)]
    e = estimate(block_id, date, start, depot, plan, cal, city=city,
                 vehicle_type=vehicle_type, confidence=confidence, conditions=cond,
                 wait_for_windows=wait)
    met = risk = missed = total = 0
    for se in e.stop_estimates:
        if se.promised_from is None or se.promised_to is None:
            continue
        total += 1
        if se.on_time is False:
            missed += 1
        elif se.at_risk:
            risk += 1
        else:
            met += 1
    return Itinerary(key=key, name=name, why=why, gives_up=gives_up,
                     order=[s.sequence for s in order], est=e,
                     windows_met=met, windows_at_risk=risk, windows_missed=missed,
                     windows_total=total, breaches=limits.breaches(e))


def suggest(block_id: str, date: str, start, depot: tuple[float, float],
            stops: list[StopPlan], cal: Calibration, city: str = "*",
            vehicle_type: str = "Large Van", confidence: int = 90,
            conditions: Conditions | None = None,
            limits: Limits | None = None,
            plan_waits: bool = False) -> list[Itinerary]:
    """At least three ways to drive this round, each costed the same way.

    `plan_waits` says whether the round as it stands is being driven to windows -
    that is, whether it waits at a door that is not open yet. It matters because
    the choice an operator makes is a policy and not only an order: picking
    "promises first" and then costing the result without its waiting would quietly
    hand back an hour the operator had decided to spend.
    """
    if not stops:
        raise ValueError("a block needs at least one stop")
    cond = (conditions or Conditions(use_traffic=False)).for_date(date)
    limits = limits or Limits()
    start_min = to_minutes(start) if isinstance(start, str) else float(start)
    ordered = sorted(stops, key=lambda s: s.sequence)

    per_km, fixed, _ = cal.travel_for(city)
    svc = cal.service_for("House", "Medium").mean          # a stand-in for pacing
    nn = _nearest_neighbour(ordered, depot)

    plans = [
        # the fifth element says whether this option waits at a door for a
        # window to open; only the one that is for keeping promises does
        ("as_planned", "As planned", "The order this round is currently stored in.",
         "Whatever the plan already gave up - this is the thing to beat.",
         ordered, plan_waits),
        ("shortest", "Least driving",
         "Nearest neighbour, then 2-opt until no swap helps. The fewest "
         "kilometres on the road.",
         "Takes no notice of the clock: it will happily drive its longest leg "
         "through the peak and miss a morning window.",
         _two_opt(nn, depot), False),
        ("windows", "Promises first",
         "Reach each stop inside its promised window - taking the nearest one "
         "that is still winnable, and waiting at the door when it arrives early.",
         "Hours. The waiting is real time the vehicle is not delivering, and it "
         "drives further to be in the right place at the right time.",
         _deadline_first(ordered, depot, start_min, per_km.mean, fixed, svc), True),
        ("steady", "Steadiest",
         "Greedy on predicted minutes rather than kilometres, at the time each "
         "leg would be driven, going around whatever is closed.",
         "Not the shortest route, and it has no special regard for windows "
         "either - it is after a predictable finish.",
         _time_opt(_steadiest(ordered, depot, start_min, date, city, per_km.mean,
                              fixed, svc, cond),
                   depot, start_min, date, city, per_km.mean, fixed, svc, cond),
         False),
    ]

    # The raw manifest, when it is on record. Not a serious option - work
    # arrives grouped by promise, not by geography - but it is the only honest
    # answer to "better than what?", so it is costed with the rest.
    if all(s.manifest_sequence is not None for s in ordered):
        plans.append(
            ("as_received", "As the work arrived",
             "The manifest order, straight off the system, unrouted.",
             "Everything. It is here to be the baseline, not to be driven.",
             sorted(ordered, key=lambda s: s.manifest_sequence), False))

    out = [_score(k, n, w, g, o, block_id, date, start, depot, cal, city,
                  vehicle_type, confidence, cond, limits, wait)
           for k, n, w, g, o, wait in plans]

    # Drop a suggestion that is the same drive as one already on the list - two
    # identical routes under different names would be a lie about having choices.
    seen, unique = {}, []
    for it in out:
        sig = (tuple(it.order), round(it.est.wait_minutes, 1))
        if sig in seen:
            seen[sig].name += f" (= {it.name.lower()})"
            continue
        seen[sig] = it
        unique.append(it)
    unique.sort(key=lambda i: i.est.expected_minutes)
    return unique


def recommend(items: list[Itinerary],
              minutes_per_missed_window: float = MINUTES_PER_MISSED_WINDOW) -> dict:
    """Which one to drive, by a rule anybody can check.

    Two steps, both stated:

    1. An option that breaks an operating limit is set aside - but only if
       something else does not. A round has to be driven by somebody, so if every
       option is over the limit the recommendation is the least bad one and says
       so, rather than refusing to answer.
    2. Among what is left, the lowest total of
       `hours + the price of the promises it breaks`.

    The price is a policy, not a measurement (see the constant above). The
    arithmetic is returned with the answer so the page can print it and the
    operator can disagree with the input rather than with a verdict.
    """
    if not items:
        return {}
    price = max(0.0, float(minutes_per_missed_window))

    def cost(it: Itinerary) -> float:
        broken = it.windows_missed + AT_RISK_SHARE * it.windows_at_risk
        return it.est.expected_minutes + price * broken

    legal = [i for i in items if not i.breaches] or items
    best = min(legal, key=cost)
    rows = []
    for it in items:
        broken = it.windows_missed + AT_RISK_SHARE * it.windows_at_risk
        rows.append({
            "key": it.key, "name": it.name,
            "minutes": round(it.est.expected_minutes, 1),
            "broken_windows": round(broken, 1),
            "penalty_minutes": round(price * broken, 1),
            "score_minutes": round(cost(it), 1),
            "set_aside": bool(it.breaches) and bool([i for i in items if not i.breaches]),
        })
    rows.sort(key=lambda r: (r["set_aside"], r["score_minutes"]))
    runner = [r for r in rows if r["key"] != best.key and not r["set_aside"]]
    margin = (runner[0]["score_minutes"] - rows[0]["score_minutes"]) if runner else 0.0
    return {
        "key": best.key,
        "name": best.name,
        "minutes_per_missed_window": price,
        "why": (f"{dur_min(best.est.expected_minutes)} of driving plus "
                f"{rows[0]['broken_windows']:.1f} broken promises at {price:.0f} min "
                f"= {rows[0]['score_minutes']:.0f} - the lowest of the "
                f"{len(rows)} options"
                + (f", by {margin:.0f} min over {runner[0]['name']}" if runner else "")),
        "all_over_the_limits": all(bool(i.breaches) for i in items),
        "scores": rows,
    }


def dur_min(m: float) -> str:
    return f"{int(m // 60)}h {int(round(m % 60)):02d}m"


def compare(items: list[Itinerary]) -> dict:
    """The trade-off, in numbers: what the spread between the options actually is."""
    if not items:
        return {}
    by_time = min(items, key=lambda i: i.est.expected_minutes)
    by_km = min(items, key=lambda i: i.km)
    by_windows = max(items, key=lambda i: (i.windows_met, -i.windows_missed))
    by_certainty = min(items, key=lambda i: i.est.sd_minutes)
    return {
        "options": len(items),
        "fastest": by_time.key, "shortest": by_km.key,
        "most_promises_kept": by_windows.key, "most_predictable": by_certainty.key,
        "minutes_between_best_and_worst": round(
            max(i.est.expected_minutes for i in items)
            - min(i.est.expected_minutes for i in items), 1),
        "km_between_best_and_worst": round(
            max(i.km for i in items) - min(i.km for i in items), 1),
        "windows_between_best_and_worst": (
            max(i.windows_met for i in items) - min(i.windows_met for i in items)),
        "agree": len({tuple(i.order) for i in items}) == 1,
    }
