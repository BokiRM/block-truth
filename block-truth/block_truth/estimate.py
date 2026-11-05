"""
Estimate: how long will this round take, at the earliest and at the latest?

This is the other half of the app, and it is kept at arm's length from the
first. `truth.py` reports what happened; nothing in it is predicted. Here
nothing is measured - a block entered by hand has no arrival times, so every
number below is a forecast and is labelled as one everywhere it is shown.

Why min and max are not the sum of per-stop extremes
----------------------------------------------------

The tempting way to get a range is to add up the best case at every stop for
the minimum and the worst case for the maximum. It is also badly wrong. Forty
stops all landing on their best case is not a fast day, it is a fantasy: the
odds are about one in a trillion, and a planner told "between 3h10 and 9h40"
learns nothing.

What actually happens is that errors partly cancel. Independent noise adds in
*variance*, not in width, so a round of N stops has a spread that grows like
sqrt(N) while its duration grows like N - the longer the round, the tighter its
interval in relative terms.

But not everything cancels. Some of the day misses together: traffic, weather,
a driver new to the patch. That part does not shrink with N at all, and without
it the interval on a long round would collapse to nothing, which is equally
wrong in the other direction. So:

    variance = SUM(per-stop variance)  +  (common_mode x duration)^2
                      shrinks with N            does not shrink at all

The common-mode fraction is measured from the recorded blocks, not assumed -
see `calibrate._common_mode`. On this data it is about 7%, which is what stops
a 40-stop round being quoted to the minute.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from .calibrate import Z, Calibration
from .fmt import hhmm, to_minutes
from .geo import ROAD_FACTOR, haversine_km
from .traffic import DISRUPTION_SD_SHARE, Conditions, leg_impact, stops_inside_closures


@dataclass
class StopPlan:
    """A stop to be visited. No times - that is what is being estimated."""
    sequence: int
    lat: float
    lon: float
    point_type: str = "House"
    parcel_size: str = "Medium"
    address: str | None = None
    postcode: str | None = None
    promised_from: str | None = None
    promised_to: str | None = None
    # where this stop sat on the manifest before anybody routed it; the routing
    # suggestions need it to have something to be better than
    manifest_sequence: int | None = None


@dataclass
class StopEstimate:
    sequence: int
    leg_km: float
    travel_minutes: float
    service_minutes: float
    arrival_minutes: float
    arrival_min: float
    arrival_max: float
    departure_minutes: float
    promised_from: float | None = None
    promised_to: float | None = None

    @property
    def on_time(self) -> bool | None:
        if self.promised_from is None or self.promised_to is None:
            return None
        return self.promised_from <= self.arrival_minutes <= self.promised_to

    @property
    def at_risk(self) -> bool:
        """Could miss the window even though the expected arrival is inside it."""
        if self.promised_from is None or self.promised_to is None:
            return False
        return not (self.promised_from <= self.arrival_min
                    and self.arrival_max <= self.promised_to)


@dataclass
class BlockEstimate:
    block_id: str
    delivery_date: str
    stops: int
    confidence: int

    start_minutes: float
    expected_minutes: float          # duration
    min_minutes: float
    max_minutes: float
    sd_minutes: float

    completion_expected: float       # clock position
    completion_min: float
    completion_max: float

    depot_minutes: float
    travel_minutes: float
    service_minutes: float
    return_minutes: float
    wait_minutes: float
    route_km: float
    expected_failures: float

    independent_sd: float
    common_mode_sd: float

    stop_estimates: list[StopEstimate] = field(default_factory=list)
    attributes: dict = field(default_factory=dict)
    basis: dict = field(default_factory=dict)
    conditions: dict = field(default_factory=dict)

    def as_dict(self, clock: bool = True) -> dict:
        d = {
            "block_id": self.block_id, "delivery_date": self.delivery_date,
            **self.attributes,
            "kind": "estimate",
            "stops": self.stops, "confidence": self.confidence,
            "route_km": round(self.route_km, 2),
            "expected_minutes": round(self.expected_minutes, 1),
            "min_minutes": round(self.min_minutes, 1),
            "max_minutes": round(self.max_minutes, 1),
            "sd_minutes": round(self.sd_minutes, 2),
            "expected_hours": round(self.expected_minutes / 60, 3),
            "depot_minutes": round(self.depot_minutes, 1),
            "travel_minutes": round(self.travel_minutes, 1),
            "service_minutes": round(self.service_minutes, 1),
            "return_minutes": round(self.return_minutes, 1),
            "wait_minutes": round(self.wait_minutes, 1),
            "expected_failures": round(self.expected_failures, 2),
            "independent_sd": round(self.independent_sd, 2),
            "common_mode_sd": round(self.common_mode_sd, 2),
            "minutes_per_stop": round(self.expected_minutes / max(self.stops, 1), 2),
            "basis": self.basis,
            "conditions": self.conditions,
        }
        if clock:
            d.update({
                "start_time": hhmm(self.start_minutes),
                "completion_expected": hhmm(self.completion_expected),
                "completion_min": hhmm(self.completion_min),
                "completion_max": hhmm(self.completion_max),
            })
        d["stop_estimates"] = [{
            "sequence": s.sequence, "leg_km": round(s.leg_km, 3),
            "travel_minutes": round(s.travel_minutes, 2),
            "service_minutes": round(s.service_minutes, 2),
            "arrival_minutes": round(s.arrival_minutes, 2),
            "arrival_time": hhmm(s.arrival_minutes),
            "arrival_min": hhmm(s.arrival_min), "arrival_max": hhmm(s.arrival_max),
            "window_from": hhmm(s.promised_from) if s.promised_from is not None else None,
            "window_to": hhmm(s.promised_to) if s.promised_to is not None else None,
            "on_time": s.on_time, "at_risk": s.at_risk,
        } for s in self.stop_estimates]
        return d


def estimate(block_id: str, delivery_date: str, start: str | float,
             depot: tuple[float, float], stops: list[StopPlan],
             cal: Calibration, city: str = "*", vehicle_type: str = "Large Van",
             confidence: int = 90, attributes: dict | None = None,
             conditions: Conditions | None = None,
             wait_for_windows: bool = False) -> BlockEstimate:
    """Expected, earliest and latest completion for a round that has not run.

    With `conditions`, the clock starts mattering. Each leg is driven at a time
    the estimate itself works out, so it gets the traffic factor for that hour
    and any declared disruption it runs into - which is why a round that starts
    at seven and one that starts at eleven are no longer the same round.
    """
    if not stops:
        raise ValueError("a block needs at least one stop")
    if confidence not in Z:
        raise ValueError(f"confidence must be one of {sorted(Z)}")
    z = Z[confidence]
    start_min = to_minutes(start) if isinstance(start, str) else float(start)
    cond = (conditions or Conditions(use_traffic=False)).for_date(delivery_date)

    per_km, fixed, leg_sd = cal.travel_for(city)
    load = cal.depot_for(vehicle_type)
    fail_any = cal.failure_rate.get("*", 0.08)

    mean = load.mean
    var = load.sd ** 2
    travel_total = service_total = route_km = exp_fail = 0.0
    traffic_added = disruption_added = detour_km = wait_total = 0.0
    factors: list[float] = []
    hit: list[str] = []
    out: list[StopEstimate] = []

    prev = depot
    for s in sorted(stops, key=lambda x: x.sequence):
        km = haversine_km(prev[0], prev[1], s.lat, s.lon) * ROAD_FACTOR
        plain = per_km.mean * km + fixed            # what a flat day would cost

        # the leg leaves now, so "now" decides what it meets on the road
        leaves = start_min + mean
        tf = cond.traffic.factor(leaves, delivery_date, city) if cond.use_traffic else 1.0
        imp = leg_impact((prev[0], prev[1]), (s.lat, s.lon), leaves,
                         delivery_date, cond.disruptions)
        # traffic and a jam slow the driving; parking is untouched by both, and
        # a detour is extra distance driven at whatever the roads are doing
        drive = per_km.mean * (km + imp.extra_km) * tf * imp.factor
        travel = drive + fixed
        factors.append(tf)
        hit += imp.hit
        traffic_added += per_km.mean * km * (tf - 1.0)
        disruption_added += travel - plain - per_km.mean * km * (tf - 1.0)
        detour_km += imp.extra_km
        route_km += km + imp.extra_km

        # a stop that fails is a short stop, so the expected service time is a
        # blend weighted by how often that kind of door fails
        p_fail = cal.failure_rate.get(s.point_type, fail_any)
        ok = cal.service_for(s.point_type, s.parcel_size)
        bad = cal.service_failed
        service = (1 - p_fail) * ok.mean + p_fail * bad.mean
        # the spread of a mixture includes the gap between what it is mixing
        gap = ok.mean - bad.mean
        svc_var = ((1 - p_fail) * ok.sd ** 2 + p_fail * bad.sd ** 2
                   + p_fail * (1 - p_fail) * gap ** 2)
        exp_fail += p_fail

        # Arriving before a window opens is not a delivery, it is a wait - and a
        # round that intends to keep its promises pays for that wait in hours.
        # Off by default: a driver who is not working to windows does not sit
        # outside a door, and charging every round for a wait nobody takes would
        # make every forecast too long.
        wait = 0.0
        if wait_for_windows and s.promised_from:
            opens = to_minutes(s.promised_from)
            arrive_at = start_min + mean + travel
            wait = max(0.0, opens - arrive_at)
            wait_total += wait

        mean += travel + wait + service
        # a detour is a guess at a route nobody has driven, so it is not just
        # more minutes, it is less certain minutes
        var += leg_sd ** 2 + svc_var
        if imp.hit:
            var += (DISRUPTION_SD_SHARE * max(travel - plain, 0.0)) ** 2
        travel_total += travel
        service_total += service

        arrival = start_min + mean - service
        sd_here = math.sqrt(var)
        out.append(StopEstimate(
            sequence=s.sequence, leg_km=km, travel_minutes=travel,
            service_minutes=service, arrival_minutes=arrival,
            arrival_min=arrival - z * sd_here, arrival_max=arrival + z * sd_here,
            departure_minutes=arrival + service,
            promised_from=to_minutes(s.promised_from) if s.promised_from else None,
            promised_to=to_minutes(s.promised_to) if s.promised_to else None))
        prev = (s.lat, s.lon)

    home_km = haversine_km(prev[0], prev[1], depot[0], depot[1]) * ROAD_FACTOR
    leaves = start_min + mean
    tf_home = cond.traffic.factor(leaves, delivery_date, city) if cond.use_traffic else 1.0
    imp_home = leg_impact((prev[0], prev[1]), (depot[0], depot[1]), leaves,
                          delivery_date, cond.disruptions)
    return_plain = cal.return_per_km * home_km + cal.return_fixed
    return_leg = (cal.return_per_km * (home_km + imp_home.extra_km) * tf_home
                  * imp_home.factor + cal.return_fixed)
    traffic_added += cal.return_per_km * home_km * (tf_home - 1.0)
    disruption_added += (return_leg - return_plain
                         - cal.return_per_km * home_km * (tf_home - 1.0))
    detour_km += imp_home.extra_km
    hit += imp_home.hit
    factors.append(tf_home)
    route_km += home_km + imp_home.extra_km
    mean += return_leg
    var += cal.return_sd ** 2
    if imp_home.hit:
        var += (DISRUPTION_SD_SHARE * max(return_leg - return_plain, 0.0)) ** 2

    independent_sd = math.sqrt(var)
    common_sd = cal.common_mode * mean
    sd = math.hypot(independent_sd, common_sd)

    # the per-stop intervals have to carry the common-mode part too, or an early
    # stop looks far more certain than the round it belongs to
    for s in out:
        frac = (s.arrival_minutes - start_min) / mean if mean > 0 else 0
        extra = cal.common_mode * mean * frac
        base = (s.arrival_max - s.arrival_minutes) / z
        full = math.hypot(base, extra)
        s.arrival_min = s.arrival_minutes - z * full
        s.arrival_max = s.arrival_minutes + z * full

    return BlockEstimate(
        block_id=block_id, delivery_date=delivery_date, stops=len(stops),
        confidence=confidence,
        start_minutes=start_min, expected_minutes=mean,
        min_minutes=max(mean - z * sd, 0.0), max_minutes=mean + z * sd, sd_minutes=sd,
        completion_expected=start_min + mean,
        completion_min=start_min + max(mean - z * sd, 0.0),
        completion_max=start_min + mean + z * sd,
        depot_minutes=load.mean, travel_minutes=travel_total,
        service_minutes=service_total, return_minutes=return_leg,
        wait_minutes=wait_total,
        route_km=route_km, expected_failures=exp_fail,
        independent_sd=independent_sd, common_mode_sd=common_sd,
        stop_estimates=out, attributes=attributes or {},
        conditions={
            "traffic_used": bool(cond.use_traffic),
            "traffic_factor_mean": round(sum(factors) / len(factors), 3) if factors else 1.0,
            "traffic_factor_worst": round(max(factors), 3) if factors else 1.0,
            "minutes_from_traffic": round(traffic_added, 1),
            "minutes_from_disruptions": round(disruption_added, 1),
            "detour_km": round(detour_km, 2),
            "disruptions_hit": sorted(set(hit)),
            "stops_inside_a_closure": stops_inside_closures(
                stops, delivery_date, cond.disruptions),
            "measured": "traffic", "declared": "disruptions",
        },
        basis={"city": city, "vehicle_type": vehicle_type,
               "minutes_per_km": round(per_km.mean, 3),
               "implied_speed_kmh": round(60 / per_km.mean, 1) if per_km.mean > 0 else None,
               "fixed_per_stop": round(fixed, 2),
               "depot_minutes": round(load.mean, 1),
               "return_per_km": round(cal.return_per_km, 2),
               "return_fixed": round(cal.return_fixed, 2),
               "common_mode": round(cal.common_mode, 4),
               "road_factor": ROAD_FACTOR,
               "blocks_behind_it": cal.blocks_used},
    )


def predict_parts(stops, depot, cal: Calibration, city: str, vehicle_type: str,
                  conditions: Conditions | None = None,
                  start: str | float = 450.0,
                  date: str = "2000-01-01") -> tuple[float, float]:
    """Expected minutes and the independent part of their spread.

    A thin wrapper over `estimate`, on purpose. The calibration needs the
    model's own view of how much of a block's error ought to cancel out, and
    that view has to be the *same* arithmetic the forecast uses - including the
    traffic, once there is traffic. This used to be a second copy of the loop,
    which is one edit away from the two disagreeing and nobody noticing.
    """
    e = estimate("_", date, start, depot, list(stops), cal, city=city,
                 vehicle_type=vehicle_type, conditions=conditions)
    return e.expected_minutes, e.independent_sd


def predict_duration(stops, depot, cal: Calibration, city: str,
                     vehicle_type: str, **kw) -> float:
    return predict_parts(stops, depot, cal, city, vehicle_type, **kw)[0]
