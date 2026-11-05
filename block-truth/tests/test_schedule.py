"""
Tests for the month ahead: traffic, disruptions, planning and routing.

The ones that matter are in `TestTrafficIsRecovered` and `TestTheOptionsDiffer`.
The first says the traffic curve is measured rather than assumed - it has to
find the shape the seed put into the records without being told it. The second
says the routing suggestions are real choices and not four names for the same
drive, which is the only thing that makes offering them honest.

    python3 -m unittest discover -s tests -v
"""
from __future__ import annotations

import datetime as dt
import os
import tempfile
import unittest

from block_truth.calibrate import fit_full
from block_truth.db import connect
from block_truth.estimate import StopPlan, estimate
from block_truth.itinerary import Limits, compare, suggest
from block_truth.repository import Repository
from block_truth.schedule import generate
from block_truth.seed import TRUE_WEEKDAY, seed
from block_truth.traffic import Conditions, Disruption, chord_km
from block_truth.truth import TruthError

DB = CAL = TRAFFIC = None


def setUpModule():
    global DB, CAL, TRAFFIC
    DB = os.path.join(tempfile.mkdtemp(), "t.db")
    seed(DB, days=28, seed_value=17)
    generate(DB, days=10)
    CAL, TRAFFIC = fit_full(DB)


def plan(n, lat=53.80, lon=-1.55, step=0.004, **kw):
    return [StopPlan(i + 1, lat + i * step, lon + i * step, **kw) for i in range(n)]


class TestTrafficIsRecovered(unittest.TestCase):
    """The seed hides a shape in the records; the fit has to find it."""

    def test_it_used_a_serious_number_of_legs(self):
        self.assertGreater(TRAFFIC.legs_used, 2000)

    def test_the_morning_peak_is_where_the_seed_put_it(self):
        worst, _ = TRAFFIC.busiest()
        self.assertIn(worst, (7, 8, 9), f"worst hour came out as {worst:02d}:00")

    def test_the_middle_of_the_day_is_quicker_than_the_peak(self):
        midday = sum(TRAFFIC.hour[h] for h in (11, 12, 13)) / 3
        peak = TRAFFIC.hour[8]
        self.assertLess(midday, peak - 0.1, f"midday {midday:.3f} against peak {peak:.3f}")

    def test_there_is_a_second_peak_in_the_afternoon(self):
        afternoon = max(TRAFFIC.hour[h] for h in (16, 17))
        midday = sum(TRAFFIC.hour[h] for h in (11, 12, 13)) / 3
        self.assertGreater(afternoon, midday + 0.08)

    def test_saturday_is_the_easiest_day_and_friday_the_worst(self):
        worked = {d: TRAFFIC.weekday[d] for d in range(6) if TRAFFIC.weekday_n.get(d, 0)}
        self.assertEqual(min(worked, key=worked.get), 5, "Saturday should be lightest")
        self.assertEqual(max(worked, key=worked.get), 4, "Friday should be heaviest")

    def test_the_weekday_figures_land_near_what_was_seeded(self):
        for d, want in TRUE_WEEKDAY.items():
            if not TRAFFIC.weekday_n.get(d):
                continue
            self.assertAlmostEqual(TRAFFIC.weekday[d], want, delta=0.06,
                                   msg=f"weekday {d}: {TRAFFIC.weekday[d]:.3f} vs {want}")

    def test_london_peaks_harder_than_the_quiet_cities(self):
        amp = TRAFFIC.city_amplitude
        if "London" not in amp or "Newcastle" not in amp:
            self.skipTest("not enough legs per city in this seed")
        self.assertGreater(amp["London"], amp["Newcastle"])

    def test_an_hour_with_no_evidence_says_nothing(self):
        for h in (2, 3, 4):
            self.assertEqual(TRAFFIC.hour_n.get(h, 0), 0)
            self.assertAlmostEqual(TRAFFIC.hour[h], 1.0, delta=0.05)

    def test_knowing_the_traffic_tightens_the_interval(self):
        # the clock explains some of what used to be "something happened today"
        self.assertTrue(CAL.traffic_aware)
        self.assertLess(CAL.common_mode, CAL.common_mode_flat)


class TestTrafficChangesTheForecast(unittest.TestCase):
    def setUp(self):
        self.cond = Conditions(traffic=TRAFFIC, use_traffic=True)

    def _run(self, start, date="2026-06-02", cond=None):
        return estimate("x", date, start, (53.80, -1.55), plan(18), CAL,
                        city="Leeds", conditions=cond)

    def test_the_peak_costs_more_than_the_middle_of_the_day(self):
        peak = self._run("07:40", cond=self.cond)
        quiet = self._run("10:40", cond=self.cond)
        self.assertGreater(peak.expected_minutes, quiet.expected_minutes)

    def test_with_traffic_off_the_start_time_makes_no_difference(self):
        a = self._run("07:40")
        b = self._run("10:40")
        self.assertAlmostEqual(a.expected_minutes, b.expected_minutes, places=6)

    def test_saturday_is_quicker_than_friday(self):
        fri = self._run("08:00", date="2026-06-05", cond=self.cond)      # a Friday
        sat = self._run("08:00", date="2026-06-06", cond=self.cond)      # a Saturday
        self.assertGreater(fri.expected_minutes, sat.expected_minutes)

    def test_the_forecast_says_how_much_of_it_was_the_traffic(self):
        e = self._run("07:40", cond=self.cond)
        self.assertTrue(e.conditions["traffic_used"])
        self.assertGreater(e.conditions["minutes_from_traffic"], 0)


class TestDisruptions(unittest.TestCase):
    """Declared, not measured - and they have to actually change the answer."""

    def setUp(self):
        self.date = "2026-06-02"
        self.stops = plan(14)
        mid = self.stops[7]
        self.closure = Disruption("D1", "closure", "bridge out", mid.lat, mid.lon,
                                  1.0, self.date, self.date)
        self.works = Disruption("D2", "congestion", "lights", mid.lat, mid.lon,
                                1.0, self.date, self.date, severity=3.0)

    def _run(self, disruptions=(), date=None):
        return estimate("x", date or self.date, "08:00", (53.80, -1.55), self.stops, CAL,
                        city="Leeds",
                        conditions=Conditions(traffic=TRAFFIC, use_traffic=True,
                                              disruptions=list(disruptions)))

    def test_a_closure_is_driven_around_so_the_distance_grows(self):
        base, shut = self._run(), self._run([self.closure])
        self.assertGreater(shut.route_km, base.route_km)
        self.assertGreater(shut.conditions["detour_km"], 0)

    def test_a_closure_makes_the_forecast_less_certain_not_just_longer(self):
        base, shut = self._run(), self._run([self.closure])
        self.assertGreater(shut.expected_minutes, base.expected_minutes)
        self.assertGreater(shut.sd_minutes, base.sd_minutes,
                           "a detour nobody has driven should widen the interval")

    def test_congestion_costs_time_without_adding_distance(self):
        base, slow = self._run(), self._run([self.works])
        self.assertAlmostEqual(slow.route_km, base.route_km, places=6)
        self.assertGreater(slow.expected_minutes, base.expected_minutes)

    def test_it_does_nothing_on_a_day_it_is_not_declared_for(self):
        base = self._run()
        other = self._run([self.closure], date="2026-06-09")
        self.assertAlmostEqual(other.expected_minutes, base.expected_minutes, places=6)

    def test_it_does_nothing_outside_its_hours(self):
        night = Disruption("D3", "congestion", "night works", self.stops[7].lat,
                           self.stops[7].lon, 1.0, self.date, self.date,
                           from_time="22:00", to_time="23:30", severity=3.0)
        self.assertAlmostEqual(self._run([night]).expected_minutes,
                               self._run().expected_minutes, places=6)

    def test_stops_inside_a_closure_are_named_not_quietly_delivered(self):
        e = self._run([self.closure])
        self.assertTrue(e.conditions["stops_inside_a_closure"],
                        "a stop inside a closed area needs a human, not a route")

    def test_a_bigger_closure_costs_more_than_a_small_one(self):
        small = self._run([self.closure])
        big = Disruption("D4", "closure", "wider", self.closure.lat, self.closure.lon,
                         2.5, self.date, self.date)
        self.assertGreater(self._run([big]).expected_minutes, small.expected_minutes)

    def test_the_chord_is_zero_when_the_leg_misses_the_area(self):
        self.assertEqual(chord_km(-1.55, 53.80, -1.54, 53.81, -2.50, 53.00, 1.0), 0.0)

    def test_the_chord_is_about_the_diameter_through_the_middle(self):
        # a leg straight through the centre of a 1 km disc crosses about 2 km of it
        c = chord_km(-1.57, 53.80, -1.53, 53.80, -1.55, 53.80, 1.0)
        self.assertTrue(1.8 < c < 2.2, f"{c:.2f} km")


class TestTheOptionsDiffer(unittest.TestCase):
    """Four suggestions are only worth offering if they are four real choices."""

    @classmethod
    def setUpClass(cls):
        cls.repo = Repository(DB)
        day = cls.repo.schedule_days(days=1)[0]["date"]
        cls.rows = cls.repo.schedule(date=day)[:8]
        cls.sets = [cls.repo.itineraries(r["block_id"]) for r in cls.rows[:5]]

    @classmethod
    def tearDownClass(cls):
        cls.repo.close()

    def test_every_planned_round_gets_at_least_three(self):
        for d in self.sets:
            self.assertGreaterEqual(len(d["options"]), 3, d["block"]["block_id"])

    def test_they_are_not_the_same_drive_under_four_names(self):
        differ = sum(1 for d in self.sets if not d["compare"]["agree"])
        self.assertGreaterEqual(differ, len(self.sets) - 1,
                                "the options collapsed into one route")

    def test_least_driving_really_does_drive_the_least(self):
        for d in self.sets:
            opts = {o["key"]: o for o in d["options"]}
            if "shortest" not in opts:
                continue                      # it merged with another, which is honest
            fewest = min(d["options"], key=lambda o: o["route_km"])
            self.assertLessEqual(opts["shortest"]["route_km"], fewest["route_km"] + 0.01)

    def test_promises_first_keeps_more_promises_than_the_plan(self):
        better = 0
        for d in self.sets:
            opts = {o["key"]: o for o in d["options"]}
            if "windows" in opts and "as_planned" in opts:
                better += opts["windows"]["windows_met"] >= opts["as_planned"]["windows_met"]
        self.assertEqual(better, len([d for d in self.sets
                                      if {"windows", "as_planned"} <=
                                      {o["key"] for o in d["options"]}]))

    def test_keeping_promises_costs_something(self):
        # If it were free there would be no choice to make and no point offering
        # one. Note the reference is whichever option *came out* shortest, not
        # the "shortest" key: the planner stores a routed order, so the
        # least-driving suggestion usually turns out to be the plan itself and
        # is merged into it rather than listed twice.
        cost, kept = [], []
        for d in self.sets:
            opts = {o["key"]: o for o in d["options"]}
            short = opts.get(d["compare"]["shortest"])
            win = opts.get("windows")
            if not short or not win or short is win:
                continue
            cost.append(win["expected_minutes"] - short["expected_minutes"])
            kept.append(win["windows_met"] - short["windows_met"])
        self.assertTrue(cost, "no round offered both a short route and a window route")
        self.assertGreater(sum(cost) / len(cost), 0,
                           "keeping windows came out free, which cannot be right")
        self.assertGreater(sum(kept), 0,
                           "the window route kept no more windows than the short one")

    def test_the_manifest_order_is_offered_as_the_baseline(self):
        keys = {o["key"] for d in self.sets for o in d["options"]}
        self.assertIn("as_received", keys)

    def test_routing_beats_the_raw_manifest(self):
        beaten = 0
        for d in self.sets:
            opts = {o["key"]: o for o in d["options"]}
            short = opts.get(d["compare"]["shortest"])
            raw = opts.get("as_received")
            if short and raw and short is not raw:
                self.assertLess(short["route_km"], raw["route_km"])
                beaten += 1
        self.assertTrue(beaten, "nothing was compared against the manifest order")

    def test_every_option_carries_its_own_map(self):
        for o in self.sets[0]["options"]:
            self.assertEqual(len(o["route"]["nodes"]), self.sets[0]["block"]["stops"]
                             if "stops" in self.sets[0]["block"] else len(o["order"]))

    def test_an_option_is_ordered_over_the_same_stops(self):
        for d in self.sets:
            manifest = {o["key"]: sorted(o["order"]) for o in d["options"]}
            first = next(iter(manifest.values()))
            for order in manifest.values():
                self.assertEqual(order, first, "an option dropped or invented a stop")

    def test_limits_are_reported_not_enforced(self):
        d = self.repo.itineraries(self.rows[0]["block_id"],
                                  limits=Limits(max_minutes=1, max_km=1))
        for o in d["options"]:
            self.assertTrue(o["breaches"], "an impossible limit should be flagged")
        self.assertEqual(len(d["options"]), len(self.sets[0]["options"]) if
                         d["block"]["block_id"] == self.sets[0]["block"]["block_id"]
                         else len(d["options"]))

    def test_a_round_that_has_run_cannot_be_rerouted(self):
        recorded = self.repo.blocks(source="recorded")[0]["block_id"]
        with self.assertRaises(TruthError):
            self.repo.itineraries(recorded)


class TestThePlanIsNeverMistakenForHistory(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo = Repository(DB)

    @classmethod
    def tearDownClass(cls):
        cls.repo.close()

    def test_planned_rounds_are_in_the_future_and_marked_as_planned(self):
        today = dt.date.today().isoformat()
        rows = self.repo.blocks(source="planned", limit=50)
        self.assertTrue(rows)
        for r in rows:
            self.assertEqual(r["source"], "planned")
            self.assertGreater(r["delivery_date"], today)

    def test_a_planned_round_has_no_times_to_pretend_with(self):
        b = self.repo.blocks(source="planned")[0]
        self.assertIsNone(b["duration_minutes"])
        for s in self.repo.stop_rows(b["block_id"]):
            self.assertIsNone(s["arrival_time"])
            self.assertIsNone(s["service_minutes"])

    def test_it_comes_back_as_an_estimate_never_a_ground_truth(self):
        v = self.repo.block_view(self.repo.blocks(source="planned")[0]["block_id"])
        self.assertEqual(v["kind"], "estimate")
        self.assertNotIn("checks", v)

    def test_the_plan_does_not_feed_the_calibration(self):
        # an estimate is not evidence; only recorded rounds may teach the model
        recorded = len(self.repo.blocks(source="recorded", limit=5000))
        self.assertEqual(self.repo.calibration().blocks_used, recorded)

    def test_the_manifest_order_is_kept_beside_the_route(self):
        b = self.repo.blocks(source="planned")[0]
        stops = self.repo.stop_rows(b["block_id"])
        seqs = sorted(s["sequence"] for s in stops)
        manifest = sorted(s["manifest_sequence"] for s in stops)
        self.assertEqual(seqs, manifest)
        self.assertNotEqual([s["sequence"] for s in stops],
                            [s["manifest_sequence"] for s in stops],
                            "the plan should be a route, not the manifest order")

    def test_the_day_totals_are_tighter_than_the_rounds_they_add_up(self):
        day = self.repo.schedule_days(days=1)[0]
        rounds = self.repo.schedule(date=day["date"])
        loose = sum((r["max_minutes"] - r["min_minutes"]) / 60 for r in rounds)
        tight = day["hours_high"] - day["hours_low"]
        self.assertLess(tight, loose,
                        "rounds do not all overrun together, so the day is tighter")

    def test_regenerating_the_month_gives_the_same_month_back(self):
        before = {b["block_id"]: b["start_time"]
                  for b in self.repo.blocks(source="planned", limit=500)}
        generate(DB, days=10)
        again = Repository(DB)
        try:
            after = {b["block_id"]: b["start_time"]
                     for b in again.blocks(source="planned", limit=500)}
        finally:
            again.close()
        self.assertEqual(before, after, "a plan someone looked at should still be there")


class TestDisruptionsAreStoredSanely(unittest.TestCase):
    def setUp(self):
        self.db = os.path.join(tempfile.mkdtemp(), "d.db")
        seed(self.db, days=7, seed_value=4)
        self.repo = Repository(self.db)

    def tearDown(self):
        self.repo.close()

    def test_a_declared_closure_comes_back(self):
        did = self.repo.add_disruption("closure", "bridge", 53.8, -1.55, 1.0,
                                       "2026-06-01", "2026-06-03")
        got = self.repo.disruptions("2026-06-02")
        self.assertEqual([d.disruption_id for d in got], [did])
        self.assertEqual(got[0].source, "manual")

    def test_it_is_not_active_outside_its_dates(self):
        self.repo.add_disruption("closure", "bridge", 53.8, -1.55, 1.0,
                                 "2026-06-01", "2026-06-03")
        self.assertEqual(self.repo.disruptions("2026-06-05"), [])

    def test_a_nameless_disruption_is_refused(self):
        with self.assertRaises(TruthError):
            self.repo.add_disruption("closure", "   ", 53.8, -1.55, 1.0,
                                     "2026-06-01", "2026-06-03")

    def test_an_unknown_kind_is_refused(self):
        with self.assertRaises(TruthError):
            self.repo.add_disruption("meteor", "x", 53.8, -1.55, 1.0,
                                     "2026-06-01", "2026-06-03")

    def test_a_backwards_date_range_is_refused(self):
        with self.assertRaises(TruthError):
            self.repo.add_disruption("closure", "x", 53.8, -1.55, 1.0,
                                     "2026-06-09", "2026-06-03")

    def test_an_absurd_radius_is_refused(self):
        with self.assertRaises(TruthError):
            self.repo.add_disruption("closure", "x", 53.8, -1.55, 500.0,
                                     "2026-06-01", "2026-06-03")

    def test_removing_one_removes_it(self):
        did = self.repo.add_disruption("works", "x", 53.8, -1.55, 1.0,
                                       "2026-06-01", "2026-06-03")
        self.repo.delete_disruption(did)
        self.assertEqual(self.repo.disruptions(), [])

    def test_removing_one_that_is_not_there_says_so(self):
        with self.assertRaises(TruthError):
            self.repo.delete_disruption("DIS-9999")


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestChoosingAnOption(unittest.TestCase):
    """The operator picks; the app remembers and follows the choice everywhere."""

    @classmethod
    def setUpClass(cls):
        cls.db = os.path.join(tempfile.mkdtemp(), "c.db")
        seed(cls.db, days=21, seed_value=31)
        generate(cls.db, days=4)

    def setUp(self):
        self.repo = Repository(self.db)
        self.day = self.repo.schedule_days(days=1)[0]["date"]
        self.bid = self.repo.schedule(date=self.day)[0]["block_id"]
        self._reset()

    def tearDown(self):
        self._reset()
        self.repo.close()

    def _reset(self):
        """Back to a round nobody has touched: these tests share one database."""
        for s in self.repo.stop_rows(self.bid):
            self.repo.con.execute(
                "UPDATE stops SET sequence = -? WHERE block_id=? AND sequence=?",
                (s["manifest_sequence"], self.bid, s["sequence"]))
        self.repo.con.execute(
            "UPDATE stops SET sequence = -sequence WHERE block_id=? AND sequence < 0",
            (self.bid,))
        self.repo.con.execute(
            "UPDATE blocks SET chosen_key=NULL, chosen_by=NULL, chosen_at=NULL "
            "WHERE block_id=?", (self.bid,))
        self.repo.con.commit()

    def _an_option(self, prefer="windows"):
        """Whichever of these is actually on offer - they merge when they agree."""
        keys = [o["key"] for o in self.repo.itineraries(self.bid)["options"]]
        for k in (prefer, "windows", "shortest", "steady", "as_received"):
            if k in keys:
                return k
        return keys[0]

    def test_nothing_is_chosen_until_somebody_chooses(self):
        self.assertIsNone(self.repo.itineraries(self.bid)["chosen"]["key"])

    def test_choosing_applies_that_order_to_the_round(self):
        key = self._an_option()
        was = {s["sequence"]: (s["lat"], s["lon"])
               for s in self.repo.stop_rows(self.bid)}
        order = next(o for o in self.repo.itineraries(self.bid)["options"]
                     if o["key"] == key)["order"]
        self.repo.choose_itinerary(self.bid, key)
        now = {s["sequence"]: (s["lat"], s["lon"])
               for s in self.repo.stop_rows(self.bid)}
        for position, old_seq in enumerate(order, start=1):
            self.assertEqual(now[position], was[old_seq],
                             f"stop {position} is not the one the option put there")

    def test_it_records_who_chose_and_when(self):
        key = self._an_option()
        self.repo.choose_itinerary(self.bid, key, by="alex")
        ch = self.repo.itineraries(self.bid)["chosen"]
        self.assertEqual(ch["key"], key)
        self.assertEqual(ch["by"], "alex")
        self.assertTrue(ch["at"])

    def test_the_choice_survives_closing_the_database(self):
        key = self._an_option()
        self.repo.choose_itinerary(self.bid, key, by="alex")
        self.repo.close()
        self.repo = Repository(self.db)
        self.assertEqual(self.repo.itineraries(self.bid)["chosen"]["key"], key)

    def test_the_choice_carries_its_policy_not_just_its_order(self):
        """Choosing the window route buys waiting; every later forecast pays for it.

        Finds a round where the window route actually waits at a door - on a
        round whose windows are all open by the time the van arrives it waits
        for nothing, and there is no policy to carry.
        """
        target = None
        for r in self.repo.schedule(date=self.day)[:12]:
            for o in self.repo.itineraries(r["block_id"])["options"]:
                if o["key"] == "windows" and o["wait_minutes"] > 0:
                    target = r["block_id"]
                    break
            if target:
                break
        if not target:
            self.skipTest("no round today has a window route that waits")
        self.bid = target
        self.repo.choose_itinerary(self.bid, "windows")
        row = next(r for r in self.repo.schedule(date=self.day)
                   if r["block_id"] == self.bid)
        view = self.repo.block_view(self.bid)
        self.assertGreater(row["wait_minutes"], 0,
                           "the schedule dropped the waiting the operator chose")
        self.assertAlmostEqual(row["wait_minutes"], view["wait_minutes"], places=3)

    def test_every_forecast_of_the_round_agrees_after_a_choice(self):
        self.repo.choose_itinerary(self.bid, self._an_option())
        row = next(r for r in self.repo.schedule(date=self.day)
                   if r["block_id"] == self.bid)
        plan = next(o for o in self.repo.itineraries(self.bid)["options"]
                    if o["key"] == "as_planned")
        self.assertAlmostEqual(row["expected_minutes"], plan["expected_minutes"],
                               places=3)

    def test_the_manifest_survives_so_the_choice_can_be_undone(self):
        before = {(s["lat"], s["lon"]): s["manifest_sequence"]
                  for s in self.repo.stop_rows(self.bid)}
        self.repo.choose_itinerary(self.bid, self._an_option())
        after = {(s["lat"], s["lon"]): s["manifest_sequence"]
                 for s in self.repo.stop_rows(self.bid)}
        self.assertEqual(before, after)

    def test_going_back_to_the_manifest_really_goes_back(self):
        self.repo.choose_itinerary(self.bid, self._an_option())
        self.repo.clear_choice(self.bid)
        for s in self.repo.stop_rows(self.bid):
            self.assertEqual(s["sequence"], s["manifest_sequence"])

    def test_choosing_twice_is_not_a_problem(self):
        self.repo.choose_itinerary(self.bid, self._an_option())
        a = sorted(s["sequence"] for s in self.repo.stop_rows(self.bid))
        self.repo.choose_itinerary(self.bid, "as_planned")
        b = sorted(s["sequence"] for s in self.repo.stop_rows(self.bid))
        self.assertEqual(a, b)
        self.assertEqual(a, list(range(1, len(a) + 1)))

    def test_choosing_what_is_already_being_driven_changes_nothing(self):
        self.repo.choose_itinerary(self.bid, "as_planned")
        before = [(s["sequence"], s["lat"]) for s in self.repo.stop_rows(self.bid)]
        self.repo.choose_itinerary(self.bid, "as_planned")
        self.assertEqual([(s["sequence"], s["lat"])
                          for s in self.repo.stop_rows(self.bid)], before)

    def test_an_option_that_is_not_offered_is_refused(self):
        with self.assertRaises(TruthError):
            self.repo.choose_itinerary(self.bid, "teleport")

    def test_a_round_that_has_run_cannot_be_chosen_for(self):
        recorded = self.repo.blocks(source="recorded")[0]["block_id"]
        with self.assertRaises(TruthError):
            self.repo.choose_itinerary(recorded, "shortest")

    def test_the_stops_are_still_the_same_stops(self):
        before = sorted((s["lat"], s["lon"]) for s in self.repo.stop_rows(self.bid))
        self.repo.choose_itinerary(self.bid, self._an_option())
        after = sorted((s["lat"], s["lon"]) for s in self.repo.stop_rows(self.bid))
        self.assertEqual(before, after)

    def test_a_recommended_choice_is_not_reported_as_a_rebellion(self):
        """The applied order comes back as "as planned", so compare drives, not names."""
        d = self.repo.itineraries(self.bid)
        self.repo.choose_itinerary(self.bid, d["recommendation"]["key"])
        after = self.repo.itineraries(self.bid)
        self.assertTrue(after["chosen"]["is_recommended"],
                        "choosing the recommendation was recorded as going against it")

class TestTheRecommendation(unittest.TestCase):
    """It has to follow its own stated rule, and move when the rule's input moves."""

    @classmethod
    def setUpClass(cls):
        cls.repo = Repository(DB)
        cls.day = cls.repo.schedule_days(days=1)[0]["date"]
        cls.bids = [r["block_id"] for r in cls.repo.schedule(date=cls.day)[:6]]

    @classmethod
    def tearDownClass(cls):
        cls.repo.close()

    def test_every_round_gets_one_and_says_why(self):
        for bid in self.bids[:3]:
            r = self.repo.itineraries(bid)["recommendation"]
            self.assertIn(r["key"], {o["key"] for o in
                                     self.repo.itineraries(bid)["options"]})
            self.assertTrue(r["why"])

    def test_the_score_is_the_arithmetic_it_claims(self):
        d = self.repo.itineraries(self.bids[0])
        rec = d["recommendation"]
        for s, o in zip(sorted(rec["scores"], key=lambda x: x["key"]),
                        sorted(d["options"], key=lambda x: x["key"])):
            self.assertEqual(s["key"], o["key"])
            want = o["expected_minutes"] + rec["minutes_per_missed_window"] * \
                s["broken_windows"]
            self.assertAlmostEqual(s["score_minutes"], want, places=0)

    def test_it_picks_the_lowest_score_that_fits_the_limits(self):
        d = self.repo.itineraries(self.bids[0], limits=Limits(max_minutes=9 * 60))
        rec = d["recommendation"]
        eligible = [s for s in rec["scores"] if not s["set_aside"]]
        self.assertEqual(rec["key"], min(eligible,
                                         key=lambda s: s["score_minutes"])["key"])

    def test_a_free_promise_means_the_fastest_wins(self):
        # price a broken window at nothing and the rule becomes "shortest day"
        d = self.repo.itineraries(self.bids[0], minutes_per_missed_window=0)
        fastest = min(d["options"], key=lambda o: o["expected_minutes"])
        self.assertEqual(d["recommendation"]["key"], fastest["key"])

    def test_an_expensive_promise_moves_the_recommendation(self):
        cheap = self.repo.itineraries(self.bids[0], minutes_per_missed_window=0)
        dear = self.repo.itineraries(self.bids[0], minutes_per_missed_window=240)
        best_windows = max(dear["options"], key=lambda o: o["windows_met"])
        self.assertEqual(dear["recommendation"]["key"], best_windows["key"])
        self.assertNotEqual(cheap["recommendation"]["key"],
                            dear["recommendation"]["key"])

    def test_options_over_a_limit_are_set_aside_but_still_shown(self):
        d = self.repo.itineraries(self.bids[0], limits=Limits(max_minutes=1))
        self.assertTrue(d["recommendation"]["all_over_the_limits"])
        self.assertTrue(all(o["breaches"] for o in d["options"]))
        self.assertTrue(d["recommendation"]["key"], "it still has to answer")

    def test_exactly_one_option_is_marked_recommended(self):
        for bid in self.bids[:3]:
            flags = [o["recommended"] for o in self.repo.itineraries(bid)["options"]]
            self.assertEqual(sum(flags), 1)


class TestBadRowsDoNotBringTheServiceDown(unittest.TestCase):
    """A real database produced a 500 from inside math.radians. Never again.

    One stop with no coordinates reached the distance arithmetic, which raised
    "must be real number, not NoneType" from a trigonometric function. The stack
    trace named `math.radians`; it did not name the block, the stop, or anything
    an operator could act on, and the whole month-ahead page went blank.

    The rule now: a row that cannot be used is skipped and *named*, never
    silently dropped - a forecast quietly made from eighty of a hundred stops is
    worse than an error - and the endpoint stays up.
    """

    @classmethod
    def setUpClass(cls):
        import shutil
        import sqlite3
        good = os.path.join(tempfile.mkdtemp(), "good.db")
        seed(good, days=14, seed_value=19)
        generate(good, days=6)
        cls.db = os.path.join(tempfile.mkdtemp(), "broken.db")
        shutil.copy(good, cls.db)
        con = sqlite3.connect(cls.db)
        # the live table must predate the NOT NULL, so relax it here to put the
        # same shape of row back
        con.executescript(
            "PRAGMA writable_schema=ON;"
            "UPDATE sqlite_master SET sql = replace(sql,"
            "  'lat               REAL NOT NULL', 'lat REAL')"
            " WHERE type='table' AND name='stops';"
            "PRAGMA writable_schema=OFF;")
        con.close()
        con = sqlite3.connect(cls.db)
        cls.bid = con.execute(
            "SELECT block_id FROM blocks WHERE source='planned' "
            "ORDER BY block_id LIMIT 1").fetchone()[0]
        con.execute("UPDATE stops SET lat=NULL WHERE block_id=? AND sequence=3",
                    (cls.bid,))
        con.execute(
            "INSERT INTO blocks (block_id, delivery_date, vehicle_id, vehicle_type,"
            " depot_id, start_time, source) SELECT 'HALF-WRITTEN', delivery_date,"
            " vehicle_id, vehicle_type, depot_id, start_time, 'planned' FROM blocks"
            " WHERE block_id=?", (cls.bid,))
        con.commit()
        con.close()

    def setUp(self):
        self.repo = Repository(self.db)

    def tearDown(self):
        self.repo.close()

    def test_the_month_ahead_still_answers(self):
        rows = self.repo.schedule_days(days=10)
        self.assertTrue(rows, "one bad stop took the whole month down")

    def test_it_says_which_block_and_which_stop(self):
        self.repo.schedule_days(days=10)
        said = " ".join(self.repo.last_problems)
        self.assertIn(self.bid, said)
        self.assertIn("coordinates", said)

    def test_the_day_counts_what_it_could_not_use(self):
        rows = self.repo.schedule_days(days=10)
        self.assertTrue(any(r.get("unusable", 0) for r in rows),
                        "a round that could not be forecast should be counted")

    def test_the_schedule_list_still_answers(self):
        day = self.repo.schedule_days(days=10)[0]["date"]
        self.assertTrue(self.repo.schedule(date=day))

    def test_a_block_can_still_be_forecast_from_the_stops_that_are_good(self):
        v = self.repo.block_view(self.bid)
        self.assertGreater(v["expected_minutes"], 0)

    def test_the_map_leaves_the_bad_stop_off_and_says_so(self):
        v = self.repo.block_view(self.bid)
        self.assertEqual(v["route"]["stops_without_coordinates"], [3])

    def test_it_can_still_be_routed(self):
        self.assertGreaterEqual(len(self.repo.itineraries(self.bid)["options"]), 2)

    def test_a_block_with_no_stops_at_all_says_that_plainly(self):
        with self.assertRaises(TruthError) as e:
            self.repo.block_view("HALF-WRITTEN")
        self.assertIn("no stops", str(e.exception))

    def test_the_geometry_names_the_problem_if_one_ever_reaches_it(self):
        from block_truth.geo import haversine_km
        with self.assertRaises(ValueError) as e:
            haversine_km(53.8, -1.5, None, -1.5)
        self.assertIn("coordinate", str(e.exception))

    def test_the_data_check_finds_every_kind_of_bad_row(self):
        con = self.repo.con
        self.assertEqual(
            con.execute("SELECT count(*) FROM stops WHERE lat IS NULL").fetchone()[0], 1)
        orphans = con.execute(
            "SELECT count(*) FROM blocks b WHERE NOT EXISTS "
            "(SELECT 1 FROM stops s WHERE s.block_id=b.block_id)").fetchone()[0]
        self.assertEqual(orphans, 1)
