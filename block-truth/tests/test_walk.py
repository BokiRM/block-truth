"""
Tests for the walked-block half of the application.

Three of these matter more than the rest:

* `TestPassesConserveTime` - the traversals derived from pings must tile the
  walk: never summing to more than it took, never losing the gaps between runs.
  Both failure modes happened during development, and each one silently changed
  every number downstream by tens of per cent.
* `TestTheModelRecoversTheSeed` - the pace, the transit bonus and the weather
  are hidden in the records by the generator and never handed to the model. If
  it cannot find them here it cannot be trusted to find them in real data.
* `TestRoutingIsAnArcProblem` - the route must cover every street with mail on
  it and come home, and must never be worse than the way the block is walked
  now. A route that misses a street is not a faster route.

    python3 -m unittest tests.test_walk -v
"""
from __future__ import annotations

import datetime as dt
import os
import statistics as st
import tempfile
import unittest

from walk.generate import (TRUE_DEADHEAD_BONUS, TRUE_NEW_WALKER, TRUE_PACE,
                           TRUE_RELIEF, TRUE_WEATHER, generate, true_season)
from walk.graph import (Edge, Graph, covers_everything, deadhead_m, habitual_route,
                        route_length_m, rural_postman)
from walk.model import ITEM_KINDS, block_statics, fit, forecast, predict_block
from walk.passes import Pass, passes_from, smooth, total_seconds
from walk.resource import plan_day, expected_mail
from walk.route import options, saving
from walk.schema import connect, counts
from walk.truth import block_time, buffer_sensitivity, overlap_report

DB = None
MODEL = None


def setUpModule():
    global DB, MODEL
    DB = os.path.join(tempfile.mkdtemp(), "walk.db")
    # Six blocks an office, not four: the model separates pace from door time
    # using the spread in metres-per-door between urban and rural blocks, and a
    # sample without enough of each cannot do it. That sensitivity is worth
    # knowing about, so it is written down here rather than discovered later.
    generate(DB, years=0.8, blocks_per_office=6, detail_days=150, ping_days=3,
             seed_value=5)
    con = connect(DB, read_only=True)
    try:
        MODEL = fit(con=con)
    finally:
        con.close()


class TestTheMockLooksLikeTheirs(unittest.TestCase):
    def setUp(self):
        self.con = connect(DB, read_only=True)

    def tearDown(self):
        self.con.close()

    def test_every_table_on_their_slides_exists_and_has_rows(self):
        c = counts(self.con)
        for table in ("delivery_offices", "blocks", "osm_segments", "delivery_points",
                      "walkers", "devices", "block_days", "pings", "ping_osm_match",
                      "segment_traversals", "block_traversal_totals"):
            self.assertGreater(c[table], 0, table)

    def test_segments_are_the_length_their_slide_says(self):
        for area, lo, hi in (("urban", 8, 30), ("rural", 40, 170)):
            r = self.con.execute(
                "SELECT avg(length_m) m FROM osm_segments WHERE area_type=?",
                (area,)).fetchone()
            self.assertTrue(lo <= r["m"] <= hi, f"{area} segments average {r['m']:.0f} m")

    def test_the_geofence_width_comes_from_the_dp_distance(self):
        rows = self.con.execute(
            "SELECT buffer_m, mean_dp_distance_m FROM segment_geofence "
            "WHERE dp_count > 0 AND mean_dp_distance_m > 8 LIMIT 500").fetchall()
        self.assertTrue(rows)
        capped = 0
        for r in rows:
            if r["buffer_m"] >= 59.9:
                capped += 1                 # a fence that wide would swallow the
                continue                    # next village, so it is held at 60 m
            self.assertGreaterEqual(r["buffer_m"], r["mean_dp_distance_m"])
        self.assertLess(capped, len(rows) * 0.5,
                        "most fences should be sized by the data, not by the cap")

    def test_every_dp_is_joined_to_a_segment(self):
        n = self.con.execute(
            "SELECT count(*) n FROM delivery_points p WHERE NOT EXISTS "
            "(SELECT 1 FROM dp_segment_join j WHERE j.dp_id = p.dp_id)").fetchone()["n"]
        self.assertEqual(n, 0)

    def test_two_years_can_be_asked_for_and_the_history_is_continuous(self):
        c = counts(self.con)
        days = self.con.execute(
            "SELECT count(DISTINCT date) n FROM block_days").fetchone()["n"]
        span = (dt.date.fromisoformat(c["to"]) - dt.date.fromisoformat(c["from"])).days
        self.assertGreater(days, span * 0.7, "too many days missing from the history")

    def test_nobody_delivers_on_a_sunday(self):
        for r in self.con.execute("SELECT DISTINCT date FROM block_days"):
            self.assertNotEqual(dt.date.fromisoformat(r["date"]).weekday(), 6)

    def test_christmas_puts_more_parcels_through_the_same_doors(self):
        rows = self.con.execute(
            "SELECT substr(date,1,7) m, avg(items_parcels*1.0/dps_served) p "
            "FROM block_days GROUP BY 1").fetchall()
        by = {r["m"]: r["p"] for r in rows}
        dec = [v for k, v in by.items() if k.endswith("-12")]
        rest = [v for k, v in by.items() if not k.endswith("-12")]
        if dec and rest:
            self.assertGreater(st.fmean(dec), 1.5 * st.fmean(rest))

    def test_the_season_curve_peaks_before_christmas(self):
        peak = max(range(1, 366), key=lambda d: true_season(dt.date(2025, 1, 1)
                                                            + dt.timedelta(days=d - 1)))
        self.assertTrue(330 <= peak <= 358, f"the peak landed on day {peak}")


class TestPassesConserveTime(unittest.TestCase):
    """The rule everything downstream rests on."""

    def test_a_run_owns_the_interval_up_to_the_next_run(self):
        tl = [(0.0, "a"), (10.0, "a"), (20.0, "b"), (30.0, "b"), (40.0, "a")]
        ps = passes_from(tl)
        self.assertEqual(total_seconds(ps), 40.0, "the walk was forty seconds long")

    def test_nothing_is_lost_between_runs(self):
        tl = [(0.0, "a"), (10.0, "b"), (20.0, "c"), (30.0, "d")]
        self.assertEqual(total_seconds(passes_from(tl)), 30.0)

    def test_time_outside_every_fence_belongs_to_nobody(self):
        tl = [(0.0, "a"), (10.0, None), (20.0, None), (30.0, "a")]
        total = total_seconds(passes_from(tl))
        self.assertLess(total, 30.0)
        self.assertGreater(total, 0.0)

    def test_a_street_walked_twice_is_two_passes(self):
        # the walker really does go away and come back: several pings elsewhere
        # in between, not one. A single ping elsewhere is a bad fix, and the
        # test below says so.
        tl = [(0.0, "a"), (10.0, "a"), (20.0, "b"), (30.0, "b"), (40.0, "b"),
              (50.0, "a"), (60.0, "a")]
        ps = [p for p in passes_from(tl) if p.segment_id == "a"]
        self.assertEqual([p.pass_no for p in ps], [1, 2])

    def test_one_ping_elsewhere_is_a_bad_fix_and_not_a_second_pass(self):
        tl = [(0.0, "a"), (10.0, "a"), (20.0, "b"), (30.0, "a"), (40.0, "a")]
        ps = [p for p in passes_from(tl) if p.segment_id == "a"]
        self.assertEqual(len(ps), 1, "a single stray fix should not split a pass")

    def test_a_single_bad_fix_does_not_become_a_street(self):
        self.assertEqual(smooth(["a", "a", "b", "a", "a"]), ["a", "a", "a", "a", "a"])

    def test_a_real_change_of_street_survives_smoothing(self):
        self.assertEqual(smooth(["a", "a", "b", "b", "b"]), ["a", "a", "b", "b", "b"])

    def test_on_real_pings_the_traversals_never_exceed_the_walk(self):
        con = connect(DB, read_only=True)
        try:
            bad = []
            for r in con.execute("SELECT DISTINCT block_day_id FROM pings LIMIT 60"):
                bt = block_time(con, r["block_day_id"])
                if bt.coverage and bt.coverage > 1.001:
                    bad.append((bt.block_id, round(bt.coverage, 3)))
            self.assertEqual(bad, [], "segment times added up to more than the walk")
        finally:
            con.close()


class TestTheAuditFindsWhatTheSlidesLeaveOpen(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.con = connect(DB, read_only=True)
        row = cls.con.execute("SELECT DISTINCT block_day_id FROM pings LIMIT 1").fetchone()
        cls.bd = row["block_day_id"]

    @classmethod
    def tearDownClass(cls):
        cls.con.close()

    def test_counting_a_ping_for_every_fence_inflates_the_walk(self):
        o = overlap_report(self.con, self.bd)
        self.assertGreater(o["mean_fences_per_ping"], 1.0,
                           "fences that never overlap would make this moot")
        self.assertGreater(o["inflation"], 1.2,
                           "the two readings of their method should differ materially")

    def test_the_honest_reading_cannot_exceed_the_walk(self):
        o = overlap_report(self.con, self.bd)
        self.assertLessEqual(o["nearest_vs_elapsed"], 1.001)

    def test_narrow_fences_lose_time_and_widening_them_finds_it(self):
        s = buffer_sensitivity(self.con, self.bd)
        first, last = s["rows"][0], s["rows"][-1]
        self.assertLess(first["coverage"], last["coverage"])
        self.assertGreater(last["coverage"], 0.9,
                           "a wide enough fence should catch nearly all of the walk")

    def test_it_says_how_much_evidence_is_behind_the_answer(self):
        bt = block_time(self.con, self.bd)
        self.assertGreater(bt.passes, 0)
        self.assertGreater(bt.segments_in_block, 0)
        self.assertIsNotNone(bt.coverage)


class TestTheModelRecoversTheSeed(unittest.TestCase):
    """The generator hides these in the records; the model has to find them."""

    def test_the_pace_is_in_the_right_place_and_the_right_order(self):
        """What can honestly be asked of the pace, and no more.

        The split between "seconds per metre" and "seconds per door" is
        identified only as well as the spread of block densities in the sample
        allows, and on a sample this size it wanders by a quarter either way
        between runs. So this asserts what is stable - that the figure is a
        walking pace at all, and that the countryside is walked faster than a
        terrace - and leaves the quality bar to the out-of-sample test, where
        the *sum* comes out within a few per cent.
        """
        for area, seeded in TRUE_PACE.items():
            if area not in MODEL.pace_mps:
                continue
            got = MODEL.pace_mps[area]
            self.assertTrue(0.6 <= got <= 1.8,
                            f"{area}: {got:.2f} m/s is not a walking pace")
            self.assertTrue(0.65 * seeded <= got <= 1.4 * seeded,
                            f"{area}: fitted {got:.2f} m/s against a seeded {seeded}")
        if {"urban", "rural"} <= set(MODEL.pace_mps):
            self.assertGreater(MODEL.pace_mps["rural"], MODEL.pace_mps["urban"],
                               "a country lane is walked faster than a terrace")

    def test_a_walker_with_nothing_to_deliver_is_found_to_be_faster(self):
        self.assertGreater(MODEL.deadhead_bonus, 1.05)
        self.assertLess(abs(MODEL.deadhead_bonus - TRUE_DEADHEAD_BONUS), 0.25)

    def test_the_weather_comes_out_in_the_right_order(self):
        clear = MODEL.weather.get("clear", 1.0)
        got = {k: v / clear for k, v in MODEL.weather.items()}
        for worse, better in (("heavy rain", "rain"), ("rain", "overcast")):
            if worse in got and better in got:
                self.assertGreater(got[worse], got[better],
                                   f"{worse} should cost more than {better}")

    def test_rain_slows_a_walker_by_something_like_what_was_seeded(self):
        clear = MODEL.weather.get("clear", 1.0)
        if "rain" not in MODEL.weather:
            self.skipTest("no rain in this sample")
        got = MODEL.weather["rain"] / clear
        self.assertTrue(1.02 < got < 1.35, f"rain came out at {got:.2f}")

    def test_covering_somebody_elses_block_is_found_to_be_slower(self):
        clear = MODEL.weather.get("clear", 1.0)
        self.assertGreater(MODEL.relief / clear, 1.0)

    def test_a_new_walker_is_found_to_be_slower_than_an_old_hand(self):
        if "first month" not in MODEL.experience:
            self.skipTest("nobody started during this sample")
        self.assertGreater(MODEL.experience["first month"],
                           MODEL.experience.get("a year or more", 1.0))

    def test_a_parcel_costs_more_than_a_letter(self):
        self.assertGreater(MODEL.item_s.get("parcels", 0), MODEL.item_s.get("letters", 0))

    def test_a_flat_takes_longer_than_a_terrace(self):
        for area in ("urban", "suburban"):
            f, t = (MODEL.service_s.get(f"{area}|flat"),
                    MODEL.service_s.get(f"{area}|terrace"))
            if f and t:
                self.assertGreater(f, t, area)

    def test_it_says_what_it_could_not_measure(self):
        self.assertTrue(MODEL.notes, "a fit that says nothing about itself is a worry")


class TestItPredictsOutOfSample(unittest.TestCase):
    """Fitted on the early walks, tested on the later ones."""

    @classmethod
    def setUpClass(cls):
        cls.con = connect(DB, read_only=True)
        dates = [r["date"] for r in cls.con.execute(
            "SELECT DISTINCT date FROM block_days ORDER BY date")]
        cls.cut = dates[int(len(dates) * 0.7)]
        cls.model = fit(con=cls.con, to_date=cls.cut)
        cls.statics = block_statics(cls.con)
        cls.errors = []
        for r in cls.con.execute(
                "SELECT d.*, b.block_id FROM block_days d JOIN blocks b USING(block_id) "
                "WHERE d.date >= ? LIMIT 4000", (cls.cut,)):
            s = cls.statics.get(r["block_id"])
            actual = cls.con.execute(
                "SELECT does_take_s FROM block_traversal_totals WHERE block_day_id=? "
                "ORDER BY (source='ping') DESC LIMIT 1", (r["block_day_id"],)).fetchone()
            if not s or not actual or not r["dps_served"]:
                continue
            per = {k: r[f"items_{k}"] / r["dps_served"] for k in ITEM_KINDS}
            pred = predict_block(cls.model, s, per)
            if pred > 0:
                cls.errors.append((actual["does_take_s"] - pred) / actual["does_take_s"])

    @classmethod
    def tearDownClass(cls):
        cls.con.close()

    def test_there_is_something_to_judge(self):
        self.assertGreater(len(self.errors), 200)

    def test_it_is_not_systematically_wrong(self):
        bias = st.fmean(self.errors)
        self.assertLess(abs(bias), 0.10, f"the forecast runs {100 * bias:+.1f}% out")

    def test_it_is_close_enough_to_plan_with(self):
        mae = st.fmean([abs(e) for e in self.errors])
        self.assertLess(mae, 0.12, f"mean absolute error {100 * mae:.1f}%")


class TestRoutingIsAnArcProblem(unittest.TestCase):
    """A walked block is streets to cover, not points to visit."""

    def _square(self) -> tuple[Graph, str]:
        g = Graph()
        pts = {"a": (53.80, -1.55), "b": (53.801, -1.55),
               "c": (53.801, -1.549), "d": (53.80, -1.549)}
        for k, (la, lo) in pts.items():
            g.add_node(k, la, lo)
        for key, x, y in (("ab", "a", "b"), ("bc", "b", "c"),
                          ("cd", "c", "d"), ("da", "d", "a")):
            g.add_edge(Edge(key, x, y, 100.0, required=True))
        return g, "a"

    def test_a_loop_is_walked_once_round(self):
        g, start = self._square()
        steps = rural_postman(g, start)
        self.assertTrue(covers_everything(g, steps))
        self.assertEqual(deadhead_m(g, steps), 0.0,
                         "a closed loop needs no street walked twice")

    def test_a_dead_end_has_to_be_walked_both_ways(self):
        g, start = self._square()
        g.add_node("e", 53.802, -1.549)
        g.add_edge(Edge("ce", "c", "e", 50.0, required=True))
        steps = rural_postman(g, start)
        self.assertTrue(covers_everything(g, steps))
        self.assertEqual(sum(1 for s in steps if s.key == "ce"), 2)

    def test_it_comes_home(self):
        g, start = self._square()
        steps = rural_postman(g, start)
        self.assertEqual(steps[-1].to_node, start)

    def test_a_street_with_no_mail_on_it_need_not_be_walked(self):
        g, start = self._square()
        g.add_node("f", 53.803, -1.546)
        g.add_edge(Edge("cf", "c", "f", 400.0, required=False))
        steps = rural_postman(g, start)
        self.assertTrue(covers_everything(g, steps))
        self.assertEqual(sum(1 for s in steps if s.key == "cf"), 0,
                         "nothing to deliver down there, so do not walk it")

    def test_it_covers_every_street_on_a_real_block(self):
        import random
        con = connect(DB, read_only=True)
        try:
            from walk.route import load_graph
            missed = []
            for b in con.execute("SELECT block_id FROM blocks LIMIT 12"):
                g, start = load_graph(con, b["block_id"])
                if not covers_everything(g, rural_postman(g, start)):
                    missed.append(b["block_id"])
            self.assertEqual(missed, [], "a route that misses a street is not a route")
        finally:
            con.close()

    def test_what_is_recommended_is_the_best_of_what_was_tried(self):
        """The heuristic is not always the winner, so the winner is what is offered.

        Frederickson's construction pairs odd junctions greedily, and on a block
        whose junctions pair badly the plain nearest-street route can come out
        shorter. Pretending otherwise would be the easy thing and the wrong one.
        """
        con = connect(DB, read_only=True)
        try:
            for b in con.execute("SELECT block_id FROM blocks LIMIT 10"):
                opts = options(con, b["block_id"], MODEL)
                self.assertEqual(opts[0].minutes, min(o.minutes for o in opts))
                for o in opts:
                    if o.key != "as_walked":
                        self.assertTrue(o.covers_all, f"{o.key} missed a street")
        finally:
            con.close()

    def test_the_saving_is_reported_in_hours_a_year(self):
        con = connect(DB, read_only=True)
        try:
            b = con.execute("SELECT block_id FROM blocks LIMIT 1").fetchone()["block_id"]
            s = saving(con, b, MODEL)
            self.assertIn("hours_a_year", s)
            self.assertGreaterEqual(s["minutes_a_day"], -0.001,
                                    "the optimiser should never lose to the status quo")
            self.assertGreaterEqual(len(s["options"]), 2)
        finally:
            con.close()


class TestResourcing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.con = connect(DB, read_only=True)
        cls.date = cls.con.execute(
            "SELECT max(date) d FROM block_days").fetchone()["d"]
        cls.rows = plan_day(cls.con, MODEL, cls.date)

    @classmethod
    def tearDownClass(cls):
        cls.con.close()

    def test_every_office_gets_a_plan(self):
        self.assertTrue(self.rows)
        for r in self.rows:
            self.assertGreater(r.blocks, 0)
            self.assertGreater(r.expected_h, 0)

    def test_the_hours_add_up_but_the_doubt_does_not(self):
        for r in self.rows:
            spread = r.high_h - r.low_h
            self.assertLess(spread, r.expected_h * 0.5,
                            "an office's day is far more certain than one block's")
            self.assertLess(r.low_h, r.expected_h)
            self.assertLess(r.expected_h, r.high_h)

    def test_a_walker_is_a_person_not_a_pool_of_hours(self):
        for r in self.rows:
            self.assertGreaterEqual(r.walkers_needed, r.blocks,
                                    "five blocks cannot be done by four people")

    def test_overtime_is_counted_as_overtime_and_not_as_people(self):
        for r in self.rows:
            self.assertGreaterEqual(r.overtime_h, 0)
            if r.overtime_h > 0:
                self.assertAlmostEqual(r.overtime_as_people, r.overtime_h / 7.5,
                                       places=6)

    def test_the_mail_forecast_says_how_much_history_is_behind_it(self):
        b = self.con.execute("SELECT block_id FROM blocks LIMIT 1").fetchone()["block_id"]
        mail = expected_mail(self.con, self.date, b)
        self.assertIn("days_behind_it", mail)
        self.assertGreater(mail["days_behind_it"], 0)

    def test_a_block_that_cannot_fit_a_shift_is_named_not_averaged_away(self):
        for r in self.rows:
            for b in r.must_split:
                self.assertIsInstance(b, str)


if __name__ == "__main__":
    unittest.main(verbosity=2)
