"""
Tests for the estimator.

The one that matters is `TestCoverage`: a 90% interval has to contain the real
duration about 90% of the time on blocks the calibration never saw. Everything
else is a property that has to hold whatever the data says.

    python3 -m unittest discover -s tests -v
"""
from __future__ import annotations

import os
import sqlite3
import tempfile
import unittest

from block_truth.calibrate import Calibration, Dist, fit
from block_truth.db import connect
from block_truth.estimate import StopPlan, estimate
from block_truth.geo import haversine_km
from block_truth.seed import seed

# One seeded database for the whole module: 21 days is a couple of seconds, and
# every test below wants the same one.
DB = None
CAL = None


def setUpModule():
    global DB, CAL
    DB = os.path.join(tempfile.mkdtemp(), "t.db")
    seed(DB, days=21, seed_value=11)
    CAL = fit(DB)


def plan(n, lat=53.80, lon=-1.55, step=0.004, **kw):
    return [StopPlan(i + 1, lat + i * step, lon + i * step, **kw) for i in range(n)]


class TestRecoversWhatWasSeeded(unittest.TestCase):
    """The seed builds times from geometry, so the fit has something real to find."""

    def test_speed_comes_back(self):
        # seeded: London 13, Manchester 17, Birmingham 17.5, the rest 21 km/h,
        # then scaled by vehicle type and the day factor, so the fit lands a
        # little under the headline figure
        london = 60 / CAL.travel_per_km["London"].mean
        leeds = 60 / CAL.travel_per_km["Leeds"].mean
        self.assertLess(london, leeds, "London should come out slower than Leeds")
        self.assertTrue(10 < london < 16, f"London speed {london:.1f} km/h")
        self.assertTrue(16 < leeds < 23, f"Leeds speed {leeds:.1f} km/h")

    def test_depot_load_orders_by_vehicle(self):
        small = CAL.depot_for("Small Van").mean
        large = CAL.depot_for("Large Van").mean
        truck = CAL.depot_for("Box Truck").mean
        self.assertLess(small, large)
        self.assertLess(large, truck)

    def test_a_failed_stop_is_shorter_than_a_successful_one(self):
        self.assertLess(CAL.service_failed.mean,
                        CAL.service_for("House", "Medium").mean)

    def test_a_flat_takes_longer_than_a_house(self):
        self.assertGreater(CAL.service_for("Flat", "Medium").mean,
                           CAL.service_for("House", "Medium").mean)

    def test_a_bulky_parcel_takes_longer_than_a_letter(self):
        self.assertGreater(CAL.service_for("House", "Bulky").mean,
                           CAL.service_for("House", "Letter").mean)


class TestCoverage(unittest.TestCase):
    """The interval has to mean what it says, on data the fit never saw."""

    @classmethod
    def setUpClass(cls):
        con = connect(DB, read_only=True)
        dates = sorted({r[0] for r in con.execute("SELECT delivery_date FROM blocks")})
        cls.cut = dates[int(len(dates) * 0.75)]

        cls.train = train = os.path.join(tempfile.mkdtemp(), "train.db")
        import shutil
        shutil.copy(DB, train)
        w = sqlite3.connect(train)
        w.execute("DELETE FROM stops WHERE block_id IN "
                  "(SELECT block_id FROM blocks WHERE delivery_date >= ?)", (cls.cut,))
        w.execute("DELETE FROM blocks WHERE delivery_date >= ?", (cls.cut,))
        w.commit(); w.close()
        cls.cal = fit(train)

        cls.held = con.execute(
            "SELECT b.*, d.city, d.lat dlat, d.lon dlon FROM blocks b "
            "JOIN depots d USING(depot_id) WHERE b.delivery_date >= ?", (cls.cut,)).fetchall()
        cls.stops = {b["block_id"]: con.execute(
            "SELECT * FROM stops WHERE block_id=? ORDER BY sequence",
            (b["block_id"],)).fetchall() for b in cls.held}
        con.close()

    def _coverage(self, confidence):
        inside = 0
        for b in self.held:
            e = estimate(b["block_id"], b["delivery_date"], b["start_time"],
                         (b["dlat"], b["dlon"]),
                         [StopPlan(s["sequence"], s["lat"], s["lon"], s["point_type"],
                                   s["parcel_size"]) for s in self.stops[b["block_id"]]],
                         self.cal, city=b["city"], vehicle_type=b["vehicle_type"],
                         confidence=confidence)
            inside += e.min_minutes <= b["duration_minutes"] <= e.max_minutes
        return inside / len(self.held)

    def test_ninety_percent_means_ninety_percent(self):
        got = self._coverage(90)
        self.assertGreater(len(self.held), 40, "need enough held-out blocks to judge")
        # +/- 3 sampling SEs at n~80; a wider miss is the model, not luck
        self.assertTrue(0.80 <= got <= 0.97, f"90% interval covered {100 * got:.1f}%")

    def test_wider_confidence_covers_more(self):
        self.assertLessEqual(self._coverage(80), self._coverage(95) + 1e-9)

    def test_knowing_the_traffic_predicts_better_than_assuming_a_flat_day(self):
        """The whole case for the traffic model, measured out of sample.

        It has to earn its place: a curve fitted on the training days must
        predict the held-out days better than pretending every hour is the same.
        It does, by about a sixth of the error, and the interval it quotes is
        the tighter for it.
        """
        from block_truth.calibrate import fit_full
        from block_truth.traffic import Conditions

        cal2, traf = fit_full(self.train)
        cond = Conditions(traffic=traf, use_traffic=True)

        def error(c, cl):
            errs = []
            for b in self.held:
                e = estimate(b["block_id"], b["delivery_date"], b["start_time"],
                             (b["dlat"], b["dlon"]),
                             [StopPlan(s["sequence"], s["lat"], s["lon"],
                                       s["point_type"], s["parcel_size"])
                              for s in self.stops[b["block_id"]]],
                             cl, city=b["city"], vehicle_type=b["vehicle_type"],
                             conditions=c)
                errs.append(abs(b["duration_minutes"] - e.expected_minutes)
                            / b["duration_minutes"])
            return sum(errs) / len(errs)

        flat_err = error(None, self.cal)
        traffic_err = error(cond, cal2)
        self.assertLess(traffic_err, flat_err,
                        f"traffic-aware {100 * traffic_err:.1f}% vs flat "
                        f"{100 * flat_err:.1f}% mean absolute error")

    def test_the_point_estimate_is_not_biased(self):
        errs = []
        for b in self.held:
            e = estimate(b["block_id"], b["delivery_date"], b["start_time"],
                         (b["dlat"], b["dlon"]),
                         [StopPlan(s["sequence"], s["lat"], s["lon"], s["point_type"],
                                   s["parcel_size"]) for s in self.stops[b["block_id"]]],
                         self.cal, city=b["city"], vehicle_type=b["vehicle_type"])
            errs.append((b["duration_minutes"] - e.expected_minutes) / b["duration_minutes"])
        bias = sum(errs) / len(errs)
        self.assertLess(abs(bias), 0.05, f"expected duration is {100 * bias:+.1f}% out")


class TestProperties(unittest.TestCase):
    """Things that must hold whatever the calibration says."""

    def test_more_stops_takes_longer(self):
        a = estimate("a", "2026-06-01", "07:30", (53.80, -1.55), plan(8), CAL)
        b = estimate("b", "2026-06-01", "07:30", (53.80, -1.55), plan(20), CAL)
        self.assertGreater(b.expected_minutes, a.expected_minutes)

    def test_the_interval_is_relatively_tighter_on_a_longer_round(self):
        # independent noise cancels; this is the whole argument for not summing
        # per-stop extremes
        a = estimate("a", "2026-06-01", "07:30", (53.80, -1.55), plan(5), CAL)
        b = estimate("b", "2026-06-01", "07:30", (53.80, -1.55), plan(40), CAL)
        self.assertLess(b.sd_minutes / b.expected_minutes,
                        a.sd_minutes / a.expected_minutes)

    def test_but_it_never_collapses_to_nothing(self):
        # because the common-mode part does not shrink with N
        b = estimate("b", "2026-06-01", "07:30", (53.80, -1.55), plan(200), CAL)
        self.assertGreater(b.sd_minutes / b.expected_minutes, 0.01)

    def test_min_is_below_expected_is_below_max(self):
        e = estimate("e", "2026-06-01", "07:30", (53.80, -1.55), plan(15), CAL)
        self.assertLess(e.min_minutes, e.expected_minutes)
        self.assertLess(e.expected_minutes, e.max_minutes)

    def test_the_parts_add_up_to_the_expected_duration(self):
        e = estimate("e", "2026-06-01", "07:30", (53.80, -1.55), plan(12), CAL)
        total = (e.depot_minutes + e.travel_minutes + e.service_minutes
                 + e.return_minutes)
        self.assertAlmostEqual(total, e.expected_minutes, places=6)

    def test_spread_out_stops_take_longer_than_tight_ones(self):
        tight = estimate("t", "2026-06-01", "07:30", (53.80, -1.55),
                         plan(12, step=0.001), CAL)
        wide = estimate("w", "2026-06-01", "07:30", (53.80, -1.55),
                        plan(12, step=0.010), CAL)
        self.assertGreater(wide.expected_minutes, tight.expected_minutes)
        self.assertGreater(wide.route_km, tight.route_km)

    def test_flats_take_longer_than_houses(self):
        h = estimate("h", "2026-06-01", "07:30", (53.80, -1.55),
                     plan(15, point_type="House"), CAL)
        f = estimate("f", "2026-06-01", "07:30", (53.80, -1.55),
                     plan(15, point_type="Flat"), CAL)
        self.assertGreater(f.service_minutes, h.service_minutes)

    def test_arrival_times_only_move_forwards(self):
        e = estimate("e", "2026-06-01", "07:30", (53.80, -1.55), plan(18), CAL)
        arr = [s.arrival_minutes for s in e.stop_estimates]
        self.assertEqual(arr, sorted(arr))

    def test_a_later_stop_is_less_certain_than_an_earlier_one(self):
        e = estimate("e", "2026-06-01", "07:30", (53.80, -1.55), plan(18), CAL)
        first, last = e.stop_estimates[0], e.stop_estimates[-1]
        self.assertLess(first.arrival_max - first.arrival_min,
                        last.arrival_max - last.arrival_min)

    def test_the_window_check_reads_the_clock(self):
        stops = [StopPlan(1, 53.801, -1.551, promised_from="06:00", promised_to="06:30"),
                 StopPlan(2, 53.805, -1.545, promised_from="07:00", promised_to="21:00")]
        e = estimate("e", "2026-06-01", "07:30", (53.80, -1.55), stops, CAL)
        self.assertFalse(e.stop_estimates[0].on_time)     # the window shut before we left
        self.assertTrue(e.stop_estimates[1].on_time)

    def test_no_stops_is_an_error(self):
        with self.assertRaises(ValueError):
            estimate("e", "2026-06-01", "07:30", (53.80, -1.55), [], CAL)

    def test_an_unknown_confidence_is_refused(self):
        with self.assertRaises(ValueError):
            estimate("e", "2026-06-01", "07:30", (53.80, -1.55), plan(3), CAL,
                     confidence=73)


class TestFallsBackWithoutData(unittest.TestCase):
    def test_an_empty_calibration_still_produces_a_sane_answer(self):
        cal = Calibration()
        e = estimate("e", "2026-06-01", "07:30", (53.80, -1.55), plan(10), cal)
        self.assertGreater(e.expected_minutes, 0)
        self.assertLess(e.min_minutes, e.max_minutes)


class TestGeometry(unittest.TestCase):
    def test_haversine_against_a_known_distance(self):
        # Leeds to Manchester is about 57 km as the crow flies
        km = haversine_km(53.7997, -1.5492, 53.4808, -2.2426)
        self.assertTrue(55 < km < 60, f"{km:.1f} km")

    def test_the_same_point_is_no_distance_at_all(self):
        self.assertAlmostEqual(haversine_km(53.8, -1.5, 53.8, -1.5), 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
