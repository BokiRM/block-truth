"""
Tests for the completion-time calculation.

Hand-built rounds where the right answer is known by construction, plus the
cases that actually go wrong in recorded data: rounding, a round past midnight,
a broken chain, a missing duration.

    python3 -m unittest discover -s tests -v
"""
from __future__ import annotations

import unittest

from block_truth.fmt import hhmm, to_minutes
from block_truth.model import Stop
from block_truth.truth import TruthError, compute


def round_of(start, legs, services, duration=None, outcomes=None, depot=0.0):
    """Build a round whose arrivals follow exactly from its parts.

    `depot` is the time spent loading before the first leg, which is not part of
    any stop's recorded travel - exactly as the real records have it.
    """
    stops, t = [], start + depot
    for i, (leg, svc) in enumerate(zip(legs, services)):
        t += leg
        stops.append(Stop(sequence=i + 1, arrival_minutes=t, service_minutes=svc,
                          travel_minutes=leg,
                          outcome=(outcomes[i] if outcomes else "Delivered")))
        t += svc
    return compute("B1", "2026-05-14", start, stops, duration_recorded=duration)


class TestArithmetic(unittest.TestCase):
    def test_the_parts_add_up_to_the_completion_time(self):
        # 07:40 start, 20 min at the depot, three stops, 15 min home
        start = to_minutes("07:40")
        t = round_of(start, legs=[10, 6, 8], services=[4, 3, 5], depot=20,
                     duration=20 + 10 + 4 + 6 + 3 + 8 + 5 + 15)
        self.assertAlmostEqual(t.service_minutes, 12)
        self.assertAlmostEqual(t.travel_minutes, 24)
        self.assertAlmostEqual(t.depot_minutes, 20)
        self.assertAlmostEqual(t.return_leg_minutes, 15)
        self.assertAlmostEqual(t.duration_minutes, 71)
        self.assertEqual(hhmm(t.completion_minutes), "08:51")
        self.assertTrue(t.ok, [c for c in t.checks if not c.ok])

    def test_depot_time_is_separated_from_the_first_leg(self):
        # the whole point: 25 min loading then a 12 min run out is not a 37 min leg
        t = round_of(to_minutes("06:00"), legs=[12, 7], services=[5, 5], depot=25,
                     duration=25 + 12 + 5 + 7 + 5 + 9)
        self.assertAlmostEqual(t.depot_minutes, 25)
        self.assertAlmostEqual(t.travel_minutes, 19)
        self.assertAlmostEqual(t.return_leg_minutes, 9)

    def test_completion_is_start_plus_duration(self):
        t = round_of(to_minutes("08:00"), [12, 5], [3, 4], duration=60)
        self.assertAlmostEqual(t.completion_minutes, to_minutes("09:00"))

    def test_door_to_door_excludes_the_depot_and_the_way_home(self):
        t = round_of(to_minutes("07:00"), [10, 10], [5, 5], depot=20, duration=100)
        # first arrival 07:30, last door finishes 07:50
        self.assertAlmostEqual(t.door_to_door_minutes, 20)

    def test_minutes_per_stop_uses_the_whole_round(self):
        t = round_of(to_minutes("08:00"), [10, 5, 5], [5, 5, 5], duration=90)
        self.assertAlmostEqual(t.as_dict()["minutes_per_stop"], 30.0)


class TestWithoutARecordedDuration(unittest.TestCase):
    def test_it_stops_at_the_last_door_rather_than_inventing_a_drive_home(self):
        t = round_of(to_minutes("08:00"), [10, 6], [4, 3], duration=None)
        self.assertAlmostEqual(t.duration_minutes, 10 + 4 + 6 + 3)
        self.assertAlmostEqual(t.return_leg_minutes, 0.0)
        self.assertIn("no recorded duration", t.completion_source)
        self.assertTrue(t.ok)


class TestRounding(unittest.TestCase):
    def test_arrivals_recorded_to_the_minute_still_pass(self):
        # the generator writes HH:MM, so every arrival carries up to 30s of error
        start = to_minutes("07:54")
        legs = [13.77, 5.01, 3.42, 4.12]
        svcs = [3.97, 2.32, 2.92, 1.56]
        stops, t = [], start
        for i, (leg, svc) in enumerate(zip(legs, svcs)):
            t += leg
            stops.append(Stop(i + 1, round(t), svc, leg, "Delivered"))  # minute precision
            t += svc
        res = compute("B", "2026-05-14", start, stops, duration_recorded=120.0)
        self.assertTrue(res.ok, [c.detail for c in res.checks if not c.ok])


class TestMidnight(unittest.TestCase):
    def test_a_round_that_runs_past_midnight_keeps_counting(self):
        start = to_minutes("22:30")
        stops = [
            Stop(1, to_minutes("23:40"), 5, 70, "Delivered"),
            Stop(2, to_minutes("00:20"), 5, 35, "Delivered"),   # wraps
        ]
        t = compute("B", "2026-05-14", start, stops, duration_recorded=140.0)
        self.assertTrue(t.crosses_midnight)
        self.assertEqual(hhmm(t.completion_minutes), "00:50 (+1d)")
        self.assertAlmostEqual(t.last_arrival_minutes, to_minutes("00:20") + 1440)
        self.assertTrue(t.ok, [c.detail for c in t.checks if not c.ok])


class TestChecksCatchBadRecords(unittest.TestCase):
    def test_a_broken_arrival_chain_is_reported(self):
        stops = [
            Stop(1, to_minutes("08:10"), 5, 10, "Delivered"),
            Stop(2, to_minutes("09:30"), 5, 5, "Delivered"),   # 70 min unaccounted for
        ]
        t = compute("B", "2026-05-14", to_minutes("08:00"), stops, 120.0)
        self.assertFalse(t.ok)
        failed = [c.name for c in t.checks if not c.ok]
        self.assertIn("arrival chain", failed)

    def test_a_duration_shorter_than_the_work_gives_a_negative_return_leg(self):
        t = round_of(to_minutes("08:00"), [10, 5], [5, 5], duration=10)
        self.assertFalse(t.ok)
        self.assertIn("return leg is not negative",
                      [c.name for c in t.checks if not c.ok])

    def test_negative_service_is_reported(self):
        stops = [Stop(1, to_minutes("08:10"), -5, 10, "Delivered")]
        t = compute("B", "2026-05-14", to_minutes("08:00"), stops, 30.0)
        self.assertIn("no negative service or travel",
                      [c.name for c in t.checks if not c.ok])

    def test_a_gap_in_the_sequence_is_reported(self):
        stops = [Stop(1, to_minutes("08:10"), 5, 10, "Delivered"),
                 Stop(3, to_minutes("08:25"), 5, 10, "Delivered")]
        t = compute("B", "2026-05-14", to_minutes("08:00"), stops, 60.0)
        self.assertIn("stop sequence is unbroken",
                      [c.name for c in t.checks if not c.ok])

    def test_the_parts_always_rebuild_the_recorded_duration(self):
        # this one is arithmetic, not data quality: it must hold for any input
        t = round_of(to_minutes("06:15"), [22, 9, 4, 11], [3, 7, 2, 6], duration=301.5)
        rebuilt = (t.depot_minutes + t.service_minutes + t.travel_minutes
                   + t.return_leg_minutes)
        self.assertAlmostEqual(rebuilt, 301.5, places=6)


class TestOutcomes(unittest.TestCase):
    def test_delivered_and_failed_are_counted(self):
        t = round_of(to_minutes("08:00"), [10, 5, 5], [4, 2, 4], duration=60,
                     outcomes=["Delivered", "Unsuccessful", "Delivered"])
        self.assertEqual((t.delivered, t.failed), (2, 1))


class TestEdges(unittest.TestCase):
    def test_no_stops_is_an_error_not_a_zero(self):
        with self.assertRaises(TruthError):
            compute("B", "2026-05-14", 480.0, [], 60.0)

    def test_a_single_stop_round_works(self):
        t = round_of(to_minutes("08:00"), [15], [6], depot=5, duration=40)
        self.assertEqual(t.stops, 1)
        self.assertAlmostEqual(t.depot_minutes, 5)
        self.assertAlmostEqual(t.return_leg_minutes, 14)
        self.assertTrue(t.ok)

    def test_stops_out_of_order_are_sorted_first(self):
        stops = [Stop(2, to_minutes("08:25"), 5, 5, "Delivered"),
                 Stop(1, to_minutes("08:15"), 5, 15, "Delivered")]
        t = compute("B", "2026-05-14", to_minutes("08:00"), stops, 60.0)
        self.assertAlmostEqual(t.first_arrival_minutes, to_minutes("08:15"))
        self.assertTrue(t.ok, [c.detail for c in t.checks if not c.ok])


class TestClock(unittest.TestCase):
    def test_rounding_happens_once_so_there_is_no_1560(self):
        self.assertEqual(hhmm(15 * 60 + 59.6), "16:00")

    def test_round_trip(self):
        for s in ("00:00", "07:54", "23:59"):
            self.assertEqual(hhmm(to_minutes(s)), s)

    def test_rejects_nonsense(self):
        for bad in ("", "7", "07:99", "ten past"):
            with self.assertRaises(ValueError):
                to_minutes(bad)


if __name__ == "__main__":
    unittest.main(verbosity=2)
