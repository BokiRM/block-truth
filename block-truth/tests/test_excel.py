"""
Tests for the workbook importer and for storing a block.

A spreadsheet from a depot is never the shape you asked for, so the importer is
tested on the shapes it will actually meet: odd column names, a title row above
the header, Excel time cells, blank trailing rows, and rows that are simply
wrong.
"""
from __future__ import annotations

import datetime as dt
import io
import os
import tempfile
import unittest

from openpyxl import Workbook

from block_truth.db import connect, counts
from block_truth.excel import parse, template
from block_truth.repository import Repository
from block_truth.seed import seed
from block_truth.truth import TruthError


def book(rows) -> bytes:
    wb = Workbook()
    ws = wb.active
    for r in rows:
        ws.append(list(r))
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


class TestTemplate(unittest.TestCase):
    def test_the_template_parses_itself(self):
        p = parse(template())
        self.assertTrue(p.ok, p.errors)
        self.assertEqual(len(p.stops), 3)

    def test_the_template_carries_the_block_details(self):
        p = parse(template())
        self.assertEqual(p.meta["depotid"], "D01")
        self.assertEqual(p.meta["starttime"], "07:30")

    def test_blank_rows_left_for_the_user_are_not_errors(self):
        p = parse(template())
        self.assertEqual(p.errors, [])


class TestLooseHeaders(unittest.TestCase):
    def test_it_finds_columns_whatever_they_are_called(self):
        p = parse(book([
            ["Stop No", "Street", "Y", "X", "Premises", "Size"],
            [1, "1 A St", 53.80, -1.55, "House", "Small"],
        ]))
        self.assertTrue(p.ok, p.errors)
        self.assertEqual(p.stops[0]["address"], "1 A St")
        self.assertEqual(p.stops[0]["point_type"], "House")

    def test_a_title_row_above_the_header_is_fine(self):
        p = parse(book([
            ["Tuesday round, Leeds"], [],
            ["sequence", "lat", "lon"],
            [1, 53.80, -1.55],
        ]))
        self.assertTrue(p.ok, p.errors)

    def test_a_sheet_with_no_usable_header_says_so(self):
        p = parse(book([["name", "colour"], ["a", "red"]]))
        self.assertFalse(p.ok)
        self.assertIn("header", p.errors[0].problem)


class TestValues(unittest.TestCase):
    def test_a_real_excel_time_cell_is_read(self):
        p = parse(book([
            ["sequence", "lat", "lon", "promised_from", "promised_to"],
            [1, 53.80, -1.55, dt.time(9, 30), dt.time(12, 0)],
        ]))
        self.assertTrue(p.ok, p.errors)
        self.assertEqual(p.stops[0]["promised_from"], "09:30")

    def test_excels_fraction_of_a_day_is_read(self):
        p = parse(book([
            ["sequence", "lat", "lon", "promised_from"],
            [1, 53.80, -1.55, 0.5],                     # noon
        ]))
        self.assertTrue(p.ok, p.errors)
        self.assertEqual(p.stops[0]["promised_from"], "12:00")

    def test_case_and_spacing_in_a_category_do_not_matter(self):
        p = parse(book([
            ["sequence", "lat", "lon", "point_type", "parcel_size"],
            [1, 53.80, -1.55, "parcel locker", "BULKY"],
        ]))
        self.assertTrue(p.ok, p.errors)
        self.assertEqual(p.stops[0]["point_type"], "Parcel Locker")
        self.assertEqual(p.stops[0]["parcel_size"], "Bulky")

    def test_stops_come_back_in_order_however_they_were_typed(self):
        p = parse(book([["sequence", "lat", "lon"],
                        [3, 53.82, -1.52], [1, 53.80, -1.55], [2, 53.81, -1.54]]))
        self.assertEqual([s["sequence"] for s in p.stops], [1, 2, 3])


class TestItReportsEveryProblem(unittest.TestCase):
    def setUp(self):
        self.p = parse(book([
            ["sequence", "lat", "lon", "point_type", "promised_from", "promised_to"],
            [1, 53.80, -1.55, "House", "08:00", "12:00"],
            [2, "oops", -1.55, "House", None, None],
            [3, 53.81, -1.54, "Mansion", "13:00", "09:00"],
            [3, 53.82, -1.53, "House", None, None],
            [5, 999, -1.52, "House", None, None],
        ]))

    def test_nothing_is_ok(self):
        self.assertFalse(self.p.ok)

    def test_every_problem_is_listed_with_its_row(self):
        probs = {(e.row, e.column) for e in self.p.errors}
        self.assertIn((3, "lat/lon"), probs)          # not a number
        self.assertIn((4, "point_type"), probs)       # not in the vocabulary
        self.assertIn((4, "promised_from/to"), probs)  # window shuts before it opens
        self.assertIn((5, "sequence"), probs)         # duplicate
        self.assertIn((6, "lat/lon"), probs)          # not on Earth

    def test_the_good_rows_still_parsed(self):
        self.assertEqual([s["sequence"] for s in self.p.stops], [1, 3])

    def test_a_row_with_details_but_no_coordinates_is_caught(self):
        p = parse(book([["sequence", "address", "lat", "lon"],
                        [1, "1 A St", 53.80, -1.55],
                        [2, "2 B St", None, None]]))
        self.assertFalse(p.ok)
        self.assertIn("coordinates", p.errors[0].problem)


class TestStoringABlock(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.db = os.path.join(tempfile.mkdtemp(), "t.db")
        seed(cls.db, days=14, seed_value=3)

    def setUp(self):
        self.repo = Repository(self.db)

    def tearDown(self):
        self.repo.close()

    def test_a_block_from_a_workbook_round_trips(self):
        stops = parse(template()).stops
        bid = self.repo.suggest_block_id("2026-06-02", "D01")
        self.repo.create_block(bid, "2026-06-02", "D01", stops, source="excel")
        v = self.repo.block_view(bid)
        self.assertEqual(v["kind"], "estimate")
        self.assertEqual(v["stops"], 3)
        self.assertEqual(len(v["route"]["nodes"]), 3)
        self.repo.delete_block(bid)

    def test_a_recorded_block_gives_a_ground_truth_not_an_estimate(self):
        b = next(x for x in self.repo.blocks() if x["source"] == "recorded")
        v = self.repo.block_view(b["block_id"])
        self.assertEqual(v["kind"], "truth")
        self.assertIn("checks", v)

    def test_the_same_id_cannot_be_used_twice(self):
        stops = parse(template()).stops
        bid = self.repo.suggest_block_id("2026-06-03", "D01")
        self.repo.create_block(bid, "2026-06-03", "D01", stops)
        with self.assertRaises(TruthError):
            self.repo.create_block(bid, "2026-06-03", "D01", stops)
        self.repo.delete_block(bid)

    def test_an_unknown_depot_is_refused(self):
        with self.assertRaises(TruthError):
            self.repo.create_block("x", "2026-06-03", "NOPE", parse(template()).stops)

    def test_a_block_with_no_stops_is_refused(self):
        with self.assertRaises(TruthError):
            self.repo.create_block("y", "2026-06-03", "D01", [])

    def test_recorded_history_is_not_deletable_here(self):
        b = next(x for x in self.repo.blocks() if x["source"] == "recorded")
        with self.assertRaises(TruthError):
            self.repo.delete_block(b["block_id"])

    def test_the_suggested_id_is_free(self):
        a = self.repo.suggest_block_id("2026-06-04", "D01")
        self.repo.create_block(a, "2026-06-04", "D01", parse(template()).stops)
        b = self.repo.suggest_block_id("2026-06-04", "D01")
        self.assertNotEqual(a, b)
        self.repo.delete_block(a)

    def test_the_route_closes_the_loop(self):
        b = next(x for x in self.repo.blocks() if x["source"] == "recorded")
        v = self.repo.block_view(b["block_id"])
        r = v["route"]
        legs = sum(n["leg_km"] for n in r["nodes"]) + r["return_km"]
        self.assertAlmostEqual(legs, r["total_km"], places=1)


class TestSeed(unittest.TestCase):
    def test_a_fresh_database_seeds_coherently(self):
        db = os.path.join(tempfile.mkdtemp(), "s.db")
        out = seed(db, days=7, seed_value=5)
        self.assertGreater(out["blocks"], 10)
        self.assertEqual(out["entered"], 0)
        repo = Repository(db)
        bad = []
        for b in repo.blocks(limit=60):
            v = repo.block_view(b["block_id"])
            if not v.get("ok", True):
                bad.append(b["block_id"])
        repo.close()
        self.assertEqual(bad, [], "seeded blocks should pass their own checks")


if __name__ == "__main__":
    unittest.main(verbosity=2)
