"""
Is the audit worth running?

Every seeded block passes its checks, which on its own proves nothing: a check
that cannot fail and a check that never fires look identical from outside. So
these tests copy the database, break one block in it on purpose, and insist that
the audit catches that block and leaves every other one alone.

This is also how the one vacuous check was found. "Parts sum to the recorded
duration" can never fail - the return leg is defined as the leftover, so the sum
is an identity - which left a recorded duration that is simply too long with
nothing to contradict it. The geometry check in `repository` closes that hole:
the stop rows do not know how far home is, but the map does.

    python3 -m unittest discover -s tests -v
"""
from __future__ import annotations

import contextlib
import io
import os
import shutil
import sqlite3
import tempfile
import unittest

from block_truth.cli import main
from block_truth.repository import Repository
from block_truth.seed import seed

GOOD = None          # one clean seeded database, built once


def setUpModule():
    global GOOD
    GOOD = os.path.join(tempfile.mkdtemp(), "good.db")
    seed(GOOD, days=7, seed_value=23)


class Hurt:
    """A copy of the clean database with exactly one block damaged."""

    def __init__(self, sql: str):
        self.path = os.path.join(tempfile.mkdtemp(), "hurt.db")
        shutil.copy(GOOD, self.path)
        con = sqlite3.connect(self.path)
        row = con.execute("SELECT block_id, delivery_date FROM blocks "
                          "WHERE source='recorded' ORDER BY block_id LIMIT 1").fetchone()
        self.block_id, self.date = row
        con.execute(sql, (self.block_id,))
        con.commit()
        con.close()

    def failures(self) -> dict[str, list[str]]:
        """{block_id: names of the checks it fails} across the whole database."""
        repo = Repository(self.path)
        try:
            out = {}
            for b in repo.blocks(limit=400):
                v = repo.block_view(b["block_id"])
                if v["kind"] == "truth" and not v["ok"]:
                    out[b["block_id"]] = [c["name"] for c in v["checks"] if not c["ok"]]
            return out
        finally:
            repo.close()


class TestTheBaselineIsClean(unittest.TestCase):
    def test_nothing_fails_before_anything_is_broken(self):
        repo = Repository(GOOD)
        try:
            bad = {b["block_id"]: [c["name"] for c in repo.block_view(b["block_id"])["checks"]
                                   if not c["ok"]]
                   for b in repo.blocks(limit=400)
                   if not repo.block_view(b["block_id"])["ok"]}
        finally:
            repo.close()
        self.assertEqual(bad, {}, "the baseline must be clean or nothing below means anything")


class TestInjectedFaults(unittest.TestCase):
    """One fault at a time: caught, by the right check, and only where it was put."""

    def _only(self, sql: str, expect: str):
        hurt = Hurt(sql)
        fails = hurt.failures()
        self.assertIn(hurt.block_id, fails,
                      f"the audit missed a block whose records are wrong ({expect})")
        self.assertTrue(any(expect in n for n in fails[hurt.block_id]),
                        f"caught, but by the wrong check: {fails[hurt.block_id]}")
        collateral = set(fails) - {hurt.block_id}
        self.assertEqual(collateral, set(),
                         f"the audit also blamed untouched blocks: {sorted(collateral)[:5]}")
        return hurt

    def test_an_arrival_time_moved_breaks_the_chain(self):
        self._only("UPDATE stops SET arrival_time='23:17' WHERE sequence=3 AND block_id=?",
                   "arrival chain")

    def test_a_service_time_inflated_breaks_the_chain(self):
        self._only("UPDATE stops SET service_minutes=service_minutes+45 "
                   "WHERE sequence=2 AND block_id=?", "arrival chain")

    def test_a_negative_service_time_is_impossible(self):
        self._only("UPDATE stops SET service_minutes=-3 WHERE sequence=2 AND block_id=?",
                   "negative")

    def test_a_duration_shorter_than_the_work_leaves_no_way_home(self):
        self._only("UPDATE blocks SET duration_minutes=duration_minutes-120 WHERE block_id=?",
                   "return leg")

    def test_a_missing_stop_is_a_hole_in_the_sequence(self):
        self._only("DELETE FROM stops WHERE sequence=3 AND block_id=?", "sequence")

    def test_a_duration_inflated_by_an_hour_is_caught_by_the_geometry(self):
        # the hole the identity check leaves: nothing in the stop rows contradicts
        # a longer day, so the distance home has to
        self._only("UPDATE blocks SET duration_minutes=duration_minutes+90 WHERE block_id=?",
                   "drive home")

    def test_a_start_time_moved_shows_up_somewhere(self):
        hurt = self._only("UPDATE blocks SET start_time='04:00' WHERE block_id=?", "")
        self.assertTrue(hurt.failures()[hurt.block_id])


class TestTheIdentityIsLabelledAsOne(unittest.TestCase):
    def test_it_does_not_claim_to_verify_the_duration(self):
        repo = Repository(GOOD)
        try:
            v = repo.block_view(next(b["block_id"] for b in repo.blocks()))
        finally:
            repo.close()
        names = [c["name"] for c in v["checks"]]
        self.assertTrue(any("identity" in n for n in names),
                        f"the sum-of-parts row must say it is an identity: {names}")
        self.assertTrue(any("drive home" in n for n in names),
                        "a recorded block needs the check the identity cannot make")


class TestTheCliReportsThem(unittest.TestCase):
    def test_a_broken_block_is_listed_and_the_exit_code_is_not_zero(self):
        hurt = Hurt("UPDATE stops SET service_minutes=-3 WHERE sequence=2 AND block_id=?")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = main(["--db", hurt.path, "--audit", hurt.date])
        self.assertEqual(code, 1, "a failing block has to be a non-zero exit")
        self.assertIn(hurt.block_id, buf.getvalue())

    def test_a_clean_day_says_so_and_exits_zero(self):
        repo = Repository(GOOD)
        date = repo.dates()[0]
        repo.close()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = main(["--db", GOOD, "--audit", date])
        self.assertEqual(code, 0)
        self.assertIn("none disagree", buf.getvalue())


if __name__ == "__main__":
    unittest.main(verbosity=2)
