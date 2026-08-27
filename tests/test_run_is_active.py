"""Tests for meta_agent.run_inspect.run_is_active -- the dashboard's
LIVE/STOPPED heuristic. Real bug this guards against: a single slow
straggler case within a batch can legitimately block any new file write for
tens of minutes (observed live, growing past 40+ minutes, well within the
evaluator's own wall_time_s_per_case=3600s default) -- too short a
staleness threshold makes a perfectly healthy, actively-running process
falsely show as "STOPPED".

    PYTHONPATH=. python3 -m unittest tests.test_run_is_active
"""
from __future__ import annotations

import os
import shutil
import tempfile
import time
import unittest
from pathlib import Path

from meta_agent.run_inspect import run_is_active


class RunIsActiveTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="run_is_active_test_"))
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def _touch(self, rel_path: str, *, age_s: float) -> None:
        path = self.tmp / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x", encoding="utf-8")
        ts = time.time() - age_s
        os.utime(path, (ts, ts))

    def test_recently_written_file_is_active(self) -> None:
        self._touch("round_000/logs/case_1.json", age_s=5)
        self.assertTrue(run_is_active(self.tmp))

    def test_a_long_running_straggler_case_still_counts_as_active(self) -> None:
        # The exact scenario that motivated bumping the default threshold:
        # no new file for 45 minutes, well within a single case's own
        # wall_time_s_per_case=3600s timeout -- this must NOT read as dead.
        self._touch("round_023/logs/case_105.json", age_s=45 * 60)
        self.assertTrue(run_is_active(self.tmp))

    def test_truly_stale_directory_is_not_active(self) -> None:
        self._touch("round_000/logs/case_1.json", age_s=3 * 3600)
        self.assertFalse(run_is_active(self.tmp))

    def test_run_summary_present_means_finished_regardless_of_mtimes(self) -> None:
        self._touch("round_000/logs/case_1.json", age_s=5)  # very fresh
        self._touch("run_summary.md", age_s=5)
        self.assertFalse(run_is_active(self.tmp))

    def test_empty_directory_is_not_active(self) -> None:
        self.assertFalse(run_is_active(self.tmp))

    def test_custom_staleness_s_is_still_respected(self) -> None:
        self._touch("round_000/logs/case_1.json", age_s=200)
        self.assertFalse(run_is_active(self.tmp, staleness_s=120.0))
        self.assertTrue(run_is_active(self.tmp, staleness_s=300.0))


if __name__ == "__main__":
    unittest.main()
