"""Opt-in live STATUS report (manager.config.status_report)."""
from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from tests.test_hgm_resume import N_TRAIN


class StatusReportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="hgm_status_"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.seed = self.tmp / "seed"
        self.seed.mkdir()
        (self.seed / "workflow.py").write_text("def run_task(t):\n    return None\n")
        self.exp = self.tmp / "exp"
        self.exp.mkdir()

    def _evolve_status(self, enabled: bool):
        from meta_agent.feedback_gatherer import DefaultFeedbackGatherer
        from meta_agent.managers.hgm import HGMManager
        from tests.test_hgm_resume import TRAIN, _HookEvaluator
        from tests.test_hgm_smoke import _StubEditor

        m = HGMManager(eval_budget=40, init_expansions=2, eval_batch_size=4, alpha=0.6, seed=7,
                       snapshot_tree=True, status_report=enabled)
        m.evolve(editor=_StubEditor(), evaluator=_HookEvaluator(self.exp), gatherer=DefaultFeedbackGatherer(),
                 seed_dir=self.seed, benchmark_dir=self.tmp / "bench", experiment_dir=self.exp,
                 max_rounds=30, score_target=None, train_case_ids=TRAIN, eval_case_ids=None)
        return m

    def test_off_by_default_writes_nothing(self) -> None:
        self._evolve_status(False)
        self.assertFalse((self.exp / "STATUS.md").exists())
        self.assertFalse((self.exp / "status.json").exists())

    def test_rows_and_paired_counts(self) -> None:
        from meta_agent import status_report

        self._evolve_status(True)
        info = status_report.collect(self.exp)
        self.assertEqual({n["node_id"] for n in info["nodes"]},
                         {int(d.name[6:]) for d in self.exp.glob("round_???")})
        md = (self.exp / "STATUS.md").read_text()
        self.assertIn("| node | parent |", md)
        self.assertIn("main_loop.py --resume", md)
        self.assertNotIn("sid_seedling", md)          # no project-specific commands
        for n in info["nodes"]:
            if n["node_id"] != 0 and n.get("paired"):
                p = n["paired"]
                self.assertEqual(p["better"] + p["worse"] + p["same"], p["n"])
                self.assertLessEqual(p["n"], N_TRAIN)


if __name__ == "__main__":
    unittest.main()
