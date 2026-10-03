"""first_batch_on_expand: every new node is evaluated once (starvation fix)."""
from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from tests.test_hgm_smoke import _StubEditor, _StubEvaluator

TRAIN = [f"c{i}" for i in range(20)]


class FirstBatchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="firstbatch_"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.seed = self.tmp / "seed"
        self.seed.mkdir()
        (self.seed / "workflow.py").write_text("def run_task(t):\n    return None\n")

    def _run(self, flag: bool, budget: int = 60):
        from meta_agent.feedback_gatherer import DefaultFeedbackGatherer
        from meta_agent.managers.hgm import HGMManager

        exp = Path(tempfile.mkdtemp(prefix="exp_", dir=self.tmp))
        m = HGMManager(eval_budget=budget, init_expansions=2, eval_batch_size=4, alpha=0.7, seed=11,
                       first_batch_on_expand=flag)
        m.evolve(editor=_StubEditor(), evaluator=_StubEvaluator(), gatherer=DefaultFeedbackGatherer(),
                 seed_dir=self.seed, benchmark_dir=self.tmp / "b", experiment_dir=exp, max_rounds=40,
                 score_target=None, train_case_ids=TRAIN, eval_case_ids=None)
        return m

    def test_on_no_real_node_is_left_unevaluated(self) -> None:
        m = self._run(True)
        unevaluated = [nid for nid, n in m._tree.nodes.items() if not n.edit_failed and n.n_evals == 0]
        self.assertEqual(unevaluated, [])
        spent = sum(n.n_evals for nid, n in m._tree.nodes.items() if nid != 0)
        self.assertEqual(m._budget_spent, 60)
        self.assertGreaterEqual(spent, m._budget_spent)   # >= : finalize top-ups are uncharged

    def test_off_keeps_the_default_schedule(self) -> None:
        a, b = self._run(False), self._run(False)
        self.assertEqual({k: n.n_evals for k, n in a._tree.nodes.items()},
                         {k: n.n_evals for k, n in b._tree.nodes.items()})
        self.assertEqual(a._budget_spent, 60)


if __name__ == "__main__":
    unittest.main()
