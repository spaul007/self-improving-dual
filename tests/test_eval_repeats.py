"""Tests for the opt-in `eval_repeats` knob (HGMManager) -- repeat-eligible
dripping for small/single-case runs, where breadth across many distinct
cases isn't available to do variance reduction for free.

    PYTHONPATH=. python3 -m unittest tests.test_eval_repeats
"""
from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from tests.test_hgm_smoke import _StubEditor, _StubEvaluator


class EvaluableAndUnevaluatedTests(unittest.TestCase):
    """Direct, hand-built-tree checks on the two gated methods, independent
    of a full evolve() run."""

    def _node(self, node_id, parent_id, case_scores: dict[str, float]):
        from meta_agent.managers.hgm_tree import HGMNode
        from meta_agent.models import CaseResult

        n = HGMNode(node_id=node_id, parent_id=parent_id, round_dir=Path("."))
        for cid, s in case_scores.items():
            n.record(CaseResult(case_id=cid, passed=s >= 1.0, score=s))
        return n

    def test_default_eval_repeats_is_one_and_unchanged(self) -> None:
        from meta_agent.managers.hgm import HGMManager

        m = HGMManager()
        self.assertEqual(m.eval_repeats, 1)

    def test_evaluable_excludes_a_fully_seen_node_by_default(self) -> None:
        from meta_agent.managers.hgm import HGMManager
        from meta_agent.managers.hgm_tree import HGMTree

        m = HGMManager()  # eval_repeats=1 (default)
        m._tree = HGMTree()
        m._train_case_ids = ["a"]
        m._tree.add(self._node(0, None, {"a": 0.7}))  # has seen its only case
        m._tree.add(self._node(1, 0, {}))  # unevaluated
        self.assertEqual(m._evaluable(), [1])

    def test_evaluable_never_excludes_with_repeats_enabled(self) -> None:
        from meta_agent.managers.hgm import HGMManager
        from meta_agent.managers.hgm_tree import HGMTree
        from meta_agent.models import CaseResult

        m = HGMManager(eval_repeats=5)
        m._tree = HGMTree()
        m._train_case_ids = ["a"]
        node = self._node(0, None, {"a": 0.7})
        for _ in range(9):  # already re-evaluated many times over
            node.record(CaseResult(case_id="a", passed=True, score=0.7))
        m._tree.add(node)
        self.assertEqual(m._evaluable(), [0])

    def test_evaluable_empty_with_repeats_enabled_and_no_train_cases(self) -> None:
        from meta_agent.managers.hgm import HGMManager
        from meta_agent.managers.hgm_tree import HGMTree

        m = HGMManager(eval_repeats=5)
        m._tree = HGMTree()
        m._train_case_ids = []
        m._tree.add(self._node(0, None, {}))
        self.assertEqual(m._evaluable(), [])

    def test_record_increments_count_instead_of_just_marking_seen(self) -> None:
        from meta_agent.models import CaseResult

        node = self._node(0, None, {})
        for _ in range(3):
            node.record(CaseResult(case_id="a", passed=True, score=0.6))
        self.assertEqual(node.evaluated_case_ids["a"], 3)
        self.assertEqual(node.n_evals, 3)
        self.assertAlmostEqual(node.mean_utility, 0.6)


class SingleCaseEvolveTests(unittest.TestCase):
    """End-to-end: a 1-train-case run stalls at the default (eval_repeats=1)
    and doesn't at eval_repeats>1 -- the actual bug this knob fixes."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="hgm_eval_repeats_"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.seed = self.tmp / "seed"
        self.seed.mkdir()
        (self.seed / "workflow.py").write_text(
            "def run_task(task):\n    return None\n", encoding="utf-8"
        )
        self.experiment = self.tmp / "exp"
        self.experiment.mkdir()

    def _run(self, *, eval_repeats: int):
        from meta_agent.feedback_gatherer import DefaultFeedbackGatherer
        from meta_agent.managers.hgm import HGMManager

        manager = HGMManager(
            eval_budget=40,
            init_expansions=2,
            eval_batch_size=4,
            alpha=0.6,
            seed=7,
            eval_repeats=eval_repeats,
        )
        outcome = manager.evolve(
            editor=_StubEditor(),
            evaluator=_StubEvaluator(),
            gatherer=DefaultFeedbackGatherer(),
            seed_dir=self.seed,
            benchmark_dir=self.tmp / "bench",
            experiment_dir=self.experiment,
            max_rounds=30,
            score_target=None,
            train_case_ids=["only_case"],  # exactly one train case
            eval_case_ids=None,
        )
        return manager, outcome

    def test_default_repeats_caps_every_node_at_one_eval(self) -> None:
        manager, _ = self._run(eval_repeats=1)
        # The actual bug this knob fixes: with one train case and no
        # repeats, the search still spends its budget (by minting a fresh
        # never-before-evaluated node every round instead), but NO node
        # ever gets more than a single, permanently-noisy observation --
        # the bandit can never gather more confidence on any one node.
        max_n_evals = max(
            (n.n_evals for n in manager._tree.nodes.values()), default=0
        )
        self.assertEqual(max_n_evals, 1)

    def test_repeats_enabled_actually_spends_the_budget(self) -> None:
        manager, _ = self._run(eval_repeats=5)
        self.assertEqual(manager._budget_spent, 40)
        # At least one node was evaluated on the same case more than once.
        max_repeat = max(
            (
                count
                for node in manager._tree.nodes.values()
                for count in node.evaluated_case_ids.values()
            ),
            default=0,
        )
        self.assertGreater(max_repeat, 1)


if __name__ == "__main__":
    unittest.main()
