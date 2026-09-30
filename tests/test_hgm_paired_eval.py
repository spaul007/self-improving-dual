"""Opt-in paired evaluation (``expand_eval_size``) in ``HGMManager``.

    PYTHONPATH=. python3 -m unittest tests.test_hgm_paired_eval
"""
from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path


class _Editor:
    """Strict stub (the manager's default-path signature): copies the
    parent's agent; fails the calls listed in ``fail``."""

    def __init__(self, fail: set[int] = frozenset()) -> None:
        self.fail = set(fail)
        self.calls = 0
        self.parents: list[str] = []

    def apply(self, feedback, base_dir, out_dir, *, context=None, has_suggestion=False):
        from meta_agent.models import EditResult, EvolutionStrategy

        self.calls += 1
        self.parents.append(Path(base_dir).name)
        dst = Path(out_dir) / "task_agent"
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(Path(base_dir) / "task_agent", dst)
        strategy = EvolutionStrategy(target_files=["workflow.py"],
                                     optimization_goal=f"edit {self.calls}", proposed_changes="p")
        if self.calls in self.fail:
            return EditResult(success=False, errors=[f"forced failure {self.calls}"], strategy=strategy)
        return EditResult(success=True, edited_files=["workflow.py"], strategy=strategy)


class _Base(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="hgm_paired_"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.seed = self.tmp / "seed"
        self.seed.mkdir()
        (self.seed / "workflow.py").write_text("def run_task(task):\n    return None\n")
        self.exp = self.tmp / "exp"
        self.exp.mkdir()

    def run_hgm(self, editor, **kw):
        from meta_agent.feedback_gatherer import DefaultFeedbackGatherer
        from meta_agent.managers.hgm import HGMManager
        from tests.test_hgm_smoke import _StubEvaluator

        opts = dict(eval_budget=40, init_expansions=2, eval_batch_size=4, alpha=0.6, seed=7,
                    snapshot_tree=True, finalize_top_k=0)
        opts.update(kw)
        manager = HGMManager(**opts)
        calls: list[tuple[int, object]] = []
        real = manager._evaluate

        def spy(node_id, evaluator, gatherer, *, batch_size=None):
            calls.append((node_id, batch_size))
            return real(node_id, evaluator, gatherer, batch_size=batch_size)

        manager._evaluate = spy
        manager.evolve(editor=editor, evaluator=_StubEvaluator(), gatherer=DefaultFeedbackGatherer(),
                       seed_dir=self.seed, benchmark_dir=self.tmp / "b", experiment_dir=self.exp,
                       max_rounds=30, score_target=None,
                       train_case_ids=[f"c{i}" for i in range(20)], eval_case_ids=None)
        return manager, calls

    def events(self) -> list[str]:
        path = self.exp / "snapshots" / "tree_snapshots.jsonl"
        return [json.loads(line)["event"] for line in path.read_text().splitlines()]


class PairedEvalTests(_Base):
    def test_every_successful_child_is_evaluated_right_after_its_edit(self) -> None:
        editor = _Editor(fail={3})
        manager, calls = self.run_hgm(editor, expand_eval_size=3)
        paired = [nid for nid, bs in calls if bs == 3]
        children = [nid for nid, n in manager._tree.nodes.items()
                    if n.parent_id is not None and not n.edit_failed]
        self.assertEqual(sorted(paired), sorted(children))
        ev = self.events()
        for i, e in enumerate(ev):
            if e == "expand_eval":
                self.assertEqual(ev[i - 1], "expand")
        failed = [nid for nid, n in manager._tree.nodes.items() if n.edit_failed]
        self.assertEqual(len(failed), 1)
        self.assertNotIn(failed[0], paired)
        self.assertLessEqual(manager._budget_spent, 40)
        self.assertEqual(manager._budget_spent, manager._node_evals_spent)

    def test_init_expansions_branch_from_the_root(self) -> None:
        editor = _Editor()
        self.run_hgm(editor, expand_eval_size=3)
        self.assertEqual(editor.parents[:2], ["round_000", "round_000"])

    def test_init_expansions_stay_on_the_root_when_children_are_better(self) -> None:
        """Paired evaluation makes each init child expandable before the next
        init expansion; with children scoring far above the seed, a Thompson
        draw would favor them. All init expansions must still use the root."""
        from meta_agent.models import CaseResult, EvaluationResult

        class _ChildrenBetter:
            def run(self, round_dir, benchmark_dir, *, case_ids=None):
                s = 0.1 if Path(round_dir).name == "round_000" else 0.95
                per_case = [CaseResult(case_id=c, passed=s >= 0.5, score=s) for c in case_ids or []]
                return EvaluationResult(score=s, passed=sum(c.passed for c in per_case),
                                        failed=sum(not c.passed for c in per_case), per_case=per_case)

        from meta_agent.feedback_gatherer import DefaultFeedbackGatherer
        from meta_agent.managers.hgm import HGMManager

        manager = HGMManager(eval_budget=60, init_expansions=3, eval_batch_size=4, alpha=0.6, seed=7,
                             finalize_top_k=0, expand_eval_size=3)
        steps: list[tuple[int, list[int]]] = []
        real_step = manager._expand_step

        def spy(parent_id, *args):
            steps.append((parent_id, manager._expandable()))
            return real_step(parent_id, *args)

        manager._expand_step = spy
        manager.evolve(editor=_Editor(), evaluator=_ChildrenBetter(), gatherer=DefaultFeedbackGatherer(),
                       seed_dir=self.seed, benchmark_dir=self.tmp / "b", experiment_dir=self.exp,
                       max_rounds=30, score_target=None,
                       train_case_ids=[f"c{i}" for i in range(20)], eval_case_ids=None)
        init = steps[:3]
        self.assertEqual([p for p, _ in init], [0, 0, 0])
        self.assertEqual(init[1][1], [0, 1])          # child 1 was already expandable
        self.assertEqual(init[2][1], [0, 1, 2])       # children 1 and 2 were
        self.assertEqual(manager._tree[0].children[:3], [1, 2, 3])

    def test_budget_guard_reserves_one_paired_batch(self) -> None:
        from meta_agent.managers.hgm import HGMManager

        self.assertEqual(HGMManager(expand_eval_size=5)._min_budget_to_expand(), 5)
        self.assertEqual(HGMManager()._min_budget_to_expand(), 0)

    def test_default_has_no_paired_evaluation(self) -> None:
        _, calls = self.run_hgm(_Editor())
        self.assertTrue(all(bs is None for _, bs in calls))
        self.assertNotIn("expand_eval", self.events())

    def test_negative_size_rejected(self) -> None:
        from meta_agent.managers.hgm import HGMManager

        with self.assertRaises(ValueError):
            HGMManager(expand_eval_size=-1)


if __name__ == "__main__":
    unittest.main()
