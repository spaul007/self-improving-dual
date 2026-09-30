"""How ``HGMManager._expand`` hands the selection to an assignment-steered
editor (the agentic editor) versus the default prose context.

    PYTHONPATH=. python3 -m unittest tests.test_hgm_assignment_wiring
"""
from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from tests.test_hgm_paired_eval import _Editor


class _AssignmentEditor(_Editor):
    steering = "assignment"

    def __init__(self) -> None:
        super().__init__()
        self.received: list[dict] = []

    def apply(self, feedback, base_dir, out_dir, *, context=None, has_suggestion=False,
              assignment=None):
        self.received.append({"context": context, "has_suggestion": has_suggestion,
                              "assignment": assignment})
        return super().apply(feedback, base_dir, out_dir, context=context,
                             has_suggestion=has_suggestion)


class _Suggester:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def suggest(self, **kw):
        self.calls.append(kw)
        return f"proposal for {kw['block']}"


class AssignmentWiringTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="hgm_assign_"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.seed = self.tmp / "seed"
        self.seed.mkdir()
        (self.seed / "workflow.py").write_text("def run_task(task):\n    return None\n")
        self.exp = self.tmp / "exp"
        self.exp.mkdir()

    def run_hgm(self, editor, suggester=None, **kw):
        from meta_agent.feedback_gatherer import DefaultFeedbackGatherer
        from meta_agent.managers.hgm import HGMManager
        from tests.test_hgm_smoke import _StubEvaluator

        opts = dict(eval_budget=24, init_expansions=2, eval_batch_size=4, alpha=0.6, seed=5,
                    finalize_top_k=0, block_selection_strategy="adaptive")
        opts.update(kw)
        manager = HGMManager(**opts)
        manager.evolve(editor=editor, evaluator=_StubEvaluator(), gatherer=DefaultFeedbackGatherer(),
                       seed_dir=self.seed, benchmark_dir=self.tmp / "b", experiment_dir=self.exp,
                       max_rounds=30, score_target=None,
                       train_case_ids=[f"c{i}" for i in range(12)], eval_case_ids=None,
                       block_suggester=suggester)
        return manager

    def test_assignment_replaces_the_prose_context(self) -> None:
        from meta_agent.block_suggester import block_scope

        editor = _AssignmentEditor()
        manager = self.run_hgm(editor)
        self.assertTrue(editor.received)
        for i, got in enumerate(editor.received, start=1):
            a = got["assignment"]
            self.assertIsNone(got["context"])
            self.assertFalse(got["has_suggestion"])
            fb = manager._feedback[i]
            self.assertEqual(a.block, fb.strategy.block)
            self.assertEqual(a.block_scope, block_scope(a.block))
            self.assertIsNone(a.suggestion)
            self.assertIsNone(a.curriculum_directive)
            self.assertIsNone(a.implementation_strategy_body)
            saved = json.loads((self.exp / f"round_{i:03d}" / "assignment.json").read_text())
            self.assertEqual(saved["block"], a.block)

    def test_suggester_runs_only_when_configured_and_is_advisory(self) -> None:
        editor, suggester = _AssignmentEditor(), _Suggester()
        self.run_hgm(editor, suggester)
        self.assertEqual(len(suggester.calls), len(editor.received))
        for got, call in zip(editor.received, suggester.calls):
            self.assertEqual(got["assignment"].suggestion, f"proposal for {call['block']}")
            self.assertTrue(got["has_suggestion"])
            self.assertEqual(sorted(call), ["agent_dir", "block", "curriculum_directive", "failure_summary",
                                            "feedback", "node_id", "out_dir", "siblings"])

    def test_implementation_strategy_body_when_that_axis_is_on(self) -> None:
        from meta_agent.implementation_strategy import _IMPLEMENTATION_STRATEGY_BODIES

        editor = _AssignmentEditor()
        self.run_hgm(editor, implementation_strategy_selection_strategy="harness_heavy")
        for got in editor.received:
            a = got["assignment"]
            self.assertEqual(a.implementation_strategy, "harness_heavy")
            self.assertEqual(a.implementation_strategy_body,
                             _IMPLEMENTATION_STRATEGY_BODIES["harness_heavy"])

    def test_strict_editors_never_get_the_new_kwargs(self) -> None:
        # _Editor.apply has exactly the pre-port signature: any extra kwarg
        # (assignment / memory_path) would raise TypeError here.
        manager = self.run_hgm(_Editor(), _Suggester())
        self.assertGreater(len(manager._tree.nodes), 2)
        self.assertFalse(any((self.exp / f"round_{i:03d}" / "assignment.json").exists()
                             for i in manager._tree.nodes))


if __name__ == "__main__":
    unittest.main()
