"""Manager-level wiring tests for HGMManager._last_suggestion_produced and
its threading into editor.apply's has_suggestion= kwarg (both the base
HGMManager._expand and HGMDualManager's Stage A).

    PYTHONPATH=. python3 -m unittest tests.test_block_suggester_has_suggestion_wiring
"""
from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from meta_agent.managers.hgm import HGMManager
from meta_agent.managers.hgm_tree import HGMNode, HGMTree
from meta_agent.models import AgentFeedback, CaseResult, EvaluationResult, EvolutionStrategy


def _feedback(round_number, base_round, project_metrics=None) -> AgentFeedback:
    return AgentFeedback(
        round_number=round_number, base_round=base_round,
        strategy=EvolutionStrategy(
            target_files=[], optimization_goal=f"goal-{round_number}", proposed_changes="x",
        ),
        eval_result=EvaluationResult(score=0.0),
        project_metrics=project_metrics or {},
    )


class _StubSuggester:
    def __init__(self, *, result=None, raises=False):
        self.result = result
        self.raises = raises
        self.calls = 0

    def suggest(self, **kwargs):
        self.calls += 1
        if self.raises:
            raise RuntimeError("boom")
        return self.result


class RenderExpandContextAttributeTests(unittest.TestCase):
    """Direct, isolated checks of _render_expand_context's four cases,
    mirroring tests/test_hgm_curriculum_wiring.py's DirectiveWiringTests
    hand-built-tree style."""

    def _manager_with_tree(self) -> HGMManager:
        m = HGMManager()
        m._tree = HGMTree()
        m._feedback = {}
        n0 = HGMNode(0, None, Path("."))
        n0.record(CaseResult(case_id="a", passed=True, score=0.6))
        m._tree.add(n0)
        m._feedback[0] = _feedback(0, 0)
        return m

    def test_false_by_default_no_suggester_configured(self) -> None:
        m = self._manager_with_tree()
        self.assertIsNone(m._block_suggester)
        m._render_expand_context(m._tree[0], "verifiers", Path("."), 1)
        self.assertFalse(m._last_suggestion_produced)

    def test_true_when_suggester_returns_text(self) -> None:
        m = self._manager_with_tree()
        m._block_suggester = _StubSuggester(result="a real suggestion")
        m._render_expand_context(m._tree[0], "verifiers", Path("."), 1)
        self.assertTrue(m._last_suggestion_produced)

    def test_false_when_suggester_returns_none(self) -> None:
        m = self._manager_with_tree()
        m._block_suggester = _StubSuggester(result=None)
        m._render_expand_context(m._tree[0], "verifiers", Path("."), 1)
        self.assertFalse(m._last_suggestion_produced)

    def test_false_when_suggester_raises(self) -> None:
        m = self._manager_with_tree()
        m._block_suggester = _StubSuggester(raises=True)
        # _render_expand_context swallows the exception internally (see
        # hgm.py's own except Exception block) -- must not propagate.
        m._render_expand_context(m._tree[0], "verifiers", Path("."), 1)
        self.assertFalse(m._last_suggestion_produced)


class EvolveEndToEndHasSuggestionTests(unittest.TestCase):
    """End-to-end: a real evolve() round with a stub block_suggester
    configured must reach the (real) AgentEditor's has_suggestion kwarg."""

    def setUp(self) -> None:
        from tests.test_hgm_smoke import _StubEditor, _StubEvaluator

        self.tmp = Path(tempfile.mkdtemp(prefix="hgm_has_suggestion_"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.seed = self.tmp / "seed"
        self.seed.mkdir()
        (self.seed / "workflow.py").write_text(
            "def run_task(task):\n    return None\n", encoding="utf-8"
        )
        self.experiment = self.tmp / "exp"
        self.experiment.mkdir()
        self._StubEditor = _StubEditor
        self._StubEvaluator = _StubEvaluator

    def test_evolve_passes_has_suggestion_true_to_editor_when_suggester_fires(self) -> None:
        from meta_agent.feedback_gatherer import DefaultFeedbackGatherer

        manager = HGMManager(
            eval_budget=8, init_expansions=1, eval_batch_size=4, alpha=0.6, seed=7,
        )
        editor = self._StubEditor()
        manager.evolve(
            editor=editor,
            evaluator=self._StubEvaluator(),
            gatherer=DefaultFeedbackGatherer(),
            seed_dir=self.seed,
            benchmark_dir=self.tmp / "bench",
            experiment_dir=self.experiment,
            max_rounds=10,
            score_target=None,
            train_case_ids=[f"c{i}" for i in range(20)],
            eval_case_ids=None,
            block_suggester=_StubSuggester(result="a real suggestion"),
        )
        self.assertTrue(editor.received_has_suggestion)
        self.assertTrue(all(editor.received_has_suggestion))

    def test_evolve_passes_has_suggestion_false_when_no_suggester(self) -> None:
        from meta_agent.feedback_gatherer import DefaultFeedbackGatherer

        manager = HGMManager(
            eval_budget=8, init_expansions=1, eval_batch_size=4, alpha=0.6, seed=7,
        )
        editor = self._StubEditor()
        manager.evolve(
            editor=editor,
            evaluator=self._StubEvaluator(),
            gatherer=DefaultFeedbackGatherer(),
            seed_dir=self.seed,
            benchmark_dir=self.tmp / "bench",
            experiment_dir=self.experiment,
            max_rounds=10,
            score_target=None,
            train_case_ids=[f"c{i}" for i in range(20)],
            eval_case_ids=None,
        )
        self.assertTrue(editor.received_has_suggestion)
        self.assertFalse(any(editor.received_has_suggestion))


class HGMDualStageAHasSuggestionTests(unittest.TestCase):
    """Stage A (which reuses the inherited _render_expand_context) must
    thread has_suggestion through; Stage B (its own _render_variant_context,
    never touches BlockSuggester) must never pass it -- both checked in the
    same run, on the same recorded call list."""

    def setUp(self) -> None:
        from meta_agent.managers.hgm_dual import HGMDualManager
        from tests.test_hgm_dual_manager import _StubEditor, _StubEvaluator

        self.tmp = Path(tempfile.mkdtemp(prefix="hgm_dual_has_suggestion_"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.seed = self.tmp / "seed"
        self.seed.mkdir()
        (self.seed / "workflow.py").write_text(
            "def run_task(task):\n    return None\n", encoding="utf-8"
        )
        self.experiment = self.tmp / "exp"
        self.experiment.mkdir()
        self._HGMDualManager = HGMDualManager
        self._StubEditor = _StubEditor
        self._StubEvaluator = _StubEvaluator

    def test_stage_a_gets_true_stage_b_gets_false(self) -> None:
        from meta_agent.feedback_gatherer import DefaultFeedbackGatherer

        manager = self._HGMDualManager(
            eval_budget=80, init_expansions=1, eval_batch_size=2, alpha=0.6,
            seed=7, num_variants=2, intra_expand_eval_size=4,
            variant_parallelism=2, categorizer_max_representative_errors=3,
            min_remaining_budget_for_dual=0,
            error_categorizer="projects.travel.travel_error_categorizer:categorize_errors",
        )
        editor = self._StubEditor()
        manager.evolve(
            editor=editor,
            evaluator=self._StubEvaluator(),
            gatherer=DefaultFeedbackGatherer(),
            seed_dir=self.seed,
            benchmark_dir=self.tmp / "bench",
            experiment_dir=self.experiment,
            max_rounds=30,
            score_target=None,
            train_case_ids=[f"c{i}" for i in range(12)],
            eval_case_ids=None,
            block_suggester=_StubSuggester(result="a real suggestion"),
        )
        # One Stage A call (True) and >=1 Stage B variant calls (False,
        # since Stage B never computes/passes has_suggestion at all).
        self.assertIn(True, editor.received_has_suggestion)
        self.assertIn(False, editor.received_has_suggestion)


if __name__ == "__main__":
    unittest.main()
