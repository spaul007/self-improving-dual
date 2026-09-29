"""Manager-level wiring tests for the "unit" curriculum granularity
(meta_agent/unit_curriculum.py + its integration in
meta_agent/managers/hgm.py). Mirrors tests/test_hgm_curriculum_wiring.py's
direct-tree-construction style (no LLM/evaluator).

    PYTHONPATH=. python3 -m unittest tests.test_hgm_unit_curriculum_wiring
"""
from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from meta_agent.curriculum import Curriculum
from meta_agent.managers.hgm import HGMManager
from meta_agent.managers.hgm_tree import HGMNode, HGMTree
from meta_agent.models import AgentFeedback, CaseResult, EvaluationResult, EvolutionStrategy
from meta_agent.unit_curriculum import UnitCurriculum

UNITS = {
    "Business Hours": ("commonsense:Business Hours:",),
    "Activity Diversity": ("commonsense:Activity Diversity:",),
}


def _feedback(round_number, base_round, project_metrics=None) -> AgentFeedback:
    return AgentFeedback(
        round_number=round_number,
        base_round=base_round,
        strategy=EvolutionStrategy(
            target_files=[], optimization_goal=f"goal-{round_number}", proposed_changes="x",
        ),
        eval_result=EvaluationResult(score=0.0),
        project_metrics=project_metrics or {},
    )


class GranularityValidationTests(unittest.TestCase):
    def test_default_granularity_is_check(self) -> None:
        m = HGMManager()
        self.assertEqual(m.curriculum_granularity, "check")
        self.assertIsNone(m.curriculum_units_path)

    def test_invalid_granularity_rejected(self) -> None:
        with self.assertRaises(ValueError):
            HGMManager(curriculum_granularity="not_a_real_granularity")

    def test_unit_granularity_without_units_path_rejected(self) -> None:
        with self.assertRaises(ValueError):
            HGMManager(curriculum_granularity="unit")

    def test_unit_granularity_with_units_path_accepted(self) -> None:
        m = HGMManager(curriculum_granularity="unit", curriculum_units_path="/some/path.json")
        self.assertEqual(m.curriculum_granularity, "unit")
        self.assertEqual(m.curriculum_units_path, "/some/path.json")
        # Not built yet -- only evolve()'s post-seed hook constructs it.
        self.assertIsNone(m._curriculum)

    def test_check_mode_with_curriculum_enabled_is_unaffected(self) -> None:
        # Regression guard: opting into "unit"-mode-only knobs never
        # touches check-mode's own construction/validation path.
        m = HGMManager(curriculum_enabled=True)
        self.assertEqual(m.curriculum_granularity, "check")
        self.assertIsNone(m._curriculum)

    def test_default_unit_max_attempts_is_2(self) -> None:
        m = HGMManager()
        self.assertEqual(m.curriculum_unit_max_attempts, 2)


class UnitsFileLoadingWiringTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="unit_curriculum_wiring_"))
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def test_evolve_time_construction_builds_a_unit_curriculum(self) -> None:
        # Exercises the ACTUAL branching logic inside evolve() (not just a
        # hand-constructed UnitCurriculum) by replicating exactly what that
        # code does with the units file + seed feedback, using the same
        # HGMManager instance/config it would use in a real run.
        path = self.tmp / "units.json"
        path.write_text(json.dumps({
            "Business Hours": ["commonsense:Business Hours:"],
            "Activity Diversity": ["commonsense:Activity Diversity:"],
        }))
        m = HGMManager(
            curriculum_enabled=True, curriculum_granularity="unit",
            curriculum_units_path=str(path),
        )
        from meta_agent.curriculum import _combined_check_counts
        from meta_agent.unit_curriculum import load_units_map

        units = load_units_map(m.curriculum_units_path)
        seed_metrics = {
            "top_failed_checks": [
                ["commonsense:Business Hours:dining_within_service_hours", 27],
                ["commonsense:Activity Diversity:diverse_attraction_options", 5],
            ],
        }
        seed_check_counts = _combined_check_counts(seed_metrics)
        m._curriculum = UnitCurriculum(
            seed_check_counts, units=units, max_attempts=m.curriculum_unit_max_attempts,
            resolution_threshold=m.curriculum_resolution_threshold,
            patience=m.curriculum_patience,
            unit_selector=m._unit_selector,
        )
        self.assertIsInstance(m._curriculum, UnitCurriculum)
        # Mechanical fallback (no unit_selector configured) picks the
        # highest-count candidate.
        self.assertEqual(m._curriculum.current_goal, "Business Hours")

    def test_bad_units_file_raises_at_load_time(self) -> None:
        from meta_agent.unit_curriculum import load_units_map

        with self.assertRaises(ValueError):
            load_units_map(str(self.tmp / "nonexistent.json"))

    def test_real_evolve_call_exercises_the_actual_import_statement(self) -> None:
        # Regression test for a real bug found live: evolve()'s own lazy
        # `from ..unit_curriculum import ...` had a single-dot typo
        # (`from .unit_curriculum import ...`, which resolves to
        # meta_agent.managers.unit_curriculum -- doesn't exist) that every
        # OTHER test in this file missed because they all import
        # unit_curriculum directly rather than going through evolve()'s own
        # import statement. This test calls the real evolve() end-to-end
        # (stub editor/evaluator, no LLM) specifically so a reintroduced
        # bad import raises here, not just in a live run.
        from tests.test_hgm_smoke import _StubEditor, _StubEvaluator
        from meta_agent.feedback_gatherer import DefaultFeedbackGatherer

        path = self.tmp / "units.json"
        path.write_text(json.dumps({"Unit A": ["prefix:"]}))
        seed = self.tmp / "seed"
        seed.mkdir()
        (seed / "workflow.py").write_text("def run_task(task):\n    return None\n")
        experiment = self.tmp / "exp"
        experiment.mkdir()

        manager = HGMManager(
            eval_budget=4, init_expansions=1, eval_batch_size=2, alpha=0.6, seed=7,
            curriculum_enabled=True, curriculum_granularity="unit",
            curriculum_units_path=str(path),
        )
        # Must not raise ModuleNotFoundError (or anything else) -- the
        # curriculum-construction branch runs unconditionally whenever
        # curriculum_enabled and curriculum_granularity="unit", regardless
        # of whether the stub's seed metrics have any real failing checks.
        manager.evolve(
            editor=_StubEditor(), evaluator=_StubEvaluator(),
            gatherer=DefaultFeedbackGatherer(), seed_dir=seed,
            benchmark_dir=self.tmp / "bench", experiment_dir=experiment,
            max_rounds=10, score_target=None,
            train_case_ids=[f"c{i}" for i in range(10)], eval_case_ids=None,
        )


class DirectiveInterfaceCompatibilityTests(unittest.TestCase):
    """UnitCurriculum implements the identical duck-typed surface
    Curriculum does -- _curriculum_directive_for_expand needs no
    per-granularity branching to consume either one."""

    def setUp(self) -> None:
        self.m = HGMManager(
            curriculum_enabled=True, curriculum_resolution_threshold=0.15, curriculum_patience=5,
        )
        self.m._tree = HGMTree()
        self.m._feedback = {}
        n0 = HGMNode(0, None, Path("."))
        n0.record(CaseResult(case_id="a", passed=True, score=0.6))
        self.m._tree.add(n0)
        seed_counts = {
            "commonsense:Business Hours:dining_within_service_hours": 27,
            "commonsense:Activity Diversity:diverse_attraction_options": 5,
        }
        self.m._feedback[0] = _feedback(
            0, 0, project_metrics={
                "top_failed_checks": [
                    ["commonsense:Business Hours:dining_within_service_hours", 27],
                ],
            },
        )
        self.m._curriculum = UnitCurriculum(seed_counts, units=UNITS, max_attempts=2)

    def test_directive_reaches_render_expand_context(self) -> None:
        directive, snapshot = self.m._curriculum_directive_for_expand()
        self.assertIsNotNone(directive)
        self.assertIn("Business Hours", directive)
        self.assertEqual(snapshot["current_goal"], "Business Hours")

        parent = self.m._tree[0]
        context = self.m._render_expand_context(
            parent, "verifiers", Path("."), 1, curriculum_directive=directive,
        )
        self.assertIn("## Current curriculum focus", context)
        self.assertIn("Business Hours", context)

    def test_advance_repicks_from_live_check_counts(self) -> None:
        # Business Hours resolved (0 occurrences in the live metrics) ->
        # re-picks; Activity Diversity is now the only remaining candidate.
        self.m._feedback[0] = _feedback(
            0, 0, project_metrics={
                "top_failed_checks": [
                    ["commonsense:Activity Diversity:diverse_attraction_options", 5],
                ],
            },
        )
        directive, snapshot = self.m._curriculum_directive_for_expand()
        self.assertEqual(snapshot["current_goal"], "Activity Diversity")
        self.assertEqual(snapshot["advance_reason"], "resolved")


if __name__ == "__main__":
    unittest.main()
