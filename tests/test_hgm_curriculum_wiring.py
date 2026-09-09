"""Manager-level wiring tests for the curriculum layer
(meta_agent/curriculum.py + its integration in meta_agent/managers/hgm.py).

Hand-built tree/feedback, no LLM/evaluator -- mirrors
tests/test_block_bandit.py's HGMManagerBlockRewardMetricWiringTests and
tests/test_hgm_smoke.py's direct-tree-construction style.

    PYTHONPATH=. python3 -m unittest tests.test_hgm_curriculum_wiring
"""
from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from meta_agent.managers.hgm import HGMManager
from meta_agent.managers.hgm_tree import HGMNode, HGMTree
from meta_agent.models import AgentFeedback, EvaluationResult, EvolutionStrategy


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


class DefaultDisabledTests(unittest.TestCase):
    def test_default_manager_has_no_curriculum(self) -> None:
        m = HGMManager()
        self.assertFalse(m.curriculum_enabled)
        self.assertIsNone(m._curriculum)

    def test_helper_is_a_pure_noop_when_disabled(self) -> None:
        m = HGMManager()
        m._tree = HGMTree()
        m._feedback = {}
        n0 = HGMNode(0, None, Path("."))
        m._tree.add(n0)
        directive, snapshot = m._curriculum_directive_for_expand()
        self.assertIsNone(directive)
        self.assertIsNone(snapshot)


class SeedGoalsOverrideTests(unittest.TestCase):
    """curriculum_goals_override: an explicit ordered check list used
    INSTEAD of the seed's own top_failed_checks ranking (for sanity-testing
    the curriculum against one hand-picked check)."""

    def _seed_fb(self) -> AgentFeedback:
        return _feedback(
            0, 0,
            project_metrics={
                "top_failed_checks": [
                    ["check_a", 40], ["check_b", 20], ["check_c", 5],
                ]
            },
        )

    def test_default_none_reproduces_seed_ranking_verbatim(self) -> None:
        # The non-override path now runs infer_curriculum's merge/rank
        # (task checks + harness checks -- empty here) -- same order
        # since it was already sorted by count descending, but as tuples
        # (infer_curriculum's own output shape), not the raw
        # project_metrics lists passed straight through.
        m = HGMManager()
        self.assertEqual(
            m._curriculum_seed_goals(self._seed_fb()),
            [("check_a", 40), ("check_b", 20), ("check_c", 5)],
        )

    def test_override_reorders_to_named_checks_with_looked_up_counts(self) -> None:
        m = HGMManager(curriculum_goals_override=["check_c", "check_a"])
        self.assertEqual(
            m._curriculum_seed_goals(self._seed_fb()),
            [("check_c", 5), ("check_a", 40)],
        )

    def test_override_naming_an_absent_check_defaults_to_zero_count(self) -> None:
        m = HGMManager(curriculum_goals_override=["check_a", "never_seen_check"])
        self.assertEqual(
            m._curriculum_seed_goals(self._seed_fb()),
            [("check_a", 40), ("never_seen_check", 0)],
        )

    def test_override_with_no_seed_feedback_still_yields_zero_counts(self) -> None:
        m = HGMManager(curriculum_goals_override=["some_check"])
        self.assertEqual(
            m._curriculum_seed_goals(None), [("some_check", 0)],
        )

    def test_empty_override_list_is_falsy_and_falls_back_to_seed_ranking(self) -> None:
        m = HGMManager(curriculum_goals_override=[])
        self.assertEqual(
            m._curriculum_seed_goals(self._seed_fb()),
            [("check_a", 40), ("check_b", 20), ("check_c", 5)],
        )

    def test_curriculum_max_goals_caps_the_default_path(self) -> None:
        m = HGMManager(curriculum_max_goals=2)
        self.assertEqual(
            m._curriculum_seed_goals(self._seed_fb()),
            [("check_a", 40), ("check_b", 20)],
        )

    def test_merges_harness_checks_with_top_failed_checks(self) -> None:
        m = HGMManager()
        fb = _feedback(
            0, 0,
            project_metrics={
                "top_failed_checks": [["check_a", 40], ["check_b", 20]],
                "harness_checks": [["sightseeing_failed", 30]],
            },
        )
        self.assertEqual(
            m._curriculum_seed_goals(fb),
            [("check_a", 40), ("sightseeing_failed", 30), ("check_b", 20)],
        )

    def test_high_no_plan_rate_prioritizes_harness_goal(self) -> None:
        m = HGMManager()
        fb = _feedback(
            0, 0,
            project_metrics={
                "top_failed_checks": [["check_a", 40]],
                "harness_checks": [["sightseeing_failed", 5]],
                "no_plan_rate": 0.35,
            },
        )
        self.assertEqual(
            m._curriculum_seed_goals(fb),
            [("sightseeing_failed", 5), ("check_a", 40)],
        )

    def test_low_no_plan_rate_keeps_flat_merge(self) -> None:
        m = HGMManager()
        fb = _feedback(
            0, 0,
            project_metrics={
                "top_failed_checks": [["check_a", 40]],
                "harness_checks": [["sightseeing_failed", 5]],
                "no_plan_rate": 0.05,
            },
        )
        self.assertEqual(
            m._curriculum_seed_goals(fb),
            [("check_a", 40), ("sightseeing_failed", 5)],
        )

    def test_custom_curriculum_harness_priority_threshold(self) -> None:
        m = HGMManager(curriculum_harness_priority_threshold=0.5)
        fb = _feedback(
            0, 0,
            project_metrics={
                "top_failed_checks": [["check_a", 40]],
                "harness_checks": [["sightseeing_failed", 5]],
                "no_plan_rate": 0.20,
            },
        )
        # 0.20 < the custom 0.5 threshold -> flat merge still applies.
        self.assertEqual(
            m._curriculum_seed_goals(fb),
            [("check_a", 40), ("sightseeing_failed", 5)],
        )


class CheckDescriptionsLoadingTests(unittest.TestCase):
    """_load_curriculum_check_descriptions -- the JSON registry feeding
    Curriculum's own check_descriptions param (see
    projects/travel_mas_refactored/adapter/error_semantics.json for a
    real example)."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="curriculum_descs_"))
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def test_none_when_path_unset(self) -> None:
        m = HGMManager()
        self.assertIsNone(m._load_curriculum_check_descriptions())

    def test_loads_a_real_json_file(self) -> None:
        path = self.tmp / "descs.json"
        path.write_text(
            json.dumps({"check_a": "does X", "check_b": "does Y"}),
            encoding="utf-8",
        )
        m = HGMManager(curriculum_check_descriptions_path=str(path))
        self.assertEqual(
            m._load_curriculum_check_descriptions(),
            {"check_a": "does X", "check_b": "does Y"},
        )

    def test_none_when_file_missing(self) -> None:
        m = HGMManager(
            curriculum_check_descriptions_path=str(self.tmp / "nope.json"),
        )
        self.assertIsNone(m._load_curriculum_check_descriptions())

    def test_none_when_malformed_json(self) -> None:
        path = self.tmp / "bad.json"
        path.write_text("{not valid json", encoding="utf-8")
        m = HGMManager(curriculum_check_descriptions_path=str(path))
        self.assertIsNone(m._load_curriculum_check_descriptions())

    def test_none_when_json_is_not_an_object(self) -> None:
        path = self.tmp / "list.json"
        path.write_text(json.dumps(["a", "b"]), encoding="utf-8")
        m = HGMManager(curriculum_check_descriptions_path=str(path))
        self.assertIsNone(m._load_curriculum_check_descriptions())

    def test_non_string_values_are_dropped(self) -> None:
        path = self.tmp / "mixed.json"
        path.write_text(
            json.dumps({"check_a": "a real description", "check_b": 123}),
            encoding="utf-8",
        )
        m = HGMManager(curriculum_check_descriptions_path=str(path))
        self.assertEqual(
            m._load_curriculum_check_descriptions(),
            {"check_a": "a real description"},
        )

    def test_end_to_end_reaches_the_directive(self) -> None:
        path = self.tmp / "descs.json"
        path.write_text(
            json.dumps({"check_a": "a specific, verified description"}),
            encoding="utf-8",
        )
        m = HGMManager(
            curriculum_enabled=True,
            curriculum_check_descriptions_path=str(path),
        )
        from meta_agent.models import CaseResult

        m._tree = HGMTree()
        m._feedback = {}
        n0 = HGMNode(0, None, Path("."))
        n0.record(CaseResult(case_id="a", passed=True, score=0.6))
        m._tree.add(n0)
        m._feedback[0] = _feedback(
            0, 0, project_metrics={"top_failed_checks": [["check_a", 5]]},
        )
        from meta_agent.curriculum import Curriculum

        m._curriculum = Curriculum(
            [("check_a", 5)],
            check_descriptions=m._load_curriculum_check_descriptions(),
        )
        directive, _ = m._curriculum_directive_for_expand()
        self.assertIn("a specific, verified description", directive)

    def test_merges_task_and_harness_error_files(self) -> None:
        task_path = self.tmp / "task_descs.json"
        task_path.write_text(
            json.dumps({"check_a": "a task-based description"}),
            encoding="utf-8",
        )
        harness_path = self.tmp / "harness_descs.json"
        harness_path.write_text(
            json.dumps({"sightseeing_failed": "a harness-based description"}),
            encoding="utf-8",
        )
        m = HGMManager(
            curriculum_check_descriptions_path=str(task_path),
            curriculum_harness_error_descriptions_path=str(harness_path),
        )
        merged = m._load_curriculum_check_descriptions()
        self.assertEqual(
            merged,
            {
                "check_a": "a task-based description",
                "sightseeing_failed": "a harness-based description",
            },
        )

    def test_harness_path_alone_works_without_a_task_file(self) -> None:
        harness_path = self.tmp / "harness_descs.json"
        harness_path.write_text(
            json.dumps({"sightseeing_failed": "a harness-based description"}),
            encoding="utf-8",
        )
        m = HGMManager(curriculum_harness_error_descriptions_path=str(harness_path))
        self.assertEqual(
            m._load_curriculum_check_descriptions(),
            {"sightseeing_failed": "a harness-based description"},
        )

    def test_both_unset_returns_none_not_empty_dict(self) -> None:
        m = HGMManager()
        self.assertIsNone(m._load_curriculum_check_descriptions())

    def test_real_error_semantics_files_load_and_merge(self) -> None:
        # The actual shipped files, loaded together, for real.
        repo_root = Path(__file__).resolve().parents[1]
        adapter_dir = (
            repo_root / "projects" / "travel_mas_refactored" / "adapter"
        )
        m = HGMManager(
            curriculum_check_descriptions_path=str(
                adapter_dir / "error_semantics.json"
            ),
            curriculum_harness_error_descriptions_path=str(
                adapter_dir / "harness_error_semantics.json"
            ),
        )
        merged = m._load_curriculum_check_descriptions()
        self.assertIsNotNone(merged)
        self.assertIn(
            "commonsense:Sandbox Compliance:validated_meals", merged,
        )
        self.assertIn("sightseeing_output_failure", merged)
        self.assertIn("sightseeing_budget_exhausted", merged)


class ValidationTests(unittest.TestCase):
    def test_invalid_resolution_threshold_rejected(self) -> None:
        with self.assertRaises(ValueError):
            HGMManager(curriculum_resolution_threshold=1.5)
        with self.assertRaises(ValueError):
            HGMManager(curriculum_resolution_threshold=-0.1)

    def test_invalid_patience_rejected(self) -> None:
        with self.assertRaises(ValueError):
            HGMManager(curriculum_patience=0)

    def test_invalid_max_goals_rejected(self) -> None:
        with self.assertRaises(ValueError):
            HGMManager(curriculum_max_goals=0)

    def test_invalid_harness_priority_threshold_rejected(self) -> None:
        with self.assertRaises(ValueError):
            HGMManager(curriculum_harness_priority_threshold=1.5)
        with self.assertRaises(ValueError):
            HGMManager(curriculum_harness_priority_threshold=-0.1)

    def test_default_harness_priority_threshold_is_015(self) -> None:
        m = HGMManager()
        self.assertEqual(m.curriculum_harness_priority_threshold, 0.15)

    def test_valid_config_accepted(self) -> None:
        m = HGMManager(
            curriculum_enabled=True,
            curriculum_resolution_threshold=0.2,
            curriculum_patience=3,
        )
        self.assertTrue(m.curriculum_enabled)
        self.assertEqual(m.curriculum_resolution_threshold, 0.2)
        self.assertEqual(m.curriculum_patience, 3)
        # Not built yet -- only evolve()'s post-seed hook constructs it.
        self.assertIsNone(m._curriculum)


class DirectiveWiringTests(unittest.TestCase):
    """Hand-build a curriculum + tree directly (bypassing evolve()/seed) to
    confirm the helper and _render_expand_context correctly surface it."""

    def setUp(self) -> None:
        from meta_agent.curriculum import Curriculum

        self.m = HGMManager(
            curriculum_enabled=True, curriculum_resolution_threshold=0.15, curriculum_patience=5,
        )
        self.m._tree = HGMTree()
        self.m._feedback = {}
        n0 = HGMNode(0, None, Path("."))
        from meta_agent.models import CaseResult

        n0.record(CaseResult(case_id="a", passed=True, score=0.6))
        self.m._tree.add(n0)
        # check_x is absent from the best node's live top_failed_checks
        # (i.e. it dropped out entirely -- 0 occurrences), so
        # failure_rate_for("check_x", ..., n_evals=1) = 0/1 = 0.0.
        self.m._feedback[0] = _feedback(
            0, 0, project_metrics={"top_failed_checks": [["check_y", 3]]}
        )
        self.m._curriculum = Curriculum(
            [("check_x", 8), ("check_y", 3)], resolution_threshold=0.15, patience=5,
        )

    def test_directive_reaches_render_expand_context(self) -> None:
        # Use a curriculum where check_x is NOT yet resolved (still present
        # in the best node's own metrics), so this test observes the
        # un-advanced directive in isolation from the advance behavior
        # covered by test_snapshot_advances_after_crossing_threshold.
        self.m._feedback[0] = _feedback(
            0, 0, project_metrics={"top_failed_checks": [["check_x", 8]]}
        )
        directive, snapshot = self.m._curriculum_directive_for_expand()
        self.assertIsNotNone(directive)
        self.assertIn("check_x", directive)
        self.assertEqual(snapshot["current_goal"], "check_x")
        self.assertEqual(snapshot["current_index"], 0)

        parent = self.m._tree[0]
        context = self.m._render_expand_context(
            parent, "verifiers", Path("."), 1, curriculum_directive=directive,
        )
        self.assertIn("## Current curriculum focus", context)
        self.assertIn("check_x", context)

    def test_snapshot_advances_after_crossing_threshold(self) -> None:
        # best node's failure rate for check_x: 0/1 = 0.0 <= 0.15 -> resolved.
        directive, snapshot = self.m._curriculum_directive_for_expand()
        self.assertEqual(snapshot["current_goal"], "check_y")
        self.assertEqual(snapshot["current_index"], 1)
        self.assertEqual(snapshot["advance_reason"], "resolved")
        self.assertEqual(snapshot["resolved_goals"], ["check_x"])
        self.assertIn("check_y", directive)


class RenderExpandContextByteIdenticalTests(unittest.TestCase):
    """Confirms curriculum_directive=None (the default) reproduces
    identical _render_expand_context output whether passed explicitly or
    omitted -- the automated proof of the plan's zero-regression trace."""

    def setUp(self) -> None:
        self.m = HGMManager()
        self.m._tree = HGMTree()
        self.m._feedback = {}
        n0 = HGMNode(0, None, Path("."))
        from meta_agent.models import CaseResult

        n0.record(CaseResult(case_id="a", passed=True, score=0.6))
        self.m._tree.add(n0)
        self.m._feedback[0] = _feedback(0, 0)

    def test_omitted_vs_explicit_none_are_byte_identical(self) -> None:
        parent = self.m._tree[0]
        omitted = self.m._render_expand_context(parent, "verifiers", Path("."), 1)
        explicit_none = self.m._render_expand_context(
            parent, "verifiers", Path("."), 1, curriculum_directive=None,
        )
        self.assertEqual(omitted, explicit_none)
        self.assertNotIn("Current curriculum focus", omitted)


if __name__ == "__main__":
    unittest.main()
