"""Tests for meta_agent/curriculum.py's Curriculum/CurriculumSnapshot --
pure logic, no LLM, no HGMTree/AgentFeedback, mirrors
tests/test_block_bandit.py's offline style.

    PYTHONPATH=. python3 -m unittest tests.test_curriculum
"""
from __future__ import annotations

import dataclasses
import json
import unittest

from meta_agent.curriculum import (
    Curriculum,
    CurriculumSnapshot,
    _combined_check_counts,
    infer_curriculum,
)


class OrderingAndBasicsTests(unittest.TestCase):
    def test_goals_preserve_seed_ranking_order(self) -> None:
        c = Curriculum(
            [("commonsense:Time Feasibility:reasonable_transfer_time", 40),
             ("hard:budget_constraint", 5)]
        )
        self.assertEqual(
            c.current_goal, "commonsense:Time Feasibility:reasonable_transfer_time"
        )
        self.assertFalse(c.done)

    def test_empty_goals_is_immediately_done(self) -> None:
        c = Curriculum([])
        self.assertTrue(c.done)
        self.assertIsNone(c.current_goal)
        self.assertIsNone(c.directive())

    def test_duplicate_check_names_collapse_to_first_occurrence(self) -> None:
        c = Curriculum([("a", 10), ("b", 5), ("a", 3)])
        self.assertEqual(c._goals, ["a", "b"])
        self.assertEqual(c._seed_counts["a"], 10)


class DirectiveTests(unittest.TestCase):
    def test_directive_none_when_done(self) -> None:
        c = Curriculum([("a", 5)], resolution_threshold=0.5, patience=1)
        c.advance_if_ready(failure_rate=0.0)  # resolves immediately
        self.assertTrue(c.done)
        self.assertIsNone(c.directive())

    def test_directive_contains_escape_hatch_and_goal_name(self) -> None:
        c = Curriculum([("some_check_name", 5)])
        d = c.directive()
        self.assertIsNotNone(d)
        self.assertIn("ESCAPE HATCH", d)
        self.assertIn("some_check_name", d)

    def test_directive_explains_both_kinds_of_failure_mode(self) -> None:
        # No more NO_PLAN_GOAL sentinel -- the directive is now generic
        # over both a task-based error (a failed check/constraint) and a
        # harness-based error (a crash cause), described in
        # project-agnostic terms (no travel-specific example names --
        # this module must work identically for any project, e.g.
        # shopping) since a bare goal name alone is meaningless to the
        # diagnosing LLM without that context.
        c = Curriculum([("some_bare_name", 5)])
        d = c.directive()
        self.assertIn("some_bare_name", d)
        self.assertIn("harness-based error", d)
        self.assertIn("task-based error", d)
        self.assertIn("ESCAPE HATCH", d)

    def test_directive_states_the_concrete_seed_count(self) -> None:
        # "we need this error to reduce" needs a real number attached,
        # not just a bare identifier -- the seed count is already stored
        # on construction (self._seed_counts), just needed to be read.
        c = Curriculum([("some_check_name", 17)])
        d = c.directive()
        self.assertIn("17", d)
        self.assertIn("seed's evaluated cases", d)

    def test_directive_uses_rich_description_when_provided(self) -> None:
        c = Curriculum(
            [("commonsense:Sandbox Compliance:validated_meals", 10)],
            check_descriptions={
                "commonsense:Sandbox Compliance:validated_meals": (
                    "Fires when a scheduled restaurant's name is not "
                    "found in the search-tool-derived restaurant index "
                    "for that case."
                ),
            },
        )
        d = c.directive()
        self.assertIn("Fires when a scheduled restaurant's name", d)
        # The generic task-check/harness-check explanation must NOT
        # appear once a real description is available -- no redundant
        # boilerplate alongside the real semantics.
        self.assertNotIn("either a task-based error", d)

    def test_directive_falls_back_to_generic_when_no_description_for_this_goal(self) -> None:
        # A description dict that covers OTHER goals but not the current
        # one must still fall back correctly for this one.
        c = Curriculum(
            [("check_a", 5), ("check_b", 3)],
            check_descriptions={"check_b": "some description"},
        )
        d = c.directive()  # current_goal is check_a, not check_b
        self.assertIn("either a task-based error", d)

    def test_directive_check_descriptions_defaults_to_generic(self) -> None:
        # No check_descriptions argument at all (most projects/tests) --
        # identical to today's generic-only behavior.
        c = Curriculum([("check_a", 5)])
        d = c.directive()
        self.assertIn("either a task-based error", d)


class AdvancementTests(unittest.TestCase):
    def test_advance_on_resolution_threshold(self) -> None:
        c = Curriculum(
            [("a", 10), ("b", 5)], resolution_threshold=0.15, patience=5,
        )
        reason = c.advance_if_ready(failure_rate=0.10)
        self.assertEqual(reason, "resolved")
        self.assertEqual(c.current_goal, "b")
        self.assertEqual(c._rounds_on_current, 0)
        self.assertEqual(c._resolved, ["a"])

    def test_no_advance_above_threshold_and_under_patience(self) -> None:
        c = Curriculum([("a", 10)], resolution_threshold=0.15, patience=5)
        c.record_expand()
        reason = c.advance_if_ready(failure_rate=0.5)
        self.assertIsNone(reason)
        self.assertEqual(c.current_goal, "a")

    def test_patience_fallback_forces_advance_despite_high_failure_rate(self) -> None:
        # patience=2 means 2 full EXPANDs (check-then-record cycles) get
        # spent on the goal; the 3rd EXPAND's check is what force-advances,
        # since the check happens BEFORE record_expand within one EXPAND.
        c = Curriculum([("a", 10), ("b", 5)], resolution_threshold=0.15, patience=2)
        self.assertIsNone(c.advance_if_ready(failure_rate=0.9))  # EXPAND 1
        c.record_expand()
        self.assertIsNone(c.advance_if_ready(failure_rate=0.9))  # EXPAND 2
        c.record_expand()
        reason = c.advance_if_ready(failure_rate=0.9)  # EXPAND 3 -- exhausted
        self.assertEqual(reason, "patience_exhausted")
        self.assertEqual(c.current_goal, "b")

    def test_resolved_takes_priority_over_patience_when_both_true(self) -> None:
        c = Curriculum([("a", 10)], resolution_threshold=0.5, patience=1)
        c.record_expand()  # rounds_on_current = 1 == patience
        # Both conditions true simultaneously: resolved AND patience-exhausted.
        reason = c.advance_if_ready(failure_rate=0.1)
        self.assertEqual(reason, "resolved")

    def test_none_failure_rate_never_resolves_but_still_burns_patience(self) -> None:
        c = Curriculum([("a", 10), ("b", 5)], resolution_threshold=0.9, patience=2)
        self.assertIsNone(c.advance_if_ready(failure_rate=None))  # EXPAND 1
        c.record_expand()
        self.assertIsNone(c.advance_if_ready(failure_rate=None))  # EXPAND 2
        c.record_expand()
        reason = c.advance_if_ready(failure_rate=None)  # EXPAND 3 -- exhausted
        self.assertEqual(reason, "patience_exhausted")

    def test_advance_if_ready_is_a_noop_when_already_done(self) -> None:
        c = Curriculum([("a", 5)], resolution_threshold=0.5, patience=1)
        c.advance_if_ready(failure_rate=0.0)
        self.assertTrue(c.done)
        self.assertIsNone(c.advance_if_ready(failure_rate=0.0))

    def test_record_expand_is_a_noop_when_done(self) -> None:
        c = Curriculum([("a", 5)], resolution_threshold=0.5, patience=1)
        c.advance_if_ready(failure_rate=0.0)
        self.assertTrue(c.done)
        c.record_expand()  # must not raise or touch rounds_on_current meaningfully
        self.assertEqual(c._rounds_on_current, 0)


class FailureRateForTests(unittest.TestCase):
    def test_missing_check_in_top_failed_checks_counts_as_zero(self) -> None:
        rate = Curriculum.failure_rate_for(
            "some_check", {"top_failed_checks": [("other_check", 5)]}, n_evals=10,
        )
        self.assertEqual(rate, 0.0)

    def test_present_check_computes_count_over_n_evals(self) -> None:
        rate = Curriculum.failure_rate_for(
            "x", {"top_failed_checks": [["x", 3], ["y", 1]]}, n_evals=12,
        )
        self.assertAlmostEqual(rate, 3 / 12)

    def test_n_evals_zero_is_none(self) -> None:
        self.assertIsNone(
            Curriculum.failure_rate_for("x", {"top_failed_checks": []}, n_evals=0)
        )

    def test_n_evals_negative_is_none(self) -> None:
        self.assertIsNone(
            Curriculum.failure_rate_for("x", {"top_failed_checks": []}, n_evals=-1)
        )

    def test_check_none_is_none(self) -> None:
        self.assertIsNone(
            Curriculum.failure_rate_for(None, {"top_failed_checks": []}, n_evals=10)
        )

    def test_missing_top_failed_checks_key_degrades_to_zero(self) -> None:
        rate = Curriculum.failure_rate_for("x", {}, n_evals=10)
        self.assertEqual(rate, 0.0)

    def test_malformed_entries_are_skipped_not_raised(self) -> None:
        rate = Curriculum.failure_rate_for(
            "x", {"top_failed_checks": [None, "not a tuple", ["x", 2]]}, n_evals=10,
        )
        self.assertAlmostEqual(rate, 0.2)

    def test_no_plan_rate_excludes_crashed_cases_from_denominator(self) -> None:
        # 8 n_evals, half of them crashed (no_plan_rate=0.5) -> denominator
        # is the 4 actually-scored cases, not the raw 8.
        rate = Curriculum.failure_rate_for(
            "x",
            {"top_failed_checks": [["x", 2]], "no_plan_rate": 0.5},
            n_evals=8,
        )
        self.assertAlmostEqual(rate, 2 / 4)

    def test_full_no_plan_rate_is_none_not_falsely_zero(self) -> None:
        # Every eval crashed: the old (buggy) behavior would compute
        # 0/n_evals = 0.0, falsely reading as "check resolved" on a node
        # that never actually ran the check-level scorer at all -- must
        # be None (no real signal) instead, mirroring
        # behavior_summarizer.py's crash-miscounting fix.
        self.assertIsNone(
            Curriculum.failure_rate_for(
                "x", {"top_failed_checks": [], "no_plan_rate": 1.0}, n_evals=4,
            )
        )

    def test_no_plan_rate_absent_is_uncorrected_denominator(self) -> None:
        # A project whose scorer doesn't emit no_plan_rate at all (most
        # projects) must reproduce today's exact behavior.
        rate = Curriculum.failure_rate_for(
            "x", {"top_failed_checks": [["x", 3]]}, n_evals=12,
        )
        self.assertAlmostEqual(rate, 3 / 12)

    def test_invalid_no_plan_rate_is_ignored(self) -> None:
        # Out-of-range values (negative, > 1) are treated as malformed --
        # ignored rather than corrupting the denominator.
        rate = Curriculum.failure_rate_for(
            "x", {"top_failed_checks": [["x", 3]], "no_plan_rate": -0.2}, n_evals=12,
        )
        self.assertAlmostEqual(rate, 3 / 12)

    def test_harness_check_hit_divides_by_raw_n_evals_not_scored_evals(self) -> None:
        # harness_checks entries are already a count of ALL evals
        # attributable to that crash cause -- no scored_evals correction,
        # unlike a top_failed_checks hit.
        rate = Curriculum.failure_rate_for(
            "sightseeing_failed",
            {"harness_checks": [["sightseeing_failed", 3]], "no_plan_rate": 0.4},
            n_evals=10,
        )
        self.assertAlmostEqual(rate, 0.3)

    def test_top_failed_checks_searched_before_harness_checks(self) -> None:
        # A name present in both (shouldn't happen in practice -- the two
        # namespaces are disjoint by construction) resolves via
        # top_failed_checks first.
        rate = Curriculum.failure_rate_for(
            "x",
            {
                "top_failed_checks": [["x", 3]],
                "harness_checks": [["x", 999]],
            },
            n_evals=12,
        )
        self.assertAlmostEqual(rate, 3 / 12)

    def test_absent_from_both_counts_as_zero(self) -> None:
        rate = Curriculum.failure_rate_for(
            "never_seen",
            {"top_failed_checks": [["x", 3]], "harness_checks": [["y", 2]]},
            n_evals=12,
        )
        self.assertEqual(rate, 0.0)

    def test_absent_from_both_is_none_when_fully_crashed(self) -> None:
        # A 100%-crashed node must never report a check "resolved" just
        # because it's absent from both empty rankings.
        rate = Curriculum.failure_rate_for(
            "never_seen", {"no_plan_rate": 1.0}, n_evals=10,
        )
        self.assertIsNone(rate)


class SnapshotTests(unittest.TestCase):
    def test_snapshot_reflects_state(self) -> None:
        c = Curriculum([("a", 10), ("b", 5)], resolution_threshold=0.15, patience=5)
        c.record_expand()
        c.advance_if_ready(failure_rate=0.1)  # resolves "a"
        snap = c.snapshot(current_failure_rate=0.1, advance_reason="resolved")
        self.assertIsInstance(snap, CurriculumSnapshot)
        self.assertEqual(snap.goals, ["a", "b"])
        self.assertEqual(snap.seed_counts, {"a": 10, "b": 5})
        self.assertEqual(snap.current_index, 1)
        self.assertEqual(snap.current_goal, "b")
        self.assertEqual(snap.rounds_on_current, 0)
        self.assertEqual(snap.resolved_goals, ["a"])
        self.assertEqual(snap.advance_reason, "resolved")
        self.assertFalse(snap.done)

    def test_snapshot_round_trips_through_json(self) -> None:
        c = Curriculum([("a", 10)], resolution_threshold=0.15, patience=5)
        snap = c.snapshot(current_failure_rate=0.3, advance_reason=None)
        as_dict = dataclasses.asdict(snap)
        text = json.dumps(as_dict, indent=2)
        reloaded = json.loads(text)
        self.assertEqual(reloaded["goals"], ["a"])
        self.assertEqual(reloaded["current_failure_rate"], 0.3)


class MultiGoalEndToEndTests(unittest.TestCase):
    def test_progression_across_many_expands_ends_done_with_history(self) -> None:
        c = Curriculum(
            [("a", 30), ("b", 20), ("c", 10)],
            resolution_threshold=0.15, patience=4,
        )
        # Simulate a loosely-improving best node across ~10 EXPANDs.
        failure_rates = [0.5, 0.4, 0.3, 0.1, 0.5, 0.4, 0.1, 0.5, 0.3, 0.1]
        for rate in failure_rates:
            reason = c.advance_if_ready(failure_rate=rate)
            c.record_expand()
            if c.done:
                break
        self.assertTrue(c.done)
        self.assertEqual(c._resolved, ["a", "b", "c"])


class CombinedCheckCountsTests(unittest.TestCase):
    def test_merges_both_sources(self) -> None:
        counts = _combined_check_counts({
            "top_failed_checks": [["check_a", 5], ["check_b", 2]],
            "harness_checks": [["sightseeing_failed", 3]],
        })
        self.assertEqual(
            counts, {"check_a": 5, "check_b": 2, "sightseeing_failed": 3}
        )

    def test_missing_keys_degrade_to_empty(self) -> None:
        self.assertEqual(_combined_check_counts({}), {})

    def test_name_in_both_sources_sums_counts(self) -> None:
        # Shouldn't happen in practice (disjoint namespaces by
        # construction), but must not silently drop one source's count.
        counts = _combined_check_counts({
            "top_failed_checks": [["x", 3]],
            "harness_checks": [["x", 4]],
        })
        self.assertEqual(counts["x"], 7)

    def test_malformed_entries_skipped_not_raised(self) -> None:
        counts = _combined_check_counts({
            "top_failed_checks": [None, "not a tuple", ["x", 2]],
            "harness_checks": [["y", "not a number"]],
        })
        self.assertEqual(counts, {"x": 2})


class InferCurriculumTests(unittest.TestCase):
    def test_merges_and_ranks_by_count_descending(self) -> None:
        goals = infer_curriculum({
            "top_failed_checks": [["check_a", 5], ["check_b", 2]],
            "harness_checks": [["sightseeing_failed", 8]],
        })
        self.assertEqual(
            goals,
            [("sightseeing_failed", 8), ("check_a", 5), ("check_b", 2)],
        )

    def test_ties_broken_alphabetically(self) -> None:
        goals = infer_curriculum({
            "top_failed_checks": [["b", 3]],
            "harness_checks": [["a", 3]],
        })
        self.assertEqual(goals, [("a", 3), ("b", 3)])

    def test_cap_k(self) -> None:
        goals = infer_curriculum(
            {"top_failed_checks": [[f"check_{i}", 20 - i] for i in range(20)]},
            k=5,
        )
        self.assertEqual(len(goals), 5)
        self.assertEqual(goals[0], ("check_0", 20))

    def test_empty_project_metrics_yields_empty_list(self) -> None:
        self.assertEqual(infer_curriculum({}), [])

    def test_result_is_directly_usable_as_curriculum_goals(self) -> None:
        goals = infer_curriculum({
            "top_failed_checks": [["check_a", 5]],
            "harness_checks": [["sightseeing_failed", 8]],
        })
        c = Curriculum(goals)
        self.assertEqual(c.current_goal, "sightseeing_failed")

    def test_high_no_plan_rate_prioritizes_harness_over_higher_count_task(self) -> None:
        # A task check with a far higher raw count (59) than the harness
        # cause (32) would win the flat merge -- but no_plan_rate=0.35 is
        # above the default 0.15 threshold, so the harness goal must still
        # come first: a task-level fix can't show up in the score for a
        # case the harness never let reach scoring.
        goals = infer_curriculum({
            "top_failed_checks": [["reasonable_transfer_time", 59]],
            "harness_checks": [["sightseeing_failed", 32]],
            "no_plan_rate": 0.35,
        })
        self.assertEqual(
            goals,
            [("sightseeing_failed", 32), ("reasonable_transfer_time", 59)],
        )

    def test_low_no_plan_rate_falls_back_to_flat_merge(self) -> None:
        # Below the threshold, raw count wins regardless of source --
        # byte-identical to the pre-tiering behavior.
        goals = infer_curriculum({
            "top_failed_checks": [["reasonable_transfer_time", 59]],
            "harness_checks": [["sightseeing_failed", 5]],
            "no_plan_rate": 0.05,
        })
        self.assertEqual(
            goals,
            [("reasonable_transfer_time", 59), ("sightseeing_failed", 5)],
        )

    def test_no_plan_rate_at_threshold_boundary_is_inclusive(self) -> None:
        goals = infer_curriculum({
            "top_failed_checks": [["check_a", 99]],
            "harness_checks": [["sightseeing_failed", 1]],
            "no_plan_rate": 0.15,
        })
        self.assertEqual(goals[0], ("sightseeing_failed", 1))

    def test_missing_no_plan_rate_falls_back_to_flat_merge(self) -> None:
        # A project/run whose scorer never emits no_plan_rate at all --
        # zero behavior change from before this feature existed.
        goals = infer_curriculum({
            "top_failed_checks": [["check_a", 10]],
            "harness_checks": [["sightseeing_failed", 3]],
        })
        self.assertEqual(goals[0], ("check_a", 10))

    def test_high_no_plan_rate_but_no_harness_checks_falls_back(self) -> None:
        # A run that predates harness_checks (e.g. the old production
        # round_000 seen live this session): no_plan_rate is high, but
        # there's nothing to prioritize, so behavior is unchanged.
        goals = infer_curriculum({
            "top_failed_checks": [["reasonable_transfer_time", 59]],
            "no_plan_rate": 0.35,
        })
        self.assertEqual(goals[0], ("reasonable_transfer_time", 59))

    def test_custom_harness_priority_threshold(self) -> None:
        goals = infer_curriculum(
            {
                "top_failed_checks": [["check_a", 99]],
                "harness_checks": [["sightseeing_failed", 1]],
                "no_plan_rate": 0.20,
            },
            harness_priority_threshold=0.5,
        )
        # 0.20 < 0.5 -> below the custom threshold, flat merge applies.
        self.assertEqual(goals[0], ("check_a", 99))

    def test_harness_first_still_dedupes_a_name_in_both_sources(self) -> None:
        # Documented as "shouldn't happen in practice" but must not
        # duplicate a goal if it ever does.
        goals = infer_curriculum({
            "top_failed_checks": [["shared_name", 4]],
            "harness_checks": [["shared_name", 7]],
            "no_plan_rate": 0.9,
        })
        self.assertEqual(goals, [("shared_name", 7)])


if __name__ == "__main__":
    unittest.main()
