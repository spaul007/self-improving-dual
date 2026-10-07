"""Tests for projects/travel_mas_refactored_flattened_scores -- the
continuous-scoring sibling of projects/travel_mas_refactored (cloned per
the plan at /users/v.kulkarni1/.claude/plans/let-us-implement-the-lazy-wand.md).

Mirrors tests/test_travel_mas_refactored_conversion_error_tracking.py's
style (unittest.TestCase, patch.object on module-level functions,
SimpleNamespace fakes). Per the plan's own collision-avoidance note: this
file imports both projects' adapters fully-qualified
(projects.travel_mas_refactored_flattened_scores.adapter.scorer_impl /
projects.travel_mas_refactored.adapter.scorer_impl) -- never through
benchmark/scorer.py's bare-"adapter" sys.path-mutating shim, which is the
pre-existing, repo-wide collision layer this must avoid triggering.

    PYTHONPATH=. python3 -m unittest tests.test_travel_mas_refactored_flattened_scores_continuous
"""
from __future__ import annotations

import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

# Pre-existing, repo-wide collision (see the plan): every project's
# scorer_impl.py does a BARE `from _eval.eval_converted import ...` (not
# `projects.<name>.benchmark._eval...`), relying on sys.path insertion
# order rather than a namespaced import. In a full-suite run,
# test_travel_mas_refactored_conversion_error_tracking.py (collected
# alphabetically before this file) already imports the ORIGINAL project's
# scorer_impl.py, which caches sys.modules["_eval"] pointing at
# travel_mas_refactored's own benchmark/_eval/ -- which lacks this
# project's new calculate_*_continuous functions. Purge it so the import
# below re-resolves "_eval" fresh, against THIS project's own
# benchmark/_eval/. Safe: any projects.*.adapter.scorer_impl module
# already fully imported elsewhere stays cached with its own functions
# already bound from whichever "_eval" was live at ITS import time; this
# purge only affects imports that haven't happened yet in this process.
for _mod_name in list(sys.modules):
    if _mod_name == "_eval" or _mod_name.startswith("_eval."):
        del sys.modules[_mod_name]

from projects.travel_mas_refactored_flattened_scores.adapter import scorer_impl
from projects.travel_mas_refactored_flattened_scores.benchmark._eval import eval_converted
from meta_agent.registry import get as registry_get

# The ORIGINAL, unmodified project -- imported fully-qualified, never via
# benchmark/scorer.py's shim, so this coexists safely with the clone's own
# import above in the same test process.
from projects.travel_mas_refactored.adapter import scorer_impl as original_scorer_impl


# One real check name per dimension (see constraints_commonsense.py's
# EVALUATION_DIMENSIONS) -- calculate_weighted_score(_continuous) only
# counts a check if its name is actually a member of some dimension's
# checks list, so fixtures below must use real names, not made-up ones.
_ALL_DIMENSIONS_FULLY_PASSING = {
    "valid_trip_duration": (True, None),
    "validated_accommodation": (True, None),
    "traceable_accommodation": (True, None),
    "no_time_overlaps": (True, None),
    "attraction_visit_within_opening_hours": (True, None),
    "reasonable_duration_at_attractions": (True, None),
    "cost_calculation_correctness": (True, None),
    "diverse_meal_options": (True, None),
}

# 4/5 hard constraints passing -- continuous 0.8, gated 0.0 (one-vote veto).
_HARD_4_OF_5_PASSING = {
    "hotel_highest_rated": (True, None),
    "restaurant_highest_rated": (True, None),
    "flight_cheapest_direct": (True, None),
    "train_cheapest_direct": (True, None),
    "attraction_must_visit_named": (False, "named attraction missing"),
}


class EvalConvertedContinuousMathTests(unittest.TestCase):
    """Pure math tests on the two new eval_converted.py functions -- no
    scorer involved."""

    def test_dimension_partial_pass_is_fractional(self) -> None:
        # Time Feasibility has 2 checks; 1 passes.
        results = {
            "no_time_overlaps": (True, None),
            "reasonable_transfer_time": (False, "too short"),
        }
        out = eval_converted.calculate_weighted_score_continuous(results)
        self.assertEqual(out["dimension_scores"]["Time Feasibility"], 0.5)
        self.assertAlmostEqual(out["dimension_details"]["Time Feasibility"]["weighted_score"], 0.5 * 0.125)
        # Gated: same input, one-vote veto -> 0.0 (not all checks pass).
        gated = eval_converted.calculate_weighted_score(results)
        self.assertEqual(gated["dimension_scores"]["Time Feasibility"], 0.0)

    def test_dimension_with_no_present_checks_scores_zero_not_vacuous(self) -> None:
        """Deliberate asymmetry vs. hard constraints: a commonsense
        dimension's check set is fixed/universal, so zero present checks
        is a wiring problem, not legitimate inapplicability -- score 0.0,
        not 1.0."""
        out = eval_converted.calculate_weighted_score_continuous({})
        for dim_score in out["dimension_scores"].values():
            self.assertEqual(dim_score, 0.0)

    def test_hard_constraint_partial_pass_is_fractional(self) -> None:
        results = {
            "hotel_highest_rated": (True, None),
            "restaurant_highest_rated": (True, None),
            "flight_cheapest_direct": (False, "not cheapest"),
        }
        out = eval_converted.calculate_hard_score_continuous(results)
        self.assertAlmostEqual(out["score"], 2 / 3)
        # Gated: same input, one-vote veto -> 0.0.
        gated = eval_converted.calculate_hard_score(results)
        self.assertEqual(gated["score"], 0.0)

    def test_zero_hard_constraints_head_to_head(self) -> None:
        """The deliberate, scoped divergence: gated stays 0.0 (quirk
        preserved for byte-identical comparability), continuous is 1.0
        (vacuously satisfied -- nothing to violate)."""
        self.assertEqual(eval_converted.calculate_hard_score({})["score"], 0.0)
        self.assertEqual(eval_converted.calculate_hard_score_continuous({})["score"], 1.0)


class ScorerGatedModeRegressionTests(unittest.TestCase):
    """Proves the clone introduced no arithmetic drift: score_mode="gated"
    reproduces travel_mas_refactored's own original numbers exactly, on
    the identical fixture."""

    def _score(self, scorer, commonsense_results, hard_results) -> dict:
        case = {"meta_info": {"some": "meta"}, "id": "1"}
        agent_output = SimpleNamespace(result="Day 1: a fine plan", metadata={})
        target = scorer_impl if isinstance(scorer, scorer_impl.TravelFlattenedScoresScorer) else original_scorer_impl
        with patch.object(target, "_convert_plan_to_json", return_value=({"days": []}, None)), \
             patch.object(target, "eval_commonsense", return_value=commonsense_results), \
             patch.object(target, "eval_hard", return_value=hard_results):
            return scorer.score(case, agent_output)

    def test_gated_mode_matches_original_project_exactly(self) -> None:
        clone = scorer_impl.TravelFlattenedScoresScorer(score_mode="gated")
        original = original_scorer_impl.TravelCompositeScorer()
        out_clone = self._score(clone, _ALL_DIMENSIONS_FULLY_PASSING, _HARD_4_OF_5_PASSING)
        out_original = self._score(original, _ALL_DIMENSIONS_FULLY_PASSING, _HARD_4_OF_5_PASSING)
        self.assertEqual(out_clone["details"]["gated_composite_score"], out_original["details"]["composite_score"])
        self.assertEqual(out_clone["score"], out_original["score"])
        self.assertEqual(out_clone["passed"], out_original["passed"])


class ScorerContinuousModeTests(unittest.TestCase):
    def _score(self, scorer, commonsense_results, hard_results) -> dict:
        case = {"meta_info": {"some": "meta"}, "id": "1"}
        agent_output = SimpleNamespace(result="Day 1: a fine plan", metadata={})
        with patch.object(scorer_impl, "_convert_plan_to_json", return_value=({"days": []}, None)), \
             patch.object(scorer_impl, "eval_commonsense", return_value=commonsense_results), \
             patch.object(scorer_impl, "eval_hard", return_value=hard_results):
            return scorer.score(case, agent_output)

    def test_continuous_mode_is_fractional_and_drives_reported_score(self) -> None:
        scorer = scorer_impl.TravelFlattenedScoresScorer()  # default: continuous
        out = self._score(scorer, _ALL_DIMENSIONS_FULLY_PASSING, _HARD_4_OF_5_PASSING)
        details = out["details"]
        self.assertAlmostEqual(details["hard_score"], 0.8)
        self.assertAlmostEqual(details["composite_score"], (1.0 + 0.8) / 2)
        self.assertEqual(out["score"], details["composite_score"])

    def test_passed_stays_gated_derived_even_when_continuous_composite_is_high(self) -> None:
        scorer = scorer_impl.TravelFlattenedScoresScorer()  # continuous
        out = self._score(scorer, _ALL_DIMENSIONS_FULLY_PASSING, _HARD_4_OF_5_PASSING)
        details = out["details"]
        # Continuous composite is high (0.9) -- but gated hard_score is 0.0
        # (one-vote veto, 4/5 isn't all 5), so passed must still be False.
        self.assertAlmostEqual(details["composite_score"], 0.9)
        self.assertLess(details["gated_composite_score"], 1.0)
        self.assertFalse(out["passed"])

    def test_details_always_carries_both_families_regardless_of_score_mode(self) -> None:
        for mode in ("continuous", "gated"):
            scorer = scorer_impl.TravelFlattenedScoresScorer(score_mode=mode)
            out = self._score(scorer, _ALL_DIMENSIONS_FULLY_PASSING, _HARD_4_OF_5_PASSING)
            details = out["details"]
            for key in (
                "commonsense_score", "hard_score", "composite_score",
                "dimension_scores", "dimension_details", "hard_constraints",
                "gated_commonsense_score", "gated_hard_score", "gated_composite_score",
                "gated_dimension_scores", "gated_dimension_details", "gated_hard_constraints",
            ):
                self.assertIn(key, details, f"missing {key!r} under score_mode={mode!r}")


class ScorerInitValidationTests(unittest.TestCase):
    def test_default_score_mode_is_continuous(self) -> None:
        self.assertEqual(scorer_impl.TravelFlattenedScoresScorer().score_mode, "continuous")

    def test_invalid_score_mode_raises(self) -> None:
        with self.assertRaises(ValueError):
            scorer_impl.TravelFlattenedScoresScorer(score_mode="bogus")


class RegistrationCollisionGuardTests(unittest.TestCase):
    def test_clone_and_original_are_registered_under_distinct_names(self) -> None:
        # Our own class, imported the one way this project is ever
        # imported in this suite -- a plain identity check is meaningful
        # here.
        self.assertIs(
            registry_get("scorer", "travel_mas_refactored_flattened_scores_default"),
            scorer_impl.TravelFlattenedScoresScorer,
        )
        # "travel_mas_refactored_default" must NOT be our clone's class --
        # that's the actual collision this guards against. It is NOT
        # asserted to be ONE specific class object identity-equal to
        # `original_scorer_impl.TravelCompositeScorer`: other, unrelated
        # tests in the full suite (e.g. test_smoke_test_validator_wiring.py)
        # load travel_mas_refactored via build_components()'s bare
        # `adapter.scorer_impl` shim path rather than this file's
        # fully-qualified `projects.travel_mas_refactored.adapter...` path
        # -- a PRE-EXISTING ambiguity (two import paths to the same
        # original project, not introduced by this clone) that makes
        # which exact class object ends up registered last an
        # implementation detail of suite ordering, not something this
        # plan's work controls.
        registered = registry_get("scorer", "travel_mas_refactored_default")
        self.assertIsNot(registered, scorer_impl.TravelFlattenedScoresScorer)
        self.assertEqual(registered.__name__, "TravelCompositeScorer")


class FullMetricsClonedTests(unittest.TestCase):
    """TravelFlattenedScoresScorer.full_metrics() -- an independent copy
    of the original's (not inherited; this project has no import
    relationship to the original, same as aggregate()), walking the
    native (continuous) dimension_details/hard_constraints. A check's
    own pass/fail boolean is identical under continuous or gated scoring
    (only the dimension/hard-level AGGREGATE differs -- see the scorer's
    own docstring), so this must give the same per-check counts as the
    original on an identical fixture."""

    def _case(self, case_id, *, dimension_details=None, hard_constraints=None):
        from meta_agent.models import CaseResult

        details: dict = {}
        if dimension_details is not None:
            details["dimension_details"] = dimension_details
        if hard_constraints is not None:
            details["hard_constraints"] = hard_constraints
        return CaseResult(case_id=case_id, passed=False, score=0.0, details=details)

    def test_matches_original_on_an_identical_fixture(self) -> None:
        per_case = [
            self._case(
                "0",
                dimension_details={
                    "Time Feasibility": {
                        "checks": [{"name": "reasonable_transfer_time", "passed": False}]
                    }
                },
                hard_constraints={"flight_seat_status": {"passed": True}},
            ),
            self._case(
                "1",
                dimension_details={
                    "Time Feasibility": {
                        "checks": [{"name": "reasonable_transfer_time", "passed": True}]
                    }
                },
                hard_constraints={"flight_seat_status": {"passed": False}},
            ),
        ]
        cloned = scorer_impl.TravelFlattenedScoresScorer().full_metrics(per_case)
        original = original_scorer_impl.TravelCompositeScorer().full_metrics(per_case)
        self.assertEqual(cloned, original)


if __name__ == "__main__":
    unittest.main()
