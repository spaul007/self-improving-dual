"""Unit tests for projects.travel.travel_error_categorizer.categorize_errors.

No LLM, no subprocesses; pure CaseResult fixtures.

    PYTHONPATH=. python3 -m unittest tests.test_travel_error_categorizer
"""
from __future__ import annotations

import unittest

from meta_agent.models import CaseResult
from projects.travel.travel_error_categorizer import categorize_errors


def _commonsense_case(
    case_id: str,
    *,
    failed_dims: dict[str, list[tuple[str, str]]] | None = None,
    failed_hard: dict[str, str] | None = None,
) -> CaseResult:
    """Build a CaseResult shaped like projects/travel/benchmark/scorer.py
    emits for non-zero scores. ``failed_dims`` maps dim_name -> list of
    (check_name, message); ``failed_hard`` maps constraint_name -> message.
    """
    dim_details: dict[str, dict] = {}
    for dim, checks in (failed_dims or {}).items():
        dim_details[dim] = {
            "score": 0.0,
            "checks": [
                {"name": name, "passed": False, "message": msg}
                for name, msg in checks
            ],
        }
    hard = {
        cname: {"passed": False, "message": msg}
        for cname, msg in (failed_hard or {}).items()
    }
    return CaseResult(
        case_id=case_id,
        passed=False,
        score=0.5,
        details={
            "commonsense_score": 0.5,
            "hard_score": 0.5,
            "composite_score": 0.5,
            "dimension_details": dim_details,
            "hard_constraints": hard,
            "failed_checks": [],
        },
    )


def _plan_failed_case(case_id: str, reason: str) -> CaseResult:
    """Matches the ``_zero_result`` shape from scorer.py:184-191."""
    return CaseResult(
        case_id=case_id,
        passed=False,
        score=0.0,
        details={
            "error": f"plan conversion failed: {reason}",
            "commonsense_score": 0.0,
            "hard_score": 0.0,
            "composite_score": 0.0,
            "raw_output_preview": "",
        },
    )


def _runtime_crash_case(case_id: str, err: str) -> CaseResult:
    return CaseResult(case_id=case_id, passed=False, score=0.0, error=err)


def _passing_case(case_id: str) -> CaseResult:
    return CaseResult(
        case_id=case_id,
        passed=True,
        score=1.0,
        details={
            "commonsense_score": 1.0,
            "hard_score": 1.0,
            "composite_score": 1.0,
            "dimension_details": {},
            "hard_constraints": {},
            "failed_checks": [],
        },
    )


class CommonsenseGroupingTests(unittest.TestCase):
    def test_groups_by_dimension_and_sorts_by_failure_rate(self) -> None:
        cases = [
            _commonsense_case(
                "0",
                failed_dims={
                    "Time Feasibility": [("clash", "trains overlap")],
                    "Route Consistency": [("loop", "city revisited")],
                },
            ),
            _commonsense_case(
                "1",
                failed_dims={"Time Feasibility": [("clash", "trains overlap")]},
            ),
            _passing_case("2"),
            _passing_case("3"),
        ]
        result = categorize_errors(cases)
        cat_ids = [c["category_id"] for c in result]
        self.assertEqual(
            cat_ids,
            ["commonsense__time_feasibility", "commonsense__route_consistency"],
        )
        time = result[0]
        self.assertEqual(time["category_type"], "commonsense")
        self.assertEqual(time["num_failing_samples"], 2)
        self.assertEqual(time["total_samples"], 4)
        self.assertAlmostEqual(time["failure_rate"], 0.5)
        self.assertEqual(len(time["representative_errors"]), 2)
        # Other dimension has only one failing sample, lower failure_rate.
        self.assertEqual(result[1]["num_failing_samples"], 1)
        self.assertAlmostEqual(result[1]["failure_rate"], 0.25)


class HardConstraintGroupingTests(unittest.TestCase):
    def test_groups_by_prefix_and_uses_other_for_unknown(self) -> None:
        cases = [
            _commonsense_case(
                "0",
                failed_hard={
                    "flight_duration_check": "too long",
                    "flight_layover_check": "overnight",
                    "novel_prefix_thing": "weird",
                },
            ),
            _commonsense_case(
                "1",
                failed_hard={"hotel_min_nights": "below 2"},
            ),
        ]
        result = categorize_errors(cases)
        cat_ids = {c["category_id"] for c in result}
        self.assertEqual(
            cat_ids,
            {
                "hard_constraint__flight",
                "hard_constraint__hotel",
                "hard_constraint__other",
            },
        )
        flight = next(c for c in result if c["category_id"] == "hard_constraint__flight")
        # Both flight constraints collapsed under a single per-case example.
        self.assertEqual(flight["num_failing_samples"], 1)
        self.assertEqual(len(flight["representative_errors"]), 1)
        self.assertEqual(
            sorted(flight["representative_errors"][0]["checks_failed"]),
            ["flight_duration_check", "flight_layover_check"],
        )


class SyntheticCategoryTests(unittest.TestCase):
    def test_plan_conversion_failed_synthesizes_generic_category(self) -> None:
        cases = [
            _plan_failed_case("0", "missing </JSON> tag"),
            _plan_failed_case("1", "json decode error"),
            _passing_case("2"),
        ]
        result = categorize_errors(cases)
        self.assertEqual(len(result), 1)
        cat = result[0]
        self.assertEqual(cat["category_id"], "generic__plan_conversion_failed")
        self.assertEqual(cat["category_type"], "generic")
        self.assertEqual(cat["num_failing_samples"], 2)
        self.assertAlmostEqual(cat["failure_rate"], 2 / 3)
        self.assertEqual(len(cat["representative_errors"]), 2)

    def test_runtime_crash_synthesizes_generic_category(self) -> None:
        cases = [
            _runtime_crash_case("0", "subprocess timed out after 60s"),
            _passing_case("1"),
        ]
        result = categorize_errors(cases)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["category_id"], "generic__runtime_error")
        self.assertEqual(result[0]["category_type"], "generic")
        self.assertEqual(result[0]["num_failing_samples"], 1)
        self.assertIn(
            "subprocess timed out",
            result[0]["representative_errors"][0]["messages"][0],
        )


class FailedChecksFallbackTests(unittest.TestCase):
    def test_uses_failed_checks_when_structured_details_missing(self) -> None:
        cases = [
            CaseResult(
                case_id="0",
                passed=False,
                score=0.3,
                details={
                    "failed_checks": [
                        "commonsense:Time Feasibility:overlap",
                        "hard:flight_duration_check",
                    ],
                    # No dimension_details / hard_constraints — forces the
                    # last-resort branch.
                },
            ),
            _passing_case("1"),
        ]
        result = categorize_errors(cases)
        cat_ids = {c["category_id"] for c in result}
        self.assertEqual(
            cat_ids,
            {"commonsense__time_feasibility", "hard_constraint__flight"},
        )


class CapsAndEmptyInputTests(unittest.TestCase):
    def test_max_representative_errors_caps_examples_but_not_counts(self) -> None:
        cases = [
            _commonsense_case(
                str(i),
                failed_dims={"Time Feasibility": [(f"chk_{i}", f"msg {i}")]},
            )
            for i in range(7)
        ]
        result = categorize_errors(cases, max_representative_errors=3)
        self.assertEqual(len(result), 1)
        cat = result[0]
        # All 7 failures counted...
        self.assertEqual(cat["num_failing_samples"], 7)
        self.assertEqual(cat["total_samples"], 7)
        self.assertAlmostEqual(cat["failure_rate"], 1.0)
        # ...but only 3 example errors kept for the prompt.
        self.assertEqual(len(cat["representative_errors"]), 3)

    def test_empty_per_case_returns_empty_list(self) -> None:
        self.assertEqual(categorize_errors([]), [])

    def test_all_passing_returns_empty_list(self) -> None:
        self.assertEqual(
            categorize_errors([_passing_case("0"), _passing_case("1")]),
            [],
        )


if __name__ == "__main__":
    unittest.main()
