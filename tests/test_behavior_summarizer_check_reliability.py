"""Tests for BehaviorSummarizer's check/constraint reliability tracking:
_check_fail_case_ids, _aggregate_check_reliability,
_rate_change_significance (Fisher's exact test), and
_compare_check_reliability_to_parent.

Requires scipy -- run via vivek_env, not system python3:
    PYTHONPATH=. /groups/AIC-MV/v.kulkarni1/unified_framework/vivek_env/bin/python3.12 \\
        -m unittest tests.test_behavior_summarizer_check_reliability
"""
from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from meta_agent.behavior_summarizer import BehaviorSummarizer
from meta_agent.models import CaseResult, EvaluationResult


def _stub_summarizer() -> BehaviorSummarizer:
    return BehaviorSummarizer(llm_caller=lambda **kw: None, model="stub-model")


class CheckFailCaseIdsTests(unittest.TestCase):
    def test_extracts_fail_sets_and_n_cases(self) -> None:
        per_case = [
            CaseResult(case_id="1", passed=False, score=0.0,
                       details={"failed_checks": ["a", "b"]}),
            CaseResult(case_id="2", passed=True, score=1.0,
                       details={"failed_checks": []}),
            CaseResult(case_id="3", passed=False, score=0.5,
                       details={"failed_checks": ["a"]}),
        ]
        fails, n = BehaviorSummarizer._check_fail_case_ids(per_case)
        self.assertEqual(n, 3)
        self.assertEqual(fails["a"], {"1", "3"})
        self.assertEqual(fails["b"], {"1"})

    def test_no_failed_checks_key_anywhere_yields_empty_dict(self) -> None:
        # No `details` at all (defaults to {}) means this case never reached
        # real constraint evaluation -- excluded from n_cases entirely (the
        # information-ceiling guard), not counted as "1 case, 0 failures".
        per_case = [CaseResult(case_id="1", passed=True, score=1.0)]
        fails, n = BehaviorSummarizer._check_fail_case_ids(per_case)
        self.assertEqual(fails, {})
        self.assertEqual(n, 0)

    def test_present_but_empty_failed_checks_counts_as_a_real_scored_case(
        self,
    ) -> None:
        # The key IS present (just empty) -- this is a case that genuinely
        # ran the scorer's real checks and passed all of them, unlike the
        # test above where the key is absent entirely.
        per_case = [
            CaseResult(case_id="1", passed=True, score=1.0,
                       details={"failed_checks": []}),
        ]
        fails, n = BehaviorSummarizer._check_fail_case_ids(per_case)
        self.assertEqual(fails, {})
        self.assertEqual(n, 1)

    def test_non_string_entries_in_failed_checks_are_skipped(self) -> None:
        per_case = [
            CaseResult(case_id="1", passed=False, score=0.0,
                       details={"failed_checks": ["a", 42, None]}),
        ]
        fails, _ = BehaviorSummarizer._check_fail_case_ids(per_case)
        self.assertEqual(set(fails), {"a"})


class AggregateCheckReliabilityTests(unittest.TestCase):
    def test_empty_when_no_project_reports_failed_checks(self) -> None:
        summ = _stub_summarizer()
        per_case = [CaseResult(case_id="1", passed=True, score=1.0)]
        self.assertEqual(summ._aggregate_check_reliability(per_case), {})

    def test_ranks_rarest_failures_first(self) -> None:
        summ = _stub_summarizer()
        per_case = [
            CaseResult(case_id="1", passed=False, score=0.0,
                       details={"failed_checks": ["common", "rare"]}),
            CaseResult(case_id="2", passed=False, score=0.0,
                       details={"failed_checks": ["common"]}),
            CaseResult(case_id="3", passed=False, score=0.0,
                       details={"failed_checks": ["common"]}),
        ]
        result = summ._aggregate_check_reliability(per_case)
        self.assertEqual(result["n_cases"], 3)
        order = list(result["checks"].keys())
        self.assertEqual(order, ["rare", "common"])
        self.assertEqual(result["checks"]["rare"]["failed_in_n_cases"], 1)
        self.assertEqual(result["checks"]["common"]["failed_in_n_cases"], 3)


class RateChangeSignificanceTests(unittest.TestCase):
    def test_large_drop_is_significant_and_improved(self) -> None:
        significant, direction = BehaviorSummarizer._rate_change_significance(
            40, 60, 5, 60
        )
        self.assertTrue(significant)
        self.assertEqual(direction, "improved")

    def test_tiny_drop_is_not_significant(self) -> None:
        significant, direction = BehaviorSummarizer._rate_change_significance(
            40, 60, 38, 60
        )
        self.assertFalse(significant)
        self.assertEqual(direction, "improved")  # direction is still honest

    def test_identical_rates_are_unchanged(self) -> None:
        significant, direction = BehaviorSummarizer._rate_change_significance(
            0, 60, 0, 60
        )
        self.assertFalse(significant)
        self.assertEqual(direction, "unchanged")

    def test_large_increase_is_significant_and_regressed(self) -> None:
        significant, direction = BehaviorSummarizer._rate_change_significance(
            5, 60, 40, 60
        )
        self.assertTrue(significant)
        self.assertEqual(direction, "regressed")

    def test_zero_n_child_is_a_safe_unchanged_fallback(self) -> None:
        significant, direction = BehaviorSummarizer._rate_change_significance(
            0, 60, 0, 0
        )
        self.assertFalse(significant)
        self.assertEqual(direction, "unchanged")

    def test_zero_n_parent_is_a_safe_unchanged_fallback(self) -> None:
        significant, direction = BehaviorSummarizer._rate_change_significance(
            0, 0, 0, 60
        )
        self.assertFalse(significant)
        self.assertEqual(direction, "unchanged")

    def test_extreme_small_count_drop_to_zero_is_still_flagged(self) -> None:
        # The exact regime Fisher's exact test is chosen for -- small n,
        # extreme rate near 0 -- where a Wald-style approximation is known
        # to be unreliable. 8/8 -> 0/8 is a clean, total improvement.
        significant, direction = BehaviorSummarizer._rate_change_significance(
            8, 8, 0, 8
        )
        self.assertTrue(significant)
        self.assertEqual(direction, "improved")


class CompareCheckReliabilityToParentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="check_reliability_test_"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.summ = _stub_summarizer()

    def _write_parent_eval_result(self, per_case: list[CaseResult]) -> Path:
        parent_dir = self.tmp / "parent_round"
        parent_dir.mkdir()
        result = EvaluationResult(
            score=0.0, passed=0, failed=len(per_case), per_case=per_case
        )
        (parent_dir / "eval_result.json").write_text(
            result.model_dump_json(), encoding="utf-8"
        )
        return parent_dir

    def test_empty_when_parent_eval_result_missing(self) -> None:
        parent_dir = self.tmp / "no_such_parent"
        parent_dir.mkdir()
        child_per_case = [
            CaseResult(case_id="1", passed=False, score=0.0,
                       details={"failed_checks": ["x"]}),
        ]
        self.assertEqual(
            self.summ._compare_check_reliability_to_parent(
                child_per_case, parent_dir
            ),
            {},
        )

    def test_significant_improvement_flagged_with_full_counts(self) -> None:
        parent_per_case = [
            CaseResult(case_id=str(i), passed=False, score=0.0,
                       details={"failed_checks": ["transfer_time"]})
            for i in range(40)
        ] + [
            CaseResult(case_id=str(i), passed=True, score=1.0, details={"failed_checks": []})
            for i in range(40, 60)
        ]
        parent_dir = self._write_parent_eval_result(parent_per_case)

        child_per_case = [
            CaseResult(case_id=str(i), passed=False, score=0.0,
                       details={"failed_checks": ["transfer_time"]})
            for i in range(5)
        ] + [
            CaseResult(case_id=str(i), passed=True, score=1.0, details={"failed_checks": []})
            for i in range(5, 60)
        ]

        result = self.summ._compare_check_reliability_to_parent(
            child_per_case, parent_dir
        )
        self.assertIn("transfer_time", result)
        row = result["transfer_time"]
        self.assertEqual(row["parent_failed_in_n_cases"], 40)
        self.assertEqual(row["parent_n_cases"], 60)
        self.assertEqual(row["child_failed_in_n_cases"], 5)
        self.assertEqual(row["child_n_cases"], 60)
        self.assertEqual(row["direction"], "improved")
        self.assertTrue(row["significant"])

    def test_check_that_dropped_to_zero_still_appears_via_union(self) -> None:
        # The bug this test guards against: a check absent from the CHILD's
        # own failed_checks (because it now never fails) must still show
        # up here -- it would be silently invisible if the comparison only
        # iterated over the child's own failing checks.
        parent_per_case = [
            CaseResult(case_id=str(i), passed=False, score=0.0,
                       details={"failed_checks": ["budget_constraint"]})
            for i in range(30)
        ] + [
            CaseResult(case_id=str(i), passed=True, score=1.0, details={"failed_checks": []})
            for i in range(30, 60)
        ]
        parent_dir = self._write_parent_eval_result(parent_per_case)

        # Child: budget_constraint never fails anywhere.
        child_per_case = [
            CaseResult(case_id=str(i), passed=True, score=1.0, details={"failed_checks": []})
            for i in range(60)
        ]

        result = self.summ._compare_check_reliability_to_parent(
            child_per_case, parent_dir
        )
        self.assertIn("budget_constraint", result)
        row = result["budget_constraint"]
        self.assertEqual(row["parent_failed_in_n_cases"], 30)
        self.assertEqual(row["child_failed_in_n_cases"], 0)
        self.assertEqual(row["direction"], "improved")
        self.assertTrue(row["significant"])

    def test_tiny_noise_level_change_not_flagged_significant(self) -> None:
        parent_per_case = [
            CaseResult(case_id=str(i), passed=False, score=0.0,
                       details={"failed_checks": ["x"]})
            for i in range(30)
        ] + [
            CaseResult(case_id=str(i), passed=True, score=1.0, details={"failed_checks": []})
            for i in range(30, 60)
        ]
        parent_dir = self._write_parent_eval_result(parent_per_case)

        child_per_case = [
            CaseResult(case_id=str(i), passed=False, score=0.0,
                       details={"failed_checks": ["x"]})
            for i in range(29)
        ] + [
            CaseResult(case_id=str(i), passed=True, score=1.0, details={"failed_checks": []})
            for i in range(29, 60)
        ]

        result = self.summ._compare_check_reliability_to_parent(
            child_per_case, parent_dir
        )
        self.assertIn("x", result)
        self.assertFalse(result["x"]["significant"])

    def test_corrupt_parent_eval_result_degrades_to_empty(self) -> None:
        parent_dir = self.tmp / "corrupt_parent"
        parent_dir.mkdir()
        (parent_dir / "eval_result.json").write_text("not json", encoding="utf-8")
        child_per_case = [
            CaseResult(case_id="1", passed=False, score=0.0,
                       details={"failed_checks": ["x"]}),
        ]
        self.assertEqual(
            self.summ._compare_check_reliability_to_parent(
                child_per_case, parent_dir
            ),
            {},
        )

    def test_all_cases_crashed_is_not_reported_as_improvement(self) -> None:
        # The exact real bug this guards against: a node whose edit crashed
        # every case (details={}, no error making it to the scorer at all)
        # must NOT show every one of the parent's failing checks as having
        # "improved to 0 failures" -- that would be silence misread as
        # success. All 32 cases crashing is information-theoretic zero
        # data, not zero failures.
        parent_per_case = [
            CaseResult(case_id=str(i), passed=False, score=0.0,
                       details={"failed_checks": ["reasonable_transfer_time"]})
            for i in range(22)
        ] + [
            CaseResult(case_id=str(i), passed=True, score=1.0,
                       details={"failed_checks": []})
            for i in range(22, 32)
        ]
        parent_dir = self._write_parent_eval_result(parent_per_case)

        # Child: every case crashed at the harness level -- no scorer
        # details at all (mirrors a real subprocess-runner crash: the
        # per-case `error` field is set, `details` stays {}).
        child_per_case = [
            CaseResult(
                case_id=str(i), passed=False, score=0.0,
                error="Traceback (most recent call last): ...",
                details={},
            )
            for i in range(32)
        ]

        result = self.summ._compare_check_reliability_to_parent(
            child_per_case, parent_dir
        )
        # With every child case excluded (no failed_checks key), there is
        # no comparable child data for any check -- the correct output is
        # "no data", not a table full of false "0/32, improved, SIGNIFICANT"
        # rows.
        self.assertEqual(result, {})


class BuildPromptRenderingTests(unittest.TestCase):
    def test_new_sections_and_significance_appear_in_rendered_prompt(self) -> None:
        summ = _stub_summarizer()
        aggregate = {
            "node_id": 2,
            "parent_id": 1,
            "score": 0.4,
            "passed": 0,
            "failed": 60,
            "n_cases": 60,
            "changed_files": ["agents/flight.py"],
            "diff": "diff --git a/x b/x",
            "mutable_log": {},
            "tool_calls": {},
            "check_reliability": {
                "n_cases": 60,
                "checks": {
                    "rare_check": {
                        "failed_in_n_cases": 1,
                        "sample_failing_case_ids": ["5"],
                    }
                },
            },
            "check_reliability_vs_parent": {
                "transfer_time": {
                    "parent_failed_in_n_cases": 40,
                    "parent_n_cases": 60,
                    "child_failed_in_n_cases": 5,
                    "child_n_cases": 60,
                    "direction": "improved",
                    "significant": True,
                }
            },
            "per_case": [],
        }
        user, system = summ._build_prompt(aggregate, prior_memory=None)
        self.assertIn("Check/constraint reliability", user)
        self.assertIn("rare_check", user)
        self.assertIn("transfer_time", user)
        self.assertIn("SIGNIFICANT", user)
        self.assertIn("40/60", user)
        self.assertIn("5/60", user)
        # System prompt must instruct the LLM not to default to "nothing
        # helped" just because case_acc is 0.
        self.assertIn("do not default to 'nothing helped'", system)


if __name__ == "__main__":
    unittest.main()
