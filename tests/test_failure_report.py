"""Tests for the self-improving error-log rendering (level + objectives block).

    PYTHONPATH=. python3 -m unittest tests.test_failure_report
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from meta_agent.failure_report import build_failure_report, render_failure_report


def _case(cid, level, score, *, query, plan, passed=False, error_log=None):
    det = {"level": level, "query": query, "raw_result": plan,
           "level_objective": f"L{level} — _SYSTEM_PROMPT_LEVEL_{level} in workflow.py"}
    if error_log is not None:
        det["error_log"] = error_log
    return SimpleNamespace(case_id=cid, score=score, passed=passed, error=None, details=det)


class FailureReportRenderTest(unittest.TestCase):
    def _report(self):
        cases = [
            _case("L1-1", 1, 0.3, query="buy a red nike top", plan='{"items": []}',
                  error_log="[L1] fails the required color constraint (color equals Red).\n[L1] size 'L' but profile requires 'XS'."),
            _case("L3-2", 3, 0.5, query="weekend coupons", plan='{"items": [], "used_coupons": []}'),
        ]
        categories = [
            {"category_id": "missing_feature", "category_name": "Unmet product feature",
             "category_type": "missing_feature", "num_failing_samples": 1, "total_samples": 2,
             "failure_rate": 0.5, "representative_errors": [
                 {"sample_id": "L1-1", "checks_failed": ["color"],
                  "messages": ["[L1] fails the required color constraint."]}]},
            {"category_id": "suboptimal_coupon", "category_name": "Suboptimal coupon usage",
             "category_type": "suboptimal_coupon", "num_failing_samples": 1, "total_samples": 2,
             "failure_rate": 0.5, "representative_errors": [
                 {"sample_id": "L3-2", "checks_failed": ["coupon"],
                  "messages": ["[L3] Missing the optimal coupon(s): VIP."]}]},
        ]
        return build_failure_report(cases, categories), cases

    def test_full_report_uses_pointer_keeps_query_plan_and_level_tags(self) -> None:
        report, _ = self._report()
        text = render_failure_report(report)
        # NO full-text objectives block any more.
        self.assertNotIn("Task objectives by level:", text)
        # per-example pointer to the level's system prompt (not the prompt text)
        self.assertIn("governing prompt: L1 — _SYSTEM_PROMPT_LEVEL_1 in workflow.py", text)
        self.assertIn("governing prompt: L3 — _SYSTEM_PROMPT_LEVEL_3 in workflow.py", text)
        # per-example header carries the category id AND the [L#] tag
        self.assertIn("[missing_feature] [L1] case L1-1", text)
        self.assertIn("[suboptimal_coupon] [L3] case L3-2", text)
        # query + plan still rendered
        self.assertIn("query: buy a red nike top", text)
        self.assertIn("agent plan:", text)
        # the FULL per-case error log renders (multi-line, indented)
        self.assertIn("error log: [L1] fails the required color constraint", text)
        self.assertIn("\n    [L1] size 'L' but profile requires 'XS'.", text)

    def test_focused_report_pointer_and_level_tag(self) -> None:
        _, cases = self._report()
        cat = {"category_id": "suboptimal_coupon", "category_name": "Suboptimal coupon usage",
               "category_type": "suboptimal_coupon", "num_failing_samples": 1, "total_samples": 2,
               "failure_rate": 0.5, "representative_errors": [
                   {"sample_id": "L3-2", "checks_failed": ["coupon"],
                    "messages": ["[L3] Missing the optimal coupon(s): VIP."]}]}
        report = build_failure_report(cases, [cat], focus_category_id="suboptimal_coupon")
        text = render_failure_report(report)
        self.assertIn("## Assigned failure category: Suboptimal coupon usage", text)
        self.assertNotIn("Task objectives by level:", text)
        self.assertIn("governing prompt: L3 — _SYSTEM_PROMPT_LEVEL_3 in workflow.py", text)
        self.assertIn("[L3] case L3-2", text)

    def test_perfect_score_case_is_excluded_from_examples(self) -> None:
        # A score-1.0 / passed case a category lists must NOT appear as an example.
        cases = [
            _case("L3-9", 3, 1.0, query="all coupons used", plan='{"items": []}', passed=True,
                  error_log="[L3] Left ¥120 of coupon savings unused."),
            _case("L3-2", 3, 0.4, query="weekend coupons", plan='{"items": []}',
                  error_log="[L3] Missing the optimal coupon(s): VIP."),
        ]
        cat = {"category_id": "suboptimal_coupon", "category_name": "Suboptimal coupon usage",
               "category_type": "suboptimal_coupon", "num_failing_samples": 2, "total_samples": 2,
               "failure_rate": 1.0, "representative_errors": [
                   {"sample_id": "L3-9", "checks_failed": ["coupon"], "messages": ["x"]},
                   {"sample_id": "L3-2", "checks_failed": ["coupon"], "messages": ["y"]}]}
        # both full and focused modes must drop the perfect case
        full = render_failure_report(build_failure_report(cases, [cat]))
        focused = render_failure_report(
            build_failure_report(cases, [cat], focus_category_id="suboptimal_coupon"))
        for text in (full, focused):
            self.assertNotIn("L3-9", text)
            self.assertIn("L3-2", text)


if __name__ == "__main__":
    unittest.main()
