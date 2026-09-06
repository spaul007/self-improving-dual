"""Tests for FailureSummarizer._aggregate's error-field fix
(meta_agent/failure_summarizer.py).

Real bug this guards against: _aggregate built each case record with
"error": c.error -- CaseResult's TOP-LEVEL error field, reserved for
harness-level crashes and always None for a scorer-judged failure (e.g.
"plan conversion failed: ..."). The actual descriptive message lives in
c.details["error"] instead, a different field _aggregate never read. So
every scorer-judged failure (the common case) reached the LLM-facing
failure summary as error=None, indistinguishable from a case with no
error info at all -- confirmed live against a real HGM run where this
made a "no plan" case caused by a self-inflicted validation gate look
identical to one caused by genuine tool-budget exhaustion.

    PYTHONPATH=. python3 -m unittest tests.test_failure_summarizer_error_field
"""
from __future__ import annotations

import unittest

from meta_agent.failure_summarizer import FailureSummarizer
from meta_agent.models import CaseResult


class AggregateErrorFieldTests(unittest.TestCase):
    def setUp(self) -> None:
        self.summarizer = FailureSummarizer(llm_caller=lambda **kw: None)

    def test_details_error_is_preferred_over_top_level_none(self) -> None:
        case = CaseResult(
            case_id="1",
            passed=False,
            score=0.0,
            error=None,
            details={"error": "plan conversion failed: agent produced no plan"},
        )
        aggregate = self.summarizer._aggregate([case], [case], node_id=0)
        self.assertEqual(
            aggregate["cases"][0]["error"],
            "plan conversion failed: agent produced no plan",
        )

    def test_top_level_error_used_when_details_has_none(self) -> None:
        # A genuine harness-level crash: details is empty, but the
        # top-level field carries the real message -- must not be lost.
        case = CaseResult(
            case_id="2", passed=False, score=0.0,
            error="child exit code 1", details={},
        )
        aggregate = self.summarizer._aggregate([case], [case], node_id=0)
        self.assertEqual(aggregate["cases"][0]["error"], "child exit code 1")

    def test_neither_present_is_none(self) -> None:
        case = CaseResult(case_id="3", passed=False, score=0.5, details={})
        aggregate = self.summarizer._aggregate([case], [case], node_id=0)
        self.assertIsNone(aggregate["cases"][0]["error"])

    def test_metadata_folded_error_reaches_the_rendered_prompt(self) -> None:
        case = CaseResult(
            case_id="4", passed=False, score=0.0,
            details={
                "error": (
                    "plan conversion failed: agent produced no plan "
                    "(agent_metadata: {'meal_validation_failed': True})"
                ),
                "query": "q", "raw_result": "",
            },
        )
        aggregate = self.summarizer._aggregate([case], [case], node_id=0)
        prompt_user, _ = self.summarizer._build_prompt(aggregate)
        self.assertIn("meal_validation_failed", prompt_user)


if __name__ == "__main__":
    unittest.main()
