"""Tests for conversion-error tracking added to
projects/travel_mas_refactored/adapter/scorer_impl.py::TravelCompositeScorer,
plus the new TRAVEL_CONVERT_ENABLE_THINKING env var on _convert_plan_to_json.

Real gap this closes: a real conversion-infra failure (timeout/parse/API
error against TRAVEL_CONVERT_MODEL, with the agent's plan text perfectly
fine) looked IDENTICAL to a genuine no-plan case (agent produced nothing)
in project_metrics -- both only ever showed up in no_plan_rate. Found live
investigating a seed pre-eval pass where 8/60 cases scored a hard 0.0
purely from conversion timeouts on an otherwise-good plan (2026-09-29).
conversion_error_type/conversion_error_rate/conversion_error_types are
purely additive: no_plan_rate's own counting/semantics are unchanged.

    PYTHONPATH=. python3 -m unittest tests.test_travel_mas_refactored_conversion_error_tracking
"""
from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from projects.travel_mas_refactored.adapter import scorer_impl


class ConversionErrorClassificationTests(unittest.TestCase):
    def test_timeout_classified(self) -> None:
        self.assertEqual(
            scorer_impl._classify_conversion_error("conversion timed out after 603s (2 attempt(s) made, last error: None)"),
            "timeout",
        )

    def test_json_parse_classified(self) -> None:
        self.assertEqual(
            scorer_impl._classify_conversion_error("could not parse JSON on attempt 1: Expecting value"),
            "json_parse",
        )

    def test_api_error_classified(self) -> None:
        self.assertEqual(
            scorer_impl._classify_conversion_error("openai call failed (attempt 1): APIStatusError(...)"),
            "api_error",
        )

    def test_unrecognized_error_classified_other(self) -> None:
        self.assertEqual(scorer_impl._classify_conversion_error("something unexpected"), "other")

    def test_none_classified_other(self) -> None:
        self.assertEqual(scorer_impl._classify_conversion_error(None), "other")


class ScoreConversionErrorFieldTests(unittest.TestCase):
    def _score_with_convert_result(self, result: str, convert_return) -> dict:
        scorer = scorer_impl.TravelCompositeScorer()
        case = {"meta_info": {"some": "meta"}, "id": "1"}
        agent_output = SimpleNamespace(result=result, metadata={})
        with patch.object(scorer_impl, "_convert_plan_to_json", return_value=convert_return):
            return scorer.score(case, agent_output)

    def test_real_plan_conversion_timeout_sets_conversion_error_type(self) -> None:
        out = self._score_with_convert_result(
            "Day 1: a perfectly fine plan",
            (None, "conversion timed out after 603s (2 attempt(s) made, last error: None)"),
        )
        self.assertEqual(out["details"]["conversion_error_type"], "timeout")
        self.assertTrue(out["details"]["error"].startswith("plan conversion failed"))

    def test_empty_plan_leaves_conversion_error_type_none(self) -> None:
        out = self._score_with_convert_result("", (None, "agent produced no plan"))
        self.assertIsNone(out["details"]["conversion_error_type"])

    def test_successful_conversion_has_no_conversion_error_type_key_issue(self) -> None:
        # Successful path's details dict doesn't set conversion_error_type
        # at all (only _zero_result does) -- aggregate() must tolerate
        # that via .get(), not assume the key is always present.
        scorer = scorer_impl.TravelCompositeScorer()
        case = {
            "meta_info": {"days": 1},
            "id": "1",
        }
        agent_output = SimpleNamespace(result="Day 1: plan", metadata={})
        fake_parsed = {"daily_plans": []}
        with patch.object(scorer_impl, "_convert_plan_to_json", return_value=(fake_parsed, None)), \
             patch.object(scorer_impl, "_evaluate", return_value={
                 "commonsense_score": 1.0, "hard_score": 1.0, "composite_score": 1.0,
                 "dimension_scores": {}, "dimension_details": {}, "hard_constraints": {},
                 "failed_checks": [], "passed": True,
             }):
            out = scorer.score(case, agent_output)
        self.assertNotIn("conversion_error_type", out["details"])


class AggregateConversionErrorRateTests(unittest.TestCase):
    def _case(self, details: dict):
        return SimpleNamespace(details=details)

    def test_conversion_errors_tallied_separately_from_genuine_no_plan(self) -> None:
        scorer = scorer_impl.TravelCompositeScorer()
        per_case = [
            self._case({"error": "plan conversion failed: agent produced no plan",
                        "conversion_error_type": None}),
            self._case({"error": "plan conversion failed: conversion timed out after 603s (...)",
                        "conversion_error_type": "timeout"}),
            self._case({"error": "plan conversion failed: conversion timed out after 603s (...)",
                        "conversion_error_type": "timeout"}),
            self._case({"error": "plan conversion failed: could not parse JSON on attempt 1: bad",
                        "conversion_error_type": "json_parse"}),
            self._case({"failed_checks": [], "dimension_scores": {}}),  # a normal scored case
        ]
        metrics = scorer.aggregate(per_case, [])
        # no_plan_rate stays byte-identical to its pre-existing definition:
        # ALL 4 _NO_PLAN_RE matches, conversion errors included.
        self.assertAlmostEqual(metrics["no_plan_rate"], 4 / 5)
        # conversion_error_rate only counts the 3 with a real
        # conversion_error_type (timeout/timeout/json_parse), not the
        # genuine no-plan case.
        self.assertAlmostEqual(metrics["conversion_error_rate"], 3 / 5)
        types = dict(metrics["conversion_error_types"])
        self.assertEqual(types, {"timeout": 2, "json_parse": 1})

    def test_no_conversion_errors_at_all(self) -> None:
        scorer = scorer_impl.TravelCompositeScorer()
        per_case = [self._case({"failed_checks": [], "dimension_scores": {}})]
        metrics = scorer.aggregate(per_case, [])
        self.assertEqual(metrics["conversion_error_rate"], 0.0)
        self.assertEqual(metrics["conversion_error_types"], [])


class ConvertPlanEnableThinkingTests(unittest.TestCase):
    def setUp(self) -> None:
        patcher = patch.dict(
            "os.environ",
            {"TRAVEL_CONVERT_BASE_URL": "http://fake-host:8000/v1"},
            clear=False,
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def _run_with_env(self, env_extra: dict, expect_extra_body):
        fake_response = MagicMock()
        fake_response.choices = [MagicMock(message=MagicMock(content="<JSON>{}</JSON>"))]
        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = fake_response
        with patch.dict("os.environ", env_extra, clear=False), \
             patch("openai.OpenAI", return_value=fake_client):
            result, err = scorer_impl._convert_plan_to_json("a real plan")
        self.assertIsNone(err)
        self.assertEqual(result, {})
        call_kwargs = fake_client.chat.completions.create.call_args.kwargs
        if expect_extra_body is None:
            self.assertNotIn("extra_body", call_kwargs)
        else:
            self.assertEqual(call_kwargs["extra_body"], expect_extra_body)

    def test_unset_sends_no_extra_body(self) -> None:
        self._run_with_env({}, None)

    def test_false_disables_thinking(self) -> None:
        self._run_with_env(
            {"TRAVEL_CONVERT_ENABLE_THINKING": "false"},
            {"chat_template_kwargs": {"enable_thinking": False}},
        )

    def test_true_enables_thinking(self) -> None:
        self._run_with_env(
            {"TRAVEL_CONVERT_ENABLE_THINKING": "true"},
            {"chat_template_kwargs": {"enable_thinking": True}},
        )


if __name__ == "__main__":
    unittest.main()
