"""Tests for the metadata-surfacing fix to the "no plan" error message
(projects/travel_mas_refactored/adapter/scorer_impl.py::TravelCompositeScorer.score)
and its consumer (meta_agent/failure_summarizer.py::FailureSummarizer._aggregate).

Real gap this closes: when an evolving agent's own workflow returns an
empty AgentOutput.result, the scorer's error message ("agent produced no
plan") looked identical whether the underlying agent genuinely never
generated anything (e.g. a tool-calling budget exhaustion) or whether it
generated a complete, valid plan that the workflow's OWN validation logic
then discarded before conversion -- confirmed live against a real HGM run
(2026-09-02/03 curriculum sanity run on travel_mas_refactored): both
looked identical to the failure summarizer, which additionally read the
wrong CaseResult field (top-level .error, always None for a scorer-judged
failure) and never saw even the generic message at all.

    PYTHONPATH=. python3 -m unittest tests.test_travel_mas_refactored_no_plan_metadata
"""
from __future__ import annotations

import unittest
from types import SimpleNamespace

from projects.travel_mas_refactored.adapter import scorer_impl


class ScoreNoPlanMetadataTests(unittest.TestCase):
    def _score_with(self, result: str, metadata) -> dict:
        scorer = scorer_impl.TravelCompositeScorer()
        case = {"meta_info": {"some": "meta"}, "id": "1"}
        agent_output = SimpleNamespace(result=result, metadata=metadata)
        return scorer.score(case, agent_output)

    def test_agent_metadata_appended_when_plan_text_is_empty(self) -> None:
        out = self._score_with(
            "", {"sightseeing_failed": True, "budget_exhausted": True}
        )
        err = out["details"]["error"]
        self.assertTrue(err.startswith("plan conversion failed: agent produced no plan"))
        self.assertIn("sightseeing_failed", err)

    def test_no_plan_regex_still_matches_the_enhanced_message(self) -> None:
        out = self._score_with("", {"anything": 1})
        err = out["details"]["error"]
        self.assertTrue(scorer_impl._NO_PLAN_RE.search(err))

    def test_empty_metadata_reproduces_the_plain_message(self) -> None:
        out = self._score_with("", {})
        self.assertEqual(
            out["details"]["error"], "plan conversion failed: agent produced no plan"
        )

    def test_agent_output_without_metadata_attribute_is_a_noop(self) -> None:
        # AgentOutput.from_dict always sets .metadata to {}, but score()'s
        # own getattr(agent_output, "result", agent_output) fallback
        # anticipates a bare string too -- confirm that path (no
        # .metadata attribute at all) doesn't raise.
        scorer = scorer_impl.TravelCompositeScorer()
        case = {"meta_info": {"some": "meta"}, "id": "1"}
        out = scorer.score(case, "")
        self.assertEqual(
            out["details"]["error"], "plan conversion failed: agent produced no plan"
        )

    def test_distinguishes_two_real_root_causes(self) -> None:
        # The two causes actually observed live: budget exhaustion (agent
        # never produced anything) vs. a validation gate discarding an
        # already-generated plan -- must not collapse into identical text.
        budget_case = self._score_with(
            "", {"sightseeing_failed": True, "budget_exhausted": True}
        )
        gate_case = self._score_with(
            "",
            {
                "meal_validation_failed": True,
                "meal_validation_error": "Hotel name used as restaurant: X",
                "budget_exhausted": False,
            },
        )
        self.assertNotEqual(
            budget_case["details"]["error"], gate_case["details"]["error"]
        )
        self.assertIn("meal_validation_failed", gate_case["details"]["error"])
        self.assertIn("sightseeing_failed", budget_case["details"]["error"])


class AggregateHarnessChecksTests(unittest.TestCase):
    """TravelCompositeScorer.aggregate()'s harness_checks: a generic tally
    of agent_metadata boolean flags across no-plan cases specifically --
    not hardcoded field names, since agent_metadata's shape is owned by
    the mutable workflow code HGM keeps rewriting."""

    def _case(self, case_id, *, error=None, agent_metadata=None,
              failed_checks=None, dimension_scores=None):
        from meta_agent.models import CaseResult

        details: dict = {}
        if error is not None:
            details["error"] = error
        if agent_metadata is not None:
            details["agent_metadata"] = agent_metadata
        if failed_checks is not None:
            details["failed_checks"] = failed_checks
        if dimension_scores is not None:
            details["dimension_scores"] = dimension_scores
        return CaseResult(case_id=case_id, passed=False, score=0.0, details=details)

    def test_tallies_agent_metadata_flags_on_no_plan_cases_only(self) -> None:
        scorer = scorer_impl.TravelCompositeScorer()
        per_case = [
            self._case(
                "0",
                error="plan conversion failed: agent produced no plan",
                agent_metadata={"sightseeing_failed": True, "budget_exhausted": True},
            ),
            self._case(
                "1",
                error="plan conversion failed: agent produced no plan",
                agent_metadata={"meal_validation_failed": True, "budget_exhausted": False},
            ),
            # A real (non-no-plan) case's own agent_metadata must never be
            # tallied -- only no-plan cases contribute.
            self._case(
                "2",
                failed_checks=[], dimension_scores={},
                agent_metadata={"budget_exhausted": True},
            ),
        ]
        m = scorer.aggregate(per_case, trace_events=[])
        checks = dict(m["harness_checks"])
        self.assertEqual(checks.get("sightseeing_failed"), 1)
        self.assertEqual(checks.get("budget_exhausted"), 1)  # only case 0's True
        self.assertEqual(checks.get("meal_validation_failed"), 1)

    def test_empty_when_no_agent_metadata_present(self) -> None:
        scorer = scorer_impl.TravelCompositeScorer()
        per_case = [
            self._case("0", error="plan conversion failed: agent produced no plan"),
        ]
        m = scorer.aggregate(per_case, trace_events=[])
        self.assertEqual(m["harness_checks"], [])

    def test_no_plan_rate_key_unaffected_by_new_field(self) -> None:
        # Guard against a careless refactor accidentally renaming/nesting
        # no_plan_rate -- Curriculum.NO_PLAN_GOAL depends on this exact
        # top-level key.
        scorer = scorer_impl.TravelCompositeScorer()
        per_case = [
            self._case("0", error="plan conversion failed: agent produced no plan"),
            self._case("1", failed_checks=[], dimension_scores={}),
        ]
        m = scorer.aggregate(per_case, trace_events=[])
        self.assertAlmostEqual(m["no_plan_rate"], 0.5)


class AggregateCheckSemanticsTests(unittest.TestCase):
    """project_metrics["check_semantics"]: pairs each check/harness flag
    actually appearing in top_failed_checks/harness_checks with its
    human-readable description from error_semantics.json /
    harness_error_semantics.json -- independent of Curriculum, so the
    improvement proposer gets rich explanations even when curriculum is
    disabled (a real production config turns curriculum off but still
    wants the proposer to understand what a bare check name means)."""

    def _case(self, case_id, *, error=None, agent_metadata=None,
              failed_checks=None, dimension_scores=None):
        from meta_agent.models import CaseResult

        details: dict = {}
        if error is not None:
            details["error"] = error
        if agent_metadata is not None:
            details["agent_metadata"] = agent_metadata
        if failed_checks is not None:
            details["failed_checks"] = failed_checks
        if dimension_scores is not None:
            details["dimension_scores"] = dimension_scores
        return CaseResult(case_id=case_id, passed=False, score=0.0, details=details)

    def test_known_task_and_harness_checks_get_real_descriptions(self) -> None:
        scorer = scorer_impl.TravelCompositeScorer()
        per_case = [
            self._case(
                "0", error="plan conversion failed: agent produced no plan",
                agent_metadata={"budget_exhausted": True},
            ),
            self._case(
                "1",
                failed_checks=["commonsense:Sandbox Compliance:validated_meals"],
                dimension_scores={},
            ),
        ]
        m = scorer.aggregate(per_case, trace_events=[])
        semantics = dict(m["check_semantics"])
        self.assertIn(
            "commonsense:Sandbox Compliance:validated_meals", semantics
        )
        self.assertIn(
            "doesn't match ground truth",
            semantics["commonsense:Sandbox Compliance:validated_meals"],
        )
        self.assertIn("budget_exhausted", semantics)
        self.assertIn(
            "genuinely used up its entire tool-c",
            semantics["budget_exhausted"],
        )

    def test_unknown_check_is_omitted_not_fabricated(self) -> None:
        scorer = scorer_impl.TravelCompositeScorer()
        per_case = [
            self._case(
                "0", failed_checks=["totally_made_up_check_name"], dimension_scores={},
            ),
        ]
        m = scorer.aggregate(per_case, trace_events=[])
        semantics = dict(m["check_semantics"])
        self.assertNotIn("totally_made_up_check_name", semantics)

    def test_check_semantics_reaches_block_suggester_digest(self) -> None:
        from meta_agent.block_suggester import BlockSuggester
        from meta_agent.models import AgentFeedback, EvaluationResult, EvolutionStrategy

        scorer = scorer_impl.TravelCompositeScorer()
        per_case = [
            self._case(
                "0",
                failed_checks=["commonsense:Sandbox Compliance:validated_meals"],
                dimension_scores={},
            ),
            self._case("1", failed_checks=[], dimension_scores={}),
        ]
        metrics = scorer.aggregate(per_case, trace_events=[])

        suggester = BlockSuggester(llm_caller=lambda **kw: None)
        feedback = AgentFeedback(
            round_number=0, base_round=0,
            strategy=EvolutionStrategy(target_files=[], optimization_goal="", proposed_changes=""),
            eval_result=EvaluationResult(score=0.5, passed=1, failed=1),
            project_metrics=metrics,
        )
        digest = suggester._format_feedback_digest(feedback, None)
        self.assertIn("check_semantics", digest)
        self.assertIn("doesn't match ground truth", digest)


class AggregateToolInputTraceabilityTests(unittest.TestCase):
    """TravelCompositeScorer.aggregate()'s tool_input_untraced_rate /
    untraced_tool_inputs -- a directional signal for tool-call argument
    values that can't be traced to the case's own query, an earlier
    tool_call's own arguments, or an earlier tool_result's result_preview
    (same case, never looking ahead, never crossing cases). See
    scorer_impl.py's _tool_input_traceability docstring for why this is
    named "untraced" rather than "hallucinated"."""

    def _case(self, case_id, *, query=None):
        from meta_agent.models import CaseResult

        details: dict = {}
        if query is not None:
            details["query"] = query
        return CaseResult(case_id=case_id, passed=False, score=0.0, details=details)

    @staticmethod
    def _call(call_id, name, arguments, case_id):
        return {
            "kind": "tool_call",
            "payload": {"id": call_id, "name": name, "arguments": arguments, "case_id": case_id},
        }

    @staticmethod
    def _result(call_id, name, result_preview, case_id):
        return {
            "kind": "tool_result",
            "payload": {"id": call_id, "name": name, "result_preview": result_preview, "case_id": case_id},
        }

    def test_value_present_in_case_query_is_not_flagged(self) -> None:
        scorer = scorer_impl.TravelCompositeScorer()
        per_case = [self._case("1", query="trip to Chongqing")]
        trace_events = [
            self._call("a", "search_location", {"place_name": "Chongqing"}, "1"),
        ]
        m = scorer.aggregate(per_case, trace_events=trace_events)
        self.assertEqual(m["tool_input_untraced_rate"], 0.0)
        self.assertEqual(m["untraced_tool_inputs"], [])

    def test_value_traced_to_earlier_tool_result_is_not_flagged(self) -> None:
        scorer = scorer_impl.TravelCompositeScorer()
        per_case = [self._case("1", query="some request")]
        trace_events = [
            self._call("a", "recommend_attractions", {"city": "Chongqing"}, "1"),
            self._result(
                "a", "recommend_attractions",
                "Nanbin Road, Riverside Avenue on the southern bank", "1",
            ),
            self._call(
                "b", "query_attraction_details",
                {"attraction_name": "Nanbin Road, Riverside Avenue on the southern bank"},
                "1",
            ),
        ]
        m = scorer.aggregate(per_case, trace_events=trace_events)
        untraced_by_tool = dict(m["untraced_tool_inputs"])
        # "Chongqing" (call a) traces to nothing -> recommend_attractions
        # is untraced; the copied-from-the-prior-result value (call b) IS
        # traced -> query_attraction_details must NOT appear at all.
        self.assertIn("recommend_attractions", untraced_by_tool)
        self.assertNotIn("query_attraction_details", untraced_by_tool)

    def test_value_present_nowhere_is_flagged_under_the_right_tool(self) -> None:
        scorer = scorer_impl.TravelCompositeScorer()
        per_case = [self._case("1", query="unrelated request text")]
        trace_events = [
            self._call("a", "query_restaurant_details", {"restaurant_name": "Sunset & You Bistro"}, "1"),
        ]
        m = scorer.aggregate(per_case, trace_events=trace_events)
        self.assertEqual(m["tool_input_untraced_rate"], 1.0)
        self.assertEqual(m["untraced_tool_inputs"], [("query_restaurant_details", 1)])

    def test_value_only_present_in_a_later_call_is_still_flagged(self) -> None:
        # No look-ahead: a value that only becomes traceable LATER in the
        # same case must not retroactively clear an earlier call's flag.
        scorer = scorer_impl.TravelCompositeScorer()
        per_case = [self._case("1", query="request")]
        trace_events = [
            self._call("a", "query_attraction_details", {"attraction_name": "Mystery Place"}, "1"),
            self._result("a", "query_attraction_details", "not found", "1"),
            self._call("b", "recommend_attractions", {"city": "X"}, "1"),
            self._result("b", "recommend_attractions", "Mystery Place is great", "1"),
        ]
        m = scorer.aggregate(per_case, trace_events=trace_events)
        untraced_by_tool = dict(m["untraced_tool_inputs"])
        self.assertIn("query_attraction_details", untraced_by_tool)

    def test_empty_trace_events_degrades_to_zero_rate_and_empty_list(self) -> None:
        scorer = scorer_impl.TravelCompositeScorer()
        per_case = [self._case("1", query="request")]
        m = scorer.aggregate(per_case, trace_events=[])
        self.assertEqual(m["tool_input_untraced_rate"], 0.0)
        self.assertEqual(m["untraced_tool_inputs"], [])

    def test_malformed_events_are_skipped_without_crashing_or_corrupting_others(self) -> None:
        scorer = scorer_impl.TravelCompositeScorer()
        per_case = [self._case("1", query="request"), self._case("2", query="another request")]
        trace_events = [
            {"kind": "tool_call", "payload": "not_a_dict"},
            {"kind": "tool_call"},  # missing payload entirely
            "not_an_event_dict_at_all",
            self._call("z", "query_hotel_info", {"destination": "SomewhereFarAway"}, "2"),
        ]
        m = scorer.aggregate(per_case, trace_events=trace_events)
        untraced_by_tool = dict(m["untraced_tool_inputs"])
        self.assertEqual(untraced_by_tool.get("query_hotel_info"), 1)


if __name__ == "__main__":
    unittest.main()
