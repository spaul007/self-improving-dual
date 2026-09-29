"""Tests for meta_agent/unit_selector.py's UnitSelector -- fake llm_caller,
no real API key needed. Mirrors tests/test_block_suggester_agentic_mode.py's
fake-tool-call convention.

    PYTHONPATH=. python3 -m unittest tests.test_unit_selector
"""
from __future__ import annotations

import unittest
from types import SimpleNamespace

from meta_agent.unit_selector import UnitSelector


def _call(name: str, arguments: dict) -> SimpleNamespace:
    return SimpleNamespace(name=name, arguments=arguments)


UNIT_COUNTS = {"Business Hours": 49, "Activity Diversity": 20}
DESCRIPTIONS = {
    "Business Hours": "Checks in this unit: dining_within_service_hours: 27",
    "Activity Diversity": "Checks in this unit: diverse_attraction_options: 20",
}


class UnitSelectorTests(unittest.TestCase):
    def test_valid_propose_unit_call_returns_that_unit(self) -> None:
        def fake_llm(**kwargs):
            self.assertIn("tools", kwargs)
            self.assertEqual(kwargs["tools"][0]["name"], "propose_unit")
            return SimpleNamespace(
                content="", tool_calls=[_call("propose_unit", {"unit": "Activity Diversity", "rationale": "single root cause"})],
            )

        selector = UnitSelector(fake_llm)
        chosen = selector.choose(UNIT_COUNTS, DESCRIPTIONS, {}, 2)
        self.assertEqual(chosen, "Activity Diversity")

    def test_unknown_unit_in_tool_call_returns_none(self) -> None:
        def fake_llm(**kwargs):
            return SimpleNamespace(
                content="", tool_calls=[_call("propose_unit", {"unit": "Not A Real Unit", "rationale": "..."})],
            )

        selector = UnitSelector(fake_llm)
        self.assertIsNone(selector.choose(UNIT_COUNTS, DESCRIPTIONS, {}, 2))

    def test_no_tool_call_returns_none(self) -> None:
        def fake_llm(**kwargs):
            return SimpleNamespace(content="I couldn't decide.", tool_calls=[])

        selector = UnitSelector(fake_llm)
        self.assertIsNone(selector.choose(UNIT_COUNTS, DESCRIPTIONS, {}, 2))

    def test_wrong_tool_name_returns_none(self) -> None:
        def fake_llm(**kwargs):
            return SimpleNamespace(
                content="", tool_calls=[_call("some_other_tool", {"unit": "Business Hours"})],
            )

        selector = UnitSelector(fake_llm)
        self.assertIsNone(selector.choose(UNIT_COUNTS, DESCRIPTIONS, {}, 2))

    def test_llm_caller_exception_propagates(self) -> None:
        # Fallback handling lives in unit_curriculum.choose_unit(), not
        # here -- this class's contract is simple: return a valid unit,
        # None, or let the exception through.
        def boom(**kwargs):
            raise RuntimeError("network error")

        selector = UnitSelector(boom)
        with self.assertRaises(RuntimeError):
            selector.choose(UNIT_COUNTS, DESCRIPTIONS, {}, 2)

    def test_candidate_enum_matches_given_unit_counts(self) -> None:
        seen = {}

        def fake_llm(**kwargs):
            seen["enum"] = kwargs["tools"][0]["input_schema"]["properties"]["unit"]["enum"]
            return SimpleNamespace(content="", tool_calls=[])

        selector = UnitSelector(fake_llm)
        selector.choose(UNIT_COUNTS, DESCRIPTIONS, {}, 2)
        self.assertEqual(set(seen["enum"]), set(UNIT_COUNTS))

    def test_model_and_reasoning_effort_forwarded(self) -> None:
        seen = {}

        def fake_llm(**kwargs):
            seen.update(kwargs)
            return SimpleNamespace(content="", tool_calls=[])

        selector = UnitSelector(fake_llm, model="deepseek/deepseek-v4-pro", reasoning_effort="medium")
        selector.choose(UNIT_COUNTS, DESCRIPTIONS, {}, 2)
        self.assertEqual(seen["model"], "deepseek/deepseek-v4-pro")
        self.assertEqual(seen["reasoning_effort"], "medium")
        self.assertNotIn("temperature", seen)  # reasoning_effort set -> no temperature, mirrors BlockSuggester


if __name__ == "__main__":
    unittest.main()
