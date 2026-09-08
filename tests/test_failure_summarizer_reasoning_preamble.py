"""Tests for FailureSummarizer's reasoning-preamble stripping
(meta_agent/failure_summarizer.py, _strip_reasoning_preamble).

Real bug this guards against: confirmed live 2026-09-07 with
Qwen3.5-122B-A10B (via OpenRouter) as the failure summarizer -- despite the
system prompt demanding "a concise markdown summary with exactly two
sections... stay under 250 words", the model's raw completion sometimes
leaked its full chain-of-thought / draft-revision scratchpad ("Thinking
Process: 1. Analyze the Request... *Draft:* ... *Revised Draft:* ...")
ahead of the actual answer, with the expected "## Main failure patterns"
header appearing multiple times (once per draft). Nothing enforced the
output contract on the way out -- _call_llm returned response.content
completely raw. block_suggester/editor then received this bloated,
mostly-redundant text as "the cross-case failure summary."

    PYTHONPATH=. python3 -m unittest tests.test_failure_summarizer_reasoning_preamble
"""
from __future__ import annotations

import unittest

from meta_agent.failure_summarizer import FailureSummarizer, _strip_reasoning_preamble


class StripReasoningPreambleTests(unittest.TestCase):
    def test_clean_response_is_unchanged(self) -> None:
        text = "## Main failure patterns\n1. thing\n\n## Hardest cases\ncase 1\n"
        self.assertEqual(_strip_reasoning_preamble(text), text)

    def test_no_header_at_all_is_a_noop(self) -> None:
        text = "I couldn't find a clear pattern in these cases."
        self.assertEqual(_strip_reasoning_preamble(text), text)

    def test_empty_string_is_a_noop(self) -> None:
        self.assertEqual(_strip_reasoning_preamble(""), "")

    def test_single_thinking_preamble_is_stripped(self) -> None:
        text = (
            "Thinking Process:\n"
            "1. Analyze the request...\n"
            "2. Look at the cases...\n\n"
            "## Main failure patterns\n"
            "1. Missing tag: cases 1, 2\n\n"
            "## Hardest cases\n"
            "case 1: description\n"
        )
        result = _strip_reasoning_preamble(text)
        self.assertTrue(result.startswith("## Main failure patterns"))
        self.assertNotIn("Thinking Process", result)
        self.assertIn("case 1: description", result)

    def test_multiple_draft_headers_keeps_only_the_last(self) -> None:
        text = (
            "Thinking Process:\n"
            "*Draft:*\n"
            "## Main failure patterns\n"
            "1. early wrong draft: cases 9, 9\n\n"
            "## Hardest cases\n"
            "case 9: wrong draft description\n\n"
            "*Revised Draft:*\n"
            "## Main failure patterns\n"
            "1. final correct pattern: cases 1, 2\n\n"
            "## Hardest cases\n"
            "case 1: final correct description\n"
        )
        result = _strip_reasoning_preamble(text)
        self.assertEqual(result.count("## Main failure patterns"), 1)
        self.assertIn("final correct pattern", result)
        self.assertIn("final correct description", result)
        self.assertNotIn("early wrong draft", result)
        self.assertNotIn("wrong draft description", result)

    def test_tolerates_heading_level_and_case(self) -> None:
        text = "noise before\n### MAIN FAILURE PATTERNS\nreal content\n"
        result = _strip_reasoning_preamble(text)
        self.assertTrue(result.startswith("### MAIN FAILURE PATTERNS"))
        self.assertNotIn("noise before", result)


class CallLlmAppliesStrippingTests(unittest.TestCase):
    def test_call_llm_strips_preamble_from_the_real_response(self) -> None:
        raw = (
            "Thinking Process: lots of deliberation here\n\n"
            "## Main failure patterns\n1. x\n\n## Hardest cases\ny\n"
        )

        def fake_llm(**kwargs):
            return type("R", (), {"content": raw})()

        summarizer = FailureSummarizer(llm_caller=fake_llm)
        result = summarizer._call_llm(system="sys", user="usr")
        self.assertTrue(result.startswith("## Main failure patterns"))
        self.assertNotIn("Thinking Process", result)

    def test_call_llm_passes_through_a_clean_response_unchanged(self) -> None:
        raw = "## Main failure patterns\n1. x\n\n## Hardest cases\ny\n"

        def fake_llm(**kwargs):
            return type("R", (), {"content": raw})()

        summarizer = FailureSummarizer(llm_caller=fake_llm)
        result = summarizer._call_llm(system="sys", user="usr")
        self.assertEqual(result, raw)


if __name__ == "__main__":
    unittest.main()
