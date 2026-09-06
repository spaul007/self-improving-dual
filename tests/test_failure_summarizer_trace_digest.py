"""Tests for FailureSummarizer's opt-in trace digest
(meta_agent/failure_summarizer.py, trace_digest_case_count and friends).

Real gap this closes: block_suggester/agent_editor only ever see a case's
final score/error string, never what actually happened at the tool-call
level. Confirmed live this session (round_000 of a production run, case
34): a hard-constraint error just named a missing attraction ("Nanbin
Road"), but reading trace.jsonl directly showed the real story --
recommend_attractions returned an attraction as unstructured free text, the
agent copied the ENTIRE descriptive sentence into query_attraction_details,
that 404'd, and the agent silently substituted an unrelated already-
verified attraction instead of retrying. A generic "verify names better"
diagnosis would have missed both real, separately-fixable bugs this
revealed. This feature lets a small, configurable number of the failure
summarizer's already-selected hardest cases carry that same ordered
tool_call -> tool_result narrative into the digest text.

    PYTHONPATH=. python3 -m unittest tests.test_failure_summarizer_trace_digest
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from meta_agent.failure_summarizer import FailureSummarizer
from meta_agent.models import CaseResult, EvaluationResult


def _case(case_id: str, score: float, query: str = "q") -> CaseResult:
    return CaseResult(
        case_id=case_id, passed=False, score=score,
        details={"query": query, "raw_result": ""},
    )


def _write_trace(path: Path, events: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8"
    )


class DefaultDisabledTests(unittest.TestCase):
    """trace_digest_case_count=0 (the default) must be byte-identical to
    behavior before this feature existed."""

    def test_no_tool_trace_key_anywhere_by_default(self) -> None:
        summarizer = FailureSummarizer(llm_caller=lambda **kw: None)
        self.assertEqual(summarizer.trace_digest_case_count, 0)
        case = _case("1", 0.0)
        aggregate = summarizer._aggregate([case], [case], node_id=0)
        self.assertNotIn("tool_trace", aggregate["cases"][0])

    def test_summarize_ignores_trace_jsonl_when_disabled(self) -> None:
        with TemporaryDirectory() as tmp:
            round_dir = Path(tmp)
            _write_trace(
                round_dir / "logs" / "trace.jsonl",
                [{"kind": "tool_call", "payload": {"id": "a", "name": "x", "arguments": {}, "case_id": "1"}}],
            )
            summarizer = FailureSummarizer(llm_caller=lambda **kw: type("R", (), {"content": "## Main failure patterns\nx\n## Hardest cases\ny"})())
            eval_result = EvaluationResult(score=0.0, per_case=[_case("1", 0.0)])
            summarizer.summarize(eval_result=eval_result, round_dir=round_dir, node_id=0)
            aggregate = json.loads((round_dir / "failure_summary_aggregate.json").read_text())
            self.assertNotIn("tool_trace", aggregate["cases"][0])


class RenderCaseToolTraceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.summarizer = FailureSummarizer(llm_caller=lambda **kw: None)

    def test_well_formed_pairs_render_numbered(self) -> None:
        events = [
            {"kind": "tool_call", "payload": {"id": "a", "name": "search_location", "arguments": {"place_name": "X"}, "case_id": "1"}},
            {"kind": "tool_result", "payload": {"id": "a", "name": "search_location", "result_preview": "found it", "case_id": "1"}},
        ]
        text = self.summarizer._render_case_tool_trace(events, max_calls=15, preview_chars=200)
        self.assertEqual(
            text,
            "1. search_location({'place_name': 'X'})\n   -> found it",
        )

    def test_unmatched_call_renders_no_result_without_crashing(self) -> None:
        events = [
            {"kind": "tool_call", "payload": {"id": "a", "name": "search_location", "arguments": {}, "case_id": "1"}},
        ]
        text = self.summarizer._render_case_tool_trace(events, max_calls=15, preview_chars=200)
        self.assertIn("-> (no result)", text)

    def test_capped_at_max_calls(self) -> None:
        events = []
        for i in range(20):
            events.append({"kind": "tool_call", "payload": {"id": str(i), "name": "t", "arguments": {}, "case_id": "1"}})
        text = self.summarizer._render_case_tool_trace(events, max_calls=3, preview_chars=200)
        self.assertEqual(text.count("\n") + 1, 3 * 2)  # 3 calls x 2 lines each

    def test_result_preview_truncated_to_preview_chars(self) -> None:
        events = [
            {"kind": "tool_call", "payload": {"id": "a", "name": "t", "arguments": {}, "case_id": "1"}},
            {"kind": "tool_result", "payload": {"id": "a", "name": "t", "result_preview": "x" * 500, "case_id": "1"}},
        ]
        text = self.summarizer._render_case_tool_trace(events, max_calls=15, preview_chars=10)
        self.assertIn("x" * 10, text)
        self.assertNotIn("x" * 11, text)


class SummarizeEndToEndTests(unittest.TestCase):
    def _llm(self, **kw):
        return type("R", (), {"content": "## Main failure patterns\nx\n## Hardest cases\ny"})()

    def test_only_first_n_of_shown_get_a_digest(self) -> None:
        with TemporaryDirectory() as tmp:
            round_dir = Path(tmp)
            events = [
                {"kind": "tool_call", "payload": {"id": "a1", "name": "t1", "arguments": {"x": "worst"}, "case_id": "1"}},
                {"kind": "tool_call", "payload": {"id": "a2", "name": "t2", "arguments": {"x": "mid"}, "case_id": "2"}},
                {"kind": "tool_call", "payload": {"id": "a3", "name": "t3", "arguments": {"x": "best"}, "case_id": "3"}},
            ]
            _write_trace(round_dir / "logs" / "trace.jsonl", events)
            summarizer = FailureSummarizer(llm_caller=self._llm, trace_digest_case_count=2)
            # Worst-scoring first: case 1 (0.0), case 2 (0.2), case 3 (0.5).
            eval_result = EvaluationResult(
                score=0.0,
                per_case=[_case("1", 0.0), _case("2", 0.2), _case("3", 0.5)],
            )
            summarizer.summarize(eval_result=eval_result, round_dir=round_dir, node_id=0)
            aggregate = json.loads((round_dir / "failure_summary_aggregate.json").read_text())
            by_id = {c["case_id"]: c for c in aggregate["cases"]}
            self.assertIn("tool_trace", by_id["1"])
            self.assertIn("tool_trace", by_id["2"])
            self.assertNotIn("tool_trace", by_id["3"])

    def test_missing_trace_file_degrades_gracefully(self) -> None:
        with TemporaryDirectory() as tmp:
            round_dir = Path(tmp)
            # No logs/trace.jsonl written at all.
            summarizer = FailureSummarizer(llm_caller=self._llm, trace_digest_case_count=1)
            eval_result = EvaluationResult(score=0.0, per_case=[_case("1", 0.0)])
            path = summarizer.summarize(eval_result=eval_result, round_dir=round_dir, node_id=0)
            self.assertIsNotNone(path)
            aggregate = json.loads((round_dir / "failure_summary_aggregate.json").read_text())
            self.assertNotIn("tool_trace", aggregate["cases"][0])

    def test_malformed_trace_file_degrades_gracefully(self) -> None:
        with TemporaryDirectory() as tmp:
            round_dir = Path(tmp)
            (round_dir / "logs").mkdir(parents=True)
            (round_dir / "logs" / "trace.jsonl").write_text("{not valid json\n", encoding="utf-8")
            summarizer = FailureSummarizer(llm_caller=self._llm, trace_digest_case_count=1)
            eval_result = EvaluationResult(score=0.0, per_case=[_case("1", 0.0)])
            path = summarizer.summarize(eval_result=eval_result, round_dir=round_dir, node_id=0)
            self.assertIsNotNone(path)


if __name__ == "__main__":
    unittest.main()
