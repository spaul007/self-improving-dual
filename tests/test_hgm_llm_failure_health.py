"""Tests for HGMManager._record_batch and its two opt-in params
(exclude_llm_call_failures, llm_call_failure_threshold_pct) -- excluding
OpenRouter LLM-call terminal-failure-corrupted cases from a node's
reward, and the always-on llm_failure_health.json artifact / loud
>threshold% warning. See meta_agent/llm_failure_health.py for the
underlying trace.jsonl parsing this builds on.

Mirrors tests/test_hgm_active_blocks.py's style (direct HGMManager()
construction, no LLM/evaluator needed) -- _record_batch only needs a
real HGMNode with a real round_dir (for the trace.jsonl read /
llm_failure_health.json write) and a fake EvaluationResult.

    PYTHONPATH=. python3 -m unittest tests.test_hgm_llm_failure_health
"""
from __future__ import annotations

import io
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory

from meta_agent.managers.hgm import HGMManager
from meta_agent.managers.hgm_tree import HGMNode
from meta_agent.models import CaseResult, EvaluationResult


def _write_trace(round_dir: Path, events: list[dict]) -> None:
    trace_path = round_dir / "logs" / "trace.jsonl"
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path.write_text(
        "\n".join(json.dumps(e) for e in events) + ("\n" if events else ""),
        encoding="utf-8",
    )


def _case(case_id: str, score: float) -> CaseResult:
    return CaseResult(case_id=case_id, passed=score >= 0.5, score=score)


def _terminal_failure_event(case_id: str) -> dict:
    return {
        "kind": "llm_response",
        "payload": {
            "case_id": case_id,
            "stop_reason": "failed",
            "response_error_code": "server_error",
        },
    }


class RecordBatchDefaultBehaviorTests(unittest.TestCase):
    """exclude_llm_call_failures=False (default) -- byte-for-byte the
    pre-feature behavior: every case is recorded, regardless of what
    trace.jsonl says."""

    def test_every_case_recorded_even_with_a_terminal_failure_in_trace(self) -> None:
        with TemporaryDirectory() as d:
            round_dir = Path(d)
            _write_trace(round_dir, [_terminal_failure_event("2")])
            node = HGMNode(0, None, round_dir)
            result = EvaluationResult(
                score=0.5,
                per_case=[_case("1", 1.0), _case("2", 0.0), _case("3", 1.0)],
            )
            m = HGMManager()  # exclude_llm_call_failures=False by default
            self.assertFalse(m.exclude_llm_call_failures)
            m._record_batch(node, result)
        self.assertEqual(node.n_evals, 3)
        self.assertEqual(sorted(node.evaluated_case_ids), ["1", "2", "3"])

    def test_no_op_when_trace_jsonl_is_missing(self) -> None:
        with TemporaryDirectory() as d:
            round_dir = Path(d)  # no logs/trace.jsonl at all
            node = HGMNode(0, None, round_dir)
            result = EvaluationResult(score=1.0, per_case=[_case("1", 1.0)])
            m = HGMManager()
            m._record_batch(node, result)
        self.assertEqual(node.n_evals, 1)


class RecordBatchExclusionTests(unittest.TestCase):
    """exclude_llm_call_failures=True -- skip exactly the cases in
    cases_with_terminal_failure."""

    def test_terminal_failure_case_excluded_others_recorded(self) -> None:
        with TemporaryDirectory() as d:
            round_dir = Path(d)
            _write_trace(round_dir, [_terminal_failure_event("2")])
            node = HGMNode(0, None, round_dir)
            result = EvaluationResult(
                score=0.5,
                per_case=[_case("1", 1.0), _case("2", 0.0), _case("3", 1.0)],
            )
            m = HGMManager(exclude_llm_call_failures=True)
            m._record_batch(node, result)
        # Case "2" (the terminal failure) never reaches node.record() --
        # only "1" and "3" do.
        self.assertEqual(node.n_evals, 2)
        self.assertEqual(sorted(node.evaluated_case_ids), ["1", "3"])
        self.assertEqual(node.mean_utility, 1.0)

    def test_a_status_failed_retry_that_recovered_is_not_excluded(self) -> None:
        """A retry that succeeded (not a terminal failure) must not be
        treated as corrupted -- only cases_with_terminal_failure counts."""
        with TemporaryDirectory() as d:
            round_dir = Path(d)
            _write_trace(round_dir, [{
                "kind": "llm_call_retry",
                "payload": {
                    "case_id": "1",
                    "error": "response status/stop_reason == 'failed' (no exception raised)",
                    "response_error_code": "server_error",
                },
            }])
            node = HGMNode(0, None, round_dir)
            result = EvaluationResult(score=1.0, per_case=[_case("1", 1.0)])
            m = HGMManager(exclude_llm_call_failures=True)
            m._record_batch(node, result)
        self.assertEqual(node.n_evals, 1)

    def test_all_cases_excluded_leaves_node_with_zero_evals_no_crash(self) -> None:
        with TemporaryDirectory() as d:
            round_dir = Path(d)
            _write_trace(round_dir, [
                _terminal_failure_event("1"), _terminal_failure_event("2"),
            ])
            node = HGMNode(0, None, round_dir)
            result = EvaluationResult(
                score=0.0, per_case=[_case("1", 0.0), _case("2", 0.0)],
            )
            m = HGMManager(exclude_llm_call_failures=True)
            m._record_batch(node, result)
        self.assertEqual(node.n_evals, 0)
        self.assertEqual(node.mean_utility, 0.0)


class LlmFailureHealthArtifactTests(unittest.TestCase):
    def test_artifact_written_regardless_of_exclude_flag(self) -> None:
        for exclude in (False, True):
            with TemporaryDirectory() as d:
                round_dir = Path(d)
                _write_trace(round_dir, [_terminal_failure_event("1")])
                node = HGMNode(0, None, round_dir)
                result = EvaluationResult(score=0.0, per_case=[_case("1", 0.0)])
                m = HGMManager(exclude_llm_call_failures=exclude)
                m._record_batch(node, result)
                artifact_path = round_dir / "llm_failure_health.json"
                self.assertTrue(artifact_path.exists())
                health = json.loads(artifact_path.read_text(encoding="utf-8"))
                self.assertEqual(health["cases_with_terminal_failure"], {"1": 1})


class LoudThresholdWarningTests(unittest.TestCase):
    def test_warning_printed_when_incidence_exceeds_threshold(self) -> None:
        with TemporaryDirectory() as d:
            round_dir = Path(d)
            # 1 response, 1 terminal failure -> 100% incidence, well over
            # the default 3.0% threshold.
            _write_trace(round_dir, [_terminal_failure_event("1")])
            node = HGMNode(0, None, round_dir)
            result = EvaluationResult(score=0.0, per_case=[_case("1", 0.0)])
            m = HGMManager()
            buf = io.StringIO()
            with redirect_stdout(buf):
                m._record_batch(node, result)
        self.assertIn("LLM call failure rate", buf.getvalue())
        self.assertIn("exceeds", buf.getvalue())

    def test_no_warning_when_incidence_under_threshold(self) -> None:
        with TemporaryDirectory() as d:
            round_dir = Path(d)
            events = [{"kind": "llm_response", "payload": {"case_id": "1", "stop_reason": "completed"}}] * 10
            _write_trace(round_dir, events)
            node = HGMNode(0, None, round_dir)
            result = EvaluationResult(score=1.0, per_case=[_case("1", 1.0)])
            m = HGMManager()
            buf = io.StringIO()
            with redirect_stdout(buf):
                m._record_batch(node, result)
        self.assertNotIn("LLM call failure rate", buf.getvalue())

    def test_custom_threshold_respected(self) -> None:
        with TemporaryDirectory() as d:
            round_dir = Path(d)
            # 10 responses, 1 terminal failure -> 10% incidence.
            events = [{"kind": "llm_response", "payload": {"case_id": str(i), "stop_reason": "completed"}} for i in range(9)]
            events.append(_terminal_failure_event("9"))
            _write_trace(round_dir, events)
            node = HGMNode(0, None, round_dir)
            result = EvaluationResult(score=0.9, per_case=[_case("9", 0.0)])
            m = HGMManager(llm_call_failure_threshold_pct=50.0)
            buf = io.StringIO()
            with redirect_stdout(buf):
                m._record_batch(node, result)
        self.assertNotIn("LLM call failure rate", buf.getvalue())


class BudgetAccountingUnaffectedTests(unittest.TestCase):
    """_evaluate()'s returned `spent` (what self._budget_spent adds) must
    equal len(batch) regardless of how many cases got excluded from
    node.record() -- exclusion only affects the node's OWN quality
    signal, never eval_budget accounting."""

    def test_evaluate_return_value_unaffected_by_exclusion(self) -> None:
        from unittest import mock

        with TemporaryDirectory() as d:
            round_dir = Path(d)
            _write_trace(round_dir, [_terminal_failure_event("1")])
            m = HGMManager(exclude_llm_call_failures=True, eval_batch_size=2)
            m._tree.add(HGMNode(0, None, round_dir))
            m._train_case_ids = ["1", "2"]
            m._task_rng = mock.Mock(sample=lambda pool, n: list(pool)[:n])
            # _evaluate's post-record housekeeping (_refresh_node_feedback)
            # looks up this node's existing feedback entry -- normally set
            # by _expand() beforehand; stub the one field it reads.
            m._feedback = {0: mock.Mock(strategy=mock.Mock())}

            fake_result = EvaluationResult(
                score=0.5, per_case=[_case("1", 0.0), _case("2", 1.0)],
            )
            fake_evaluator = mock.Mock(run=mock.Mock(return_value=fake_result))
            fake_gatherer = mock.Mock(compile=mock.Mock(return_value=None))

            spent = m._evaluate(0, fake_evaluator, fake_gatherer)
        self.assertEqual(spent, 2)  # len(batch), not len(recorded cases)
        node = m._tree[0]
        self.assertEqual(node.n_evals, 1)  # case "1" was excluded


if __name__ == "__main__":
    unittest.main()
