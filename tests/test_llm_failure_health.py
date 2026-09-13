"""Tests for meta_agent.llm_failure_health -- the shared trace.jsonl
failure-detection logic extracted from analyze_llm_call_failures.py so
HGMManager can reuse it (see tests/test_hgm_llm_failure_health.py for
its consumer-side tests).

Run from the repo root:
    PYTHONPATH=. python -m unittest tests.test_llm_failure_health
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from meta_agent.llm_failure_health import (
    analyze_trace_file,
    incidence_rate_pct,
    iter_trace_files,
)


def _write_trace(path: Path, events: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(e) for e in events) + ("\n" if events else ""),
        encoding="utf-8",
    )


class AnalyzeTraceFileTests(unittest.TestCase):
    def test_missing_file_returns_all_zero_summary(self) -> None:
        with TemporaryDirectory() as d:
            health = analyze_trace_file(Path(d) / "does_not_exist.jsonl")
        self.assertEqual(health["n_llm_calls"], 0)
        self.assertEqual(health["n_llm_responses"], 0)
        self.assertEqual(health["n_status_failed_retries"], 0)
        self.assertEqual(health["n_terminal_failed_responses"], 0)
        self.assertEqual(health["cases_with_terminal_failure"], {})
        self.assertEqual(health["error_codes"], {})

    def test_empty_file_returns_all_zero_summary(self) -> None:
        with TemporaryDirectory() as d:
            trace_path = Path(d) / "trace.jsonl"
            _write_trace(trace_path, [])
            health = analyze_trace_file(trace_path)
        self.assertEqual(health["n_llm_calls"], 0)

    def test_pure_retry_recovered_no_terminal_failure(self) -> None:
        with TemporaryDirectory() as d:
            trace_path = Path(d) / "trace.jsonl"
            _write_trace(trace_path, [
                {"kind": "llm_call", "payload": {"id": "a"}},
                {
                    "kind": "llm_call_retry",
                    "payload": {
                        "case_id": "5",
                        "error": "response status/stop_reason == 'failed' (no exception raised)",
                        "response_error_code": "server_error",
                    },
                },
                {
                    "kind": "llm_response",
                    "payload": {"case_id": "5", "stop_reason": "completed"},
                },
            ])
            health = analyze_trace_file(trace_path)
        self.assertEqual(health["n_llm_calls"], 1)
        self.assertEqual(health["n_llm_responses"], 1)
        self.assertEqual(health["n_status_failed_retries"], 1)
        self.assertEqual(health["n_terminal_failed_responses"], 0)
        self.assertEqual(health["cases_with_status_failed_retry"], {"5": 1})
        self.assertEqual(health["cases_with_terminal_failure"], {})
        self.assertEqual(health["error_codes"], {"server_error": 1})

    def test_terminal_failure_recorded(self) -> None:
        with TemporaryDirectory() as d:
            trace_path = Path(d) / "trace.jsonl"
            _write_trace(trace_path, [
                {"kind": "llm_call", "payload": {"id": "a"}},
                {
                    "kind": "llm_response",
                    "payload": {
                        "case_id": "9",
                        "stop_reason": "failed",
                        "response_error_code": "invalid_prompt",
                    },
                },
            ])
            health = analyze_trace_file(trace_path)
        self.assertEqual(health["n_terminal_failed_responses"], 1)
        self.assertEqual(health["cases_with_terminal_failure"], {"9": 1})
        self.assertEqual(health["error_codes"], {"invalid_prompt": 1})

    def test_exception_retry_not_counted_as_status_failed(self) -> None:
        with TemporaryDirectory() as d:
            trace_path = Path(d) / "trace.jsonl"
            _write_trace(trace_path, [
                {
                    "kind": "llm_call_retry",
                    "payload": {"case_id": "1", "error": "ConnectionError(...)"},
                },
            ])
            health = analyze_trace_file(trace_path)
        self.assertEqual(health["n_status_failed_retries"], 0)
        self.assertEqual(health["n_exception_retries"], 1)
        self.assertEqual(health["cases_with_status_failed_retry"], {})

    def test_mixed_error_codes_across_events(self) -> None:
        with TemporaryDirectory() as d:
            trace_path = Path(d) / "trace.jsonl"
            _write_trace(trace_path, [
                {
                    "kind": "llm_call_retry",
                    "payload": {
                        "case_id": "1",
                        "error": "response status/stop_reason == 'failed' (no exception raised)",
                        "response_error_code": "invalid_prompt",
                    },
                },
                {
                    "kind": "llm_response",
                    "payload": {
                        "case_id": "2",
                        "stop_reason": "failed",
                        "response_error_code": "server_error",
                    },
                },
                {
                    "kind": "llm_response",
                    "payload": {
                        "case_id": "3",
                        "stop_reason": "failed",
                        "response_error_code": None,
                    },
                },
            ])
            health = analyze_trace_file(trace_path)
        self.assertEqual(
            health["error_codes"],
            {"invalid_prompt": 1, "server_error": 1, "(none given by API)": 1},
        )

    def test_malformed_json_line_skipped_not_raised(self) -> None:
        with TemporaryDirectory() as d:
            trace_path = Path(d) / "trace.jsonl"
            trace_path.write_text(
                'not valid json\n{"kind": "llm_call", "payload": {}}\n',
                encoding="utf-8",
            )
            health = analyze_trace_file(trace_path)
        self.assertEqual(health["n_llm_calls"], 1)


class IterTraceFilesTests(unittest.TestCase):
    def test_direct_file_path_returned_as_single_item_list(self) -> None:
        with TemporaryDirectory() as d:
            trace_path = Path(d) / "trace.jsonl"
            _write_trace(trace_path, [])
            found = iter_trace_files(trace_path)
        self.assertEqual(found, [trace_path])

    def test_run_dir_globs_every_round(self) -> None:
        with TemporaryDirectory() as d:
            run_dir = Path(d)
            t1 = run_dir / "round_001" / "logs" / "trace.jsonl"
            t2 = run_dir / "round_002" / "logs" / "trace.jsonl"
            _write_trace(t1, [])
            _write_trace(t2, [])
            found = iter_trace_files(run_dir)
        self.assertEqual(sorted(found), sorted([t1, t2]))

    def test_missing_run_dir_returns_empty(self) -> None:
        with TemporaryDirectory() as d:
            found = iter_trace_files(Path(d) / "nonexistent_run")
        self.assertEqual(found, [])


class IncidenceRatePctTests(unittest.TestCase):
    def test_zero_responses_is_zero_percent(self) -> None:
        self.assertEqual(incidence_rate_pct({"n_llm_responses": 0}), 0.0)

    def test_computed_correctly(self) -> None:
        health = {
            "n_llm_responses": 100,
            "n_status_failed_retries": 2,
            "n_terminal_failed_responses": 1,
        }
        self.assertAlmostEqual(incidence_rate_pct(health), 3.0)

    def test_missing_keys_default_to_zero(self) -> None:
        self.assertEqual(incidence_rate_pct({"n_llm_responses": 10}), 0.0)


if __name__ == "__main__":
    unittest.main()
