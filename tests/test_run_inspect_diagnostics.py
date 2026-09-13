"""Tests for meta_agent.run_inspect.extract_diagnostics -- specifically the
is_active disambiguation for a node whose eval_result.json is behind its
logs/case_*.json (RoundInfo.eval_result flagged
"_synthesized_from_case_logs" by discover_rounds). A single filesystem
snapshot can't tell "genuinely crashed mid-batch" apart from "still being
evaluated right now, the aggregate just hasn't been persisted yet" -- only
the run's own liveness can.

    PYTHONPATH=. python3 -m unittest tests.test_run_inspect_diagnostics
"""
from __future__ import annotations

import unittest
from pathlib import Path

from meta_agent.run_inspect import RoundInfo, extract_diagnostics


def _synthesized_round(node_id: int) -> RoundInfo:
    return RoundInfo(
        round_dir=Path("."),
        node_id=node_id,
        hgm_node={"n_evals": 0, "mean_utility": 0.0, "edit_failed": False},
        eval_result={"per_case": [], "_synthesized_from_case_logs": True},
    )


class ExtractDiagnosticsActiveRunTests(unittest.TestCase):
    def test_synthesized_round_is_not_flagged_while_run_is_active(self) -> None:
        alerts = extract_diagnostics([_synthesized_round(5)], is_active=True)
        self.assertEqual(alerts, [])

    def test_synthesized_round_is_flagged_once_run_is_no_longer_active(self) -> None:
        alerts = extract_diagnostics([_synthesized_round(5)], is_active=False)
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0].severity, "error")
        self.assertEqual(alerts[0].node_id, 5)
        self.assertIn("crashed", alerts[0].message)

    def test_other_diagnostics_unaffected_by_is_active(self) -> None:
        # A genuine per-case error, and an edit_failed node, must still be
        # reported regardless of whether the run is live -- only the
        # "_synthesized_from_case_logs" heuristic depends on liveness.
        edit_failed_round = RoundInfo(
            round_dir=Path("."), node_id=1, hgm_node={"edit_failed": True},
            feedback={"edit_errors": ["boom"]},
        )
        error_case_round = RoundInfo(
            round_dir=Path("."), node_id=2,
            hgm_node={"n_evals": 1, "mean_utility": 0.0, "edit_failed": False},
            eval_result={"per_case": [{"case_id": "a", "error": "boom"}]},
        )
        for is_active in (True, False):
            alerts = extract_diagnostics(
                [edit_failed_round, error_case_round], is_active=is_active
            )
            messages = [a.message for a in alerts]
            self.assertTrue(any("edit failed" in m for m in messages))
            self.assertTrue(any("case a" in m for m in messages))


class LlmFailureHealthDiagnosticsTests(unittest.TestCase):
    """RoundInfo.llm_failure_health / llm_failure_rate_pct -- see
    meta_agent.llm_failure_health for the underlying trace.jsonl signal
    HGMManager._record_batch writes into llm_failure_health.json."""

    def test_round_over_threshold_flagged(self) -> None:
        round_ = RoundInfo(
            round_dir=Path("."), node_id=7,
            hgm_node={"n_evals": 32, "mean_utility": 0.5, "edit_failed": False},
            llm_failure_health={
                "n_llm_responses": 100,
                "n_status_failed_retries": 2,
                "n_terminal_failed_responses": 5,
            },
        )
        self.assertAlmostEqual(round_.llm_failure_rate_pct, 7.0)
        alerts = extract_diagnostics([round_], is_active=True)
        messages = [a.message for a in alerts if a.node_id == 7]
        self.assertTrue(any("LLM call failure rate" in m for m in messages))
        self.assertTrue(any(a.severity == "error" for a in alerts if a.node_id == 7))

    def test_round_under_threshold_not_flagged(self) -> None:
        round_ = RoundInfo(
            round_dir=Path("."), node_id=8,
            hgm_node={"n_evals": 32, "mean_utility": 0.5, "edit_failed": False},
            llm_failure_health={
                "n_llm_responses": 1000,
                "n_status_failed_retries": 1,
                "n_terminal_failed_responses": 0,
            },
        )
        alerts = extract_diagnostics([round_], is_active=True)
        messages = [a.message for a in alerts if a.node_id == 8]
        self.assertFalse(any("LLM call failure rate" in m for m in messages))

    def test_round_without_llm_failure_health_not_flagged_no_crash(self) -> None:
        # A round from before this feature existed -- llm_failure_health
        # stays None; must not crash or produce a false alert.
        round_ = RoundInfo(
            round_dir=Path("."), node_id=9,
            hgm_node={"n_evals": 32, "mean_utility": 0.5, "edit_failed": False},
        )
        self.assertIsNone(round_.llm_failure_rate_pct)
        alerts = extract_diagnostics([round_], is_active=True)
        self.assertFalse(any(a.node_id == 9 for a in alerts))

    def test_zero_responses_treated_as_nothing_to_report(self) -> None:
        round_ = RoundInfo(
            round_dir=Path("."), node_id=10,
            llm_failure_health={
                "n_llm_responses": 0,
                "n_status_failed_retries": 0,
                "n_terminal_failed_responses": 0,
            },
        )
        self.assertIsNone(round_.llm_failure_rate_pct)


if __name__ == "__main__":
    unittest.main()
