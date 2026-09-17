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


if __name__ == "__main__":
    unittest.main()
