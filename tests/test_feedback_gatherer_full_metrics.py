"""DefaultFeedbackGatherer._write_full_metrics -- the same opt-in
detection convention as _project_metrics/scorer.aggregate(), but the
result goes straight to round_dir/full_metrics.json rather than into
project_metrics (which feeds the prompt and stays capped by design).

    PYTHONPATH=. python3 -m unittest tests.test_feedback_gatherer_full_metrics
"""
from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from meta_agent.feedback_gatherer import DefaultFeedbackGatherer
from meta_agent.models import CaseResult, EvaluationResult

_ONE_CASE = [CaseResult(case_id="1", passed=False, score=0.5, details={})]


class WriteFullMetricsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.round_dir = Path(tempfile.mkdtemp(prefix="gatherer_full_metrics_"))
        self.addCleanup(lambda: shutil.rmtree(self.round_dir, ignore_errors=True))

    def test_writes_the_file_when_scorer_defines_full_metrics(self) -> None:
        scorer = SimpleNamespace(full_metrics=lambda per_case: {"hard:x": {"fail_rate": 0.5}})
        gatherer = DefaultFeedbackGatherer(scorer=scorer)
        gatherer._write_full_metrics(self.round_dir, EvaluationResult(score=0.5, per_case=_ONE_CASE))
        written = json.loads((self.round_dir / "full_metrics.json").read_text())
        self.assertEqual(written, {"hard:x": {"fail_rate": 0.5}})

    def test_noop_when_scorer_has_no_full_metrics(self) -> None:
        scorer = SimpleNamespace()  # no full_metrics attribute at all
        gatherer = DefaultFeedbackGatherer(scorer=scorer)
        gatherer._write_full_metrics(self.round_dir, EvaluationResult(score=0.5, per_case=_ONE_CASE))
        self.assertFalse((self.round_dir / "full_metrics.json").exists())

    def test_noop_when_no_scorer_at_all(self) -> None:
        gatherer = DefaultFeedbackGatherer(scorer=None)
        gatherer._write_full_metrics(self.round_dir, EvaluationResult(score=0.5, per_case=_ONE_CASE))
        self.assertFalse((self.round_dir / "full_metrics.json").exists())

    def test_noop_when_node_has_not_been_evaluated_yet(self) -> None:
        """Confirmed live: compile() is also called right after EXPAND,
        before any EVALUATE, to persist a placeholder round folder (see
        hgm.py's own "fresh child starts unevaluated" comment) -- that
        call's per_case is always []. Writing full_metrics.json there
        produced a content-free {} artifact for every brand-new,
        not-yet-evaluated node. The real file should only appear once
        the node is actually evaluated."""
        scorer = SimpleNamespace(full_metrics=lambda per_case: {"hard:x": {"fail_rate": 0.5}})
        gatherer = DefaultFeedbackGatherer(scorer=scorer)
        gatherer._write_full_metrics(self.round_dir, EvaluationResult(score=0.0, per_case=[]))
        self.assertFalse((self.round_dir / "full_metrics.json").exists())

    def test_exception_in_full_metrics_does_not_crash_the_round(self) -> None:
        def _boom(per_case):
            raise ValueError("broken scorer")

        scorer = SimpleNamespace(full_metrics=_boom)
        gatherer = DefaultFeedbackGatherer(scorer=scorer)
        # Must not raise.
        gatherer._write_full_metrics(self.round_dir, EvaluationResult(score=0.5, per_case=_ONE_CASE))
        self.assertFalse((self.round_dir / "full_metrics.json").exists())


if __name__ == "__main__":
    unittest.main()
