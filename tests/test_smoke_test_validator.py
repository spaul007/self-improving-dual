"""Tests for SmokeTestValidator (meta_agent/editor_validators.py) -- runs
the agent on ONE real case and rejects a genuine code-level crash, not a
low score. Uses a fake evaluator (no real LLM calls, no network).

    PYTHONPATH=. python3 -m unittest tests.test_smoke_test_validator
"""
from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from meta_agent.editor_validators import SmokeTestValidator
from meta_agent.models import CaseResult, EvaluationResult


class _FakeEvaluator:
    """Records what it was called with; returns a canned EvaluationResult
    (or raises, if configured to) instead of ever touching a real LLM."""

    def __init__(self, *, result: EvaluationResult = None, raises: bool = False):
        self.result = result
        self.raises = raises
        self.calls: list[dict] = []

    def run(self, round_dir, benchmark_dir, *, case_ids=None):
        self.calls.append(
            {"round_dir": round_dir, "benchmark_dir": benchmark_dir, "case_ids": case_ids}
        )
        if self.raises:
            raise RuntimeError("simulated evaluator/infra failure")
        return self.result


class SmokeTestValidatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="smoke_test_validator_"))
        self.addCleanup(shutil.rmtree, self.tmp, True)

        self.benchmark_dir = self.tmp / "benchmark"
        self.benchmark_dir.mkdir()
        (self.benchmark_dir / "cases.jsonl").write_text(
            '{"id": "7", "input": "first case"}\n'
            '{"id": "8", "input": "second case"}\n',
            encoding="utf-8",
        )

        self.out_dir = self.tmp / "round_001"
        agent_dir = self.out_dir / "task_agent"
        agent_dir.mkdir(parents=True)
        (agent_dir / "workflow.py").write_text("def run_task(t): return None\n")

        self.base_dir = self.tmp / "round_000"
        self.base_dir.mkdir()

    def _crash_result(self, case_id: str) -> EvaluationResult:
        return EvaluationResult(
            score=0.0, passed=0, failed=1,
            per_case=[
                CaseResult(
                    case_id=case_id, passed=False, score=0.0,
                    error="Traceback (most recent call last):\nValueError: boom",
                )
            ],
        )

    def _clean_result(self, case_id: str, *, score: float = 0.0) -> EvaluationResult:
        return EvaluationResult(
            score=score, passed=int(score >= 1.0), failed=int(score < 1.0),
            per_case=[
                CaseResult(
                    case_id=case_id, passed=score >= 1.0, score=score,
                    details={"failed_checks": ["some_check"]} if score < 1.0 else {"failed_checks": []},
                )
            ],
        )

    def test_missing_evaluator_is_a_noop(self) -> None:
        v = SmokeTestValidator(evaluator=None, benchmark_dir=self.benchmark_dir)
        self.assertEqual(v.validate(self.out_dir, self.base_dir), [])

    def test_missing_benchmark_dir_is_a_noop(self) -> None:
        v = SmokeTestValidator(evaluator=_FakeEvaluator(), benchmark_dir=None)
        self.assertEqual(v.validate(self.out_dir, self.base_dir), [])

    def test_missing_task_agent_dir_is_reported(self) -> None:
        v = SmokeTestValidator(
            evaluator=_FakeEvaluator(), benchmark_dir=self.benchmark_dir
        )
        errs = v.validate(self.tmp / "no_such_round", self.base_dir)
        self.assertEqual(len(errs), 1)
        self.assertIn("task_agent directory missing", errs[0])

    def test_real_crash_is_reported_with_case_id_and_error_text(self) -> None:
        fake = _FakeEvaluator(result=self._crash_result("7"))
        v = SmokeTestValidator(evaluator=fake, benchmark_dir=self.benchmark_dir)
        errs = v.validate(self.out_dir, self.base_dir)
        self.assertEqual(len(errs), 1)
        self.assertIn("case 7", errs[0])
        self.assertIn("ValueError: boom", errs[0])

    def test_low_score_without_a_crash_is_not_a_failure(self) -> None:
        # The whole point: a case that runs to completion and scores 0 is
        # NOT a validator failure -- only a genuine exception is.
        fake = _FakeEvaluator(result=self._clean_result("7", score=0.0))
        v = SmokeTestValidator(evaluator=fake, benchmark_dir=self.benchmark_dir)
        self.assertEqual(v.validate(self.out_dir, self.base_dir), [])

    def test_perfect_score_without_a_crash_is_not_a_failure(self) -> None:
        fake = _FakeEvaluator(result=self._clean_result("7", score=1.0))
        v = SmokeTestValidator(evaluator=fake, benchmark_dir=self.benchmark_dir)
        self.assertEqual(v.validate(self.out_dir, self.base_dir), [])

    def test_default_case_id_is_the_first_in_the_benchmark(self) -> None:
        fake = _FakeEvaluator(result=self._clean_result("7"))
        v = SmokeTestValidator(evaluator=fake, benchmark_dir=self.benchmark_dir)
        v.validate(self.out_dir, self.base_dir)
        self.assertEqual(fake.calls[0]["case_ids"], ["7"])

    def test_explicit_case_id_overrides_the_default(self) -> None:
        fake = _FakeEvaluator(result=self._clean_result("8"))
        v = SmokeTestValidator(
            evaluator=fake, benchmark_dir=self.benchmark_dir, case_id="8"
        )
        v.validate(self.out_dir, self.base_dir)
        self.assertEqual(fake.calls[0]["case_ids"], ["8"])

    def test_evaluator_raising_degrades_to_noop_not_a_failure(self) -> None:
        # An infra-level problem (bad benchmark_dir, disk error, ...) is not
        # evidence the AGENT's code crashed -- don't fail the edit over it.
        fake = _FakeEvaluator(raises=True)
        v = SmokeTestValidator(evaluator=fake, benchmark_dir=self.benchmark_dir)
        self.assertEqual(v.validate(self.out_dir, self.base_dir), [])

    def test_scratch_dir_is_cleaned_up_after_success(self) -> None:
        fake = _FakeEvaluator(result=self._clean_result("7"))
        v = SmokeTestValidator(evaluator=fake, benchmark_dir=self.benchmark_dir)
        v.validate(self.out_dir, self.base_dir)
        self.assertFalse((self.out_dir / "_smoke_test").exists())

    def test_scratch_dir_is_cleaned_up_after_a_detected_crash(self) -> None:
        fake = _FakeEvaluator(result=self._crash_result("7"))
        v = SmokeTestValidator(evaluator=fake, benchmark_dir=self.benchmark_dir)
        v.validate(self.out_dir, self.base_dir)
        self.assertFalse((self.out_dir / "_smoke_test").exists())

    def test_scratch_dir_is_cleaned_up_after_evaluator_raises(self) -> None:
        fake = _FakeEvaluator(raises=True)
        v = SmokeTestValidator(evaluator=fake, benchmark_dir=self.benchmark_dir)
        v.validate(self.out_dir, self.base_dir)
        self.assertFalse((self.out_dir / "_smoke_test").exists())

    def test_scratch_symlinks_the_real_edited_task_agent(self) -> None:
        # The evaluator must see the ACTUAL edited code, not a stale copy.
        captured_round_dir = {}

        class _CapturingEvaluator(_FakeEvaluator):
            def run(self, round_dir, benchmark_dir, *, case_ids=None):
                captured_round_dir["path"] = round_dir
                agent = round_dir / "task_agent"
                captured_round_dir["workflow_text"] = (
                    (agent / "workflow.py").read_text()
                    if (agent / "workflow.py").exists() else None
                )
                return super().run(round_dir, benchmark_dir, case_ids=case_ids)

        fake = _CapturingEvaluator(result=self._clean_result("7"))
        v = SmokeTestValidator(evaluator=fake, benchmark_dir=self.benchmark_dir)
        v.validate(self.out_dir, self.base_dir)
        self.assertEqual(
            captured_round_dir["workflow_text"], "def run_task(t): return None\n"
        )


if __name__ == "__main__":
    unittest.main()
