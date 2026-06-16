"""Unit tests for BehaviorSummarizer — exercises the deterministic
pre-aggregation (diff, mutable_log cross-tab, per-case roll-up), the
prompt assembly, and the disk artifacts. Uses a stub LLM (no network).

    PYTHONPATH=. python3 -m unittest tests.test_behavior_summarizer
"""
from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

from meta_agent.behavior_summarizer import (
    BehaviorSummarizer,
    render_memory_for_steering,
)
from meta_agent.models import CaseResult, EvaluationResult


# --------------------------------------------------------------------------- #
# Stub LLM response — captures the kwargs it was called with so the test can
# assert prompt contents.
# --------------------------------------------------------------------------- #


@dataclass
class _StubResponse:
    content: str


class _StubLLM:
    def __init__(self, *, response: str = "## What was added\nStub summary.\n"):
        self.calls: list[dict] = []
        self.response = response

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return _StubResponse(content=self.response)


# --------------------------------------------------------------------------- #
# Helpers — build round dirs that look like real ones.
# --------------------------------------------------------------------------- #


def _write_agent(round_dir: Path, *, workflow: str, mutable_tools: dict | None = None) -> None:
    """Create round_dir/task_agent/{workflow.py, ...mutable_tools/...}."""
    agent = round_dir / "task_agent"
    (agent / "mutable_tools").mkdir(parents=True, exist_ok=True)
    (agent / "workflow.py").write_text(workflow, encoding="utf-8")
    (agent / "tool_wrapper.py").write_text("def x(): return None\n", encoding="utf-8")
    (agent / "tools_schema.json").write_text("[]", encoding="utf-8")
    (agent / "mutable_tools" / "__init__.py").write_text("", encoding="utf-8")
    for name, body in (mutable_tools or {}).items():
        (agent / "mutable_tools" / name).write_text(body, encoding="utf-8")


def _write_trace(round_dir: Path, events: list[dict]) -> None:
    log_dir = round_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    with (log_dir / "trace.jsonl").open("w", encoding="utf-8") as fh:
        for ev in events:
            fh.write(json.dumps(ev) + "\n")


def _ev(kind: str, **payload) -> dict:
    return {"timestamp": 0.0, "kind": kind, "payload": payload}


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #


class BehaviorSummarizerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="bsumm_"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.parent_dir = self.tmp / "round_000"
        self.child_dir = self.tmp / "round_001"
        self.parent_dir.mkdir()
        self.child_dir.mkdir()

    def _build_summarizer(self, **kwargs) -> tuple[BehaviorSummarizer, _StubLLM]:
        llm = _StubLLM(**{k: v for k, v in kwargs.items() if k == "response"})
        return BehaviorSummarizer(llm_caller=llm, model="stub-model"), llm

    # ---- Skips silently when there's no parent (seed round) ----
    def test_skips_when_no_parent(self) -> None:
        summ, llm = self._build_summarizer()
        result = summ.summarize(
            round_dir=self.child_dir,
            parent_round_dir=None,
            eval_dir=self.child_dir,
            eval_result=EvaluationResult(score=0.0, per_case=[]),
            node_id=0,
            parent_id=None,
        )
        self.assertIsNone(result)
        self.assertEqual(llm.calls, [])
        self.assertFalse((self.child_dir / "behavior_memory.md").exists())

    # ---- Aggregate captures diff + mutable_log + per-case correlation ----
    def test_aggregate_captures_diff_log_cross_tab(self) -> None:
        _write_agent(self.parent_dir, workflow="def run_task(task):\n    return None\n")
        _write_agent(
            self.child_dir,
            workflow=(
                "from platform_core.trace import log\n"
                "def run_task(task):\n"
                "    log('verifier_fired', name='budget', verdict='pass')\n"
                "    return None\n"
            ),
            mutable_tools={"helper.py": "# new helper\n"},
        )
        _write_trace(
            self.child_dir,
            [
                _ev("mutable_log", label="verifier_fired", name="budget",
                    verdict="pass", case_id="c1"),
                _ev("mutable_log", label="verifier_fired", name="budget",
                    verdict="fail", reason="exceeds", case_id="c2"),
                _ev("mutable_log", label="verifier_fired", name="budget",
                    verdict="pass", case_id="c3"),
                _ev("tool_call", name="query_hotel_info", id="x"),  # ignored
            ],
        )
        per_case = [
            CaseResult(case_id="c1", passed=True, score=1.0, details={"failed_checks": []}),
            CaseResult(case_id="c2", passed=False, score=0.3,
                       details={"failed_checks": ["budget_total"]}),
            CaseResult(case_id="c3", passed=True, score=1.0, details={"failed_checks": []}),
        ]
        eval_result = EvaluationResult(score=0.77, passed=2, failed=1, per_case=per_case)

        summ, llm = self._build_summarizer()
        path = summ.summarize(
            round_dir=self.child_dir,
            parent_round_dir=self.parent_dir,
            eval_dir=self.child_dir,
            eval_result=eval_result,
            node_id=1,
            parent_id=0,
        )

        # behavior_memory.md was written (stub content).
        self.assertEqual(path, self.child_dir / "behavior_memory.md")
        self.assertTrue(path.exists())
        self.assertIn("Stub summary", path.read_text())

        # behavior_aggregate.json was persisted.
        agg = json.loads((self.child_dir / "behavior_aggregate.json").read_text())
        self.assertEqual(agg["node_id"], 1)
        self.assertEqual(agg["parent_id"], 0)
        self.assertIn("workflow.py", agg["changed_files"])
        self.assertIn("mutable_tools/helper.py", agg["changed_files"])
        self.assertIn("verifier_fired", agg["mutable_log"])

        vf = agg["mutable_log"]["verifier_fired"]
        self.assertEqual(vf["total_fires"], 3)
        self.assertEqual(vf["by_verdict"]["pass"], 2)
        self.assertEqual(vf["by_verdict"]["fail"], 1)
        # c1 + c3 passed and had verifier_fired → 2; c2 failed → 1.
        self.assertEqual(vf["cases_pass_when_fired"], 2)
        self.assertEqual(vf["cases_fail_when_fired"], 1)

        # Per-case roll-up is included.
        per_case_rows = {r["case_id"]: r for r in agg["per_case"]}
        self.assertTrue(per_case_rows["c1"]["passed"])
        self.assertFalse(per_case_rows["c2"]["passed"])
        self.assertIn("budget_total", per_case_rows["c2"]["failure_hint"])

        # Prompt was sent to the LLM with the right structure.
        self.assertEqual(len(llm.calls), 1)
        call = llm.calls[0]
        msgs = call["messages"]
        self.assertEqual(msgs[0]["role"], "system")
        self.assertEqual(msgs[1]["role"], "user")
        user_text = msgs[1]["content"]
        self.assertIn("## Round summary", user_text)
        self.assertIn("## Diff vs parent", user_text)
        self.assertIn("## mutable_log aggregates", user_text)
        self.assertIn("## Per-case outcomes", user_text)
        self.assertIn("verifier_fired", user_text)
        # Per-case lines render passed marker.
        self.assertIn("case c1", user_text)
        self.assertIn("case c2", user_text)

        # The literal prompt is persisted for forensics.
        prompt_path = self.child_dir / "behavior_summarizer_prompt.txt"
        self.assertTrue(prompt_path.exists())
        dumped = prompt_path.read_text()
        self.assertIn("### SYSTEM", dumped)
        self.assertIn("### USER", dumped)

    # ---- Empty mutable_log: aggregate notes "no events emitted" ----
    def test_no_mutable_log_events_surfaces_clear_signal(self) -> None:
        _write_agent(self.parent_dir, workflow="def run_task(task):\n    return None\n")
        _write_agent(self.child_dir, workflow="def run_task(task):\n    return 'x'\n")
        _write_trace(self.child_dir, [])  # empty trace
        per_case = [CaseResult(case_id="c1", passed=True, score=1.0)]
        eval_result = EvaluationResult(score=1.0, passed=1, failed=0, per_case=per_case)
        summ, llm = self._build_summarizer()
        summ.summarize(
            round_dir=self.child_dir,
            parent_round_dir=self.parent_dir,
            eval_dir=self.child_dir,
            eval_result=eval_result,
            node_id=1, parent_id=0,
        )
        user_text = llm.calls[0]["messages"][1]["content"]
        self.assertIn("no mutable_log events emitted", user_text)

    # ---- LLM failure: memory file not written, but aggregate still persisted ----
    def test_llm_failure_persists_aggregate_only(self) -> None:
        _write_agent(self.parent_dir, workflow="def run_task(task):\n    return None\n")
        _write_agent(self.child_dir, workflow="def run_task(task):\n    return 'x'\n")
        _write_trace(self.child_dir, [])

        class _FailingLLM:
            def __call__(self, **kwargs):
                raise RuntimeError("network down")

        summ = BehaviorSummarizer(llm_caller=_FailingLLM(), model="stub-model")
        result = summ.summarize(
            round_dir=self.child_dir,
            parent_round_dir=self.parent_dir,
            eval_dir=self.child_dir,
            eval_result=EvaluationResult(score=1.0, per_case=[
                CaseResult(case_id="c1", passed=True, score=1.0)
            ]),
            node_id=1, parent_id=0,
        )
        self.assertIsNone(result)
        self.assertTrue((self.child_dir / "behavior_aggregate.json").exists())
        self.assertTrue((self.child_dir / "behavior_summarizer_prompt.txt").exists())
        self.assertFalse((self.child_dir / "behavior_memory.md").exists())

    # ---- Diff truncation: very large diffs get head/tail with elision marker ----
    def test_diff_truncation_keeps_head_and_tail(self) -> None:
        _write_agent(self.parent_dir, workflow="x = 0\n" * 5)
        big_workflow = "\n".join([f"# line {i} of a large file" for i in range(2000)])
        _write_agent(self.child_dir, workflow=big_workflow)
        _write_trace(self.child_dir, [])
        per_case = [CaseResult(case_id="c1", passed=False, score=0.0)]
        eval_result = EvaluationResult(score=0.0, passed=0, failed=1, per_case=per_case)
        summ, llm = self._build_summarizer()
        summ.summarize(
            round_dir=self.child_dir,
            parent_round_dir=self.parent_dir,
            eval_dir=self.child_dir,
            eval_result=eval_result,
            node_id=1, parent_id=0,
        )
        agg = json.loads((self.child_dir / "behavior_aggregate.json").read_text())
        self.assertIn("chars elided", agg["diff"])

    # ---- render_memory_for_steering: caps and missing-file handling ----
    def test_render_memory_for_steering(self) -> None:
        self.assertIsNone(render_memory_for_steering(self.child_dir))  # no file
        (self.child_dir / "behavior_memory.md").write_text("A" * 1500)
        full = render_memory_for_steering(self.child_dir)
        self.assertEqual(len(full), 1500)
        capped = render_memory_for_steering(self.child_dir, cap_chars=500)
        self.assertTrue(capped.endswith("<... truncated ...>"))
        self.assertLess(len(capped), 600)


if __name__ == "__main__":
    unittest.main()
