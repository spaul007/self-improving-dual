"""Tests for AgentEditor's has_suggestion gating (meta_agent/agent_editor.py).

Formalizes the diagnosis/implementation role split: once block_suggester.py
has produced a real suggestion for an EXPAND, the editor's own
project_metrics rendering is redundant (the suggester already reasoned
about those same numbers) and is trimmed -- but only then, and never for
failure_report, which stays visible regardless.

    PYTHONPATH=. python3 -m unittest tests.test_agent_editor_suggestion_gating
"""
from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace

from meta_agent.agent_editor import AgentEditor
from meta_agent.models import AgentFeedback, EvaluationResult, EvolutionStrategy


def _feedback(*, project_metrics=None, failure_report=None) -> AgentFeedback:
    return AgentFeedback(
        round_number=1, base_round=0,
        strategy=EvolutionStrategy(
            target_files=[], optimization_goal="g", proposed_changes="x",
        ),
        eval_result=EvaluationResult(score=0.3, passed=0, failed=10),
        project_metrics=project_metrics or {},
        failure_report=failure_report or {},
    )


class FormatFeedbackGatingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.editor = AgentEditor(llm_caller=lambda **kw: None, validators=[])

    def test_shows_project_metrics_by_default(self) -> None:
        fb = _feedback(project_metrics={"top_failed_checks": [["x", 3]]})
        text = self.editor._format_feedback(fb)
        self.assertIn("project metrics:", text)

    def test_hides_project_metrics_when_has_suggestion_true(self) -> None:
        fb = _feedback(project_metrics={"top_failed_checks": [["x", 3]]})
        text = self.editor._format_feedback(fb, has_suggestion=True)
        self.assertNotIn("project metrics:", text)

    def test_keeps_failure_report_regardless_of_has_suggestion(self) -> None:
        report = {"summary": {"n_failing": 1, "total_cases": 4, "mean_score": 0.1}}
        fb = _feedback(
            project_metrics={"top_failed_checks": [["x", 3]]},
            failure_report=report,
        )
        without = self.editor._format_feedback(fb, has_suggestion=False)
        with_suggestion = self.editor._format_feedback(fb, has_suggestion=True)
        self.assertIn("Failure analysis", without)
        self.assertIn("Failure analysis", with_suggestion)
        # Only project_metrics toggles between the two.
        self.assertIn("project metrics:", without)
        self.assertNotIn("project metrics:", with_suggestion)


class ApplyThreadsHasSuggestionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_base = Path(
            __import__("tempfile").mkdtemp(prefix="agent_editor_gating_")
        )
        self.addCleanup(
            lambda: __import__("shutil").rmtree(self.tmp_base, ignore_errors=True)
        )
        agent_dir = self.tmp_base / "base" / "task_agent"
        agent_dir.mkdir(parents=True)
        (agent_dir / "workflow.py").write_text(
            "def run_task(task):\n    return None\n", encoding="utf-8"
        )

    def test_apply_threads_has_suggestion_into_format_feedback(self) -> None:
        captured: dict[str, str] = {}

        def fake_llm(**kwargs):
            captured["user"] = kwargs["messages"][1]["content"]
            return SimpleNamespace(content=None, tool_calls=[])

        editor = AgentEditor(llm_caller=fake_llm, validators=[])
        fb = _feedback(project_metrics={"top_failed_checks": [["x", 3]]})
        editor.apply(
            fb, self.tmp_base / "base", self.tmp_base / "out",
            context="steer", has_suggestion=True,
        )
        self.assertNotIn("project metrics:", captured["user"])

    def test_apply_default_has_suggestion_shows_project_metrics(self) -> None:
        captured: dict[str, str] = {}

        def fake_llm(**kwargs):
            captured["user"] = kwargs["messages"][1]["content"]
            return SimpleNamespace(content=None, tool_calls=[])

        editor = AgentEditor(llm_caller=fake_llm, validators=[])
        fb = _feedback(project_metrics={"top_failed_checks": [["x", 3]]})
        editor.apply(
            fb, self.tmp_base / "base", self.tmp_base / "out2", context="steer",
        )
        self.assertIn("project metrics:", captured["user"])


if __name__ == "__main__":
    unittest.main()
