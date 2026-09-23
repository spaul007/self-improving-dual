"""AgentEditor ``agentic_log_access``: the editor may read/grep the PARENT node's
evaluation logs (paged, read-only), is told so in its prompt, and records every
tool call to editor_tools.jsonl.

    PYTHONPATH=. python3 -m pytest -q tests/test_editor_log_access.py
"""
from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from meta_agent.agent_editor import AgentEditor, _AGENTIC_LOG_EVIDENCE
from meta_agent.models import AgentFeedback, EvaluationResult, EvolutionStrategy


def _fb() -> AgentFeedback:
    return AgentFeedback(round_number=1, base_round=0,
                         strategy=EvolutionStrategy(target_files=[], optimization_goal="g", proposed_changes="x"),
                         eval_result=EvaluationResult(score=0.3, passed=0, failed=10))


def _call(name, arguments, cid):
    return SimpleNamespace(name=name, arguments=arguments, id=cid)


SUMMARY = ("submit_self_improvement_summary",
           {"optimization_goal": "g", "proposed_changes": "p", "rationale": "task t1: r2 untested"})


class EditorLogAccessTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="edlog_"))
        self.addCleanup(lambda: shutil.rmtree(self.tmp, ignore_errors=True))
        base = self.tmp / "base"
        (base / "task_agent").mkdir(parents=True)
        (base / "task_agent" / "workflow.py").write_text("def run_task(task):\n    return None\n")
        (base / "task_agent" / "roles.py").write_text("VERIFY = 1\n")
        logs = base / "logs"
        (logs / "scratch" / "t1" / "r1").mkdir(parents=True)
        (logs / "DOSSIERS.md").write_text("1. t1 near_miss -> logs/scratch/t1/r1/dossier.md\n")
        (logs / "scratch" / "t1" / "r1" / "dossier.md").write_text(
            "\n".join(f"line {i}" for i in range(450)) + "\nr7 NEVER TESTED\n")
        (base / "eval_result.json").write_text(json.dumps({"score": 0.25}))
        (self.tmp / "secret.txt").write_text("hidden")

    def _apply(self, script, **kw):
        turns: list[dict] = []

        def fake(**kwargs):
            turns.append({**kwargs, "messages": list(kwargs["messages"])})
            return SimpleNamespace(content="", tool_calls=[script(len(turns))])
        ed = AgentEditor(llm_caller=fake, validators=[], max_attempts=1, agentic_editing=True,
                         mutable_exclude=["workflow.py"], **kw)
        return ed.apply(_fb(), self.tmp / "base", self.tmp / "out"), turns

    @staticmethod
    def _out(turn) -> str:
        return turn["messages"][-1].get("output", "")

    def test_default_off_unchanged(self) -> None:
        _, turns = self._apply(lambda n: _call(*SUMMARY, "s"))
        names = {t["name"] for t in turns[0]["tools"]}
        self.assertNotIn("grep", names)
        self.assertNotIn("EVALUATION EVIDENCE", turns[0]["messages"][0]["content"])
        self.assertNotIn("Evaluation evidence you can read", turns[0]["messages"][1]["content"])
        self.assertFalse((self.tmp / "out" / "editor_tools.jsonl").exists())

    def test_prompt_and_tools_reach_the_model(self) -> None:
        _, turns = self._apply(lambda n: _call(*SUMMARY, "s"), agentic_log_access=True)
        names = [t["name"] for t in turns[0]["tools"]]
        self.assertIn("grep", names)
        self.assertEqual(names.count("read_file"), 1)
        self.assertIn(_AGENTIC_LOG_EVIDENCE.strip()[:40], turns[0]["messages"][0]["content"])
        user = turns[0]["messages"][1]["content"]
        self.assertIn("logs/DOSSIERS.md", user)
        self.assertIn("eval_result.json", user)

    def test_read_paged_grep_escape_and_tool_log(self) -> None:
        def script(n):
            return {1: _call("read_file", {"path": "logs/DOSSIERS.md"}, "a"),
                    2: _call("read_file", {"path": "logs/scratch/t1/r1/dossier.md"}, "b"),
                    3: _call("read_file", {"path": "logs/scratch/t1/r1/dossier.md", "offset": 440}, "c"),
                    4: _call("grep", {"path": "logs/scratch/t1/r1/dossier.md", "pattern": "NEVER"}, "d"),
                    5: _call("read_file", {"path": "logs/../../secret.txt"}, "e"),
                    6: _call("grep", {"path": "roles.py", "pattern": "VERIFY"}, "f"),
                    7: _call("read_file", {"path": "eval_result.json"}, "g"),
                    }.get(n, _call(*SUMMARY, "s"))
        res, turns = self._apply(script, agentic_log_access=True)
        # (no file was written, so apply() reports 'no file edits' -- irrelevant here)
        self.assertIn("t1 near_miss", self._out(turns[1]))
        self.assertIn("more lines", self._out(turns[2]))               # paged at 200
        self.assertIn("r7 NEVER TESTED", self._out(turns[3]))          # offset works
        self.assertIn("r7 NEVER TESTED", self._out(turns[4]))          # grep
        self.assertIn("escapes the logs/ root", self._out(turns[5]))   # no escape
        self.assertIn("VERIFY = 1", self._out(turns[6]))               # grep source file
        self.assertIn("0.25", self._out(turns[7]))
        rows = [json.loads(l) for l in (self.tmp / "out" / "editor_tools.jsonl").read_text().splitlines()]
        self.assertEqual(len(rows), 7)   # the final summary call returns, not recorded
        self.assertEqual(rows[0]["path"], "logs/DOSSIERS.md")
        self.assertTrue(rows[4]["error"])

    def test_logs_are_never_writable(self) -> None:
        def script(n):
            if n == 1:
                return _call("write_file", {"path": "logs/DOSSIERS.md", "content": "x"}, "w")
            return _call(*SUMMARY, "s")
        _, turns = self._apply(script, agentic_log_access=True)
        self.assertIn("ERROR", self._out(turns[1]))
        self.assertIn("t1 near_miss", (self.tmp / "base" / "logs" / "DOSSIERS.md").read_text())
        self.assertFalse((self.tmp / "out" / "task_agent" / "logs").exists())   # no decoy file


if __name__ == "__main__":
    unittest.main()
