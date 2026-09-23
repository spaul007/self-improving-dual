"""Opt-in agentic editor/suggester widenings for large vendored seeds:
``agentic_edit_file`` (exact string replacement) and ``readonly_reference``
(frozen files readable, never writable).

    PYTHONPATH=. python3 -m pytest -q tests/test_agentic_edit_file_readonly.py
"""
from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from meta_agent.agent_editor import AgentEditor, apply_string_edit
from meta_agent.block_suggester import BlockSuggester
from meta_agent.models import AgentFeedback, EvaluationResult, EvolutionStrategy


def _fb() -> AgentFeedback:
    return AgentFeedback(
        round_number=1, base_round=0,
        strategy=EvolutionStrategy(target_files=[], optimization_goal="g", proposed_changes="x"),
        eval_result=EvaluationResult(score=0.3, passed=0, failed=10),
    )


def _call(name, arguments, cid):
    return SimpleNamespace(name=name, arguments=arguments, id=cid)


def _last_output(kwargs) -> str:
    return kwargs["messages"][-1].get("output", "")


class ApplyStringEditTests(unittest.TestCase):
    def test_unique_replacement(self) -> None:
        self.assertEqual(apply_string_edit("a b c", "b", "X")[0], "a X c")

    def test_missing_ambiguous_noop_refused(self) -> None:
        self.assertIsNone(apply_string_edit("a b", "z", "y")[0])
        self.assertIsNone(apply_string_edit("b b", "b", "y")[0])
        self.assertIsNone(apply_string_edit("a b", "b", "b")[0])
        self.assertIsNone(apply_string_edit("a b", "", "y")[0])

    def test_replace_all(self) -> None:
        self.assertEqual(apply_string_edit("b b", "b", "y", True)[0], "y y")


class EditorWideningTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="edit_ro_"))
        self.addCleanup(lambda: shutil.rmtree(self.tmp, ignore_errors=True))
        a = self.tmp / "base" / "task_agent"
        (a / "pkg").mkdir(parents=True)
        (a / "workflow.py").write_text("def run_task(task):\n    return None\n")
        (a / "pkg" / "roles.py").write_text("X = 1\nY = 2\n")
        (a / "pkg" / "frozen.py").write_text("def api(x):\n    return x\n")

    def _apply(self, script, **kw):
        turns: list[dict] = []

        def fake_llm(**kwargs):
            turns.append({**kwargs, "messages": list(kwargs["messages"])})
            return SimpleNamespace(content="", tool_calls=[script(len(turns))])

        ed = AgentEditor(llm_caller=fake_llm, validators=[], max_attempts=1,
                         agentic_editing=True,
                         mutable_exclude=["workflow.py", "pkg/frozen.py"], **kw)
        return ed.apply(_fb(), self.tmp / "base", self.tmp / "out"), turns

    def test_default_off_offers_no_edit_file_tool(self) -> None:
        res, turns = self._apply(lambda n: _call(
            "submit_self_improvement_summary",
            {"optimization_goal": "g", "proposed_changes": "p", "rationale": "r"}, "s"))
        self.assertNotIn("edit_file", {t["name"] for t in turns[0]["tools"]})

    def test_edit_file_applies_and_is_recorded(self) -> None:
        def script(n):
            if n == 1:
                return _call("edit_file", {"path": "pkg/roles.py", "old_string": "Y = 2", "new_string": "Y = 3"}, "e")
            return _call("submit_self_improvement_summary",
                         {"optimization_goal": "g", "proposed_changes": "p", "rationale": "r"}, "s")
        res, turns = self._apply(script, agentic_edit_file=True)
        self.assertTrue(res.success, res.errors)
        self.assertIn("edit_file", {t["name"] for t in turns[0]["tools"]})
        self.assertIn("replaced 1", _last_output(turns[1]))
        self.assertEqual((self.tmp / "out" / "task_agent" / "pkg" / "roles.py").read_text(), "X = 1\nY = 3\n")
        self.assertEqual(res.edited_files, ["pkg/roles.py"])

    def test_readonly_reference_is_readable_but_not_writable(self) -> None:
        def script(n):
            if n == 1:
                return _call("read_file", {"path": "pkg/frozen.py"}, "r")
            if n == 2:
                return _call("write_file", {"path": "pkg/frozen.py", "content": "hacked"}, "w")
            if n == 3:
                return _call("edit_file", {"path": "pkg/frozen.py", "old_string": "x", "new_string": "y"}, "e")
            return _call("submit_self_improvement_summary",
                         {"optimization_goal": "g", "proposed_changes": "p", "rationale": "r"}, "s")
        res, turns = self._apply(script, agentic_edit_file=True, readonly_reference=["pkg/*.py"])
        self.assertIn("def api", _last_output(turns[1]))           # read ok
        self.assertIn("ERROR", _last_output(turns[2]))              # write refused
        self.assertIn("ERROR", _last_output(turns[3]))              # edit refused
        self.assertEqual((self.tmp / "out" / "task_agent" / "pkg" / "frozen.py").read_text(),
                         "def api(x):\n    return x\n")
        self.assertIn("Read-only reference files", turns[0]["messages"][1]["content"])
        self.assertIn("pkg/frozen.py", turns[0]["messages"][1]["content"])

    def test_readonly_absent_means_frozen_file_unreadable(self) -> None:
        def script(n):
            if n == 1:
                return _call("read_file", {"path": "pkg/frozen.py"}, "r")
            return _call("submit_self_improvement_summary",
                         {"optimization_goal": "g", "proposed_changes": "p", "rationale": "r"}, "s")
        res, turns = self._apply(script)
        self.assertIn("ERROR", _last_output(turns[1]))


class SuggesterReadonlyTests(unittest.TestCase):
    def test_frozen_files_join_sources_labelled(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            a = Path(d)
            (a / "pkg").mkdir()
            (a / "pkg" / "roles.py").write_text("X = 1\n")
            (a / "pkg" / "frozen.py").write_text("def api(): ...\n")
            bs = BlockSuggester(llm_caller=lambda **k: None, readonly_reference=["pkg/*.py"])
            frozen = bs._read_readonly_reference(a, {"pkg/roles.py": "X = 1\n"})
        self.assertEqual(sorted(frozen), ["pkg/frozen.py"])


if __name__ == "__main__":
    unittest.main()
