"""Tests for AgentEditor's opt-in agentic self-improvement mode
(meta_agent/agent_editor.py, ``agentic_editing=True``).

Real problem this replaces: confirmed live this session with DeepSeek v4
(via OpenRouter) as the editor -- bundling every changed file's full
content into one ``submit_self_improvement`` tool call causes a real,
reproducible malformed-JSON failure rate (~65% over 81 EXPANDs) on large
multi-file edits, root-caused to that bundled shape specifically. A
standalone prototype (one file's content per ``write_file`` call, in a
multi-turn loop with real validator feedback) eliminated the failure
entirely (0/40 malformed calls across single- and multi-file trials).
These tests wire that architecture into the real ``agent_editor.py``/
``apply()`` path and pin its behavior with a fake ``llm_caller`` -- no
network calls, matching the existing convention in
``tests/test_agent_editor_malformed_json_recovery.py``.

    PYTHONPATH=. python3 -m unittest tests.test_agent_editor_agentic_mode
"""
from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from meta_agent.agent_editor import (
    _AGENTIC_MALFORMED_SUMMARY_GOAL,
    _AGENTIC_TURN_BUDGET_GOAL,
    SELF_IMPROVEMENT_TOOL,
    AgentEditor,
)
from meta_agent.models import AgentFeedback, EvaluationResult, EvolutionStrategy


def _feedback() -> AgentFeedback:
    return AgentFeedback(
        round_number=1, base_round=0,
        strategy=EvolutionStrategy(
            target_files=[], optimization_goal="g", proposed_changes="x",
        ),
        eval_result=EvaluationResult(score=0.3, passed=0, failed=10),
    )


def _call(name: str, arguments, call_id: str | None = None):
    """A minimal tool-call stand-in matching this repo's established fake
    shape (bare SimpleNamespace, no .id unless explicitly given -- proves
    the new code tolerates the same fakes the pre-existing malformed-JSON
    tests already use)."""
    ns = SimpleNamespace(name=name, arguments=arguments)
    if call_id is not None:
        ns.id = call_id
    return ns


class AgenticModeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_base = Path(tempfile.mkdtemp(prefix="agent_editor_agentic_"))
        self.addCleanup(lambda: shutil.rmtree(self.tmp_base, ignore_errors=True))
        agent_dir = self.tmp_base / "base" / "task_agent"
        agent_dir.mkdir(parents=True)
        (agent_dir / "workflow.py").write_text(
            "def run_task(task):\n    return None\n", encoding="utf-8"
        )
        (agent_dir / "tool_wrapper.py").write_text("", encoding="utf-8")
        (agent_dir / "tools_schema.json").write_text("[]", encoding="utf-8")

    def _apply(self, fake_llm, **editor_kwargs):
        editor = AgentEditor(
            llm_caller=fake_llm, validators=[], max_attempts=1, **editor_kwargs
        )
        return editor.apply(_feedback(), self.tmp_base / "base", self.tmp_base / "out")

    def test_default_off_still_uses_single_shot_tool(self) -> None:
        calls: list[dict] = []

        def fake_llm(**kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                content="",
                tool_calls=[
                    _call(
                        "submit_self_improvement",
                        {
                            "optimization_goal": "g",
                            "proposed_changes": "p",
                            "rationale": "r",
                            "files": [{"path": "workflow.py", "content": "def run_task(task):\n    return 1\n"}],
                        },
                    )
                ],
            )

        result = self._apply(fake_llm)  # agentic_editing defaults to False
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["tools"], [SELF_IMPROVEMENT_TOOL])
        self.assertTrue(result.success)

    def test_happy_path_read_write_validate_submit(self) -> None:
        turns: list[dict] = []

        def fake_llm(**kwargs):
            turns.append(kwargs)
            n = len(turns)
            if n == 1:
                return SimpleNamespace(content="", tool_calls=[_call("read_file", {"path": "workflow.py"}, "c1")])
            if n == 2:
                return SimpleNamespace(content="", tool_calls=[_call(
                    "write_file",
                    {"path": "workflow.py", "content": "def run_task(task):\n    return 1\n"},
                    "c2",
                )])
            if n == 3:
                return SimpleNamespace(content="", tool_calls=[_call("run_code_validators", {}, "c3")])
            return SimpleNamespace(content="", tool_calls=[_call(
                "submit_self_improvement_summary",
                {"optimization_goal": "g", "proposed_changes": "p", "rationale": "r"},
                "c4",
            )])

        result = self._apply(fake_llm, agentic_editing=True)
        self.assertTrue(result.success, result.errors)
        self.assertEqual(result.edited_files, ["workflow.py"])
        self.assertEqual(result.strategy.optimization_goal, "g")
        # Tool schema offered must be the agentic set, not the bundled one.
        self.assertEqual(
            {t["name"] for t in turns[0]["tools"]},
            {"read_file", "write_file", "run_code_validators", "submit_self_improvement_summary"},
        )

    def test_multiple_files_separate_write_file_calls(self) -> None:
        (self.tmp_base / "base" / "task_agent" / "mutable_tools").mkdir()

        def fake_llm(**kwargs):
            n = sum(1 for m in kwargs["messages"] if m.get("type") == "function_call_output")
            if n == 0:
                return SimpleNamespace(content="", tool_calls=[
                    _call("write_file", {"path": "workflow.py", "content": "def run_task(task):\n    return 1\n"}, "a"),
                    _call("write_file", {"path": "mutable_tools/helper.py", "content": "X = 1\n"}, "b"),
                ])
            return SimpleNamespace(content="", tool_calls=[_call(
                "submit_self_improvement_summary",
                {"optimization_goal": "g", "proposed_changes": "p", "rationale": "r"},
                "c",
            )])

        result = self._apply(fake_llm, agentic_editing=True)
        self.assertTrue(result.success, result.errors)
        self.assertEqual(sorted(result.edited_files), ["mutable_tools/helper.py", "workflow.py"])

    def test_forbidden_write_path_not_written_and_reported_as_error(self) -> None:
        def fake_llm(**kwargs):
            n = sum(1 for m in kwargs["messages"] if m.get("type") == "function_call_output")
            if n == 0:
                return SimpleNamespace(content="", tool_calls=[
                    _call("write_file", {"path": "not_allowed.py", "content": "x = 1\n"}, "a"),
                ])
            # The forbidden-path error should be visible in the next turn's
            # tool output so the model can see what happened.
            last_output = kwargs["messages"][-1]
            self.assertIn("forbidden", last_output["output"])
            return SimpleNamespace(content="", tool_calls=[_call(
                "submit_self_improvement_summary",
                {"optimization_goal": "g", "proposed_changes": "p", "rationale": "r"},
                "c",
            )])

        result = self._apply(fake_llm, agentic_editing=True)
        self.assertEqual(result.edited_files, [])

    def test_read_file_on_a_directory_path_does_not_crash(self) -> None:
        """Real production crash (confirmed live 2026-09-08): in
        mutable_exclude mode, _is_path_allowed only checks the exclude
        list, not whether the path is actually a file -- a bare directory
        name like "agents" is allowed (nothing excludes it) but
        Path.read_text() on a directory raises IsADirectoryError, which
        used to propagate uncaught and kill the whole HGM process the
        first time an agentic-editing model asked to read a directory
        instead of one of its files."""
        agents_dir = self.tmp_base / "base" / "task_agent" / "agents"
        agents_dir.mkdir()
        (agents_dir / "sightseeing.py").write_text("X = 1\n", encoding="utf-8")

        turns: list[dict] = []

        def fake_llm(**kwargs):
            turns.append(kwargs)
            if len(turns) == 1:
                return SimpleNamespace(
                    content="", tool_calls=[_call("read_file", {"path": "agents"}, "c1")]
                )
            last_output = kwargs["messages"][-1]
            self.assertIn("directory", last_output["output"])
            return SimpleNamespace(content="", tool_calls=[_call(
                "submit_self_improvement_summary",
                {"optimization_goal": "g", "proposed_changes": "p", "rationale": "r"},
                "c2",
            )])

        # The real point of this test: apply() returns a normal result
        # object at all -- IsADirectoryError never propagates and crashes
        # the whole HGM process. Since the model never wrote a real file
        # in this scripted trial, apply()'s existing "no file edits"
        # outcome is the expected (not a new) failure mode here.
        result = self._apply(fake_llm, agentic_editing=True, mutable_exclude=[])
        self.assertEqual(result.errors, ["editor returned no file edits"])
        self.assertEqual(result.edited_files, [])

    def test_write_file_on_a_directory_path_does_not_crash(self) -> None:
        agents_dir = self.tmp_base / "base" / "task_agent" / "agents"
        agents_dir.mkdir()

        turns: list[dict] = []

        def fake_llm(**kwargs):
            turns.append(kwargs)
            if len(turns) == 1:
                return SimpleNamespace(
                    content="",
                    tool_calls=[_call("write_file", {"path": "agents", "content": "x = 1\n"}, "c1")],
                )
            last_output = kwargs["messages"][-1]
            self.assertIn("directory", last_output["output"])
            return SimpleNamespace(content="", tool_calls=[_call(
                "submit_self_improvement_summary",
                {"optimization_goal": "g", "proposed_changes": "p", "rationale": "r"},
                "c2",
            )])

        # The real point of this test: apply() returns a normal result
        # object at all -- IsADirectoryError never propagates and crashes
        # the whole HGM process. Since the model never wrote a real file
        # in this scripted trial, apply()'s existing "no file edits"
        # outcome is the expected (not a new) failure mode here.
        result = self._apply(fake_llm, agentic_editing=True, mutable_exclude=[])
        self.assertEqual(result.errors, ["editor returned no file edits"])
        self.assertEqual(result.edited_files, [])

    def test_malformed_write_file_args_recovers_on_retry(self) -> None:
        def fake_llm(**kwargs):
            n = sum(1 for m in kwargs["messages"] if m.get("type") == "function_call_output")
            if n == 0:
                return SimpleNamespace(content="", tool_calls=[_call("write_file", ["not", "a", "dict"], "a")])
            if n == 1:
                return SimpleNamespace(content="", tool_calls=[
                    _call("write_file", {"path": "workflow.py", "content": "def run_task(task):\n    return 1\n"}, "b"),
                ])
            return SimpleNamespace(content="", tool_calls=[_call(
                "submit_self_improvement_summary",
                {"optimization_goal": "g", "proposed_changes": "p", "rationale": "r"},
                "c",
            )])

        result = self._apply(fake_llm, agentic_editing=True)
        self.assertTrue(result.success, result.errors)
        self.assertEqual(result.edited_files, ["workflow.py"])

    def test_malformed_summary_after_real_writes_keeps_the_real_files(self) -> None:
        def fake_llm(**kwargs):
            n = sum(1 for m in kwargs["messages"] if m.get("type") == "function_call_output")
            if n == 0:
                return SimpleNamespace(content="", tool_calls=[
                    _call("write_file", {"path": "workflow.py", "content": "def run_task(task):\n    return 1\n"}, "a"),
                ])
            return SimpleNamespace(content="", tool_calls=[
                _call("submit_self_improvement_summary", {"_raw_arguments": "{not valid json"}, "b"),
            ])

        result = self._apply(fake_llm, agentic_editing=True)
        self.assertEqual(result.strategy.optimization_goal, _AGENTIC_MALFORMED_SUMMARY_GOAL)
        self.assertEqual(result.strategy.target_files, ["workflow.py"])
        # The real content must have actually been carried through to
        # apply()'s own _write_edits/_run_validators path, not just claimed.
        self.assertEqual(result.edited_files, ["workflow.py"])

    def test_turn_budget_exhausted_with_nothing_written_is_the_generic_no_edits_case(self) -> None:
        def fake_llm(**kwargs):
            return SimpleNamespace(content="thinking out loud", tool_calls=[])

        result = self._apply(fake_llm, agentic_editing=True, agentic_max_turns=1)
        self.assertFalse(result.success)
        self.assertEqual(result.strategy.target_files, [])

    def test_turn_budget_exhausted_with_partial_writes_keeps_them(self) -> None:
        def fake_llm(**kwargs):
            n = sum(1 for m in kwargs["messages"] if m.get("type") == "function_call_output")
            if n == 0:
                return SimpleNamespace(content="", tool_calls=[
                    _call("write_file", {"path": "workflow.py", "content": "def run_task(task):\n    return 1\n"}, "a"),
                ])
            # Never calls submit_self_improvement_summary -- budget runs out.
            return SimpleNamespace(content="", tool_calls=[_call("read_file", {"path": "workflow.py"}, "b")])

        result = self._apply(fake_llm, agentic_editing=True, agentic_max_turns=2)
        self.assertEqual(result.strategy.optimization_goal, _AGENTIC_TURN_BUDGET_GOAL)
        self.assertEqual(result.strategy.target_files, ["workflow.py"])

    def test_call_id_less_fakes_do_not_crash(self) -> None:
        # Matches tests/test_agent_editor_malformed_json_recovery.py's
        # existing fake shape exactly: no .id/.call_id attribute at all.
        def fake_llm(**kwargs):
            n = sum(1 for m in kwargs["messages"] if m.get("type") == "function_call_output")
            if n == 0:
                return SimpleNamespace(
                    content="", tool_calls=[SimpleNamespace(name="read_file", arguments={"path": "workflow.py"})]
                )
            return SimpleNamespace(
                content="",
                tool_calls=[SimpleNamespace(
                    name="submit_self_improvement_summary",
                    arguments={"optimization_goal": "g", "proposed_changes": "p", "rationale": "r"},
                )],
            )

        result = self._apply(fake_llm, agentic_editing=True)
        # Nothing was written (only a read happened), but the important
        # thing is this didn't crash building the function_call history.
        self.assertFalse(result.success)


if __name__ == "__main__":
    unittest.main()
