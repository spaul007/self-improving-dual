"""Tests for BlockSuggester's opt-in agentic mode
(meta_agent/block_suggester.py, ``agentic_access=True``).

Mirrors tests/test_agent_editor_agentic_mode.py's conventions and fake-LLM
shape: instead of dumping every mutable source file upfront, the model
gets read-only `read_file`/`grep` tools (alias-rooted at 'harness/...',
'logs/...', 'eval_result.json') and a turn with no tool calls is the
normal, successful terminal case (its `content` IS the suggestion) --
unlike AgentEditor's agentic mode, suggest() only ever needs free-text
output, never a structured "submit" tool call.

    PYTHONPATH=. python3 -m unittest tests.test_block_suggester_agentic_mode
"""
from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from meta_agent.block_suggester import BlockSuggester


def _call(name: str, arguments, call_id: str | None = None):
    """Same minimal fake tool-call shape test_agent_editor_agentic_mode.py
    uses -- bare SimpleNamespace, no .id unless explicitly given."""
    ns = SimpleNamespace(name=name, arguments=arguments)
    if call_id is not None:
        ns.id = call_id
    return ns


class BlockSuggesterAgenticModeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="block_suggester_agentic_"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        # agent_dir = the PARENT's task_agent/; its parent (round_dir) is
        # where suggest() looks for this node's own real eval logs.
        self.round_dir = self.tmp / "round_005"
        self.agent_dir = self.round_dir / "task_agent"
        self.agent_dir.mkdir(parents=True)
        (self.agent_dir / "workflow.py").write_text(
            "def run_task(task):\n    return None\n", encoding="utf-8"
        )
        logs_dir = self.round_dir / "logs"
        logs_dir.mkdir()
        (logs_dir / "trace.jsonl").write_text(
            '{"event": "tool_call", "case_id": "1", "name": "search_location"}\n',
            encoding="utf-8",
        )
        (self.round_dir / "eval_result.json").write_text(
            '{"cases": [{"case_id": "1", "score": 0.0}]}', encoding="utf-8"
        )
        self.out_dir = self.tmp / "out_child"
        self.out_dir.mkdir()

    def _suggest(self, fake_llm, **kwargs):
        bs = BlockSuggester(llm_caller=fake_llm, agentic_access=True, **kwargs)
        return bs.suggest(
            block="verifiers",
            agent_dir=self.agent_dir,
            out_dir=self.out_dir,
            node_id=7,
        )

    def test_default_off_still_uses_upfront_source_dump(self) -> None:
        captured: dict[str, object] = {}

        def fake_llm(**kwargs):
            captured.update(kwargs)
            return SimpleNamespace(content="a plain suggestion", tool_calls=[])

        bs = BlockSuggester(llm_caller=fake_llm)
        result = bs.suggest(
            block="verifiers", agent_dir=self.agent_dir, out_dir=self.out_dir,
            node_id=7,
        )
        self.assertEqual(result, "a plain suggestion")
        self.assertNotIn("tools", captured)
        self.assertIn("## Current sources", captured["messages"][1]["content"])

    def test_no_tool_call_turn_is_the_terminal_case(self) -> None:
        def fake_llm(**kwargs):
            return SimpleNamespace(content="my grounded suggestion", tool_calls=[])

        result = self._suggest(fake_llm)
        self.assertEqual(result, "my grounded suggestion")
        self.assertEqual(
            (self.out_dir / "block_suggestion.md").read_text(encoding="utf-8"),
            "my grounded suggestion",
        )

    def test_read_file_serves_harness_source_from_memory(self) -> None:
        turns = iter([
            SimpleNamespace(
                content=None,
                tool_calls=[_call("read_file", {"path": "harness/workflow.py"}, "c1")],
            ),
            SimpleNamespace(content="found it: run_task returns None", tool_calls=[]),
        ])

        def fake_llm(**kwargs):
            return next(turns)

        result = self._suggest(fake_llm)
        self.assertEqual(result, "found it: run_task returns None")

    def test_read_file_can_reach_parent_round_dirs_eval_result(self) -> None:
        captured_outputs: list[str] = []
        turns = iter([
            SimpleNamespace(
                content=None,
                tool_calls=[_call("read_file", {"path": "eval_result.json"}, "c1")],
            ),
            SimpleNamespace(content="score was 0.0", tool_calls=[]),
        ])

        def fake_llm(**kwargs):
            response = next(turns)
            history = kwargs["messages"]
            for item in history:
                if item.get("type") == "function_call_output":
                    captured_outputs.append(item["output"])
            return response

        result = self._suggest(fake_llm)
        self.assertEqual(result, "score was 0.0")
        # Second call's history includes the first call's tool output.
        self.assertTrue(any('"score": 0.0' in o for o in captured_outputs))

    def test_grep_finds_match_deep_in_a_long_line(self) -> None:
        long_line = ("x" * 500) + "NEEDLE" + ("y" * 500)
        (self.round_dir / "logs" / "trace.jsonl").write_text(long_line, encoding="utf-8")
        turns = iter([
            SimpleNamespace(
                content=None,
                tool_calls=[_call(
                    "grep", {"path": "logs/trace.jsonl", "pattern": "NEEDLE"}, "c1"
                )],
            ),
            SimpleNamespace(content="found NEEDLE", tool_calls=[]),
        ])
        outputs: list[str] = []

        def fake_llm(**kwargs):
            for item in kwargs["messages"]:
                if item.get("type") == "function_call_output":
                    outputs.append(item["output"])
            return next(turns)

        result = self._suggest(fake_llm)
        self.assertEqual(result, "found NEEDLE")
        self.assertTrue(any("NEEDLE" in o for o in outputs))
        # A window, not the whole 1006-char line.
        self.assertTrue(all(len(o) < 600 for o in outputs))

    def test_read_file_can_reach_parent_round_dirs_full_metrics(self) -> None:
        (self.round_dir / "full_metrics.json").write_text(
            '{"hard:flight_seat_status": {"fail_rate": 0.5}}', encoding="utf-8"
        )
        turns = iter([
            SimpleNamespace(
                content=None,
                tool_calls=[_call("read_file", {"path": "full_metrics.json"}, "c1")],
            ),
            SimpleNamespace(content="rate was 0.5", tool_calls=[]),
        ])

        def fake_llm(**kwargs):
            return next(turns)

        result = self._suggest(fake_llm)
        self.assertEqual(result, "rate was 0.5")

    def test_read_file_full_metrics_absent_is_a_clear_not_found_not_a_crash(self) -> None:
        turns = iter([
            SimpleNamespace(
                content=None,
                tool_calls=[_call("read_file", {"path": "full_metrics.json"}, "c1")],
            ),
            SimpleNamespace(content="none available", tool_calls=[]),
        ])
        outputs: list[str] = []

        def fake_llm(**kwargs):
            for item in kwargs["messages"]:
                if item.get("type") == "function_call_output":
                    outputs.append(item["output"])
            return next(turns)

        result = self._suggest(fake_llm)
        self.assertEqual(result, "none available")
        self.assertTrue(any("not found" in o for o in outputs))

    def test_read_file_hard_character_ceiling_on_a_huge_single_line(self) -> None:
        """read_file's line-count limit alone doesn't bound worst-case
        size -- a real trace.jsonl event can be one enormous physical
        line (a full LLM call payload; these files run 100MB+ in
        production). The hard character ceiling must still cut it,
        separately from -- and in addition to -- the line-count cap."""
        from meta_agent.block_suggester import _READ_FILE_MAX_CHARS

        (self.round_dir / "logs" / "trace.jsonl").write_text(
            "x" * (_READ_FILE_MAX_CHARS * 3), encoding="utf-8"
        )
        outputs: list[str] = []
        turns = iter([
            SimpleNamespace(
                content=None,
                tool_calls=[_call("read_file", {"path": "logs/trace.jsonl"}, "c1")],
            ),
            SimpleNamespace(content="too big to read whole", tool_calls=[]),
        ])

        def fake_llm(**kwargs):
            for item in kwargs["messages"]:
                if item.get("type") == "function_call_output":
                    outputs.append(item["output"])
            return next(turns)

        result = self._suggest(fake_llm)
        self.assertEqual(result, "too big to read whole")
        self.assertTrue(outputs)
        self.assertLess(len(outputs[0]), _READ_FILE_MAX_CHARS * 3)
        self.assertIn(f"cut at {_READ_FILE_MAX_CHARS} characters", outputs[0])
        self.assertIn("grep", outputs[0])

    def test_forbidden_harness_path_is_rejected(self) -> None:
        outputs: list[str] = []
        turns = iter([
            SimpleNamespace(
                content=None,
                tool_calls=[_call("read_file", {"path": "harness/../secret.py"}, "c1")],
            ),
            SimpleNamespace(content="never mind", tool_calls=[]),
        ])

        def fake_llm(**kwargs):
            for item in kwargs["messages"]:
                if item.get("type") == "function_call_output":
                    outputs.append(item["output"])
            return next(turns)

        result = self._suggest(fake_llm)
        self.assertEqual(result, "never mind")
        self.assertTrue(any("ERROR" in o for o in outputs))

    def test_logs_path_cannot_escape_its_root(self) -> None:
        outputs: list[str] = []
        turns = iter([
            SimpleNamespace(
                content=None,
                tool_calls=[_call(
                    "read_file", {"path": "logs/../../secret.py"}, "c1"
                )],
            ),
            SimpleNamespace(content="never mind", tool_calls=[]),
        ])

        def fake_llm(**kwargs):
            for item in kwargs["messages"]:
                if item.get("type") == "function_call_output":
                    outputs.append(item["output"])
            return next(turns)

        result = self._suggest(fake_llm)
        self.assertEqual(result, "never mind")
        self.assertTrue(any("escapes the logs/ root" in o for o in outputs))

    def test_malformed_tool_args_do_not_crash(self) -> None:
        outputs: list[str] = []
        turns = iter([
            SimpleNamespace(
                content=None,
                tool_calls=[_call("read_file", {"_raw_arguments": "{not json"}, "c1")],
            ),
            SimpleNamespace(content="recovered anyway", tool_calls=[]),
        ])

        def fake_llm(**kwargs):
            for item in kwargs["messages"]:
                if item.get("type") == "function_call_output":
                    outputs.append(item["output"])
            return next(turns)

        result = self._suggest(fake_llm)
        self.assertEqual(result, "recovered anyway")
        self.assertTrue(any("ERROR" in o for o in outputs))

    def test_call_id_less_fake_tool_calls_dont_crash(self) -> None:
        turns = iter([
            SimpleNamespace(
                content=None,
                tool_calls=[_call("read_file", {"path": "harness/workflow.py"})],
            ),
            SimpleNamespace(content="ok", tool_calls=[]),
        ])

        def fake_llm(**kwargs):
            return next(turns)

        result = self._suggest(fake_llm)
        self.assertEqual(result, "ok")

    def test_turn_budget_exhausted_without_final_answer_returns_none(self) -> None:
        def fake_llm(**kwargs):
            return SimpleNamespace(
                content=None,
                tool_calls=[_call("read_file", {"path": "harness/workflow.py"}, "c1")],
            )

        result = self._suggest(fake_llm, agentic_max_turns=2)
        self.assertIsNone(result)
        self.assertFalse((self.out_dir / "block_suggestion.md").exists())

    def test_llm_call_failure_returns_none(self) -> None:
        def fake_llm(**kwargs):
            raise RuntimeError("boom")

        result = self._suggest(fake_llm)
        self.assertIsNone(result)


class BlockSuggesterRunPythonTests(unittest.TestCase):
    """Tests for the run_python agentic tool added to BlockSuggester --
    lets a diagnosis be falsified against every one of this node's own
    evaluated cases (not just the one example cited), mirroring
    AgentEditor's own run_python (meta_agent/agent_editor.py). See
    block_suggester.py's _SYSTEM_CLOSING_AGENTIC for the falsification
    discipline this tool exists to support.

    PYTHONPATH=. python3 -m unittest tests.test_block_suggester_agentic_mode
    """

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="block_suggester_run_python_"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.round_dir = self.tmp / "round_005"
        self.agent_dir = self.round_dir / "task_agent"
        self.agent_dir.mkdir(parents=True)
        (self.agent_dir / "workflow.py").write_text(
            "def run_task(task):\n    return None\n", encoding="utf-8"
        )
        (self.round_dir / "logs").mkdir()
        self.out_dir = self.tmp / "out_child"
        self.out_dir.mkdir()

    def _feedback(self, per_case):
        from meta_agent.models import AgentFeedback, EvaluationResult, EvolutionStrategy

        return AgentFeedback(
            round_number=5, base_round=0,
            strategy=EvolutionStrategy(optimization_goal="", proposed_changes=""),
            eval_result=EvaluationResult(score=0.0, per_case=per_case),
        )

    def _suggest(self, fake_llm, feedback=None, **kwargs):
        from meta_agent.block_suggester import BlockSuggester

        kwargs.setdefault("agentic_tools", ["read_file", "grep", "run_python"])
        bs = BlockSuggester(llm_caller=fake_llm, agentic_access=True, **kwargs)
        return bs.suggest(
            block="verifiers", agent_dir=self.agent_dir, out_dir=self.out_dir,
            node_id=7, feedback=feedback,
        )

    def test_run_python_sees_this_nodes_own_cases(self) -> None:
        from meta_agent.models import CaseResult

        feedback = self._feedback([
            CaseResult(case_id="1", passed=True, score=1.0, details={}),
            CaseResult(case_id="2", passed=False, score=0.0,
                       error="plan conversion failed: agent produced no plan"),
        ])
        turns = iter([
            SimpleNamespace(
                content=None,
                tool_calls=[_call(
                    "run_python",
                    {"code": "import train_data\nprint(len(train_data.CASES))\nprint(train_data.by_id('2')['passed'])"},
                    "c1",
                )],
            ),
            SimpleNamespace(content="confirmed: 2 cases, case 2 failed", tool_calls=[]),
        ])
        captured_outputs: list[str] = []

        def fake_llm(**kwargs):
            response = next(turns)
            for item in kwargs["messages"]:
                if item.get("type") == "function_call_output":
                    captured_outputs.append(item["output"])
            return response

        result = self._suggest(fake_llm, feedback=feedback)
        self.assertEqual(result, "confirmed: 2 cases, case 2 failed")
        self.assertTrue(any("exit code 0" in o and "2\nFalse" in o for o in captured_outputs))

    def test_run_python_with_no_feedback_sees_empty_cases(self) -> None:
        turns = iter([
            SimpleNamespace(
                content=None,
                tool_calls=[_call("run_python", {"code": "import train_data\nprint(train_data.CASES)"}, "c1")],
            ),
            SimpleNamespace(content="no cases available", tool_calls=[]),
        ])

        def fake_llm(**kwargs):
            return next(turns)

        result = self._suggest(fake_llm, feedback=None)
        self.assertEqual(result, "no cases available")

    def test_run_python_rejects_disallowed_import(self) -> None:
        turns = iter([
            SimpleNamespace(
                content=None,
                tool_calls=[_call("run_python", {"code": "import os\nprint(os.getcwd())"}, "c1")],
            ),
            SimpleNamespace(content="can't do that", tool_calls=[]),
        ])
        captured_outputs: list[str] = []

        def fake_llm(**kwargs):
            response = next(turns)
            for item in kwargs["messages"]:
                if item.get("type") == "function_call_output":
                    captured_outputs.append(item["output"])
            return response

        result = self._suggest(fake_llm)
        self.assertEqual(result, "can't do that")
        self.assertTrue(any("REJECTED" in o for o in captured_outputs))

    def test_run_python_missing_code_is_an_error(self) -> None:
        turns = iter([
            SimpleNamespace(content=None, tool_calls=[_call("run_python", {}, "c1")]),
            SimpleNamespace(content="ok", tool_calls=[]),
        ])
        captured_outputs: list[str] = []

        def fake_llm(**kwargs):
            response = next(turns)
            for item in kwargs["messages"]:
                if item.get("type") == "function_call_output":
                    captured_outputs.append(item["output"])
            return response

        result = self._suggest(fake_llm)
        self.assertEqual(result, "ok")
        self.assertTrue(any("requires `code`" in o for o in captured_outputs))

    def test_agentic_access_defaults_to_the_original_pre_expansion_tool_set(self) -> None:
        """agentic_tools=None (the default) must resolve to exactly the
        tools that existed before list_cases/show_case/run_python were
        added (see BlockSuggester.__init__'s agentic_tools param and
        _DEFAULT_AGENTIC_TOOL_NAMES) -- read_file and grep only."""
        captured: dict[str, object] = {}

        def fake_llm(**kwargs):
            captured.update(kwargs)
            return SimpleNamespace(content="a plain suggestion", tool_calls=[])

        from meta_agent.block_suggester import BlockSuggester
        bs = BlockSuggester(llm_caller=fake_llm, agentic_access=True)  # no agentic_tools override
        result = bs.suggest(
            block="verifiers", agent_dir=self.agent_dir, out_dir=self.out_dir, node_id=7,
        )
        self.assertEqual(result, "a plain suggestion")
        self.assertEqual({t["name"] for t in captured["tools"]}, {"read_file", "grep"})

    def test_disabled_tool_called_anyway_is_rejected(self) -> None:
        """Defense in depth: calling a tool outside the configured
        agentic_tools must be rejected, not silently run."""
        turns = iter([
            SimpleNamespace(
                content=None,
                tool_calls=[_call("run_python", {"code": "print(1)"}, "c1")],
            ),
            SimpleNamespace(content="done", tool_calls=[]),
        ])
        captured_outputs: list[str] = []

        def fake_llm(**kwargs):
            response = next(turns)
            for item in kwargs["messages"]:
                if item.get("type") == "function_call_output":
                    captured_outputs.append(item["output"])
            return response

        # read_file/grep only -- no run_python -- so the call must be rejected.
        result = self._suggest(fake_llm, agentic_tools=["read_file", "grep"])
        self.assertEqual(result, "done")
        self.assertTrue(any("not enabled" in o for o in captured_outputs))

    def test_hypothesis_discipline_defaults_false_and_is_byte_identical(self) -> None:
        """hypothesis_discipline defaults to False -- every existing config
        that doesn't set this new key gets the lean run_python description,
        with no falsification-discipline paragraph."""
        bs_default = BlockSuggester(
            llm_caller=lambda **kw: None, agentic_tools=["read_file", "grep", "run_python"],
        )
        bs_explicit_false = BlockSuggester(
            llm_caller=lambda **kw: None, agentic_tools=["read_file", "grep", "run_python"],
            hypothesis_discipline=False,
        )
        self.assertEqual(bs_default._agentic_closing(), bs_explicit_false._agentic_closing())
        self.assertNotIn("a hypothesis,", bs_default._agentic_closing())

    def test_hypothesis_discipline_true_adds_paragraph_keeps_tool(self) -> None:
        """hypothesis_discipline=True opts into the falsification-discipline
        paragraph on top of run_python's own mechanical description (and
        every other tool's), which stays untouched."""
        bs = BlockSuggester(
            llm_caller=lambda **kw: None, agentic_tools=["read_file", "grep", "run_python"],
            hypothesis_discipline=True,
        )
        out = bs._agentic_closing()
        self.assertIn("a hypothesis,", out)
        self.assertIn("run_python(code)", out)
        self.assertIn("train_data", out)


if __name__ == "__main__":
    unittest.main()
