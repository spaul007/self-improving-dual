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


if __name__ == "__main__":
    unittest.main()
