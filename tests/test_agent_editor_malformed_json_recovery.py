"""Tests for AgentEditor's malformed-JSON recovery path
(meta_agent/agent_editor.py).

Real bug this guards against: confirmed live 2026-09-04 with DeepSeek v4
(via OpenRouter) as the editor model -- a complete, well-reasoned fix (a
new validator stage, full file contents) was produced, but the tool call's
own JSON failed to parse (platform_core.llm_wrapper.call_llm wraps this as
{"_raw_arguments": <raw string>}), which agent_editor.py silently treated
as "editor returned no file edits" -- discarding real work with no
actionable feedback for the retry. This adds a distinct, informative error
message on retry ("make sure you return a valid JSON object...") instead
of the generic one.

    PYTHONPATH=. python3 -m unittest tests.test_agent_editor_malformed_json_recovery
"""
from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from meta_agent.agent_editor import _MALFORMED_JSON_GOAL, AgentEditor
from meta_agent.models import AgentFeedback, EvaluationResult, EvolutionStrategy


def _feedback() -> AgentFeedback:
    return AgentFeedback(
        round_number=1, base_round=0,
        strategy=EvolutionStrategy(
            target_files=[], optimization_goal="g", proposed_changes="x",
        ),
        eval_result=EvaluationResult(score=0.3, passed=0, failed=10),
    )


class MalformedJsonRecoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_base = Path(tempfile.mkdtemp(prefix="agent_editor_malformed_json_"))
        self.addCleanup(lambda: shutil.rmtree(self.tmp_base, ignore_errors=True))
        agent_dir = self.tmp_base / "base" / "task_agent"
        agent_dir.mkdir(parents=True)
        (agent_dir / "workflow.py").write_text(
            "def run_task(task):\n    return None\n", encoding="utf-8"
        )

    def test_raw_arguments_response_is_treated_as_zero_files_with_marker_goal(self) -> None:
        def fake_llm(**kwargs):
            return SimpleNamespace(
                content="some reasoning",
                tool_calls=[
                    SimpleNamespace(
                        name="submit_self_improvement",
                        arguments={"_raw_arguments": '{"files": [{"path": "x.py"'},
                    )
                ],
            )

        editor = AgentEditor(llm_caller=fake_llm, validators=[], max_attempts=1)
        result = editor.apply(_feedback(), self.tmp_base / "base", self.tmp_base / "out")
        self.assertFalse(result.success)
        self.assertEqual(result.strategy.optimization_goal, _MALFORMED_JSON_GOAL)
        self.assertEqual(result.edited_files, [])
        # Nothing was actually parsed -- claiming a target file (even a
        # placeholder like workflow.py) would misrepresent what happened.
        self.assertEqual(result.strategy.target_files, [])

    def test_non_dict_arguments_does_not_crash_and_is_treated_as_malformed(self) -> None:
        # Valid JSON that isn't an object (e.g. a bare list) never hits
        # llm_wrapper.py's _raw_arguments fallback (json.loads succeeds),
        # but args.get("files") would previously raise AttributeError --
        # confirmed live, this crashed the entire HGM run, not just one
        # EXPAND, since nothing up to main_loop.py's evolve() call catches
        # it. Must degrade the same way as a real JSON parse failure.
        for bad_arguments in (["not", "a", "dict"], "just a string", None, 42):
            with self.subTest(bad_arguments=bad_arguments):
                def fake_llm(**kwargs):
                    return SimpleNamespace(
                        content="",
                        tool_calls=[
                            SimpleNamespace(
                                name="submit_self_improvement",
                                arguments=bad_arguments,
                            )
                        ],
                    )

                editor = AgentEditor(llm_caller=fake_llm, validators=[], max_attempts=1)
                result = editor.apply(
                    _feedback(), self.tmp_base / "base", self.tmp_base / "out"
                )
                self.assertFalse(result.success)
                self.assertEqual(result.strategy.optimization_goal, _MALFORMED_JSON_GOAL)
                self.assertEqual(result.strategy.target_files, [])

    def test_retry_prompt_gets_actionable_json_guidance_not_generic_message(self) -> None:
        captured_prompts: list[str] = []
        calls = {"n": 0}

        def fake_llm(**kwargs):
            calls["n"] += 1
            captured_prompts.append(kwargs["messages"][1]["content"])
            if calls["n"] == 1:
                return SimpleNamespace(
                    content="",
                    tool_calls=[
                        SimpleNamespace(
                            name="submit_self_improvement",
                            arguments={"_raw_arguments": "{not valid json"},
                        )
                    ],
                )
            return SimpleNamespace(
                content="",
                tool_calls=[
                    SimpleNamespace(
                        name="submit_self_improvement",
                        arguments={
                            "optimization_goal": "fix it",
                            "proposed_changes": "fix it",
                            "rationale": "because",
                            "files": [
                                {
                                    "path": "workflow.py",
                                    "content": "def run_task(task):\n    return 1\n",
                                }
                            ],
                        },
                    )
                ],
            )

        editor = AgentEditor(llm_caller=fake_llm, validators=[], max_attempts=2)
        result = editor.apply(_feedback(), self.tmp_base / "base", self.tmp_base / "out")
        self.assertEqual(calls["n"], 2)
        self.assertIn("valid JSON object", captured_prompts[1])
        self.assertNotIn("editor returned no file edits", captured_prompts[1])
        self.assertTrue(result.success)
        self.assertEqual(result.edited_files, ["workflow.py"])

    def test_genuinely_empty_files_still_gets_the_generic_message(self) -> None:
        # A model that calls the tool with well-formed but empty `files`
        # (not a JSON parse failure) must keep the original generic
        # message -- only the _raw_arguments case gets the JSON-specific one.
        captured_prompts: list[str] = []

        def fake_llm(**kwargs):
            captured_prompts.append(kwargs["messages"][1]["content"])
            return SimpleNamespace(
                content="",
                tool_calls=[
                    SimpleNamespace(
                        name="submit_self_improvement",
                        arguments={
                            "optimization_goal": "g",
                            "proposed_changes": "p",
                            "rationale": "r",
                            "files": [],
                        },
                    )
                ],
            )

        editor = AgentEditor(llm_caller=fake_llm, validators=[], max_attempts=2)
        editor.apply(_feedback(), self.tmp_base / "base", self.tmp_base / "out")
        self.assertIn("editor returned no file edits", captured_prompts[1])
        self.assertNotIn("valid JSON object", captured_prompts[1])


class NoToolCallFallbackTargetFilesTests(unittest.TestCase):
    """The OTHER pre-existing fallback (model didn't call
    submit_self_improvement at all) had the exact same misleading
    target_files=["workflow.py"] placeholder -- fixed alongside the
    malformed-JSON case above for consistency."""

    def setUp(self) -> None:
        self.tmp_base = Path(tempfile.mkdtemp(prefix="agent_editor_no_tool_call_"))
        self.addCleanup(lambda: shutil.rmtree(self.tmp_base, ignore_errors=True))
        agent_dir = self.tmp_base / "base" / "task_agent"
        agent_dir.mkdir(parents=True)
        (agent_dir / "workflow.py").write_text(
            "def run_task(task):\n    return None\n", encoding="utf-8"
        )

    def test_no_recovery_yields_empty_target_files_not_workflow_py(self) -> None:
        def fake_llm(**kwargs):
            return SimpleNamespace(content="I decided not to call the tool.", tool_calls=[])

        editor = AgentEditor(llm_caller=fake_llm, validators=[], max_attempts=1)
        result = editor.apply(_feedback(), self.tmp_base / "base", self.tmp_base / "out")
        self.assertEqual(result.strategy.target_files, [])

    def test_fenced_json_recovery_reports_the_real_recovered_paths(self) -> None:
        content = (
            "Here is my fix:\n```json\n"
            '{"files": [{"path": "agents/sightseeing.py", "content": "x"}]}\n'
            "```\n"
        )

        def fake_llm(**kwargs):
            return SimpleNamespace(content=content, tool_calls=[])

        editor = AgentEditor(llm_caller=fake_llm, validators=[], max_attempts=1)
        result = editor.apply(_feedback(), self.tmp_base / "base", self.tmp_base / "out")
        self.assertEqual(result.strategy.target_files, ["agents/sightseeing.py"])


if __name__ == "__main__":
    unittest.main()
