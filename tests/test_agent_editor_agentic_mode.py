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
    _AGENTIC_EMPTY_RESPONSE_GOAL,
    _AGENTIC_MALFORMED_SUMMARY_GOAL,
    _run_python_problems,
    _AGENTIC_TURN_BUDGET_GOAL,
    SELF_IMPROVEMENT_TOOL,
    AgentEditor,
)
from meta_agent.models import AgentFeedback, CaseResult, EvaluationResult, EvolutionStrategy


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

        result = self._apply(
            fake_llm, agentic_editing=True,
            agentic_tools=[
                "read_file", "grep", "write_file", "str_replace_file",
                "run_code_validators", "run_python", "list_cases", "show_case",
                "submit_self_improvement_summary",
            ],
        )
        self.assertTrue(result.success, result.errors)
        self.assertEqual(result.edited_files, ["workflow.py"])
        self.assertEqual(result.strategy.optimization_goal, "g")
        # Tool schema offered must be the agentic set, not the bundled one.
        self.assertEqual(
            {t["name"] for t in turns[0]["tools"]},
            {
                "read_file", "grep", "write_file", "str_replace_file",
                "run_code_validators", "run_python", "list_cases", "show_case",
                "submit_self_improvement_summary",
            },
        )

    def test_agentic_editing_defaults_to_the_original_pre_expansion_tool_set(self) -> None:
        """agentic_tools=None (the default) must resolve to exactly the
        tools that existed before grep/str_replace_file/run_python/
        list_cases/show_case/evaluate_variant were added -- so every
        existing config's behavior is unchanged unless it explicitly
        opts into the newer tools (see AgentEditor.__init__'s
        agentic_tools param and _DEFAULT_AGENTIC_TOOL_NAMES)."""
        turns: list[dict] = []

        def fake_llm(**kwargs):
            turns.append(kwargs)
            if len(turns) == 1:
                return SimpleNamespace(content="", tool_calls=[_call(
                    "write_file",
                    {"path": "workflow.py", "content": "def run_task(task):\n    return 1\n"},
                    "c1",
                )])
            return SimpleNamespace(content="", tool_calls=[_call(
                "submit_self_improvement_summary",
                {"optimization_goal": "g", "proposed_changes": "p", "rationale": "r"},
                "c2",
            )])

        result = self._apply(fake_llm, agentic_editing=True)
        self.assertTrue(result.success, result.errors)
        self.assertEqual(
            {t["name"] for t in turns[0]["tools"]},
            {"read_file", "write_file", "run_code_validators", "submit_self_improvement_summary"},
        )

    def test_disabled_tool_called_anyway_is_rejected(self) -> None:
        """Defense in depth: even if a model somehow calls a tool outside
        the configured agentic_tools (hallucinated name, stale cached
        tool list), the dispatch must reject it rather than silently
        running it."""
        turns: list[dict] = []

        def fake_llm(**kwargs):
            turns.append(kwargs)
            if len(turns) == 1:
                return SimpleNamespace(content="", tool_calls=[_call("grep", {"path": "workflow.py", "pattern": "x"}, "c1")])
            if len(turns) == 2:
                last_output = kwargs["messages"][-1]
                self.assertIn("not enabled", last_output["output"])
                return SimpleNamespace(content="", tool_calls=[_call(
                    "write_file",
                    {"path": "workflow.py", "content": "def run_task(task):\n    return 1\n"},
                    "c2",
                )])
            return SimpleNamespace(content="", tool_calls=[_call(
                "submit_self_improvement_summary",
                {"optimization_goal": "g", "proposed_changes": "p", "rationale": "r"},
                "c3",
            )])

        result = self._apply(fake_llm, agentic_editing=True)  # default tool set -- no grep
        self.assertTrue(result.success, result.errors)

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

    def test_model_going_silent_early_is_labeled_distinctly_from_true_budget_exhaustion(self) -> None:
        """Confirmed live: DeepSeek v4 can return a totally empty turn (no
        content, no tool_calls) well before agentic_max_turns is reached,
        after already writing a real edit. That must NOT be reported as
        _AGENTIC_TURN_BUDGET_GOAL (false -- the budget wasn't exhausted) --
        it gets its own, accurate label, while still keeping whatever was
        written."""
        def fake_llm(**kwargs):
            n = sum(1 for m in kwargs["messages"] if m.get("type") == "function_call_output")
            if n == 0:
                return SimpleNamespace(content="", tool_calls=[
                    _call("write_file", {"path": "workflow.py", "content": "def run_task(task):\n    return 1\n"}, "a"),
                ])
            # Turn 2 of a 10-turn budget: model goes completely silent.
            return SimpleNamespace(content=None, tool_calls=[])

        result = self._apply(fake_llm, agentic_editing=True, agentic_max_turns=10)
        self.assertEqual(result.strategy.optimization_goal, _AGENTIC_EMPTY_RESPONSE_GOAL)
        self.assertEqual(result.strategy.target_files, ["workflow.py"])

    def test_empty_response_is_retried_without_consuming_a_turn(self) -> None:
        """The exact glitch confirmed live: a genuinely empty response
        (no content, no tool_calls, no exception) right after a
        substantive turn that was clearly mid-flow. A bare retry on the
        identical call should recover via ordinary sampling randomness
        -- and crucially must not cost an extra turn, so this succeeds
        even with agentic_max_turns=1."""
        raw_calls: list[int] = []

        def fake_llm(**kwargs):
            raw_calls.append(1)
            n = len(raw_calls)
            if n == 1:
                return SimpleNamespace(content=None, tool_calls=[])  # the glitch
            if n == 2:
                # The recovered response, still inside turn 0's single slot.
                return SimpleNamespace(content="", tool_calls=[
                    _call("write_file", {"path": "workflow.py", "content": "def run_task(task):\n    return 1\n"}, "a"),
                ])
            return SimpleNamespace(content="", tool_calls=[_call(
                "submit_self_improvement_summary",
                {"optimization_goal": "g", "proposed_changes": "p", "rationale": "r"}, "b",
            )])

        # agentic_max_turns=2: one turn for write_file, one for submit --
        # the glitch+retry both happen inside turn 0's own slot, not a
        # third turn, which is exactly the "doesn't consume a turn" claim.
        result = self._apply(
            fake_llm, agentic_editing=True, agentic_max_turns=2,
            agentic_tools=["read_file", "write_file", "submit_self_improvement_summary"],
        )
        self.assertEqual(len(raw_calls), 3)  # 1 glitch + write_file + submit
        self.assertTrue(result.success, result.errors)
        self.assertEqual(result.strategy.optimization_goal, "g")

    def test_empty_response_retry_gives_up_after_the_cap(self) -> None:
        """A persistently empty model must not retry forever -- it falls
        through to the existing, accurately-labeled stop after
        max_empty_response_retries attempts."""
        raw_calls: list[int] = []

        def fake_llm(**kwargs):
            raw_calls.append(1)
            return SimpleNamespace(content=None, tool_calls=[])

        result = self._apply(
            fake_llm, agentic_editing=True, agentic_max_turns=10,
            max_empty_response_retries=2,
        )
        self.assertEqual(len(raw_calls), 3)  # 1 original + 2 retries, then gives up
        self.assertEqual(result.strategy.optimization_goal, _AGENTIC_EMPTY_RESPONSE_GOAL)

    def test_list_cases_and_show_case_carry_an_explicit_stale_data_reminder(self) -> None:
        """Confirmed live: even with the system-prompt wording already in
        place, the model can still, deep into a long session, reason as
        if show_case might reflect its OWN str_replace_file/write_file
        edits ("the show_case may reflect that current code..."). A
        one-time system-prompt mention isn't enough by itself -- the
        reminder needs to be on the output itself, every time."""
        feedback = AgentFeedback(
            round_number=1, base_round=0,
            strategy=EvolutionStrategy(target_files=[], optimization_goal="g", proposed_changes="x"),
            eval_result=EvaluationResult(
                score=0.3, passed=0, failed=1,
                per_case=[CaseResult(case_id="1", passed=False, score=0.3, details={})],
            ),
        )
        turns: list[dict] = []

        def fake_llm(**kwargs):
            turns.append(kwargs)
            if len(turns) == 1:
                return SimpleNamespace(content="", tool_calls=[
                    _call("list_cases", {}, "a"),
                    _call("show_case", {"case_id": "1"}, "b"),
                ])
            return SimpleNamespace(content="", tool_calls=[_call(
                "submit_self_improvement_summary",
                {"optimization_goal": "g", "proposed_changes": "p", "rationale": "r"}, "c",
            )])

        editor = AgentEditor(
            llm_caller=fake_llm, validators=[], max_attempts=1,
            agentic_editing=True,
            agentic_tools=["list_cases", "show_case", "write_file", "submit_self_improvement_summary"],
        )
        editor.apply(feedback, self.tmp_base / "base", self.tmp_base / "out")
        outputs = [
            m["output"] for m in turns[1]["messages"]
            if m.get("type") == "function_call_output"
        ]
        self.assertEqual(len(outputs), 2)
        for o in outputs:
            self.assertIn("does NOT reflect your write_file/str_replace_file changes", o)

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


class _FakeEvaluator:
    """Mimics SubprocessEvaluator.run's side effect (writes round_dir/logs/
    trace.jsonl + case_<id>.json) without the real evaluator's machinery --
    enough to test isolation/aliasing, not to re-test the evaluator itself.
    Each call's written content is tagged with its own call number so
    tests can tell two calls' logs apart."""

    def __init__(self) -> None:
        self.calls: list[Path] = []

    def run(self, round_dir: Path, benchmark_dir: Path, *, case_ids):
        self.calls.append(round_dir)
        n = len(self.calls)
        logs_dir = round_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        (logs_dir / "trace.jsonl").write_text(
            f'{{"kind": "llm_call", "payload": {{"call_no": {n}}}}}\n', encoding="utf-8"
        )
        (logs_dir / "case_1.json").write_text(
            f'{{"case_id": "1", "call_no": {n}}}', encoding="utf-8"
        )
        return EvaluationResult(
            score=0.5, passed=0, failed=1,
            per_case=[CaseResult(case_id="1", passed=False, score=0.5, details={})],
        )


class InternalRunsAliasTests(unittest.TestCase):
    """read_file/grep's 'internal_runs/<rel>' alias -- reads from the MOST
    RECENT evaluate_variant call's own, isolated out_dir/internal_runs/
    call_<n>/logs/ dir. Never out_dir/logs/ itself (reserved for the real,
    framework-triggered evaluation that happens later) and never the
    parent's logs either (see list_cases/show_case for those)."""

    def setUp(self) -> None:
        self.tmp_base = Path(tempfile.mkdtemp(prefix="agent_editor_internal_runs_"))
        self.addCleanup(lambda: shutil.rmtree(self.tmp_base, ignore_errors=True))
        agent_dir = self.tmp_base / "base" / "task_agent"
        agent_dir.mkdir(parents=True)
        (agent_dir / "workflow.py").write_text(
            "def run_task(task):\n    return None\n", encoding="utf-8"
        )
        (agent_dir / "tool_wrapper.py").write_text("", encoding="utf-8")
        (agent_dir / "tools_schema.json").write_text("[]", encoding="utf-8")
        self.fake_evaluator = _FakeEvaluator()

    def _apply(self, fake_llm, **editor_kwargs):
        editor = AgentEditor(
            llm_caller=fake_llm, validators=[], max_attempts=1,
            evaluator=self.fake_evaluator, benchmark_dir=self.tmp_base / "bench",
            train_case_ids=["1"],
            **editor_kwargs,
        )
        return editor.apply(_feedback(), self.tmp_base / "base", self.tmp_base / "out")

    def _outputs(self, turns, index):
        return [
            m["output"] for m in turns[index]["messages"]
            if m.get("type") == "function_call_output"
        ]

    def test_internal_runs_unavailable_before_any_evaluate_variant_call(self) -> None:
        turns: list[dict] = []

        def fake_llm(**kwargs):
            turns.append(kwargs)
            if len(turns) == 1:
                return SimpleNamespace(
                    content="",
                    tool_calls=[_call("read_file", {"path": "internal_runs/trace.jsonl"}, "c1")],
                )
            return SimpleNamespace(content="", tool_calls=[_call(
                "submit_self_improvement_summary",
                {"optimization_goal": "g", "proposed_changes": "p", "rationale": "r"},
                "c2",
            )])

        self._apply(
            fake_llm, agentic_editing=True,
            agentic_tools=["read_file", "write_file", "submit_self_improvement_summary"],
        )
        self.assertIn("no evaluate_variant call has been made yet", self._outputs(turns, 1)[0])

    def test_evaluate_variant_then_read_file_internal_runs_reads_that_calls_logs(self) -> None:
        turns: list[dict] = []

        def fake_llm(**kwargs):
            turns.append(kwargs)
            if len(turns) == 1:
                return SimpleNamespace(
                    content="", tool_calls=[_call("evaluate_variant", {}, "c1")]
                )
            if len(turns) == 2:
                return SimpleNamespace(
                    content="",
                    tool_calls=[_call("read_file", {"path": "internal_runs/trace.jsonl"}, "c2")],
                )
            return SimpleNamespace(content="", tool_calls=[_call(
                "submit_self_improvement_summary",
                {"optimization_goal": "g", "proposed_changes": "p", "rationale": "r"},
                "c3",
            )])

        self._apply(
            fake_llm, agentic_editing=True,
            agentic_tools=["read_file", "write_file", "evaluate_variant", "submit_self_improvement_summary"],
        )
        # evaluate_variant's own output pointed at internal_runs/.
        self.assertIn("internal_runs/trace.jsonl", self._outputs(turns, 1)[0])
        # And read_file('internal_runs/trace.jsonl') actually reads THIS
        # call's own trace, tagged call_no=1.
        self.assertIn('"call_no": 1', self._outputs(turns, 2)[-1])

    def test_two_evaluate_variant_calls_do_not_clobber_each_other_on_disk(self) -> None:
        turns: list[dict] = []

        def fake_llm(**kwargs):
            turns.append(kwargs)
            if len(turns) in (1, 2):
                return SimpleNamespace(
                    content="", tool_calls=[_call("evaluate_variant", {}, f"c{len(turns)}")]
                )
            return SimpleNamespace(content="", tool_calls=[_call(
                "submit_self_improvement_summary",
                {"optimization_goal": "g", "proposed_changes": "p", "rationale": "r"},
                "c3",
            )])

        self._apply(
            fake_llm, agentic_editing=True,
            agentic_tools=["read_file", "write_file", "evaluate_variant", "submit_self_improvement_summary"],
        )
        call_1_trace = self.tmp_base / "out" / "internal_runs" / "call_1" / "logs" / "trace.jsonl"
        call_2_trace = self.tmp_base / "out" / "internal_runs" / "call_2" / "logs" / "trace.jsonl"
        self.assertIn('"call_no": 1', call_1_trace.read_text())
        self.assertIn('"call_no": 2', call_2_trace.read_text())

    def test_out_dir_logs_is_never_touched_by_evaluate_variant(self) -> None:
        """The core guarantee: out_dir/logs/ (reserved for the real,
        framework-triggered evaluation) must stay untouched by anything
        evaluate_variant does."""
        turns: list[dict] = []

        def fake_llm(**kwargs):
            turns.append(kwargs)
            if len(turns) == 1:
                return SimpleNamespace(
                    content="", tool_calls=[_call("evaluate_variant", {}, "c1")]
                )
            return SimpleNamespace(content="", tool_calls=[_call(
                "submit_self_improvement_summary",
                {"optimization_goal": "g", "proposed_changes": "p", "rationale": "r"},
                "c2",
            )])

        self._apply(
            fake_llm, agentic_editing=True,
            agentic_tools=["read_file", "write_file", "evaluate_variant", "submit_self_improvement_summary"],
        )
        self.assertFalse((self.tmp_base / "out" / "logs").exists())
        self.assertEqual(self.fake_evaluator.calls, [
            self.tmp_base / "out" / "internal_runs" / "call_1",
        ])

    def test_internal_runs_escape_guard_rejects_traversal(self) -> None:
        turns: list[dict] = []

        def fake_llm(**kwargs):
            turns.append(kwargs)
            if len(turns) == 1:
                return SimpleNamespace(
                    content="", tool_calls=[_call("evaluate_variant", {}, "c1")]
                )
            if len(turns) == 2:
                return SimpleNamespace(
                    content="",
                    tool_calls=[_call("read_file", {"path": "internal_runs/../../../etc/passwd"}, "c2")],
                )
            return SimpleNamespace(content="", tool_calls=[_call(
                "submit_self_improvement_summary",
                {"optimization_goal": "g", "proposed_changes": "p", "rationale": "r"},
                "c3",
            )])

        self._apply(
            fake_llm, agentic_editing=True,
            agentic_tools=["read_file", "write_file", "evaluate_variant", "submit_self_improvement_summary"],
        )
        self.assertIn("escapes the internal_runs/ root", self._outputs(turns, 2)[-1])

    def test_non_internal_runs_paths_still_resolve_against_agent_dir_as_before(self) -> None:
        """Regression guard: adding the internal_runs/ alias must not
        change resolution for every other path, which still goes through
        _is_path_allowed against agent_dir exactly as before."""
        turns: list[dict] = []

        def fake_llm(**kwargs):
            turns.append(kwargs)
            if len(turns) == 1:
                return SimpleNamespace(
                    content="", tool_calls=[_call("read_file", {"path": "workflow.py"}, "c1")]
                )
            return SimpleNamespace(content="", tool_calls=[_call(
                "submit_self_improvement_summary",
                {"optimization_goal": "g", "proposed_changes": "p", "rationale": "r"},
                "c2",
            )])

        self._apply(
            fake_llm, agentic_editing=True,
            agentic_tools=["read_file", "write_file", "submit_self_improvement_summary"],
        )
        self.assertIn("def run_task", self._outputs(turns, 1)[0])


class ReadFilePaginationTests(unittest.TestCase):
    """read_file's offset/limit pagination plus the hard character
    ceiling on top of it (see AgentEditor._paginated_read) -- added after
    a real, confirmed production failure: a read_file call on
    internal_runs/trace.jsonl returned 47,258,272 characters and blew
    past OpenRouter's 8MB total-request-size limit on the very next
    turn, ending that EXPAND early."""

    def setUp(self) -> None:
        self.tmp_base = Path(tempfile.mkdtemp(prefix="agent_editor_read_pagination_"))
        self.addCleanup(lambda: shutil.rmtree(self.tmp_base, ignore_errors=True))
        self.agent_dir = self.tmp_base / "base" / "task_agent"
        self.agent_dir.mkdir(parents=True)
        (self.agent_dir / "workflow.py").write_text(
            "def run_task(task):\n    return None\n", encoding="utf-8"
        )
        (self.agent_dir / "tool_wrapper.py").write_text("", encoding="utf-8")
        (self.agent_dir / "tools_schema.json").write_text("[]", encoding="utf-8")

    def _apply(self, fake_llm, **editor_kwargs):
        editor = AgentEditor(
            llm_caller=fake_llm, validators=[], max_attempts=1, **editor_kwargs
        )
        return editor.apply(_feedback(), self.tmp_base / "base", self.tmp_base / "out")

    def _outputs(self, turns, index):
        return [
            m["output"] for m in turns[index]["messages"]
            if m.get("type") == "function_call_output"
        ]

    def _read(self, path_args):
        turns: list[dict] = []

        def fake_llm(**kwargs):
            turns.append(kwargs)
            if len(turns) == 1:
                return SimpleNamespace(
                    content="", tool_calls=[_call("read_file", path_args, "c1")]
                )
            return SimpleNamespace(content="", tool_calls=[_call(
                "submit_self_improvement_summary",
                {"optimization_goal": "g", "proposed_changes": "p", "rationale": "r"},
                "c2",
            )])

        self._apply(
            fake_llm, agentic_editing=True,
            agentic_tools=["read_file", "write_file", "submit_self_improvement_summary"],
        )
        return self._outputs(turns, 1)[0]

    def test_small_file_returned_whole_unchanged(self) -> None:
        output = self._read({"path": "workflow.py"})
        self.assertEqual(output, "def run_task(task):\n    return None\n")

    def test_many_lines_truncates_with_offset_continuation_note(self) -> None:
        from meta_agent.agent_editor import _READ_FILE_DEFAULT_LINE_LIMIT

        n_lines = _READ_FILE_DEFAULT_LINE_LIMIT + 50
        (self.agent_dir / "workflow.py").write_text(
            "\n".join(f"line{i}" for i in range(n_lines)), encoding="utf-8"
        )
        output = self._read({"path": "workflow.py"})
        self.assertIn(f"line{_READ_FILE_DEFAULT_LINE_LIMIT - 1}", output)
        self.assertNotIn(f"line{_READ_FILE_DEFAULT_LINE_LIMIT}\n", output + "\n")
        self.assertIn(f"offset={_READ_FILE_DEFAULT_LINE_LIMIT}", output)
        self.assertIn("grep", output)

    def test_explicit_offset_and_limit_slice_correctly(self) -> None:
        (self.agent_dir / "workflow.py").write_text(
            "\n".join(f"line{i}" for i in range(100)), encoding="utf-8"
        )
        output = self._read({"path": "workflow.py", "offset": 10, "limit": 5})
        self.assertTrue(output.startswith("line10\nline11\nline12\nline13\nline14"))
        self.assertIn("offset=15", output)

    def test_single_huge_line_hits_hard_character_ceiling(self) -> None:
        from meta_agent.agent_editor import _READ_FILE_MAX_CHARS

        (self.agent_dir / "tool_wrapper.py").write_text(
            "x" * (_READ_FILE_MAX_CHARS * 3), encoding="utf-8"
        )
        output = self._read({"path": "tool_wrapper.py"})
        self.assertLess(len(output), _READ_FILE_MAX_CHARS * 3)
        self.assertIn(f"cut at {_READ_FILE_MAX_CHARS} characters", output)
        self.assertIn("grep", output)


class ToolsDirAliasTests(unittest.TestCase):
    """'tools/<rel>' read_file/grep alias -- the project's own tool
    implementation, read-only, gated behind tools_dir_alias_enabled
    (default False, so an existing config is unaffected; set True to A/B
    test whether letting the editor read tool source helps vs. relying
    on the capped upfront tools_source bundle alone)."""

    def setUp(self) -> None:
        self.tmp_base = Path(tempfile.mkdtemp(prefix="agent_editor_tools_dir_"))
        self.addCleanup(lambda: shutil.rmtree(self.tmp_base, ignore_errors=True))
        agent_dir = self.tmp_base / "base" / "task_agent"
        agent_dir.mkdir(parents=True)
        (agent_dir / "workflow.py").write_text("def run_task(task):\n    return None\n", encoding="utf-8")
        (agent_dir / "tool_wrapper.py").write_text("", encoding="utf-8")
        (agent_dir / "tools_schema.json").write_text("[]", encoding="utf-8")
        self.tools_dir = self.tmp_base / "project_tools"
        self.tools_dir.mkdir()
        (self.tools_dir / "roadroute.py").write_text(
            "def run(origin, destination):\n    return {'duration_in_minutes': 16}\n",
            encoding="utf-8",
        )

    def _apply(self, fake_llm, **editor_kwargs):
        editor = AgentEditor(
            llm_caller=fake_llm, validators=[], max_attempts=1, **editor_kwargs
        )
        return editor.apply(_feedback(), self.tmp_base / "base", self.tmp_base / "out")

    def _outputs(self, turns, index):
        return [
            m["output"] for m in turns[index]["messages"]
            if m.get("type") == "function_call_output"
        ]

    def test_disabled_by_default_even_with_tools_dir_set(self) -> None:
        turns: list[dict] = []

        def fake_llm(**kwargs):
            turns.append(kwargs)
            if len(turns) == 1:
                return SimpleNamespace(
                    content="", tool_calls=[_call("read_file", {"path": "tools/roadroute.py"}, "c1")]
                )
            return SimpleNamespace(content="", tool_calls=[_call(
                "submit_self_improvement_summary",
                {"optimization_goal": "g", "proposed_changes": "p", "rationale": "r"}, "c2",
            )])

        self._apply(
            fake_llm, agentic_editing=True, tools_dir=self.tools_dir,
            agentic_tools=["read_file", "write_file", "submit_self_improvement_summary"],
        )
        self.assertIn("tools_dir_alias_enabled is off", self._outputs(turns, 1)[0])

    def test_enabled_reads_the_real_tool_source(self) -> None:
        turns: list[dict] = []

        def fake_llm(**kwargs):
            turns.append(kwargs)
            if len(turns) == 1:
                return SimpleNamespace(
                    content="", tool_calls=[_call("read_file", {"path": "tools/roadroute.py"}, "c1")]
                )
            return SimpleNamespace(content="", tool_calls=[_call(
                "submit_self_improvement_summary",
                {"optimization_goal": "g", "proposed_changes": "p", "rationale": "r"}, "c2",
            )])

        self._apply(
            fake_llm, agentic_editing=True, tools_dir=self.tools_dir,
            tools_dir_alias_enabled=True,
            agentic_tools=["read_file", "write_file", "submit_self_improvement_summary"],
        )
        self.assertIn("duration_in_minutes", self._outputs(turns, 1)[0])

    def test_prompt_only_mentions_tools_alias_when_enabled(self) -> None:
        off = AgentEditor(
            llm_caller=lambda **kw: None, validators=[], agentic_editing=True,
            tools_dir=self.tools_dir, agentic_tools=["read_file"],
        )
        on = AgentEditor(
            llm_caller=lambda **kw: None, validators=[], agentic_editing=True,
            tools_dir=self.tools_dir, tools_dir_alias_enabled=True,
            agentic_tools=["read_file"],
        )
        self.assertNotIn("tools/", off._agentic_closing())
        self.assertIn("tools/roadroute.py", on._agentic_closing())


class FullMetricsJsonAliasTests(unittest.TestCase):
    """'full_metrics.json' read_file/grep alias -- resolves against
    base_dir (the PARENT round's own dir, where feedback_gatherer.py's
    _write_full_metrics actually writes it), not agent_dir. Gated by
    existence, not a flag -- whether the file exists depends on the
    PROJECT's scorer (full_metrics() is opt-in), not on anything
    AgentEditor itself configures."""

    def setUp(self) -> None:
        self.tmp_base = Path(tempfile.mkdtemp(prefix="agent_editor_full_metrics_"))
        self.addCleanup(lambda: shutil.rmtree(self.tmp_base, ignore_errors=True))
        self.base_dir = self.tmp_base / "base"
        agent_dir = self.base_dir / "task_agent"
        agent_dir.mkdir(parents=True)
        (agent_dir / "workflow.py").write_text("def run_task(task):\n    return None\n", encoding="utf-8")
        (agent_dir / "tool_wrapper.py").write_text("", encoding="utf-8")
        (agent_dir / "tools_schema.json").write_text("[]", encoding="utf-8")

    def _apply(self, fake_llm, **editor_kwargs):
        editor = AgentEditor(
            llm_caller=fake_llm, validators=[], max_attempts=1, **editor_kwargs
        )
        return editor.apply(_feedback(), self.base_dir, self.tmp_base / "out")

    def _outputs(self, turns, index):
        return [
            m["output"] for m in turns[index]["messages"]
            if m.get("type") == "function_call_output"
        ]

    def test_absent_gives_a_clear_error_not_a_crash(self) -> None:
        turns: list[dict] = []

        def fake_llm(**kwargs):
            turns.append(kwargs)
            if len(turns) == 1:
                return SimpleNamespace(
                    content="", tool_calls=[_call("read_file", {"path": "full_metrics.json"}, "c1")]
                )
            return SimpleNamespace(content="", tool_calls=[_call(
                "submit_self_improvement_summary",
                {"optimization_goal": "g", "proposed_changes": "p", "rationale": "r"}, "c2",
            )])

        self._apply(
            fake_llm, agentic_editing=True,
            agentic_tools=["read_file", "write_file", "submit_self_improvement_summary"],
        )
        self.assertIn("not available for the parent round", self._outputs(turns, 1)[0])

    def test_present_reads_from_base_dir_not_agent_dir(self) -> None:
        (self.base_dir / "full_metrics.json").write_text(
            '{"hard:flight_seat_status": {"fail_rate": 0.5}}', encoding="utf-8"
        )
        turns: list[dict] = []

        def fake_llm(**kwargs):
            turns.append(kwargs)
            if len(turns) == 1:
                return SimpleNamespace(
                    content="", tool_calls=[_call("read_file", {"path": "full_metrics.json"}, "c1")]
                )
            return SimpleNamespace(content="", tool_calls=[_call(
                "submit_self_improvement_summary",
                {"optimization_goal": "g", "proposed_changes": "p", "rationale": "r"}, "c2",
            )])

        self._apply(
            fake_llm, agentic_editing=True,
            agentic_tools=["read_file", "write_file", "submit_self_improvement_summary"],
        )
        self.assertIn("flight_seat_status", self._outputs(turns, 1)[0])

    def test_prompt_only_mentions_it_when_the_file_actually_exists(self) -> None:
        editor = AgentEditor(
            llm_caller=lambda **kw: None, validators=[], agentic_editing=True,
            agentic_tools=["read_file"],
        )
        self.assertNotIn("full_metrics.json", editor._agentic_closing(base_dir=self.base_dir))
        (self.base_dir / "full_metrics.json").write_text("{}", encoding="utf-8")
        self.assertIn("full_metrics.json", editor._agentic_closing(base_dir=self.base_dir))


class RunPythonOpenRejectionTests(unittest.TestCase):
    """Confirmed live: round_005's editor reached for `open()` inside
    run_python to read internal_runs/case_N.json's content, got a bare
    "name 'open' not allowed" with no path forward, and never retried
    with read_file (which was available and would have worked). The
    rejection message now says what to use instead."""

    def test_open_rejection_points_at_read_file(self) -> None:
        probs = _run_python_problems('open("internal_runs/case_34.json")', set())
        self.assertEqual(len(probs), 1)
        self.assertIn("run_python has no file I/O", probs[0])
        self.assertIn("read_file", probs[0])

    def test_other_blocked_names_unaffected(self) -> None:
        probs = _run_python_problems("x = exit", set())
        self.assertEqual(probs, ["name 'exit' not allowed"])


if __name__ == "__main__":
    unittest.main()
