"""AgenticEditor end to end with a scripted LLM (no API key, no bwrap).

Each test scripts the sequence of responses the "model" returns; the editor
runs its real tools (fallback sandbox), real validators and real bookkeeping
on a temp run directory.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from meta_agent import registry
from meta_agent.agent_editor_agentic import AgenticEditor
from meta_agent.agentic.policy import RUN_ROOT_MARKER
from meta_agent.agentic.session import (
    AGENTIC_SYSTEM_PROMPT,
    NUDGE_MESSAGE,
    SESSION_NAME,
    TRANSCRIPT_NAME,
)
from meta_agent.agentic.tools import SUBMIT_TOOL, SUBMIT_TOOL_NAME
from meta_agent.config import ComponentSpec, _build_with_injection, _ensure_builtins_loaded
from meta_agent.edit_beliefs import PREDICTION_NAME
from meta_agent.editor_validators import (
    ImmutableFilesValidator,
    SignatureValidator,
    SyntaxValidator,
)
from tests.test_edit_code import _agent

WF = "def run_task(task):\n    x = 1\n    return None\n"


@dataclass
class _Call:
    id: str
    name: str
    arguments: dict


@dataclass
class _Resp:
    content: str = ""
    tool_calls: list = field(default_factory=list)
    raw: object = None


class _ScriptedLLM:
    """Returns the scripted responses in order; records every call."""

    def __init__(self, steps: list) -> None:
        self.steps = list(steps)
        self.calls: list[dict] = []

    def __call__(self, **kw):
        # Snapshot: the session keeps appending to the same messages list.
        self.calls.append({**kw, "messages": list(kw["messages"])})
        if not self.steps:
            raise AssertionError("scripted LLM ran out of steps")
        step = self.steps.pop(0)
        if isinstance(step, Exception):
            raise step
        return step


def _submit(cid="s1", **extra) -> _Call:
    return _Call(cid, SUBMIT_TOOL_NAME, {"optimization_goal": "goal", "proposed_changes": "p", "rationale": "r", **extra})


class EditorBase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self.run = root / "run"
        self.run.mkdir()
        (self.run / RUN_ROOT_MARKER).write_text("")
        (self.run / "edit_memory_beliefs.md").write_text("### belief:b — x\n")
        self.base = self.run / "round_001"
        self.out = self.run / "round_002"
        _agent(self.base, WF)
        (self.base / "feedback.json").write_text('{"score": 0.4}')
        self.wf = self.out / "task_agent" / "workflow.py"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def editor(self, steps, **kw) -> tuple[AgenticEditor, _ScriptedLLM]:
        llm = _ScriptedLLM(steps)
        opts = dict(sandbox="none", max_llm_calls=10, timeout_s=60, max_attempts=3)
        opts.update(kw)
        ed = AgenticEditor(
            llm, [SyntaxValidator(), SignatureValidator(), ImmutableFilesValidator()],
            **opts,
        )
        return ed, llm

    def replace(self, cid, old, new) -> _Call:
        return _Call(cid, "editor", {"command": "str_replace", "path": str(self.wf),
                                     "old_str": old, "new_str": new})

    def transcript(self) -> list[dict]:
        lines = (self.out / "agentic" / TRANSCRIPT_NAME).read_text().splitlines()
        return [json.loads(l) for l in lines]

    def session(self) -> dict:
        return json.loads((self.out / "agentic" / SESSION_NAME).read_text())


class TestHappyPath(EditorBase):
    def test_view_edit_validate_submit(self) -> None:
        ed, llm = self.editor([
            _Resp(content="looking", tool_calls=[_Call("c1", "editor", {"command": "view", "path": str(self.wf)})]),
            _Resp(tool_calls=[self.replace("c2", "x = 1", "x = 2")]),
            _Resp(tool_calls=[_Call("c3", "validate", {})]),
            _Resp(tool_calls=[_submit(prediction={"belief_id": "belief:b", "expected_direction": "up", "expected_delta": 0.02, "why": "w"})]),
        ])
        res = ed.apply(None, self.base, self.out, context="STEERING TEXT")
        self.assertTrue(res.success, res.errors)
        self.assertEqual(res.edited_files, ["workflow.py"])
        self.assertEqual(res.strategy.optimization_goal, "goal")
        self.assertEqual(res.strategy.target_files, ["workflow.py"])
        self.assertEqual(self.wf.read_text(), "def run_task(task):\n    x = 2\n    return None\n")
        # Untouched files are byte-identical to the parent.
        self.assertEqual((self.out / "task_agent" / "tool_wrapper.py").read_text(),
                         (self.base / "task_agent" / "tool_wrapper.py").read_text())

        # Prompts: system carries the shared rules; the instruction has paths
        # and budget only — no source, no feedback, no steering context.
        first = llm.calls[0]
        self.assertEqual(first["messages"][0]["content"], AGENTIC_SYSTEM_PROMPT)
        instr = first["messages"][1]["content"]
        self.assertIn(str(self.out / "task_agent" / "workflow.py"), instr)
        self.assertIn(str(self.base), instr)
        self.assertIn("edit_memory_beliefs.md", instr)
        self.assertIn("at most 10 model calls", instr)
        self.assertNotIn("x = 1", instr)
        self.assertNotIn("STEERING TEXT", instr)
        self.assertNotIn("Last round's feedback", instr)
        self.assertEqual([t["name"] for t in first["tools"]],
                         ["bash", "editor", "validate", SUBMIT_TOOL_NAME])
        self.assertEqual(first["temperature"], 0.2)

        # Multi-turn history: the second call carries the echoed function_call
        # and a function_call_output keyed by the first call id.
        second = llm.calls[1]["messages"]
        fc = [m for m in second if isinstance(m, dict) and m.get("type") == "function_call"]
        self.assertEqual(fc[0]["call_id"], "c1")
        outs = [m for m in second if isinstance(m, dict) and m.get("type") == "function_call_output"]
        self.assertEqual(outs[0]["call_id"], "c1")
        self.assertIn("cat -n", outs[0]["output"])
        self.assertEqual([m for m in second if m.get("role") == "assistant"][0]["content"], "looking")
        # validate result reached the model
        self.assertIn("All validators passed", llm.calls[3]["messages"][-1]["output"])

        # Artifacts
        kinds = [e["kind"] for e in self.transcript()]
        self.assertEqual(kinds.count("llm_call"), 4)
        self.assertEqual(kinds.count("tool_call"), 4)
        self.assertIn("validation", kinds)
        self.assertEqual(self.transcript()[-1]["reason"], "submitted")
        s = self.session()
        self.assertTrue(s["success"])
        self.assertEqual(s["end_reason"], "submitted")
        self.assertEqual(s["validation_rounds"], 1)
        self.assertEqual(s["n_llm_calls"], 4)
        self.assertEqual(s["n_tool_calls"], {"editor": 2, "validate": 1, SUBMIT_TOOL_NAME: 1})
        self.assertEqual(s["sandbox_mode"], "none")
        self.assertTrue((self.out / "agentic" / "scratch").is_dir())
        pred = json.loads((self.out / PREDICTION_NAME).read_text())
        self.assertEqual(pred["belief_id"], "b")
        self.assertEqual(pred["expected_direction"], "up")

    def test_manager_context_opt_in(self) -> None:
        ed, llm = self.editor([_Resp(tool_calls=[self.replace("c1", "x = 1", "x = 3"), _submit()])],
                              include_manager_context=True)
        self.assertTrue(ed.apply(None, self.base, self.out, context="STEERING TEXT").success)
        self.assertIn("## Steering context\nSTEERING TEXT", llm.calls[0]["messages"][1]["content"])

    def test_multiple_tool_calls_in_one_response(self) -> None:
        ed, llm = self.editor([_Resp(tool_calls=[
            _Call("c1", "editor", {"command": "view", "path": str(self.wf), "view_range": [1, 1]}),
            self.replace("c2", "x = 1", "x = 5"),
            _submit("c3"),
            _Call("c4", "bash", {"command": "echo never"}),
        ])])
        res = ed.apply(None, self.base, self.out)
        self.assertTrue(res.success)
        self.assertEqual(llm.calls.__len__(), 1)
        outs = [e for e in self.transcript() if e["kind"] == "tool_call"]
        self.assertEqual([o["call_id"] for o in outs], ["c1", "c2", "c3", "c4"])
        self.assertIn("already accepted", outs[3]["result"])
        self.assertEqual(self.session()["n_tool_calls"]["bash"], 1)

    def test_raw_output_items_are_echoed_and_reasoning_stripped_only_on_flag(self) -> None:
        import os
        raw = type("Raw", (), {})()
        raw.output = [{"type": "reasoning", "id": "r1"},
                      {"type": "function_call", "call_id": "c1", "name": "validate", "arguments": "{}"}]
        raw.usage = type("U", (), {"input_tokens": 10, "output_tokens": 5, "output_tokens_details": None})()
        ed, llm = self.editor([
            _Resp(tool_calls=[_Call("c1", "validate", {})], raw=raw),
            _Resp(tool_calls=[self.replace("c2", "x = 1", "x = 4"), _submit()]),
        ])
        os.environ["META_AGENT_STRIP_REASONING"] = "1"
        try:
            self.assertTrue(ed.apply(None, self.base, self.out).success)
        finally:
            del os.environ["META_AGENT_STRIP_REASONING"]
        second = llm.calls[1]["messages"]
        self.assertNotIn({"type": "reasoning", "id": "r1"}, second)
        self.assertIn(raw.output[1], second)
        self.assertEqual(self.session()["usage"], {"input_tokens": 10, "output_tokens": 5})


class TestValidationLoop(EditorBase):
    def test_failure_is_fed_back_and_workspace_kept(self) -> None:
        ed, llm = self.editor([
            _Resp(tool_calls=[self.replace("c1", "x = 1", "x = (")]),      # syntax error
            _Resp(tool_calls=[_submit("s1")]),
            _Resp(tool_calls=[self.replace("c2", "x = (", "x = 2")]),      # fix in place
            _Resp(tool_calls=[_submit("s2")]),
        ])
        res = ed.apply(None, self.base, self.out)
        self.assertTrue(res.success, res.errors)
        fail_out = llm.calls[2]["messages"][-1]["output"]
        self.assertIn("Validation failed (attempt 1/3)", fail_out)
        self.assertIn("workspace is kept", fail_out)
        self.assertTrue(any("syntax" in l.lower() or "Syntax" in l for l in fail_out.splitlines()))
        # After the failed submit the broken edit was still on disk (call 3
        # fixed it via str_replace of the broken text, which only works if
        # the workspace was not reset).
        self.assertEqual(self.wf.read_text(), "def run_task(task):\n    x = 2\n    return None\n")
        self.assertEqual(self.session()["validation_rounds"], 2)

    def test_repeated_failure_exhausts_attempts(self) -> None:
        ed, llm = self.editor([
            _Resp(tool_calls=[self.replace("c1", "x = 1", "x = (")]),
            _Resp(tool_calls=[_submit("s1")]),
            _Resp(tool_calls=[_submit("s2")]),
            _Resp(tool_calls=[_submit("s3")]),
        ])
        res = ed.apply(None, self.base, self.out)
        self.assertFalse(res.success)
        self.assertEqual(self.session()["end_reason"], "max_attempts")
        self.assertTrue(any("syntax" in e.lower() for e in res.errors), res.errors)
        self.assertIsNone(res.strategy)
        last_tool = [e for e in self.transcript() if e["kind"] == "tool_call"][-1]
        self.assertIn("No attempts left", last_tool["result"])
        self.assertEqual(len(llm.calls), 4)  # no wrap-up call after max_attempts

    def test_submit_without_changes(self) -> None:
        ed, llm = self.editor([
            _Resp(tool_calls=[_submit("s1")]),
            _Resp(tool_calls=[self.replace("c1", "x = 1", "x = 9"), _submit("s2")]),
        ])
        res = ed.apply(None, self.base, self.out)
        self.assertTrue(res.success)
        self.assertIn("no changes detected", llm.calls[1]["messages"][-1]["output"])
        self.assertEqual(self.session()["validation_rounds"], 2)

    def test_immutable_file_change_is_caught(self) -> None:
        # The model's bash writes a stray file into task_agent (possible in
        # fallback mode, impossible under bwrap): the immutable-files
        # validator rejects the submission.
        ed, llm = self.editor([
            _Resp(tool_calls=[
                _Call("c1", "bash", {"command": "echo x > extra.py"}),
                self.replace("c2", "x = 1", "x = 2"),
                _submit("s1"),
            ]),
            _Resp(tool_calls=[_submit("s2")]),
            _Resp(tool_calls=[_submit("s3")]),
        ])
        res = ed.apply(None, self.base, self.out)
        self.assertFalse(res.success)
        self.assertTrue(any("extra.py" in e for e in res.errors), res.errors)


class TestTermination(EditorBase):
    def test_nudge_then_submit(self) -> None:
        ed, llm = self.editor([
            _Resp(tool_calls=[self.replace("c1", "x = 1", "x = 2")]),
            _Resp(content="I think I'm done."),
            _Resp(tool_calls=[_submit()]),
        ])
        self.assertTrue(ed.apply(None, self.base, self.out).success)
        third = llm.calls[2]["messages"]
        self.assertEqual(third[-1], {"role": "user", "content": NUDGE_MESSAGE})
        self.assertEqual(third[-2], {"role": "assistant", "content": "I think I'm done."})
        self.assertIn("nudge", [e["kind"] for e in self.transcript()])

    def test_two_silent_turns_then_wrap_up_submits(self) -> None:
        ed, llm = self.editor([
            _Resp(tool_calls=[self.replace("c1", "x = 1", "x = 2")]),
            _Resp(content="done"),
            _Resp(content="really done"),
            _Resp(tool_calls=[_submit()]),
        ])
        res = ed.apply(None, self.base, self.out)
        self.assertTrue(res.success)
        self.assertEqual(self.session()["end_reason"], "submitted_after_no_tool_calls")
        wrap = llm.calls[3]
        self.assertEqual(wrap["tools"], [SUBMIT_TOOL])
        self.assertIn("budget is exhausted (no_tool_calls)", wrap["messages"][-1]["content"])

    def test_max_llm_calls_wrap_up_variants(self) -> None:
        # (a) wrap-up returns a submit → success with the model's summary
        ed, llm = self.editor([
            _Resp(tool_calls=[self.replace("c1", "x = 1", "x = 2")]),
            _Resp(tool_calls=[_Call("c2", "bash", {"command": "echo hi"})]),
            _Resp(tool_calls=[_submit()]),
        ], max_llm_calls=2)
        res = ed.apply(None, self.base, self.out)
        self.assertTrue(res.success)
        self.assertEqual(res.strategy.optimization_goal, "goal")
        self.assertEqual(self.session()["end_reason"], "submitted_after_max_llm_calls")
        self.assertEqual(self.session()["n_llm_calls"], 3)

        # (b) wrap-up yields nothing, edit validates → accepted with placeholder
        ed, llm = self.editor([
            _Resp(tool_calls=[self.replace("c1", "x = 1", "x = 2")]),
            _Resp(content="thinking..."),
        ], max_llm_calls=1)
        res = ed.apply(None, self.base, self.out)
        self.assertTrue(res.success)
        self.assertTrue(res.strategy.optimization_goal.startswith("(agentic editor: no summary submitted"))
        self.assertEqual(res.strategy.proposed_changes, "thinking...")
        self.assertEqual(self.session()["end_reason"], "max_llm_calls")

        # (c) no changes at all → failure
        ed, llm = self.editor([
            _Resp(tool_calls=[_Call("c1", "bash", {"command": "echo hi"})]),
            _Resp(content="nothing"),
        ], max_llm_calls=1)
        res = ed.apply(None, self.base, self.out)
        self.assertFalse(res.success)
        self.assertIn("no changes made", res.errors)
        self.assertTrue(res.errors[0].startswith("agentic session ended without"))

    def test_budget_reminders_fire_at_halfway_and_final_stretch(self) -> None:
        from meta_agent.agentic.session import (
            FINAL_STRETCH_MESSAGE, HALFWAY_MESSAGE, budget_thresholds,
        )
        self.assertEqual(budget_thresholds(10), (3, 5, 8))
        self.assertEqual(budget_thresholds(5), (0, 0, 0))
        self.assertEqual(budget_thresholds(150), (50, 75, 142))
        bash = lambda cid: _Call(cid, "bash", {"command": "echo hi"})
        ed, llm = self.editor(
            [_Resp(tool_calls=[bash(f"c{i}")]) for i in range(8)]
            + [_Resp(tool_calls=[self.replace("c8", "x = 1", "x = 2"), _submit()])],
            max_llm_calls=10,
        )
        res = ed.apply(None, self.base, self.out)
        self.assertTrue(res.success)
        # Reminders appended after calls 3 and 5 (no edit yet) and after
        # call 8 (final stretch); nowhere else.
        for idx, kw in enumerate(llm.calls):
            last = kw["messages"][-1]
            is_user = isinstance(last, dict) and last.get("role") == "user"
            if idx in (3, 5):
                self.assertEqual(last["content"], HALFWAY_MESSAGE.format(used=idx, total=10))
            elif idx == 8:
                self.assertEqual(last["content"],
                                 FINAL_STRETCH_MESSAGE.format(left=2, used=8, total=10))
            elif idx > 0:
                self.assertFalse(is_user, (idx, last))
        self.assertEqual([e["used"] for e in self.transcript() if e["kind"] == "budget_reminder"], [3, 5, 8])

    def test_halfway_reminder_skipped_once_an_edit_exists(self) -> None:
        bash = lambda cid: _Call(cid, "bash", {"command": "echo hi"})
        ed, llm = self.editor(
            [_Resp(tool_calls=[self.replace("c0", "x = 1", "x = 2")])]
            + [_Resp(tool_calls=[bash(f"c{i}")]) for i in range(1, 5)]
            + [_Resp(tool_calls=[_submit()])],
            max_llm_calls=10,
        )
        self.assertTrue(ed.apply(None, self.base, self.out).success)
        self.assertEqual([e["used"] for e in self.transcript() if e["kind"] == "budget_reminder"], [])

    def test_llm_exception_ends_session(self) -> None:
        ed, llm = self.editor([
            _Resp(tool_calls=[self.replace("c1", "x = 1", "x = 2")]),
            RuntimeError("endpoint down"),
        ])
        res = ed.apply(None, self.base, self.out)
        self.assertFalse(res.success)
        self.assertIn("endpoint down", res.errors[0])
        self.assertEqual(self.session()["end_reason"], "llm_error")
        self.assertEqual(len(llm.calls), 2)  # no wrap-up on a dead endpoint
        self.assertIn("llm_error", [e["kind"] for e in self.transcript()])

    def test_malformed_arguments_do_not_break_the_loop(self) -> None:
        ed, llm = self.editor([
            _Resp(tool_calls=[_Call("c1", "editor", {"_raw_arguments": "{not json"})]),
            _Resp(tool_calls=[_Call("c2", "nosuchtool", {"a": 1})]),
            _Resp(tool_calls=[self.replace("c3", "x = 1", "x = 2"), _submit()]),
        ])
        res = ed.apply(None, self.base, self.out)
        self.assertTrue(res.success)
        self.assertIn("could not parse", llm.calls[1]["messages"][-1]["output"])
        self.assertIn("not found", llm.calls[2]["messages"][-1]["output"])

    def test_timeout_is_checked_before_each_call(self) -> None:
        ed, llm = self.editor([
            _Resp(tool_calls=[self.replace("c1", "x = 1", "x = 2"),
                              _Call("c2", "bash", {"command": "sleep 0.6"})]),
            _Resp(tool_calls=[_submit()]),
        ], timeout_s=0.5)
        res = ed.apply(None, self.base, self.out)
        # After the first turn 0.9 * 0.5 s have elapsed → the loop stops before
        # a second regular call; the wrap-up call submits.
        self.assertTrue(res.success)
        self.assertEqual(self.session()["end_reason"], "submitted_after_timeout")
        self.assertEqual(len(llm.calls), 2)
        self.assertEqual(llm.calls[1]["tools"], [SUBMIT_TOOL])


class TestConcurrencyAndWiring(EditorBase):
    def test_parallel_applies_are_independent(self) -> None:
        outs = [self.run / f"round_00{i}" for i in (3, 4)]
        eds = []
        for i, o in enumerate(outs):
            wf = o / "task_agent" / "workflow.py"
            ed, _ = self.editor([_Resp(tool_calls=[
                _Call("c1", "editor", {"command": "str_replace", "path": str(wf),
                                        "old_str": "x = 1", "new_str": f"x = {i + 7}"}),
                _submit(),
            ])])
            eds.append((ed, o, i))
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda t: t[0].apply(None, self.base, t[1]), eds))
        self.assertTrue(all(r.success for r in results))
        for _, o, i in eds:
            self.assertIn(f"x = {i + 7}", (o / "task_agent" / "workflow.py").read_text())

    def test_registry_and_injection(self) -> None:
        _ensure_builtins_loaded()
        self.assertIs(registry.get("editor", "agentic"), AgenticEditor)
        ed = _build_with_injection(
            ComponentSpec(type="agentic", config={"max_llm_calls": 5, "sandbox": "none"}),
            "editor",
            {"llm_caller": lambda **kw: None, "validators": [], "tools_source": "TS",
             "db_schema": "DB", "project_root": Path("/p/travel"),
             "eval_visibility": "whitebox", "scorer_source": "SC"},
        )
        self.assertIsInstance(ed, AgenticEditor)
        self.assertEqual(ed.max_llm_calls, 5)
        self.assertEqual(ed.project_root, Path("/p/travel"))
        self.assertEqual(ed.sandbox, "none")
        self.assertFalse(ed.include_manager_context)
        self.assertIsNone(ed.api_key_env)
        self.assertIsNone(ed.llm_timeout_s)

    def test_api_key_env_and_timeout_reach_the_llm_call(self) -> None:
        ed, llm = self.editor([_Resp(tool_calls=[self.replace("c1", "x = 1", "x = 3"), _submit()])],
                              api_key_env="OpenRouter_API_KEY", llm_timeout_s=600,
                              model="deepseek/deepseek-v4-pro-0813",
                              base_url="https://openrouter.ai/api/v1", reasoning_effort="low")
        self.assertTrue(ed.apply(None, self.base, self.out).success)
        kw = llm.calls[0]
        self.assertEqual(kw["api_key_env"], "OpenRouter_API_KEY")
        self.assertEqual(kw["timeout_s"], 600.0)
        self.assertEqual(kw["model"], "deepseek/deepseek-v4-pro-0813")
        self.assertEqual(kw["base_url"], "https://openrouter.ai/api/v1")
        self.assertEqual(kw["reasoning_effort"], "low")
        self.assertNotIn("temperature", kw)
        # Default editor: neither key is sent, so stubs/other providers see
        # exactly the historical kwargs.
        ed2, llm2 = self.editor([_Resp(tool_calls=[self.replace("c1", "x = 1", "x = 3"), _submit()])])
        ed2.apply(None, self.base, self.out)
        self.assertNotIn("api_key_env", llm2.calls[0])
        self.assertNotIn("timeout_s", llm2.calls[0])

    def test_build_components_from_agentic_configs(self) -> None:
        from meta_agent.config import REPO_ROOT, build_components, load
        for name in ("hgm_travel_smoke_agentic",
                     "hgm_travel_1000_qwen122b_node5_agentic_editmem",
                     "hgm_travel_100_dsv4pro_agentic_editmem",
                     "hgm_travel_100_dsv4pro_agentic_no_editmem"):
            cfg = load(REPO_ROOT / "configs" / f"{name}.yaml")
            fw = build_components(cfg)
            self.assertIsInstance(fw.editor, AgenticEditor, name)
            self.assertEqual(fw.editor.project_root, REPO_ROOT / "projects" / "travel")
            self.assertEqual(len(fw.validators), 8)
            self.assertEqual(fw.editor.api_key_env, "OpenRouter_API_KEY")
            self.assertEqual(fw.editor.llm_timeout_s, 600.0)
            self.assertTrue(fw.editor.model.startswith("deepseek/deepseek-v4-pro"))
            self.assertEqual(fw.editor.base_url, "https://openrouter.ai/api/v1")
            self.assertFalse(fw.editor.include_manager_context)

    def test_sanity_pair_differs_only_in_edit_memory(self) -> None:
        from meta_agent.config import REPO_ROOT, build_components, load
        a = load(REPO_ROOT / "configs" / "hgm_travel_100_dsv4pro_agentic_editmem.yaml")
        b = load(REPO_ROOT / "configs" / "hgm_travel_100_dsv4pro_agentic_no_editmem.yaml")
        self.assertIsNotNone(a.edit_memory)
        self.assertIsNone(b.edit_memory)
        for cfg in (a, b):
            self.assertEqual(cfg.task_agent.model, "deepseek/deepseek-v4-pro-0813")
            self.assertEqual(cfg.task_agent.reasoning_effort, "none")
            self.assertTrue(cfg.verbose)
            self.assertEqual(cfg.env["LLM_API_KEY_ENV"], "OpenRouter_API_KEY")
            self.assertEqual(cfg.manager.config["eval_budget"], 100)
            self.assertEqual(cfg.manager.config["finalize_top_k"], 0)
            fw = build_components(cfg)
            self.assertEqual(len(fw.train_case_ids), 60)
        da, db = a.model_dump(), b.model_dump()
        for k in ("edit_memory", "experiment_name"):
            da.pop(k), db.pop(k)
        # Each arm reuses ITS OWN earlier seed pre-eval (same seed code).
        for d in (da, db):
            self.assertIn("seed_round_dir", d["manager"]["config"])
            d["manager"]["config"].pop("seed_round_dir")
        self.assertEqual(da, db)


class TestPromptContract(unittest.TestCase):
    def test_agentic_prompt_carries_shared_rules(self) -> None:
        for needle in ("  1. workflow.py MUST define", "  7. Instrument your edits",
                       "from platform_core import tools", "ONLY in tool_wrapper.py",
                       "str_replace", "submit_self_improvement", "validate"):
            self.assertIn(needle, AGENTIC_SYSTEM_PROMPT)
        self.assertNotIn("shown below", AGENTIC_SYSTEM_PROMPT)


if __name__ == "__main__":
    unittest.main()
