"""Unit tests for the two-stage editor (propose -> retrieve -> edit).

Stub LLM, no network; empty validator list.

    PYTHONPATH=. python3 -m unittest tests.test_two_stage_editor
"""
from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from dataclasses import dataclass, field
from pathlib import Path

from meta_agent import registry
from meta_agent.agent_editor import AgentEditor
from meta_agent.agent_editor_two_stage import TwoStageEditor
from meta_agent.config import ComponentSpec, _build_with_injection, _ensure_builtins_loaded
from meta_agent.edit_archive import RETRIEVAL_MANIFEST
from meta_agent.edit_beliefs import PREDICTION_NAME
from tests.test_edit_archive import write_code, write_record
from tests.test_edit_code import _agent


@dataclass
class _Resp:
    content: str = ""
    tool_calls: list = field(default_factory=list)


@dataclass
class _Call:
    name: str
    arguments: dict


PROPOSAL = {
    "edits": [{"goal": "repair the route gate", "mechanism": "check X first",
               "strategy": "add-verifier", "area": "routing"}],
    "prediction": {"belief_id": "add-verifier", "expected_direction": "up",
                   "expected_delta": 0.03, "why": "gate repair should help"},
    "memory_query": {"nodes": [2], "include_code": True},
}
IMPROVEMENT = {
    "optimization_goal": "g", "proposed_changes": "p", "rationale": "r",
    "files": [{"path": "workflow.py",
               "content": "def run_task(task):\n    return None\n"}],
}


class _EditorStub:
    """Anthropic-style tool dispatch (the editor modules' schema style)."""

    def __init__(self, propose_junk: bool = False):
        self.propose_junk = propose_junk
        self.calls: list[tuple[str, dict]] = []

    def __call__(self, **kw):
        name = kw["tools"][0]["name"]
        self.calls.append((name, kw))
        if name == "submit_edit_proposal":
            if self.propose_junk:
                return _Resp(content="not calling the tool")
            return _Resp(tool_calls=[_Call(name, PROPOSAL)])
        return _Resp(tool_calls=[_Call(name, IMPROVEMENT)])


class TwoStageBase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.base = self.tmp / "round_001"
        _agent(self.base, "def run_task(task):\n    return 1\n")
        write_record(self.tmp, 1, 0, "## Edit 1\n- **what**: seed-ish")
        rd2 = write_record(self.tmp, 2, 1,
                           "## Edit 1\n- **what**: Adds a route verifier")
        write_code(rd2, 2, 1)
        self.out = self.tmp / "round_003"

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)


class TestTwoStageApply(TwoStageBase):
    def test_full_flow_writes_artifacts_and_stage2_context(self):
        stub = _EditorStub()
        editor = TwoStageEditor(stub, [])
        result = editor.apply(None, self.base, self.out,
                              context="steering block here")
        self.assertTrue(result.success)
        self.assertEqual([n for n, _ in stub.calls],
                         ["submit_edit_proposal", "submit_self_improvement"])
        pred = json.loads((self.out / PREDICTION_NAME)
                          .read_text(encoding="utf-8"))
        self.assertEqual(pred["belief_id"], "add-verifier")
        self.assertEqual(pred["expected_direction"], "up")
        self.assertEqual(pred["query"]["nodes"], [2])
        manifest = json.loads((self.out / RETRIEVAL_MANIFEST)
                              .read_text(encoding="utf-8"))
        self.assertEqual(manifest["selected"][0]["node"], 2)
        stage2_user = stub.calls[1][1]["messages"][1]["content"]
        self.assertIn("steering block here", stage2_user)
        self.assertIn("Advisory proposal", stage2_user)
        self.assertIn("repair the route gate", stage2_user)
        self.assertIn("Retrieved node 2 (explicit)", stage2_user)
        self.assertIn("guard_2", stage2_user)  # code slice made it through
        # Stage 1 saw the parent's sources, read-only.
        stage1_user = stub.calls[0][1]["messages"][1]["content"]
        self.assertIn("Current sources", stage1_user)
        self.assertIn("return 1", stage1_user)

    def test_propose_junk_falls_back_to_single_call(self):
        stub = _EditorStub(propose_junk=True)
        editor = TwoStageEditor(stub, [])
        result = editor.apply(None, self.base, self.out, context="ctx")
        self.assertTrue(result.success)
        self.assertFalse((self.out / PREDICTION_NAME).exists())
        self.assertFalse((self.out / RETRIEVAL_MANIFEST).exists())
        stage2_user = stub.calls[-1][1]["messages"][1]["content"]
        self.assertNotIn("Advisory proposal", stage2_user)

    def test_propose_exception_falls_back(self):
        stub = _EditorStub()
        editor = TwoStageEditor(stub, [])

        def _boom(**_kw):
            raise RuntimeError("propose exploded")

        real = editor.llm

        def _dispatch(**kw):
            if kw["tools"][0]["name"] == "submit_edit_proposal":
                return _boom(**kw)
            return real(**kw)

        editor.llm = _dispatch
        result = editor.apply(None, self.base, self.out, context="ctx")
        self.assertTrue(result.success)
        self.assertFalse((self.out / PREDICTION_NAME).exists())

    def test_propose_disabled_is_plain_editor(self):
        stub = _EditorStub()
        editor = TwoStageEditor(stub, [], propose_enabled=False)
        result = editor.apply(None, self.base, self.out, context="ctx")
        self.assertTrue(result.success)
        self.assertEqual([n for n, _ in stub.calls],
                         ["submit_self_improvement"])

    def test_result_matches_plain_editor_output(self):
        stub = _EditorStub()
        TwoStageEditor(stub, []).apply(None, self.base, self.out, context="c")
        out_plain = self.tmp / "round_004"
        AgentEditor(_EditorStub(), []).apply(None, self.base, out_plain,
                                             context="c")
        self.assertEqual(
            (self.out / "task_agent" / "workflow.py").read_text(encoding="utf-8"),
            (out_plain / "task_agent" / "workflow.py").read_text(encoding="utf-8"))


class TestWiring(unittest.TestCase):
    def test_registry_resolves_two_stage(self):
        _ensure_builtins_loaded()
        self.assertIs(registry.get("editor", "two_stage"), TwoStageEditor)

    def test_build_with_injection_reaches_subclass(self):
        _ensure_builtins_loaded()
        stub = _EditorStub()
        spec = ComponentSpec(type="two_stage",
                             config={"max_attempts": 1,
                                     "retrieval_char_budget": 1234})
        obj = _build_with_injection(spec, "editor", {
            "llm_caller": stub, "validators": [],
            "tools_source": "TS", "db_schema": None, "scorer_source": None})
        self.assertIsInstance(obj, TwoStageEditor)
        self.assertIs(obj.llm, stub)              # injection landed
        self.assertEqual(obj.tools_source, "TS")  # parent kwarg forwarded
        self.assertEqual(obj.max_attempts, 1)     # config wins
        self.assertEqual(obj.retrieval_char_budget, 1234)


if __name__ == "__main__":
    unittest.main()
