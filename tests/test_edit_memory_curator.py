"""The curator sessions: read/write surface, the document check on submit,
and the wrap-up fallback (scripted LLM, sandbox "none")."""
from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from dataclasses import dataclass, field
from pathlib import Path

from meta_agent.edit_memory import generator as G
from meta_agent.edit_memory import prompts as P
from meta_agent.edit_memory.curator import CuratorConfig, run_curator
from meta_agent.edit_memory.policy import build_curator_policy
from tests.test_agentic_policy import _fake_repo


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
    def __init__(self, steps):
        self.steps = list(steps)
        self.calls = []

    def __call__(self, **kw):
        self.calls.append({**kw, "messages": list(kw["messages"])})
        if not self.steps:
            raise AssertionError("out of steps")
        return self.steps.pop(0)


def _round(exp: Path, nid: int, code: str) -> Path:
    rd = exp / f"round_{nid:03d}"
    (rd / "task_agent" / "mutable_tools").mkdir(parents=True)
    (rd / "task_agent" / "workflow.py").write_text(code)
    (rd / "task_agent" / "tool_wrapper.py").write_text("")
    (rd / "task_agent" / "tools_schema.json").write_text("[]")
    (rd / "logs").mkdir()
    (rd / "logs" / "case_1.json").write_text(json.dumps({"case_id": "1", "passed": nid % 2 == 0, "score": 0.5}))
    (rd / "strategy.json").write_text(json.dumps({"optimization_goal": f"goal of {nid}"}))
    (rd / "feedback.json").write_text("{}")
    (rd / "agentic").mkdir()
    (rd / "agentic" / "transcript.jsonl").write_text('{"kind": "llm_call"}\n')
    return rd


def _curation(ids) -> str:
    out = ""
    for nid in ids:
        out += P.NODE_SECTION_HEADING.format(node_id=nid) + "\n" + \
            "".join(f"### {s}\nx\n" for s in P.NODE_SUBSECTIONS)
    return out + P.CURATION_CROSS_HEADING + "\nx\n" + P.CURATION_GRADIENT_HEADING + "\nx\n"


class CuratorBase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="curator_"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.repo, self.proj = _fake_repo(self.tmp)
        self.exp = self.tmp / "runs" / "exp"
        self.exp.mkdir(parents=True)
        (self.exp / "config.snapshot.yaml").write_text("")
        self.r0 = _round(self.exp, 0, "def run_task(task):\n    return None\n")
        self.r1 = _round(self.exp, 1, "def run_task(task):\n    return 1\n")
        self.r2 = _round(self.exp, 2, "def run_task(task):\n    return 2\n")
        self.r3 = _round(self.exp, 3, "def run_task(task):\n    return 3\n")   # not in the window
        self.mem = self.exp / "edit_memory"
        self.mem.mkdir()
        (self.mem / "edit_memory.md").write_text("# memory v1\n")
        (self.mem / "instruction.md").write_text("- addendum\n")
        self.work = self.mem / "window_001"
        self.policy = build_curator_policy(
            workspace=self.work, memory_dir=self.mem, node_dirs=[self.r1, self.r2],
            parent_dirs=[self.r0, self.r0], repo_root=self.repo, project_root=self.proj,
        )

    def run_memory_curator(self, steps, **cfg):
        llm = _ScriptedLLM(steps)
        res = run_curator(
            llm, policy=self.policy, system_prompt=P.MEMORY_CURATOR_SYSTEM,
            instruction=P.render_memory_curation_instruction(
                nodes=[{"node_id": 1, "parent_id": 0, "memory_arm": "none", "memory_version": None,
                        "n_evals": 4, "mean_utility": 0.5, "edit_failed": False},
                       {"node_id": 2, "parent_id": 0, "memory_arm": "none", "memory_version": None,
                        "n_evals": 4, "mean_utility": 0.6, "edit_failed": False}],
                previous_memory_exists=True, addendum="- addendum", max_llm_calls=8,
                timeout_s=60, max_attempts=2),
            output_file=P.CURATION_FILE,
            validate=lambda t: G.validate_curation(t, node_ids=[1, 2]),
            cfg=CuratorConfig(sandbox="none", max_llm_calls=cfg.pop("max_llm_calls", 8),
                              timeout_s=60, max_attempts=2),
            llm_kwargs={"model": "m", "reasoning_effort": "low", "base_url": None,
                        "api_key_env": "K", "llm_timeout_s": 30,
                        "extra_body": {"provider": {"order": ["Baidu"]}}},
        )
        return res, llm


class TestPolicy(CuratorBase):
    def test_roots_and_surface(self) -> None:
        pol = self.policy
        self.assertEqual(list(pol.roots()), ["WORK_DIR", "MEMORY_DIR", "NODE_1", "PARENT_1", "NODE_2", "PARENT_2", "REPO_DIR"])
        r = pol.resolve
        for p in (r("$NODE_1/strategy.json"), r("$NODE_2/task_agent/workflow.py"), r("$PARENT_1/logs/case_1.json"),
                  r("$MEMORY_DIR/edit_memory.md"), r("$MEMORY_DIR/instruction.md"),
                  r("$REPO_DIR/platform_core/runner.py"), r("$REPO_DIR/projects/travel/tools/flight.py")):
            self.assertTrue(pol.can_read(p), p)
        for p in (r(str(self.r3 / "strategy.json")), r(str(self.proj / "benchmark" / "cases.jsonl")),
                  r(str(self.proj / "data" / "db.csv")), r(str(self.repo / "meta_agent" / "config.py"))):
            self.assertFalse(pol.can_read(p), p)
        self.assertTrue(pol.can_write(r("$WORK_DIR/curation.md")))
        self.assertTrue(pol.can_write(r("notes.md")))                    # relative → WORK_DIR
        self.assertFalse(pol.can_write(r("$NODE_1/strategy.json")))
        self.assertFalse(pol.can_write(r("$MEMORY_DIR/edit_memory.md")))
        self.assertTrue(self.work.is_dir())


class TestSession(CuratorBase):
    def test_write_then_submit(self) -> None:
        res, llm = self.run_memory_curator([
            _Resp(tool_calls=[_Call("b1", "bash", {"command": "diff -u $PARENT_1/task_agent/workflow.py $NODE_1/task_agent/workflow.py; cat $MEMORY_DIR/edit_memory.md"})]),
            _Resp(tool_calls=[_Call("e1", "editor", {"command": "create", "path": "$WORK_DIR/curation.md", "file_text": _curation([1, 2])})]),
            _Resp(tool_calls=[_Call("s1", P.SUBMIT_CURATION_NAME, {"summary": "two edits, one helped"})]),
        ])
        self.assertTrue(res.success, res.errors)
        self.assertEqual(res.end_reason, "submitted")
        self.assertEqual(res.summary, "two edits, one helped")
        self.assertEqual(res.output_path.read_text(), _curation([1, 2]))
        first = llm.calls[0]
        self.assertEqual([t["name"] for t in first["tools"]], ["bash", "editor", P.SUBMIT_CURATION_NAME])
        self.assertIn("WORK_DIR, MEMORY_DIR, NODE_1, PARENT_1, NODE_2, PARENT_2 and REPO_DIR",
                      first["tools"][0]["description"])
        instr = first["messages"][1]["content"]
        self.assertIn("$NODE_1        1       0  none", instr)
        self.assertIn("- addendum", instr)
        self.assertNotIn(str(self.exp), instr)
        self.assertEqual(first["messages"][0]["content"], P.MEMORY_CURATOR_SYSTEM)
        self.assertEqual((first["model"], first["api_key_env"], first["timeout_s"]), ("m", "K", 30))
        self.assertEqual(first["extra_body"], {"provider": {"order": ["Baidu"]}})
        outs = [e for e in map(json.loads, (self.work / "agentic" / "transcript.jsonl").read_text().splitlines())
                if e["kind"] == "tool_call"]
        self.assertIn("-    return None", outs[0]["result"])
        self.assertIn("+    return 1", outs[0]["result"])
        self.assertIn("# memory v1", outs[0]["result"])
        sess = json.loads((self.work / "agentic" / "session.json").read_text())
        self.assertEqual(sess["summary"], "two edits, one helped")
        self.assertEqual(sess["changed_files"], ["curation.md"])
        self.assertEqual(sess["n_tool_calls"][P.SUBMIT_CURATION_NAME], 1)

    def test_missing_section_is_fed_back_then_fixed(self) -> None:
        res, llm = self.run_memory_curator([
            _Resp(tool_calls=[_Call("e1", "editor", {"command": "create", "path": "$WORK_DIR/curation.md", "file_text": _curation([1])})]),
            _Resp(tool_calls=[_Call("s1", P.SUBMIT_CURATION_NAME, {"summary": "x"})]),
            _Resp(tool_calls=[_Call("e2", "editor", {"command": "insert", "path": "$WORK_DIR/curation.md", "insert_line": 0,
                                                     "new_str": _curation([2]).split(P.CURATION_CROSS_HEADING)[0]})]),
            _Resp(tool_calls=[_Call("s2", P.SUBMIT_CURATION_NAME, {"summary": "fixed"})]),
        ])
        self.assertTrue(res.success, res.errors)
        fail_out = llm.calls[2]["messages"][-1]["output"]
        self.assertIn("Document check failed (attempt 1/2)", fail_out)
        self.assertIn("missing section '## Node 2'", fail_out)
        self.assertIn(f"call {P.SUBMIT_CURATION_NAME} again", fail_out)
        self.assertEqual(json.loads((self.work / "agentic" / "session.json").read_text())["validation_rounds"], 2)

    def test_submit_without_a_file(self) -> None:
        res, llm = self.run_memory_curator([
            _Resp(tool_calls=[_Call("s1", P.SUBMIT_CURATION_NAME, {"summary": "nothing"})]),
            _Resp(tool_calls=[_Call("s2", P.SUBMIT_CURATION_NAME, {"summary": "still nothing"})]),
        ])
        self.assertFalse(res.success)
        self.assertEqual(res.end_reason, "max_attempts")
        self.assertIn("does not exist or is empty", llm.calls[1]["messages"][-1]["output"])

    def test_writes_outside_work_dir_refused(self) -> None:
        res, llm = self.run_memory_curator([
            _Resp(tool_calls=[_Call("e1", "editor", {"command": "create", "path": "$NODE_1/notes.md", "file_text": "x"}),
                              _Call("e2", "editor", {"command": "str_replace", "path": "$MEMORY_DIR/edit_memory.md",
                                                     "old_str": "v1", "new_str": "v9"})]),
            _Resp(tool_calls=[_Call("e3", "editor", {"command": "create", "path": "$WORK_DIR/curation.md", "file_text": _curation([1, 2])}),
                              _Call("s1", P.SUBMIT_CURATION_NAME, {"summary": "ok"})]),
        ])
        self.assertTrue(res.success)
        outs = [e for e in map(json.loads, (self.work / "agentic" / "transcript.jsonl").read_text().splitlines())
                if e["kind"] == "tool_call"]
        self.assertTrue(outs[0]["result"].startswith("Error"))
        self.assertTrue(outs[1]["result"].startswith("Error"))
        self.assertEqual((self.mem / "edit_memory.md").read_text(), "# memory v1\n")
        self.assertFalse((self.r1 / "notes.md").exists())

    def test_wrap_up_accepts_a_partial_document(self) -> None:
        # Budget of 2 calls: the model writes an incomplete file and never submits.
        res, llm = self.run_memory_curator([
            _Resp(tool_calls=[_Call("e1", "editor", {"command": "create", "path": "$WORK_DIR/curation.md", "file_text": _curation([1])})]),
            _Resp(tool_calls=[_Call("b1", "bash", {"command": "echo more reading"})]),
            _Resp(content="I ran out of time."),          # the wrap-up submit-only call: no tool call
        ], max_llm_calls=2)
        self.assertTrue(res.success)
        self.assertEqual(res.end_reason, "max_llm_calls")
        self.assertEqual(res.errors, ["missing section '## Node 2'"])
        self.assertIn(P.SUBMIT_CURATION_NAME, llm.calls[2]["messages"][-1]["content"])
        self.assertEqual([t["name"] for t in llm.calls[2]["tools"]], [P.SUBMIT_CURATION_NAME])
        sess = json.loads((self.work / "agentic" / "session.json").read_text())
        self.assertEqual(sess["fallback_errors"], ["missing section '## Node 2'"])


if __name__ == "__main__":
    unittest.main()
