"""The agentic editor on an exclude-list (multi-agent) surface.

Covers the write rule (``PathPolicy.write_allowed_rel`` / ``can_write``), the
``describe()`` map, the bwrap bind order, the built-in scope check,
``changed_agent_files``, an end-to-end session on the real
``travel_mas_refactored`` seed with a scripted LLM, and -- when bubblewrap
works on this node -- the real sandbox.

    PYTHONPATH=. python3 -m unittest tests.test_agentic_exclude_surface
"""
from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from pathlib import Path

from meta_agent.agent_editor_agentic import AgenticEditor
from meta_agent.agentic.policy import REPO_ROOT, RUN_ROOT_MARKER, build_policy
from meta_agent.agentic.sandbox import Sandbox, bwrap_argv, probe_bwrap
from meta_agent.agentic.tools import SUBMIT_TOOL_NAME
from meta_agent.edit_diff import changed_agent_files
from meta_agent.editor_validators import ImmutableFilesValidator, SyntaxValidator

from tests.test_agentic_editor import _Call, _Resp, _ScriptedLLM, _submit

PROJECT = REPO_ROOT / "projects" / "travel_mas_refactored"
SEED = PROJECT / "seed"
EXCLUDE = ["agents/immutable/", "benchmark/", "workflow.py"]


class _MasRun(unittest.TestCase):
    """A run dir whose parent round holds a copy of the real MAS seed."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.run = Path(self._tmp.name) / "run"
        self.run.mkdir()
        (self.run / RUN_ROOT_MARKER).write_text("")
        self.base = self.run / "round_001"
        self.out = self.run / "round_002"
        shutil.copytree(SEED, self.base / "task_agent",
                        ignore=shutil.ignore_patterns("__pycache__"))
        (self.base / "feedback.json").write_text('{"score": 0.4}')
        (self.out / "task_agent").parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(self.base / "task_agent", self.out / "task_agent")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def policy(self, **kw):
        return build_policy(out_dir=self.out, base_dir=self.base, repo_root=REPO_ROOT,
                            project_root=PROJECT, mutable_exclude=EXCLUDE, **kw)


class WriteRuleTests(_MasRun):
    def test_write_rule(self) -> None:
        pol = self.policy()
        ta = pol.task_agent
        for rel in ("agents/flight.py", "agents/new_helper.py", "mas_workflow.py",
                    "mas_llm_backbone.yaml", "mutable_tools/__init__.py", "tools_schema.json"):
            self.assertTrue(pol.can_write(ta / rel), rel)
        for rel in ("workflow.py", "agents/immutable/message.py", "agents/immutable/new.py",
                    "__pycache__/x.pyc", "agents/results/x.json"):
            self.assertFalse(pol.can_write(ta / rel), rel)
        self.assertFalse(pol.can_write(ta))
        self.assertFalse(pol.can_write(self.base / "task_agent" / "agents" / "flight.py"))
        self.assertTrue(pol.can_write(pol.scratch / "t.py"))

    def test_symlink_out_of_task_agent_is_not_writable(self) -> None:
        link = self.out / "task_agent" / "agents" / "escape.py"
        os.symlink(self.base / "task_agent" / "agents" / "flight.py", link)
        pol = self.policy()
        self.assertFalse(pol.can_write(pol.resolve(str(link))))

    def test_readonly_paths_are_the_existing_excluded_ones(self) -> None:
        pol = self.policy()
        ta = pol.task_agent
        self.assertEqual(set(pol.readonly_paths),
                         {ta / "agents" / "immutable", ta / "workflow.py"})

    def test_adapter_benchmark_data_are_denied(self) -> None:
        pol = self.policy()
        for sub in ("adapter", "benchmark", "data"):
            self.assertFalse(pol.can_read(PROJECT / sub))
            self.assertIn(PROJECT / sub, pol.deny_roots)
        self.assertTrue(pol.can_read(PROJECT / "tools"))


class DescribeAndArgvTests(_MasRun):
    def test_describe_exclude_block(self) -> None:
        text = self.policy().describe()
        self.assertIn(
            "  WRITABLE (edit with the editor tool; new files may be created): "
            "everything under $NODE_DIR/task_agent/ except\n"
            "    $NODE_DIR/task_agent/agents/immutable/\n"
            "    $NODE_DIR/task_agent/benchmark/\n"
            "    $NODE_DIR/task_agent/workflow.py\n"
            "    (those stay read-only; __pycache__/ and results/ are ignored)\n",
            text,
        )
        self.assertNotIn("new *.py files allowed", text)
        self.assertNotIn("lineage:", text)          # evidence_hints off

    def test_evidence_hints_add_lineage_and_siblings(self) -> None:
        (self.base / "failure_summary.md").write_text("x")
        text = self.policy(evidence_hints=True).describe()
        self.assertIn("failure_summary.md", text)
        self.assertIn("lineage: hgm_node.json's parent_id", text)
        self.assertIn('"base_round": <parent id>', text)

    def test_bind_order(self) -> None:
        pol = self.policy()
        argv = bwrap_argv(pol, command="true", prefixes=[])
        ta, immut = str(pol.task_agent), str(pol.task_agent / "agents" / "immutable")
        i_rw = _index(argv, "--bind", ta)
        i_ro = _index(argv, "--ro-bind", immut)
        i_seal = argv.index("--remount-ro")
        self.assertLess(i_rw, i_ro)
        self.assertLess(i_ro, i_seal)


def _index(argv: list[str], flag: str, path: str) -> int:
    for i in range(len(argv) - 2):
        if argv[i] == flag and argv[i + 1] == path and argv[i + 2] == path:
            return i
    raise AssertionError(f"{flag} {path} not in argv")


class ScopeCheckTests(_MasRun):
    def editor(self) -> AgenticEditor:
        return AgenticEditor(lambda **kw: None, [], mutable_exclude=EXCLUDE,
                             project_root=PROJECT, sandbox="none")

    def test_clean_tree_has_no_errors(self) -> None:
        (self.out / "task_agent" / "agents" / "flight.py").write_text("changed\n")
        (self.out / "task_agent" / "agents" / "helper.py").write_text("new\n")
        self.assertEqual(self.editor()._scope_errors(self.out, self.base), [])

    def test_excluded_modify_delete_create(self) -> None:
        ta = self.out / "task_agent"
        (ta / "workflow.py").write_text("changed\n")
        (ta / "agents" / "immutable" / "message.py").unlink()
        (ta / "agents" / "immutable" / "extra.py").write_text("x\n")
        errors = self.editor()._scope_errors(self.out, self.base)
        self.assertEqual(len(errors), 3, errors)
        joined = "\n".join(errors)
        for frag in ("workflow.py was modified", "agents/immutable/message.py was deleted",
                     "agents/immutable/extra.py was created"):
            self.assertIn(frag, joined)

    def test_new_symlink_is_rejected(self) -> None:
        os.symlink("/etc/passwd", self.out / "task_agent" / "agents" / "p.txt")
        errors = self.editor()._scope_errors(self.out, self.base)
        self.assertEqual(len(errors), 1)
        self.assertIn("new symlink", errors[0])

    def test_pycache_is_ignored(self) -> None:
        cache = self.out / "task_agent" / "agents" / "immutable" / "__pycache__"
        cache.mkdir(parents=True)
        (cache / "message.cpython-312.pyc").write_bytes(b"\0")
        self.assertEqual(self.editor()._scope_errors(self.out, self.base), [])


class ChangedAgentFilesTests(_MasRun):
    def test_exclude_mode(self) -> None:
        ta = self.out / "task_agent"
        (ta / "mutable_tools" / "__init__.py").write_text("# touched\n")
        (ta / "agents" / "new.py").write_text("x\n")
        (ta / "workflow.py").write_text("excluded change\n")
        (ta / "__pycache__").mkdir(exist_ok=True)
        (ta / "__pycache__" / "x.pyc").write_bytes(b"\0")
        self.assertEqual(
            changed_agent_files(self.base, self.out, mutable_exclude=EXCLUDE),
            ["agents/new.py", "mutable_tools/__init__.py"],
        )

    def test_none_is_the_single_agent_surface(self) -> None:
        from meta_agent.edit_diff import changed_mutable_files
        self.assertEqual(changed_agent_files(self.base, self.out),
                         changed_mutable_files(self.base, self.out))


class EndToEndTests(_MasRun):
    def setUp(self) -> None:
        super().setUp()
        shutil.rmtree(self.out)   # apply() copies the parent itself

    def run_editor(self, steps, **kw):
        llm = _ScriptedLLM(steps)
        ed = AgenticEditor(
            llm, [SyntaxValidator(), ImmutableFilesValidator(mutable_exclude=EXCLUDE)],
            mutable_exclude=EXCLUDE, project_root=PROJECT, sandbox="none",
            max_llm_calls=8, timeout_s=60, **kw,
        )
        return ed.apply(None, self.base, self.out), llm

    def test_edit_a_role_and_submit(self) -> None:
        flight = self.base / "task_agent" / "agents" / "flight.py"
        first = flight.read_text().split("\n")[0]
        res, llm = self.run_editor([
            _Resp(tool_calls=[_Call("e1", "editor", {
                "command": "str_replace", "path": "agents/flight.py",
                "old_str": first, "new_str": first + "  # edited"})]),
            _Resp(tool_calls=[_submit()]),
        ])
        self.assertTrue(res.success, res.errors)
        self.assertEqual(res.strategy.target_files, ["agents/flight.py"])
        self.assertEqual(res.edited_files, ["agents/flight.py"])
        self.assertTrue((self.out / "agentic" / "prompt_system.txt").is_file())
        self.assertTrue((self.out / "agentic" / "prompt_instruction.txt").is_file())

    def test_editor_tool_refuses_excluded_file(self) -> None:
        res, llm = self.run_editor([
            _Resp(tool_calls=[_Call("e1", "editor", {
                "command": "str_replace", "path": "workflow.py",
                "old_str": "import mas_workflow", "new_str": "import os"})]),
            _Resp(tool_calls=[_submit()]),
        ], max_attempts=1)
        refusal = next(m for m in llm.calls[1]["messages"]
                       if isinstance(m, dict) and m.get("type") == "function_call_output")
        self.assertIn("is not writable; everything under", refusal["output"])
        self.assertIn("agents/immutable/, benchmark/, workflow.py", refusal["output"])
        self.assertFalse(res.success)   # nothing changed -> submit rejected

    def test_bash_write_to_excluded_file_fails_the_submit(self) -> None:
        res, _ = self.run_editor([
            _Resp(tool_calls=[_Call("b1", "bash", {"command": "echo '# x' >> workflow.py"})]),
            _Resp(tool_calls=[_Call("e1", "editor", {
                "command": "create", "path": "agents/helper.py", "file_text": "X = 1\n"})]),
            _Resp(tool_calls=[_submit()]),
        ], max_attempts=1)
        self.assertFalse(res.success)
        self.assertTrue(any("scope violation: workflow.py was modified" in e for e in res.errors),
                        res.errors)


@unittest.skipUnless(probe_bwrap()[0], "bubblewrap unusable on this node")
class RealBwrapTests(_MasRun):
    def run_cmd(self, cmd: str) -> str:
        r = Sandbox(self.policy(), mode="bwrap").run(cmd)
        return (r.stdout + r.stderr).strip()

    def test_agent_imports_and_surface_is_enforced(self) -> None:
        self.assertEqual(self.run_cmd('python3 -c "import workflow; print(1)"'), "1")
        self.assertIn("rc=0", self.run_cmd("sed -i '1s/^/# x /' agents/flight.py; echo rc=$?"))
        self.assertIn("Read-only file system", self.run_cmd("echo x >> workflow.py"))
        self.assertIn("Read-only file system", self.run_cmd("touch agents/immutable/n.py"))
        self.assertEqual(self.run_cmd("echo X=1 > agents/n.py && echo ok"), "ok")
        self.assertIn("Read-only file system",
                      self.run_cmd("touch $PARENT_DIR/task_agent/agents/n.py"))

    def test_project_internals_are_absent(self) -> None:
        listing = self.run_cmd("ls $REPO_DIR/projects/travel_mas_refactored").split()
        self.assertEqual(sorted(listing), ["__init__.py", "tools"])
        self.assertEqual(self.run_cmd("env | grep -c -i -E 'key|token|secret' || true"), "0")


if __name__ == "__main__":
    unittest.main()
