"""The agentic editor's tools and sandbox.

Editor tool semantics (view/create/str_replace/insert, unique match, policy
enforcement), bash tool result shaping, the pure bwrap argv builder, and —
when bubblewrap works on this machine — the real confinement guarantees.
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
import unittest
from pathlib import Path

from meta_agent.agentic.policy import RUN_ROOT_MARKER, REPO_ROOT, build_policy, resolve
from meta_agent.agentic.sandbox import Sandbox, bwrap_argv, fallback_env, probe_bwrap
from meta_agent.agentic.tools import (
    SUBMIT_TOOL,
    BashTool,
    EditorTool,
    ToolSet,
    ValidateTool,
    bash_tool_info,
    editor_tool_info,
    truncate_head_tail,
)
from tests.test_edit_code import _agent

WF = "def run_task(task):\n    x = 1\n    return None\n"


class ToolBase(unittest.TestCase):
    """A run dir with a base round and an out round, no project root."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self.run = root / "run"
        self.run.mkdir()
        (self.run / RUN_ROOT_MARKER).write_text("")
        self.base = self.run / "round_001"
        self.out = self.run / "round_002"
        _agent(self.base, WF)
        _agent(self.out, WF)
        (self.out / "agentic" / "scratch").mkdir(parents=True)
        (self.base / "feedback.json").write_text('{"score": 0.5}')
        self.policy = build_policy(out_dir=self.out, base_dir=self.base)
        self.wf = resolve(str(self.out / "task_agent" / "workflow.py"))
        self.editor = EditorTool(self.policy, max_view_chars=400)

    def tearDown(self) -> None:
        self._tmp.cleanup()


class TestEditorView(ToolBase):
    def test_view_numbered_and_range(self) -> None:
        out = self.editor("view", str(self.wf))
        self.assertIn("Here's the result of running `cat -n`", out)
        self.assertIn("     1\tdef run_task(task):", out)
        self.assertIn("     3\t    return None", out)
        sliced = self.editor("view", str(self.wf), view_range=[2, 2])
        self.assertIn("     2\t    x = 1", sliced)
        self.assertNotIn("run_task", sliced)
        tail = self.editor("view", str(self.wf), view_range=["2", "-1"])
        self.assertIn("return None", tail)

    def test_view_range_errors_name_line_count(self) -> None:
        self.assertIn("file has 4 lines", self.editor("view", str(self.wf), view_range=[9, 10]))
        self.assertIn("Error: invalid view_range", self.editor("view", str(self.wf), view_range="x"))

    def test_view_clip(self) -> None:
        self.wf.write_text("a" * 1000)
        out = self.editor("view", str(self.wf))
        self.assertIn("<response clipped", out)
        self.assertIn("view_range", out)
        self.assertLess(len(out), 600)

    def test_view_directory_and_policy(self) -> None:
        out = self.editor("view", str(self.out / "task_agent"))
        self.assertIn("files and directories", out)
        self.assertIn("workflow.py", out)
        self.assertIn("mutable_tools/", out)
        self.assertIn("feedback.json", self.editor("view", str(self.base)))
        denied = self.editor("view", "/etc/passwd")
        self.assertTrue(denied.startswith("Error:"))
        self.assertIn("not readable", denied)
        self.assertIn("does not exist", self.editor("view", str(self.out / "nope.py")))

    def test_relative_path_rejected(self) -> None:
        self.assertIn("must be absolute", self.editor("view", "workflow.py"))

    def test_unknown_command(self) -> None:
        self.assertIn("unknown command", self.editor("edit", str(self.wf)))


class TestEditorEdits(ToolBase):
    def test_create_rules(self) -> None:
        new_tool = self.out / "task_agent" / "mutable_tools" / "helper.py"
        self.assertIn("created successfully", self.editor("create", str(new_tool), file_text="def run(): pass\n"))
        self.assertEqual(new_tool.read_text(), "def run(): pass\n")
        self.assertIn("already exists", self.editor("create", str(new_tool), file_text="x"))
        self.assertIn("missing required 'file_text'", self.editor("create", str(self.out / "task_agent" / "mutable_tools" / "b.py")))
        self.assertIn("not writable", self.editor("create", str(self.out / "task_agent" / "extra.py"), file_text="x"))
        self.assertIn("not writable", self.editor("create", str(self.out / "task_agent" / "mutable_tools" / "notes.txt"), file_text="x"))
        scratch = self.out / "agentic" / "scratch" / "t.py"
        self.assertIn("created successfully", self.editor("create", str(scratch), file_text="print(1)\n"))

    def test_str_replace_unique_match(self) -> None:
        out = self.editor("str_replace", str(self.wf), old_str="    x = 1\n", new_str="    x = 2\n")
        self.assertIn("has been edited", out)
        self.assertIn("x = 2", out)  # snippet
        self.assertEqual(self.wf.read_text(), "def run_task(task):\n    x = 2\n    return None\n")

    def test_str_replace_zero_and_multiple(self) -> None:
        self.assertIn("did not appear", self.editor("str_replace", str(self.wf), old_str="nope", new_str="y"))
        self.wf.write_text("a\nb\na\n")
        out = self.editor("str_replace", str(self.wf), old_str="a", new_str="c")
        self.assertIn("occurs 2 times", out)
        self.assertIn("[1, 3]", out)
        self.assertEqual(self.wf.read_text(), "a\nb\na\n")  # untouched

    def test_str_replace_policy_and_missing(self) -> None:
        self.assertIn("missing required 'old_str'", self.editor("str_replace", str(self.wf), new_str="y"))
        ro = self.base / "task_agent" / "workflow.py"
        self.assertIn("not writable", self.editor("str_replace", str(ro), old_str="x", new_str="y"))
        self.assertIn("does not exist", self.editor("str_replace", str(self.out / "task_agent" / "mutable_tools" / "gone.py"), old_str="x", new_str="y"))

    def test_insert(self) -> None:
        out = self.editor("insert", str(self.wf), insert_line=1, new_str="    import json\n")
        self.assertIn("has been edited", out)
        self.assertEqual(self.wf.read_text(), "def run_task(task):\n    import json\n    x = 1\n    return None\n")
        self.assertIn("between 0 and", self.editor("insert", str(self.wf), insert_line=99, new_str="x"))
        self.assertIn("must be an integer", self.editor("insert", str(self.wf), insert_line="q", new_str="x"))
        top = self.editor("insert", str(self.wf), insert_line=0, new_str="# top")
        self.assertIn("has been edited", top)
        self.assertTrue(self.wf.read_text().startswith("# top\ndef run_task"))


class TestToolSetAndHelpers(ToolBase):
    def test_dispatch_errors(self) -> None:
        ts = ToolSet([(editor_tool_info(max_view_chars=10), self.editor)])
        self.assertIn("not found", ts.call("nope", {}))
        self.assertIn("submit_self_improvement", ts.call("nope", {}))
        self.assertIn("could not parse", ts.call("editor", {"_raw_arguments": "{bad"}))
        self.assertIn("bad arguments", ts.call("editor", {"command": "view"}))  # missing path
        self.assertIn("bad arguments", ts.call("editor", {"command": "view", "path": "/x", "bogus": 1}))
        self.assertEqual(ts.names(), ["editor"])
        self.assertEqual([t["name"] for t in ts.infos()], ["editor"])

    def test_validate_tool(self) -> None:
        calls = []
        vt = ValidateTool(lambda: calls.append(1) or [])
        self.assertEqual(vt(), "All validators passed.")
        vt2 = ValidateTool(lambda: ["bad thing", "worse"])
        self.assertIn("  - bad thing", vt2())

    def test_truncate_head_tail(self) -> None:
        text = "H" * 600 + "T" * 400
        out = truncate_head_tail(text, 100)
        self.assertTrue(out.startswith("H" * 60))
        self.assertTrue(out.endswith("T" * 40))
        self.assertIn("[truncated 900 chars]", out)
        self.assertEqual(truncate_head_tail("short", 100), "short")

    def test_schemas_are_anthropic_shape(self) -> None:
        for info in (bash_tool_info(bash_timeout_s=1, max_output_chars=1),
                     editor_tool_info(max_view_chars=1), SUBMIT_TOOL):
            self.assertIn("input_schema", info)
            self.assertIn("name", info)
        self.assertNotIn("files", SUBMIT_TOOL["input_schema"]["properties"])
        self.assertIn("prediction", SUBMIT_TOOL["input_schema"]["properties"])


class TestBashFallback(ToolBase):
    def setUp(self) -> None:
        super().setUp()
        self.sb = Sandbox(self.policy, mode="none", bash_timeout_s=0.7)
        self.bash = BashTool(self.sb, max_output_chars=200)

    def test_echo_cwd_and_exit_code(self) -> None:
        self.assertEqual(self.bash("echo hello"), "hello")
        self.assertEqual(self.bash("pwd"), str(self.policy.task_agent))
        out = self.bash("ls /nonexistent/dir")
        self.assertIn("Error:", out)
        self.assertIn("No such file", out)
        self.assertIn("[exit code 2]", out)
        self.assertEqual(self.bash("true"), "(no output)")
        self.assertIn("non-empty", self.bash(""))

    def test_timeout_kills(self) -> None:
        t0 = time.time()
        out = self.bash("sleep 5; echo late")
        self.assertLess(time.time() - t0, 3.0)
        self.assertIn("[killed: exceeded 0.7s]", out)
        self.assertNotIn("late", out)

    def test_truncation(self) -> None:
        out = self.bash("for i in $(seq 1 200); do echo line$i; done")
        self.assertIn("line1\n", out)
        self.assertIn("line200", out)
        self.assertIn("[truncated", out)

    def test_env_is_scrubbed(self) -> None:
        os.environ["OPENAI_API_KEY"] = "sk-test-secret"
        os.environ["TRAVEL_DATABASE_ROOT"] = "/x"
        try:
            env = fallback_env(self.policy)
            self.assertNotIn("OPENAI_API_KEY", env)
            self.assertNotIn("TRAVEL_DATABASE_ROOT", env)
            self.assertNotIn("sk-test-secret", self.bash("env"))
            self.assertIn("PYTHONPATH", env)
        finally:
            del os.environ["OPENAI_API_KEY"]
            del os.environ["TRAVEL_DATABASE_ROOT"]


class TestBwrapArgv(ToolBase):
    def test_argv_shape(self) -> None:
        argv = bwrap_argv(self.policy, command="true", prefixes=[Path("/nonexistent/prefix")])
        self.assertEqual(argv[0], "bwrap")
        self.assertEqual(argv[-3:], ["bash", "-c", "true"])
        # Mount ordering: run root ro, then mutable file rw, then remount-ro last.
        s = " ".join(argv)
        run = str(self.policy.run_root)
        wf = str(self.wf)
        self.assertLess(s.index(f"--ro-bind-try {run} {run}"), s.index(f"--bind {wf} {wf}"))
        mounts = [i for i, a in enumerate(argv) if a in ("--ro-bind", "--ro-bind-try", "--bind", "--tmpfs", "--proc", "--dev", "--symlink", "--remount-ro")]
        self.assertEqual(argv[mounts[-1]], "--remount-ro")
        self.assertEqual(argv[mounts[-1] + 1], "/")
        for flag in ("--unshare-all", "--clearenv", "--die-with-parent", "--new-session"):
            self.assertIn(flag, argv)
        self.assertIn("--chdir", argv)
        self.assertEqual(argv[argv.index("--chdir") + 1], str(self.policy.task_agent))
        setenv_keys = [argv[i + 1] for i, a in enumerate(argv) if a == "--setenv"]
        self.assertIn("PYTHONPATH", setenv_keys)
        self.assertFalse(any("KEY" in k or "LLM_" in k or "DATABASE" in k for k in setenv_keys))
        self.assertNotIn("benchmark", s)
        # Symlinked system dirs become --symlink, real dirs --ro-bind.
        for p in ("/bin", "/lib", "/lib64", "/sbin"):
            if os.path.islink(p):
                self.assertIn(f"--symlink {os.readlink(p)} {p}", s)
            elif os.path.isdir(p):
                self.assertIn(f"--ro-bind {p} {p}", s)
        self.assertIn("--ro-bind /usr /usr", s)
        self.assertIn("--ro-bind /nonexistent/prefix /nonexistent/prefix", s)


def _bwrap_available() -> bool:
    ok, _ = probe_bwrap()
    return ok


class TestBwrapConfinement(unittest.TestCase):
    """Real sandbox behaviour — skipped where bubblewrap cannot run."""

    @classmethod
    def setUpClass(cls) -> None:
        if not _bwrap_available():
            raise unittest.SkipTest(f"bwrap unusable here: {probe_bwrap()[1]}")

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        # Fake project with tools/ + benchmark/ + data/ so we can prove the
        # latter two are absent inside; real platform_core so `import
        # workflow` (the seed's) resolves through PYTHONPATH.
        self.proj = root / "projects" / "travel"
        (self.proj / "tools").mkdir(parents=True)
        (self.proj / "__init__.py").write_text("")
        (self.proj / "tools" / "__init__.py").write_text("")
        (self.proj / "benchmark").mkdir()
        (self.proj / "benchmark" / "cases.jsonl").write_text("{}\n")
        (self.proj / "data").mkdir()
        (self.proj / "data" / "db.csv").write_text("secret\n")
        self.run = root / "run"
        self.run.mkdir()
        (self.run / RUN_ROOT_MARKER).write_text("")
        self.base, self.out = self.run / "round_001", self.run / "round_002"
        _agent(self.base, WF)
        _agent(self.out, WF)
        (self.out / "agentic" / "scratch").mkdir(parents=True)
        (self.run / "edit_memory_beliefs.md").write_text("### belief:a — b\n")
        self.policy = build_policy(out_dir=self.out, base_dir=self.base,
                                   repo_root=REPO_ROOT, project_root=self.proj)
        self.sb = Sandbox(self.policy, mode="bwrap", bash_timeout_s=30)
        self.bash = BashTool(self.sb, max_output_chars=5000)
        self.wf = str(self.policy.task_agent / "workflow.py")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_mode(self) -> None:
        self.assertEqual(self.sb.effective_mode, "bwrap")

    def test_python_imports_workflow_and_platform(self) -> None:
        out = self.bash('python3 -c "import workflow, platform_core.trace; print(\'ok\')"')
        self.assertEqual(out, "ok", out)

    def test_writes_confined_to_mutable_surface(self) -> None:
        self.assertEqual(self.bash(f'python3 -c "open(\'{self.wf}\',\'a\').write(\'# x\\n\')" && echo done'), "done")
        self.assertTrue(Path(self.wf).read_text().endswith("# x\n"))
        out = self.bash(f"touch {self.policy.task_agent}/other.txt")
        self.assertIn("Read-only file system", out)
        self.assertIn("Read-only file system", self.bash("touch /tmp_probe_outside_x 2>&1; touch /groups/x 2>&1; touch /users/x 2>&1"))
        self.assertEqual(self.bash(f"echo hi > {self.policy.scratch}/t.txt && cat {self.policy.scratch}/t.txt"), "hi")
        self.assertEqual(self.bash(f"echo 'def run(): pass' > {self.policy.task_agent}/mutable_tools/n.py && echo ok"), "ok")
        sed = self.bash(f"sed -i 's/x/y/' {self.wf}")
        self.assertTrue("Read-only file system" in sed or "Device or resource busy" in sed, sed)
        self.assertIn("x = 1", Path(self.wf).read_text())

    def test_reads_confined(self) -> None:
        self.assertIn("workflow.py", self.bash(f"ls {self.base}/task_agent"))
        self.assertIn("belief:a", self.bash(f"cat {self.run}/edit_memory_beliefs.md"))
        listing = self.bash(f"ls {self.proj}")
        self.assertIn("tools", listing)
        self.assertNotIn("benchmark", listing)
        self.assertNotIn("data", listing)
        self.assertIn("No such file", self.bash(f"cat {self.proj}/benchmark/cases.jsonl"))
        self.assertIn("No such file", self.bash(f"cat {self.proj}/data/db.csv"))
        self.assertIn("No such file", self.bash(f"ls {REPO_ROOT}/meta_agent"))

    def test_env_and_network(self) -> None:
        os.environ["OPENAI_API_KEY"] = "sk-test-secret"
        try:
            env = self.bash("env | sort")
        finally:
            del os.environ["OPENAI_API_KEY"]
        self.assertNotIn("sk-test-secret", env)
        self.assertNotIn("OPENAI_API_KEY", env)
        self.assertIn("PYTHONPATH=", env)
        self.assertIn("META_AGENT_PROJECT=travel", env)
        net = self.bash('python3 -c "import socket; socket.create_connection((\'1.1.1.1\', 80), 2)"')
        self.assertIn("unreachable", net.lower())

    def test_fresh_shell_per_call(self) -> None:
        self.bash("export FOO=1; cd /tmp")
        self.assertEqual(self.bash("echo \"${FOO:-unset}\" $(pwd)"), f"unset {self.policy.task_agent}")


if __name__ == "__main__":
    unittest.main()
