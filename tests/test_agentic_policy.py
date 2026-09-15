"""PathPolicy: the agentic editor's read/write surface.

The write set must be exactly the validators' mutable surface plus scratch;
the read set must cover the whole run directory under read_scope "run" (so
dynamically generated node files are visible) — and only the parent's and
this node's round dirs under read_scope "parent" — plus the platform +
project tool code, and must NEVER include the benchmark, the database, the
categorizer or the meta-agent itself.
"""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from meta_agent.agentic.policy import (
    RUN_ROOT_MARKER,
    build_policy,
    find_run_root,
    resolve,
)
from tests.test_agentic_editor import _agent


def _fake_repo(root: Path) -> tuple[Path, Path]:
    """A repo skeleton: platform_core/, projects/__init__.py, and a project
    with tools/, db_schema.md, benchmark/, data/ and a categorizer."""
    repo = root / "repo"
    (repo / "platform_core").mkdir(parents=True)
    (repo / "platform_core" / "runner.py").write_text("x = 1\n")
    (repo / "meta_agent").mkdir()
    (repo / "meta_agent" / "config.py").write_text("")
    (repo / "tests").mkdir()
    (repo / "projects").mkdir()
    (repo / "projects" / "__init__.py").write_text("")
    proj = repo / "projects" / "travel"
    (proj / "tools").mkdir(parents=True)
    (proj / "__init__.py").write_text("")
    (proj / "tools" / "flight.py").write_text("NAME='q'\n")
    (proj / "db_schema.md").write_text("# schema\n")
    (proj / "benchmark" / "_eval").mkdir(parents=True)
    (proj / "benchmark" / "scorer.py").write_text("def score(): ...\n")
    (proj / "benchmark" / "_eval" / "hard.py").write_text("")
    (proj / "benchmark" / "cases.jsonl").write_text("{}\n")
    (proj / "data").mkdir()
    (proj / "data" / "db.csv").write_text("secret\n")
    (proj / "travel_error_categorizer.py").write_text("")
    return repo, proj


def _fake_run(root: Path) -> tuple[Path, Path, Path]:
    run = root / "runs" / "20260101_run"
    run.mkdir(parents=True)
    (run / RUN_ROOT_MARKER).write_text("project: travel\n")
    base = run / "round_001"
    _agent(base, "def run_task(task):\n    return None\n")
    (base / "logs").mkdir()
    (base / "logs" / "case_7.json").write_text("{}")
    (base / "feedback.json").write_text("{}")
    (base / "strategy.json").write_text("{}")
    (run / "tree_snapshots.jsonl").write_text("{}\n")
    # A sibling node (another branch of the run).
    sib = run / "round_000"
    _agent(sib, "def run_task(task):\n    return None\n")
    (sib / "strategy.json").write_text('{"optimization_goal": "sibling"}')
    out = run / "round_002" / "variants" / "var_0"
    _agent(out, "def run_task(task):\n    return None\n")
    (out / "agentic" / "scratch").mkdir(parents=True)
    return run, base, out


class PolicyBase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self.repo, self.proj = _fake_repo(root)
        self.run, self.base, self.out = _fake_run(root)
        self.policy = build_policy(
            out_dir=self.out, base_dir=self.base, repo_root=self.repo,
            project_root=self.proj,
        )

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def r(self, p: Path) -> Path:
        return resolve(str(p))


class TestRunRoot(PolicyBase):
    def test_walks_up_through_variants(self) -> None:
        self.assertEqual(find_run_root(self.out), self.r(self.run))
        self.assertEqual(self.policy.run_root, self.r(self.run))

    def test_none_without_marker(self) -> None:
        self.assertIsNone(find_run_root(self.proj))


class TestWriteSurface(PolicyBase):
    def test_mutable_files_and_tools_writable(self) -> None:
        ta = self.out / "task_agent"
        for name in ("workflow.py", "tool_wrapper.py", "tools_schema.json"):
            self.assertTrue(self.policy.can_write(self.r(ta / name)), name)
        self.assertTrue(self.policy.can_write(self.r(ta / "mutable_tools" / "new_tool.py")))
        self.assertTrue(self.policy.can_write(self.r(self.out / "agentic" / "scratch" / "t.sh")))
        self.assertTrue(self.policy.can_write(self.r(self.out / "agentic" / "scratch" / "d" / "x.txt")))

    def test_everything_else_denied(self) -> None:
        ta = self.out / "task_agent"
        for p in (
            ta / "other.py",
            ta / "mutable_tools" / "notes.txt",
            ta / "mutable_tools" / "sub" / "x.py",
            ta / "mutable_tools",
            self.base / "task_agent" / "workflow.py",
            self.out / "agentic" / "transcript.jsonl",
            self.repo / "platform_core" / "runner.py",
        ):
            self.assertFalse(self.policy.can_write(self.r(p)), str(p))

    def test_write_implies_read(self) -> None:
        self.assertTrue(self.policy.can_read(self.r(self.out / "task_agent" / "workflow.py")))


class TestReadSurface(PolicyBase):
    def test_whole_run_readable_including_dynamic_files(self) -> None:
        for p in (
            self.base / "logs" / "case_7.json",
            self.base / "feedback.json",
            self.base / "strategy.json",
            self.base / "task_agent" / "workflow.py",
            self.run / "round_000" / "strategy.json",
            self.run / "tree_snapshots.jsonl",
            self.run / RUN_ROOT_MARKER,
        ):
            self.assertTrue(self.policy.can_read(self.r(p)), str(p))
        # Written after the policy was built (the framework does this between
        # rounds and during a run): still readable, no rebuild needed.
        later = self.run / "round_003" / "hgm_node.json"
        later.parent.mkdir()
        later.write_text("{}")
        self.assertTrue(self.policy.can_read(self.r(later)))
        sibling = self.run / "round_003"
        _agent(sibling, "def run_task(task):\n    return 1\n")
        self.assertTrue(self.policy.can_read(self.r(sibling / "task_agent" / "workflow.py")))

    def test_platform_and_project_code_readable(self) -> None:
        for p in (
            self.repo / "platform_core" / "runner.py",
            self.repo / "projects" / "__init__.py",
            self.proj / "__init__.py",
            self.proj / "tools" / "flight.py",
            self.proj / "db_schema.md",
        ):
            self.assertTrue(self.policy.can_read(self.r(p)), str(p))

    def test_evaluation_scripts_database_and_meta_agent_never_readable(self) -> None:
        for p in (
            self.proj / "benchmark" / "scorer.py",
            self.proj / "benchmark" / "_eval" / "hard.py",
            self.proj / "benchmark" / "cases.jsonl",
            self.proj / "benchmark",
            self.proj / "data" / "db.csv",
            self.proj / "data",
            self.proj / "travel_error_categorizer.py",
            self.proj,  # the project root itself is not a root
            self.repo / "meta_agent" / "config.py",
            self.repo / "tests",
            self.repo,
            Path("/etc/passwd"),
        ):
            self.assertFalse(self.policy.can_read(self.r(p)), str(p))
        self.assertNotIn(self.r(self.proj), self.policy.read_roots)
        self.assertFalse(any("benchmark" in str(r) or r.name == "data"
                             for r in self.policy.read_roots))

    def test_deny_list_beats_a_misconfigured_root(self) -> None:
        # Even if a run were placed inside the project dir, benchmark/ and
        # cases.jsonl stay unreadable.
        run = self.proj / "runs" / "r"
        run.mkdir(parents=True)
        (run / RUN_ROOT_MARKER).write_text("")
        base, out = run / "round_001", run / "round_002"
        _agent(base, "def run_task(task):\n    return None\n")
        _agent(out, "def run_task(task):\n    return None\n")
        pol = build_policy(out_dir=out, base_dir=base, repo_root=self.repo,
                           project_root=self.proj)
        self.assertTrue(pol.can_read(self.r(base / "task_agent" / "workflow.py")))
        self.assertFalse(pol.can_read(self.r(self.proj / "benchmark" / "scorer.py")))
        self.assertFalse(pol.can_read(self.r(self.proj / "data" / "db.csv")))
        (run / "cases.jsonl").write_text("")
        self.assertFalse(pol.can_read(self.r(run / "cases.jsonl")))


class TestSymlinkEscape(PolicyBase):
    def test_symlink_inside_writable_dir_is_judged_by_its_target(self) -> None:
        link = self.out / "task_agent" / "mutable_tools" / "evil.py"
        os.symlink(self.proj / "data" / "db.csv", link)
        target = self.r(link)
        self.assertEqual(target, self.r(self.proj / "data" / "db.csv"))
        self.assertFalse(self.policy.can_read(target))
        self.assertFalse(self.policy.can_write(target))

    def test_symlinked_parent_dir_denied(self) -> None:
        link_dir = self.out / "agentic" / "scratch" / "esc"
        os.symlink(self.proj / "benchmark", link_dir)
        self.assertFalse(self.policy.can_write(self.r(link_dir / "new.py")))
        self.assertFalse(self.policy.can_read(self.r(link_dir / "scorer.py")))


class TestResolveAndDescribe(PolicyBase):
    def test_resolve_requires_absolute(self) -> None:
        with self.assertRaises(ValueError) as cm:
            resolve("task_agent/workflow.py")
        self.assertIn("absolute", str(cm.exception))
        with self.assertRaises(ValueError):
            resolve("")

    def test_describe_uses_root_variables_and_omits_missing(self) -> None:
        text = self.policy.describe()
        # No absolute path anywhere: everything is expressed via $VAR roots.
        for absolute in (str(self.r(self.out)), str(self.r(self.base)),
                         str(self.r(self.run)), str(self.r(self.repo))):
            self.assertNotIn(absolute, text)
        self.assertIn("NODE_DIR    $RUN_DIR/round_002/variants/var_0/   this node", text)
        self.assertIn("PARENT_DIR  $RUN_DIR/round_001/   its parent", text)
        self.assertIn("$NODE_DIR/task_agent/workflow.py", text)
        self.assertIn("$NODE_DIR/agentic/scratch/", text)
        self.assertIn("logs/case_<id>.json", text)
        self.assertNotIn("hgm_node.json", text)  # not present in this base
        self.assertIn("$REPO_DIR/projects/travel/tools/", text)
        self.assertIn("$REPO_DIR/projects/travel/db_schema.md", text)
        self.assertNotIn("benchmark", text.split("NOT available")[0])
        self.assertIn("workflow.py", text.split("Listing of")[1])
        self.assertIn("every other node $RUN_DIR/round_NNN/ has the same layout.", text)
        for word in ("memory", "belief", "accumulated"):
            self.assertNotIn(word, text)

    def test_roots_and_resolve_forms(self) -> None:
        roots = self.policy.roots()
        self.assertEqual(roots["RUN_DIR"], self.r(self.run))
        self.assertEqual(roots["NODE_DIR"], self.r(self.out))
        self.assertEqual(roots["PARENT_DIR"], self.r(self.base))
        self.assertEqual(roots["REPO_DIR"], self.r(self.repo))
        wf = self.r(self.out / "task_agent" / "workflow.py")
        self.assertEqual(self.policy.resolve("$NODE_DIR/task_agent/workflow.py"), wf)
        self.assertEqual(self.policy.resolve("${NODE_DIR}/task_agent/workflow.py"), wf)
        self.assertEqual(self.policy.resolve("workflow.py"), wf)              # relative to task_agent
        self.assertEqual(self.policy.resolve("./mutable_tools/../workflow.py"), wf)
        self.assertEqual(self.policy.resolve(str(wf)), wf)                    # absolute
        self.assertEqual(self.policy.resolve("$PARENT_DIR/feedback.json"),
                         self.r(self.base / "feedback.json"))
        self.assertEqual(self.policy.resolve("$RUN_DIR"), self.r(self.run))
        with self.assertRaises(ValueError) as cm:
            self.policy.resolve("$NOPE/x")
        self.assertIn("unknown root", str(cm.exception))
        with self.assertRaises(ValueError):
            self.policy.resolve("")
        self.assertEqual(self.policy.var_path(wf), "$NODE_DIR/task_agent/workflow.py")
        self.assertEqual(self.policy.var_path(self.r(self.base) / "x"), "$PARENT_DIR/x")



class TestParentReadScope(PolicyBase):
    def setUp(self) -> None:
        super().setUp()
        self.parent_policy = build_policy(
            out_dir=self.out, base_dir=self.base, repo_root=self.repo,
            project_root=self.proj, read_scope="parent",
        )

    def test_read_roots_are_the_two_round_dirs_only(self) -> None:
        pol = self.parent_policy
        self.assertEqual(pol.read_scope, "parent")
        self.assertEqual(pol.run_root, self.r(self.run))       # still located ...
        self.assertNotIn(self.r(self.run), pol.read_roots)     # ... but not readable
        self.assertIn(self.r(self.base), pol.read_roots)
        self.assertIn(self.r(self.out), pol.read_roots)
        for p in (self.base / "feedback.json", self.base / "strategy.json",
                  self.base / "logs" / "case_7.json", self.base / "task_agent" / "workflow.py",
                  self.out / "task_agent" / "workflow.py", self.out / "agentic" / "scratch",
                  self.repo / "platform_core" / "runner.py", self.proj / "tools" / "flight.py"):
            self.assertTrue(pol.can_read(self.r(p)), str(p))
        for p in (self.run / "round_000" / "strategy.json", self.run / "round_000" / "task_agent" / "workflow.py",
                  self.run / "tree_snapshots.jsonl", self.run / RUN_ROOT_MARKER, self.run):
            self.assertFalse(pol.can_read(self.r(p)), str(p))
        # The write surface is unchanged.
        self.assertTrue(pol.can_write(self.r(self.out / "task_agent" / "workflow.py")))
        self.assertFalse(pol.can_write(self.r(self.base / "task_agent" / "workflow.py")))

    def test_no_run_dir_root(self) -> None:
        pol = self.parent_policy
        self.assertEqual(list(pol.roots()), ["NODE_DIR", "PARENT_DIR", "REPO_DIR"])
        self.assertEqual(pol.root_vars(), ("NODE_DIR", "PARENT_DIR", "REPO_DIR"))
        with self.assertRaises(ValueError) as cm:
            pol.resolve("$RUN_DIR/round_000/strategy.json")
        self.assertIn("unknown root $RUN_DIR", str(cm.exception))
        self.assertIn("$NODE_DIR, $PARENT_DIR, $REPO_DIR", str(cm.exception))
        self.assertEqual(pol.resolve("$PARENT_DIR/feedback.json"), self.r(self.base / "feedback.json"))
        # A path outside every root renders as-is (never as $RUN_DIR/...).
        self.assertEqual(pol.var_path(self.r(self.run / "round_000")), str(self.r(self.run / "round_000")))
        self.assertEqual(pol.var_path(self.r(self.base) / "x"), "$PARENT_DIR/x")

    def test_describe_never_mentions_other_nodes(self) -> None:
        text = self.parent_policy.describe()
        for needle in ("RUN_DIR", "round_NNN", "every other node", "run directory"):
            self.assertNotIn(needle, text)
        self.assertIn("  NODE_DIR    this node", text)
        self.assertIn("  PARENT_DIR  its parent", text)
        self.assertIn("parent node $PARENT_DIR/ — the agent you are improving", text)
        self.assertIn("feedback.json", text)
        self.assertIn("$NODE_DIR/task_agent/workflow.py", text)
        self.assertIn("$REPO_DIR/projects/travel/tools/", text)
        for absolute in (str(self.r(self.out)), str(self.r(self.base)),
                         str(self.r(self.run)), str(self.r(self.repo))):
            self.assertNotIn(absolute, text)

    def test_run_scope_is_the_default_and_bad_values_raise(self) -> None:
        self.assertEqual(self.policy.read_scope, "run")
        with self.assertRaises(ValueError):
            build_policy(out_dir=self.out, base_dir=self.base, repo_root=self.repo,
                         project_root=self.proj, read_scope="node")

    def test_sandbox_binds_follow_the_scope(self) -> None:
        from meta_agent.agentic.sandbox import bwrap_argv, sandbox_env
        run_argv = bwrap_argv(self.policy, command="true", prefixes=[Path("/nonexistent/prefix")])
        par_argv = bwrap_argv(self.parent_policy, command="true", prefixes=[Path("/nonexistent/prefix")])
        self.assertIn(str(self.r(self.run)), run_argv)
        self.assertNotIn(str(self.r(self.run)), par_argv)
        self.assertNotIn(str(self.r(self.run / "round_000")), par_argv)
        self.assertIn(str(self.r(self.base)), par_argv)
        self.assertIn(str(self.r(self.out)), par_argv)
        env = sandbox_env(self.parent_policy, path="/bin")
        self.assertNotIn("RUN_DIR", env)
        self.assertEqual(env["PARENT_DIR"], str(self.r(self.base)))
        self.assertEqual(env["NODE_DIR"], str(self.r(self.out)))
        self.assertIn("RUN_DIR", sandbox_env(self.policy, path="/bin"))
        i = par_argv.index("--setenv", par_argv.index("--clearenv"))
        self.assertNotIn("RUN_DIR", par_argv[i:])

    def test_list_dir_filters_unreadable_and_hidden(self) -> None:
        (self.out / "task_agent" / ".hidden").write_text("")
        (self.out / "task_agent" / "__pycache__").mkdir()
        listing = self.policy.list_dir(self.r(self.out / "task_agent"))
        self.assertIn("workflow.py", listing)
        self.assertIn("mutable_tools/", listing)
        self.assertNotIn(".hidden", listing)
        self.assertNotIn("__pycache__", listing)
        proj_listing = self.policy.list_dir(self.r(self.proj))
        self.assertIn("tools/", proj_listing)
        self.assertNotIn("benchmark", proj_listing)
        self.assertNotIn("data", proj_listing)


if __name__ == "__main__":
    unittest.main()
