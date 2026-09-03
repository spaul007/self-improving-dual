"""Unit tests for the per-node code record (edit_code.md).

Deterministic module — no LLM anywhere.

    PYTHONPATH=. python3 -m unittest tests.test_edit_code
"""
from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from meta_agent.edit_code import (
    CODE_NAME,
    extract_changed_defs,
    map_subedits,
    render_edit_code,
    write_edit_code,
)


def _agent(round_dir: Path, workflow: str) -> None:
    a = round_dir / "task_agent"
    (a / "mutable_tools").mkdir(parents=True, exist_ok=True)
    (a / "workflow.py").write_text(workflow, encoding="utf-8")
    (a / "tool_wrapper.py").write_text("def x(): return None\n", encoding="utf-8")
    (a / "tools_schema.json").write_text("[]", encoding="utf-8")
    (a / "mutable_tools" / "__init__.py").write_text("", encoding="utf-8")


PARENT_WF = """def run_task(task):
    return None

def helper(x):
    return x
"""

CHILD_WF = """def run_task(task):
    return validate(None)

def helper(x):
    return x

def validate(out):
    return out

class RouteGuard:
    pass
"""


class TestExtractChangedDefs(unittest.TestCase):
    def test_added_changed_untouched(self):
        got = extract_changed_defs(PARENT_WF, CHILD_WF)
        by_name = {name: (kind, status) for name, kind, status, _src in got}
        self.assertEqual(by_name["run_task"], ("function", "changed"))
        self.assertEqual(by_name["validate"], ("function", "added"))
        self.assertEqual(by_name["RouteGuard"], ("class", "added"))
        self.assertNotIn("helper", by_name)  # untouched def is skipped

    def test_child_syntax_error_yields_empty(self):
        self.assertEqual(extract_changed_defs(PARENT_WF, "def broken(:\n"), [])

    def test_parent_syntax_error_degrades_to_added(self):
        got = extract_changed_defs("def broken(:\n", "def a():\n    return 1\n")
        self.assertEqual([(n, s) for n, _k, s, _ in got], [("a", "added")])

    def test_source_segment_is_full_def(self):
        got = extract_changed_defs(PARENT_WF, CHILD_WF)
        src = next(s for n, _k, _s, s in got if n == "validate")
        self.assertIn("def validate(out):", src)
        self.assertIn("return out", src)


class TestMapSubedits(unittest.TestCase):
    def test_token_overlap_maps_subedit_to_def(self):
        changed = {"workflow.py": [("validate", "function", "added", "src")]}
        lines = map_subedits(
            [{"name": "add-validate-gate", "what": "Adds a validate gate"}],
            changed, ["workflow.py"])
        joined = "\n".join(lines)
        self.assertIn("`add-validate-gate` (Edit 1) -> workflow.py :: validate",
                      joined)

    def test_unmatched_def_is_unattributed(self):
        changed = {"workflow.py": [("zorp", "function", "added", "src")]}
        lines = map_subedits(
            [{"name": "tune-prompt", "what": "Rewrites the system prompt"}],
            changed, ["workflow.py"])
        joined = "\n".join(lines)
        self.assertIn("(unattributed) workflow.py :: zorp", joined)

    def test_no_subedits_lists_files_and_defs(self):
        changed = {"workflow.py": [("a", "function", "added", "src")]}
        lines = map_subedits(None, changed, ["workflow.py"])
        self.assertIn("- changed files: workflow.py", lines[0])
        self.assertIn("(unattributed) workflow.py :: a", "\n".join(lines))


class TestRenderAndWrite(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.parent = self.tmp / "round_000"
        self.child = self.tmp / "round_001"
        _agent(self.parent, PARENT_WF)
        _agent(self.child, CHILD_WF)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_layout_and_sections(self):
        text = render_edit_code(self.parent, self.child, node_id=1, parent_id=0)
        self.assertTrue(text.startswith("---\nnode: 1\nparent: 0\n---"))
        self.assertIn("## Sub-edit map", text)
        self.assertIn("## Diff vs parent", text)
        self.assertIn("## Final-state definitions", text)
        self.assertIn("workflow.py :: validate (function, added)", text)
        self.assertIn("class RouteGuard:", text)

    def test_deterministic(self):
        a = render_edit_code(self.parent, self.child, node_id=1, parent_id=0)
        b = render_edit_code(self.parent, self.child, node_id=1, parent_id=0)
        self.assertEqual(a, b)

    def test_diff_cap_honored(self):
        big = CHILD_WF + "\n" + "\n".join(
            f"def filler_{i}():\n    return {i}" for i in range(200))
        _agent(self.child, big)
        text = render_edit_code(self.parent, self.child, node_id=1,
                                parent_id=0, diff_char_cap=500)
        self.assertIn("chars elided", text)

    def test_defs_cap_honored(self):
        big = PARENT_WF + "\n" + "\n".join(
            f"def filler_{i}():\n    return {i}" for i in range(200))
        _agent(self.child, big)
        text = render_edit_code(self.parent, self.child, node_id=1,
                                parent_id=0, defs_char_cap=300)
        self.assertIn("remaining definitions elided", text)

    def test_write_then_rewrite_with_map_is_idempotent(self):
        p1 = write_edit_code(self.parent, self.child, node_id=1, parent_id=0)
        self.assertIsNotNone(p1)
        first = p1.read_text(encoding="utf-8")
        sub = [{"name": "add-validate-gate", "what": "Adds a validate gate"}]
        write_edit_code(self.parent, self.child, node_id=1, parent_id=0,
                        sub_edits=sub)
        second = (self.child / CODE_NAME).read_text(encoding="utf-8")
        self.assertIn("add-validate-gate", second)
        # Everything except the map is unchanged between the two writes.
        self.assertEqual(first.split("## Diff vs parent", 1)[1],
                         second.split("## Diff vs parent", 1)[1])
        write_edit_code(self.parent, self.child, node_id=1, parent_id=0,
                        sub_edits=sub)
        self.assertEqual(second,
                         (self.child / CODE_NAME).read_text(encoding="utf-8"))

    def test_no_changes_still_renders(self):
        _agent(self.child, PARENT_WF)
        text = render_edit_code(self.parent, self.child, node_id=1, parent_id=0)
        self.assertIn("changed files: (none)", text)


class TestEditMemoryIntegration(unittest.TestCase):
    """record_node writes edit_code.md alongside the prose record."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.parent = self.tmp / "round_000"
        self.child = self.tmp / "round_001"
        _agent(self.parent, PARENT_WF)
        _agent(self.child, CHILD_WF)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _record(self, **kwargs):
        from meta_agent.edit_memory import EditMemory
        from tests.test_edit_memory import _StubLLM
        em = EditMemory(_StubLLM(), setup_pass=False, usage_tracking=False,
                        analysis_mode="off", **kwargs)
        em.setup(self.tmp, self.parent, [])
        return em.record_node(round_dir=self.child,
                              parent_round_dir=self.parent,
                              node_id=1, parent_id=0, ancestors=[0])

    def test_record_node_writes_code_record_with_map(self):
        self.assertIsNotNone(self._record())
        text = (self.child / CODE_NAME).read_text(encoding="utf-8")
        self.assertIn("## Final-state definitions", text)
        # The post-tagger rewrite carries the sub-edit map (stub tags one
        # sub-edit named "route-check").
        self.assertIn("(Edit 1)", text)

    def test_code_record_disabled(self):
        self.assertIsNotNone(self._record(code_record=False))
        self.assertFalse((self.child / CODE_NAME).exists())


if __name__ == "__main__":
    unittest.main()
