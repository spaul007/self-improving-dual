"""Unit tests for deterministic archive retrieval (edit_archive).

    PYTHONPATH=. python3 -m unittest tests.test_edit_archive
"""
from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from meta_agent.edit_archive import (
    RETRIEVAL_MANIFEST,
    render_retrieved,
    resolve_query,
    write_manifest,
)
from meta_agent.edit_code import CODE_NAME
from meta_agent.edit_memory import REGISTRY_NAME, RECORD_NAME, render_record
from meta_agent.edit_outcome import EditOutcome


def write_record(experiment_dir: Path, node: int, parent: int, body: str,
                 *, delta: float = 0.05, n_shared: int = 8) -> Path:
    """A parseable edit_memory.md in round_{node:03d}."""
    rd = experiment_dir / f"round_{node:03d}"
    rd.mkdir(parents=True, exist_ok=True)
    oc = EditOutcome(
        n_shared=n_shared, parent_mean_shared=0.45,
        child_mean_shared=round(0.45 + delta, 4), delta_shared=delta,
        child_mean_all=round(0.45 + delta, 4), child_n_all=n_shared + 2)
    fm = {"node": node, "parent": parent, "depth": 1,
          "lineage": f"0 > {node}"}
    (rd / RECORD_NAME).write_text(render_record(fm, body, oc),
                                  encoding="utf-8")
    return rd


def write_code(round_dir: Path, node: int, parent: int) -> None:
    (round_dir / CODE_NAME).write_text(
        "---\n"
        f"node: {node}\nparent: {parent}\n---\n\n"
        "## Sub-edit map\n- changed files: workflow.py\n\n"
        "## Diff vs parent (cap 20000 chars)\n```diff\n+ diff line\n```\n\n"
        "## Final-state definitions (added/changed, from child sources)\n"
        f"### workflow.py :: guard_{node} (function, added)\n"
        f"```python\ndef guard_{node}():\n    return True\n```\n",
        encoding="utf-8")


class TestResolveQuery(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        rd1 = write_record(self.tmp, 1, 0,
                           "## Edit 1\n- **what**: Adds a route verifier")
        rd2 = write_record(self.tmp, 2, 0,
                           "## Edit 1\n- **what**: Reworks the hotel budget")
        write_code(rd1, 1, 0)
        write_code(rd2, 2, 0)
        (self.tmp / REGISTRY_NAME).write_text(json.dumps({
            "strategies": {"add-verifier": {
                "definition": "adds a check", "first_node": 1,
                "edits": [{"node": 1, "edit_index": 1, "name": "route-check"},
                          {"node": 2, "edit_index": 1, "name": "budget-check"}],
            }},
            "areas": {"routing": {
                "definition": "route problems", "first_node": 1,
                "edits": [{"node": 1, "edit_index": 1, "name": "route-check"}],
            }},
        }), encoding="utf-8")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_explicit_node(self):
        r = resolve_query(self.tmp, {"nodes": [2]})
        self.assertEqual([row["node"] for row in r.manifest["selected"]], [2])
        self.assertEqual(r.manifest["selected"][0]["why"], "explicit")
        self.assertIn("Retrieved node 2 (explicit)", r.blocks[0])
        self.assertIn("hotel budget", r.blocks[0])

    def test_strategy_expansion(self):
        r = resolve_query(self.tmp, {"strategies": ["add-verifier"]})
        self.assertEqual([row["node"] for row in r.manifest["selected"]], [1, 2])
        self.assertTrue(all(row["why"].startswith("strategy:add-verifier")
                            for row in r.manifest["selected"]))

    def test_area_expansion(self):
        r = resolve_query(self.tmp, {"areas": ["routing"]})
        self.assertEqual([row["node"] for row in r.manifest["selected"]], [1])

    def test_keyword_fallback_newest_first(self):
        r = resolve_query(self.tmp, {"keywords": ["verifier"]})
        self.assertEqual([row["node"] for row in r.manifest["selected"]], [1])
        r2 = resolve_query(self.tmp, {"keywords": ["Edit"]})  # both match
        self.assertEqual([row["node"] for row in r2.manifest["selected"]],
                         [2, 1])  # equal hits -> newest first

    def test_explicit_outranks_keyword_dedup(self):
        r = resolve_query(self.tmp, {"nodes": [1], "keywords": ["verifier"]})
        self.assertEqual([row["node"] for row in r.manifest["selected"]], [1])
        self.assertEqual(r.manifest["selected"][0]["why"], "explicit")

    def test_unknown_node_dropped(self):
        r = resolve_query(self.tmp, {"nodes": [99]})
        self.assertEqual(r.blocks, [])
        self.assertEqual(r.manifest["dropped"][0]["node"], 99)
        self.assertEqual(r.manifest["dropped"][0]["reason"], "no record")

    def test_max_nodes_overflow_dropped(self):
        r = resolve_query(self.tmp, {"nodes": [1, 2]}, max_nodes=1)
        self.assertEqual(len(r.manifest["selected"]), 1)
        self.assertEqual(r.manifest["dropped"][0]["reason"], "over max_nodes")

    def test_budget_truncation_flagged(self):
        r = resolve_query(self.tmp, {"nodes": [1]}, char_budget=200)
        self.assertTrue(r.manifest["selected"][0]["truncated"])
        self.assertLessEqual(r.manifest["selected"][0]["chars"], 260)

    def test_code_slice_defs_before_diff(self):
        r = resolve_query(self.tmp, {"nodes": [1]})
        block = r.blocks[0]
        self.assertIn("guard_1", block)
        self.assertLess(block.find("Final-state definitions"),
                        block.find("Diff vs parent"))

    def test_include_code_false(self):
        r = resolve_query(self.tmp, {"nodes": [1], "include_code": False})
        self.assertNotIn("guard_1", r.blocks[0])

    def test_deterministic(self):
        q = {"strategies": ["add-verifier"], "keywords": ["verifier"]}
        a, b = resolve_query(self.tmp, q), resolve_query(self.tmp, q)
        self.assertEqual(a.blocks, b.blocks)
        self.assertEqual(a.manifest, b.manifest)

    def test_empty_archive(self):
        empty = Path(tempfile.mkdtemp())
        try:
            r = resolve_query(empty, {"nodes": [1], "keywords": ["x"]})
            self.assertEqual(r.blocks, [])
            self.assertIn("matched no recorded node", render_retrieved(r))
        finally:
            shutil.rmtree(empty, ignore_errors=True)

    def test_manifest_write(self):
        r = resolve_query(self.tmp, {"nodes": [1]})
        out = self.tmp / "round_003"
        out.mkdir()
        write_manifest(out, r)
        data = json.loads((out / RETRIEVAL_MANIFEST).read_text(encoding="utf-8"))
        self.assertEqual(data["version"], 1)
        self.assertEqual(data["selected"][0]["node"], 1)


if __name__ == "__main__":
    unittest.main()
