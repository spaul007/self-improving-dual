"""Tests for meta_agent.run_inspect against a tiny fake experiment dir."""
from __future__ import annotations

import json
import os
import tempfile
import time
import unittest
from pathlib import Path

from meta_agent import run_inspect as ri


def _w(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _wj(path: Path, obj) -> None:
    _w(path, json.dumps(obj))


def _case(case_id: str, score: float, *, dims: dict, hard: dict, failed: list, error=None) -> dict:
    if error:
        return {"case_id": case_id, "passed": False, "score": 0.0, "error": error, "details": {}}
    return {
        "case_id": case_id,
        "passed": score >= 0.5,
        "score": score,
        "error": None,
        "details": {
            "composite_score": score,
            "commonsense_score": score,
            "hard_score": 1.0,
            "dimension_scores": dims,
            "hard_constraints": {k: {"passed": v, "message": ""} for k, v in hard.items()},
            "failed_checks": failed,
            "agent_metadata": {"iterations": 3},
        },
    }


def _task_agent(round_dir: Path, workflow: str) -> None:
    _w(round_dir / "task_agent" / "workflow.py", workflow)
    _w(round_dir / "task_agent" / "tool_wrapper.py", "def wrap(): pass\n")
    _w(round_dir / "task_agent" / "tools_schema.json", "[]")
    _w(round_dir / "task_agent" / "immutable.py", "X = 1\n")
    _w(round_dir / "task_agent" / "mutable_tools" / "__init__.py", "")


def make_experiment(root: Path) -> Path:
    exp = root / "20260917_000000_fake_exp"
    _w(
        exp / "config.snapshot.yaml",
        "experiment_name: fake\nproject: travel\n"
        "task_agent:\n  model: qwen\n"
        "editor:\n  type: agentic\n  config:\n    model: dsv4\n    read_scope: run\n"
        "manager:\n  type: hgm\n  config:\n    eval_budget: 100\n    snapshot_tree: true\n"
        "edit_memory:\n  type: agentic\n  config:\n    window_size: 4\n",
    )
    # round_000 -- seed
    r0 = exp / "round_000"
    _task_agent(r0, "def run(): return 1\n")
    _wj(r0 / "hgm_node.json", {"node_id": 0, "parent_id": None, "children": [], "edit_failed": False,
                                "n_evals": 2, "mean_utility": 0.5, "cmp": 0.5, "memory_arm": "none", "memory_version": None})
    _wj(r0 / "eval_result.json", {"score": 0.5, "passed": 1, "failed": 1, "crashed": False, "per_case": [
        _case("a", 1.0, dims={"D1": 1.0, "D2": 0.5}, hard={"h1": True, "h2": False}, failed=["c:D2:x"]),
        _case("b", 0.0, dims={"D1": 0.0, "D2": 0.5}, hard={"h1": False, "h2": False}, failed=["c:D2:x", "c:D1:y"]),
    ]})
    # round_001 -- full successful node, arm "with"
    r1 = exp / "round_001"
    _task_agent(r1, "def run():\n    return 2\n")
    _w(r1 / "task_agent" / "immutable.py", "X = 2\n")  # changed but NOT mutable -> must not show in diff
    _wj(r1 / "hgm_node.json", {"node_id": 1, "parent_id": 0, "children": [], "edit_failed": False,
                                "n_evals": 4, "mean_utility": 0.75, "cmp": 0.7, "memory_arm": "with", "memory_version": 1})
    _wj(r1 / "strategy.json", {"target_files": ["workflow.py"], "optimization_goal": "go", "proposed_changes": "x", "rationale": "y"})
    _wj(r1 / "feedback.json", {"round_number": 1, "base_round": 0, "edit_errors": [], "log_excerpt": "Z" * 10000,
                                "project_metrics": {"no_plan_rate": 0.1}})
    _wj(r1 / "eval_result.json", {"score": 0.75, "passed": 3, "failed": 1, "crashed": False, "per_case": [
        _case("a", 1.0, dims={"D1": 1.0, "D2": 1.0}, hard={"h1": True, "h2": True}, failed=[]),
        _case("c", 0.0, dims={}, hard={}, failed=[], error="timeout after 1s"),
    ]})
    _wj(r1 / "agentic" / "session.json", {"success": True, "end_reason": "submitted", "n_llm_calls": 3,
                                          "changed_files": ["workflow.py"], "memory_path": "/x/edit_memory/edit_memory_v001.md"})
    _w(r1 / "agentic" / "transcript.jsonl", '{"t": 1.0, "kind": "end", "reason": "submitted", "success": true}\n')
    # round_002 -- edit failed mid-run: no hgm_node.json
    r2 = exp / "round_002"
    _wj(r2 / "strategy.json", {"target_files": [], "optimization_goal": "", "proposed_changes": "", "rationale": ""})
    _wj(r2 / "feedback.json", {"round_number": 2, "base_round": 1, "edit_errors": ["validator boom"], "log_excerpt": ""})
    _wj(r2 / "agentic" / "session.json", {"success": False, "end_reason": "max_llm_calls", "memory_path": None})
    # round_003 -- session without memory but no hgm_node yet, in-progress eval via case logs only
    r3 = exp / "round_003"
    _wj(r3 / "agentic" / "session.json", {"success": True, "end_reason": "submitted", "memory_path": "/x/m.md"})
    _wj(r3 / "logs" / "case_a.json", _case("a", 1.0, dims={"D1": 1.0}, hard={}, failed=[]))
    # snapshots -- duplicate budget 4 (expand after evaluate) and a dip that must be clamped
    snaps = [
        {"snapshot_idx": 0, "event": "seed", "budget_spent": 0, "best_node_id": 0, "best_mean_utility": 0.5,
         "best_round_dir": "round_000", "n_nodes": 1, "nodes": [{"node_id": 0, "parent_id": None, "edit_failed": False}]},
        {"snapshot_idx": 1, "event": "evaluate", "budget_spent": 4, "best_node_id": 1, "best_mean_utility": 0.75,
         "best_round_dir": "round_001", "n_nodes": 2, "nodes": [{"node_id": 0}, {"node_id": 1, "edit_failed": False}]},
        {"snapshot_idx": 2, "event": "expand", "budget_spent": 4, "best_node_id": 1, "best_mean_utility": 0.75,
         "best_round_dir": "round_001", "n_nodes": 3, "nodes": [{"node_id": 0}, {"node_id": 1}, {"node_id": 2, "edit_failed": True}]},
        {"snapshot_idx": 3, "event": "evaluate", "budget_spent": 8, "best_node_id": 1, "best_mean_utility": 0.70,
         "best_round_dir": "round_001", "n_nodes": 3, "nodes": [{"node_id": 0}, {"node_id": 1}, {"node_id": 2, "edit_failed": True}]},
    ]
    _w(exp / "snapshots" / "tree_snapshots.jsonl", "\n".join(json.dumps(s) for s in snaps) + "\n")
    _wj(exp / "snapshots" / "eval_at_budget_4.json", {"requested_budget": 4, "node_id": 1, "composite_score": 0.6})
    _wj(exp / "edit_memory" / "state.json", {"memory_version": 1, "pulls": {"with": 2, "without": 1}, "events": []})
    # a standalone evaluate.py output dir that must NOT count as an experiment
    ev = root / "eval_20260917_x_seed"
    _w(ev / "config.snapshot.yaml", "project: travel\n")
    _wj(ev / "round_eval" / "eval_result.json", {"score": 0.1, "per_case": []})
    # stray file
    _w(root / "console.log", "hi\n")
    return exp


class RunInspectTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.exp = make_experiment(self.root)

    def tearDown(self):
        self._tmp.cleanup()

    def test_list_experiments_skips_eval_dirs_and_files(self):
        names = [p.name for p in ri.list_experiments(self.root)]
        self.assertEqual(names, [self.exp.name])

    def test_config_helpers(self):
        cfg = ri.load_config_snapshot(self.exp)
        self.assertEqual(ri.task_agent_model(cfg), "qwen")
        self.assertEqual(ri.editor_model(cfg), "dsv4")
        self.assertEqual(ri.editor_read_scope(cfg), "run")
        self.assertEqual(ri.edit_memory_config(cfg), {"window_size": 4})
        self.assertIsNone(ri.edit_memory_config({"project": "x"}))
        self.assertEqual(ri.task_agent_model({"task_agent": {"config": {"model": "old"}}}), "old")

    def test_discover_rounds_shapes(self):
        rounds = ri.discover_rounds(self.exp)
        self.assertEqual([r.node_id for r in rounds], [0, 1, 2, 3])
        r0, r1, r2, r3 = rounds
        self.assertIsNone(r0.parent_id)
        self.assertEqual(r1.parent_id, 0)
        # edit-failed round without hgm_node.json
        self.assertIsNone(r2.hgm_node)
        self.assertTrue(r2.edit_failed)
        self.assertEqual(r2.parent_id, 1)
        self.assertEqual(r2.n_evals, 0)
        self.assertIsNone(r2.mean_utility)
        self.assertEqual(r2.session_end_reason, "max_llm_calls")
        # in-progress round: per_case backfilled from logs
        self.assertTrue(r3.eval_result.get("_synthesized_from_case_logs"))
        self.assertEqual(len(r3.eval_result["per_case"]), 1)
        self.assertFalse(r3.edit_failed)

    def test_log_excerpt_popped(self):
        r1 = ri.discover_rounds(self.exp)[1]
        self.assertNotIn("log_excerpt", r1.feedback)
        self.assertEqual(r1.feedback["project_metrics"], {"no_plan_rate": 0.1})

    def test_memory_arm_fallbacks(self):
        rounds = {r.node_id: r for r in ri.discover_rounds(self.exp)}
        self.assertEqual(rounds[0].memory_arm, "none")
        self.assertEqual(rounds[1].memory_arm, "with")
        self.assertEqual(rounds[1].memory_version, 1)
        self.assertEqual(rounds[2].memory_arm, "none")     # session.memory_path None
        self.assertEqual(rounds[3].memory_arm, "with")     # no hgm_node, session has memory_path
        self.assertIsNone(rounds[3].memory_version)
        self.assertEqual(rounds[1].changed_files, ["workflow.py"])

    def test_diff_round_files_uses_mutable_surface(self):
        diffs = ri.diff_round_files(self.exp / "round_000", self.exp / "round_001")
        self.assertEqual(list(diffs), ["workflow.py"])  # immutable.py changed but excluded
        d = diffs["workflow.py"]
        self.assertEqual(d.status, "modified")
        self.assertEqual((d.lines_added, d.lines_removed), (2, 1))
        self.assertIn("+++ child/workflow.py", d.diff_text)
        self.assertEqual(ri.diff_totals(diffs), (2, 1))
        self.assertEqual(ri.diff_round_files(self.exp / "round_001", self.exp / "round_002"), {})

    def test_diagnostics_flags_failed_session_and_edit(self):
        rounds = ri.discover_rounds(self.exp)
        alerts = ri.extract_diagnostics(rounds, is_active=True)
        by_node = {}
        for a in alerts:
            by_node.setdefault(a.node_id, []).append((a.severity, a.message))
        self.assertTrue(any(sev == "error" and m.startswith("edit failed: validator boom") for sev, m in by_node[2]))
        # not double-reported as a failed session
        self.assertFalse(any("editor session ended" in m for _, m in by_node[2]))
        self.assertTrue(any("case c: timeout" in m for _, m in by_node[1]))
        # in-progress synthesized round is fine while active, an error once stopped
        self.assertNotIn(3, by_node)
        stopped = ri.extract_diagnostics(rounds, is_active=False)
        self.assertTrue(any(a.node_id == 3 and a.severity == "error" for a in stopped))

    def test_best_so_far_curve_dedup_and_monotone(self):
        curve = ri.best_so_far_curve(ri.load_tree_snapshots(self.exp))
        self.assertEqual([(c.budget_spent, c.best_mean_utility) for c in curve], [(0, 0.5), (4, 0.75), (8, 0.75)])
        self.assertEqual(curve[1].snapshot_idx, 2)  # last snapshot at budget 4 wins

    def test_budget_progress_exact_vs_approx(self):
        cfg = ri.load_config_snapshot(self.exp)
        rounds = ri.discover_rounds(self.exp)
        exact = ri.budget_progress(cfg, ri.load_tree_snapshots(self.exp), rounds)
        self.assertEqual((exact.spent, exact.total, exact.exact), (8, 100, True))
        approx = ri.budget_progress(cfg, [], rounds)
        self.assertEqual((approx.spent, approx.exact), (4, False))  # root's 2 evals excluded

    def test_dimension_and_constraint_helpers(self):
        r0 = ri.discover_rounds(self.exp)[0]
        self.assertEqual(ri.dimension_means(r0.eval_result), {"D1": 0.5, "D2": 0.5})
        self.assertEqual(ri.hard_constraint_failure_counts(r0.eval_result), [("h2", 2), ("h1", 1)])
        self.assertEqual(ri.failed_check_counts(r0.eval_result), [("c:D2:x", 2), ("c:D1:y", 1)])
        rows = ri.per_case_dimension_rows(ri.discover_rounds(self.exp)[1].eval_result)
        self.assertEqual([r["case_id"] for r in rows], ["a", "c"])
        self.assertEqual(rows[0]["D1"], 1.0)
        self.assertIsNone(rows[1]["D1"])
        self.assertEqual(rows[1]["error"], "timeout after 1s")
        self.assertEqual(ri.dimension_means(None), {})

    def test_arm_utility_summary(self):
        s = ri.arm_utility_summary(ri.discover_rounds(self.exp))
        self.assertEqual(set(s), {"with", "none"})
        self.assertEqual(s["with"]["n_nodes"], 2)       # nodes 1 and 3
        self.assertEqual(s["with"]["n_evaluated"], 1)
        self.assertEqual(s["with"]["mean_of_means"], 0.75)
        self.assertEqual(s["none"]["n_edit_failed"], 1)  # node 2
        self.assertIsNone(s["none"]["mean_of_means"])

    def test_experiment_summary_and_eval_at_budget(self):
        s = ri.experiment_summary(self.exp)
        self.assertEqual((s.n_nodes, s.n_edit_failed, s.best_node_id, s.best_mean), (3, 1, 1, 0.70))
        self.assertEqual((s.budget_spent, s.budget_total), (8, 100))
        self.assertEqual(s.pulls, {"with": 2, "without": 1})
        self.assertEqual(s.memory_version, 1)
        self.assertTrue(s.has_edit_memory and s.has_snapshots and not s.finished)
        self.assertEqual(s.best_dimension_means, {"D1": 1.0, "D2": 1.0})
        self.assertEqual(len(s.curve), 3)
        ev = ri.load_eval_at_budget(self.exp)
        self.assertEqual([(e["requested_budget"], e["composite_score"]) for e in ev], [(4, 0.6)])

    def test_experiment_summary_without_snapshots(self):
        (self.exp / "snapshots" / "tree_snapshots.jsonl").unlink()
        s = ri.experiment_summary(self.exp)
        self.assertFalse(s.has_snapshots)
        self.assertEqual((s.n_nodes, s.n_edit_failed, s.best_node_id), (4, 1, 1))
        self.assertEqual(s.curve, [])

    def test_run_is_active(self):
        self.assertTrue(ri.run_is_active(self.exp))
        old = time.time() - 10_000
        for p in self.exp.rglob("*"):
            os.utime(p, (old, old))
        os.utime(self.exp, (old, old))
        self.assertFalse(ri.run_is_active(self.exp))
        _w(self.exp / "round_003" / "hgm_node.json", "{}")
        self.assertTrue(ri.run_is_active(self.exp))
        _w(self.exp / "run_summary.md", "# done\n")
        self.assertFalse(ri.run_is_active(self.exp))
        self.assertEqual(ri.load_run_summary(self.exp), "# done\n")


if __name__ == "__main__":
    unittest.main()
