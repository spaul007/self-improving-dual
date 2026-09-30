"""Golden lock on the HGM default path (no agentic editor, no edit memory).

Runs deterministic ``evolve`` loops on ``hgm`` and ``hgm_block_tagged`` --
adaptive blocks, ``eval_repeats=2``, one forced edit failure, a stub block
suggester, a crc32-scored stub evaluator (``hash()`` is salted per process) --
and pins what every existing config's behavior depends on: each
``editor.apply`` call (parent, a hash of the steering context,
``has_suggestion``, the kwarg names), each suggester call's kwarg names, the
final tree, the budget counters, the ``hgm_node.json`` key sets and the
tree-snapshot records. The fixture was generated from unmodified vivek_mas @
1bd6884, before the agentic-editor / edit-memory port touched ``hgm.py``.

    PYTHONPATH=. python3 -m unittest tests.test_hgm_default_path_golden
    # deliberate default-path behavior changes only:
    PYTHONPATH=. python3 -m tests.test_hgm_default_path_golden --regen
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
import tempfile
import unittest
import zlib
from pathlib import Path

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "hgm_default_path.json"


class _GoldenEditor:
    """Copies the parent's task_agent, fails the ``fail_call``-th call, and
    records exactly what the manager passed."""

    def __init__(self, tmp: Path, fail_call: int) -> None:
        self.tmp = tmp
        self.fail_call = fail_call
        self.calls: list[dict] = []

    def apply(self, feedback, base_dir, out_dir, *, context=None, has_suggestion=False):
        from meta_agent.models import EditResult, EvolutionStrategy

        ctx = (context or "").replace(str(self.tmp), "<TMP>")
        self.calls.append({
            "parent": Path(base_dir).name,
            "out": Path(out_dir).name,
            "context_sha256": hashlib.sha256(ctx.encode("utf-8")).hexdigest(),
            "context_len": len(ctx),
            "has_suggestion": has_suggestion,
            "feedback_round": feedback.round_number if feedback is not None else None,
        })
        dst = Path(out_dir) / "task_agent"
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(Path(base_dir) / "task_agent", dst)
        n = len(self.calls)
        strategy = EvolutionStrategy(
            target_files=["workflow.py"], optimization_goal=f"stub goal {n}",
            proposed_changes=f"stub change {n}", rationale="stub",
        )
        if n == self.fail_call:
            return EditResult(success=False, errors=["forced failure"], strategy=strategy)
        return EditResult(success=True, edited_files=["workflow.py"], strategy=strategy)


class _GoldenSuggester:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def suggest(self, **kwargs):
        self.calls.append({"kwargs": sorted(kwargs), "block": kwargs.get("block"),
                           "node_id": kwargs.get("node_id")})
        return f"Target: {kwargs.get('block')}\nDiagnosis: stub\nProposed change: stub {kwargs.get('node_id')}"


class _Crc32Evaluator:
    @staticmethod
    def score(case_id: str, round_name: str) -> float:
        return (zlib.crc32(f"{round_name}:{case_id}".encode()) % 1000) / 1000.0

    def run(self, round_dir, benchmark_dir, *, case_ids=None):
        from meta_agent.models import CaseResult, EvaluationResult

        name = Path(round_dir).name
        per_case = [
            CaseResult(case_id=c, passed=self.score(c, name) >= 0.5, score=self.score(c, name))
            for c in (case_ids or [])
        ]
        passed = sum(1 for c in per_case if c.passed)
        return EvaluationResult(
            score=sum(c.score for c in per_case) / max(len(per_case), 1),
            passed=passed, failed=len(per_case) - passed, per_case=per_case,
        )


SCENARIOS = {
    "hgm_adaptive_repeats": dict(manager="hgm", block_selection_strategy="adaptive",
                                 eval_repeats=2, implementation=None),
    "block_tagged_adaptive_repeats": dict(manager="hgm_block_tagged", block_selection_strategy="adaptive",
                                          eval_repeats=2, implementation=None),
    "block_tagged_two_tier": dict(manager="hgm_block_tagged", block_selection_strategy="adaptive",
                                  eval_repeats=1, implementation="adaptive"),
    "hgm_non_adaptive_no_suggester": dict(manager="hgm", block_selection_strategy="non_adaptive",
                                          eval_repeats=1, implementation=None, suggester=False),
}


def _run(spec: dict) -> dict:
    from meta_agent import registry
    from meta_agent.config import _ensure_builtins_loaded
    from meta_agent.feedback_gatherer import DefaultFeedbackGatherer

    _ensure_builtins_loaded()
    tmp = Path(tempfile.mkdtemp(prefix="hgm_default_golden_"))
    try:
        seed = tmp / "seed"
        seed.mkdir()
        (seed / "workflow.py").write_text("def run_task(task):\n    return None\n", encoding="utf-8")
        exp = tmp / "exp"
        exp.mkdir()
        kwargs = dict(
            eval_budget=48, init_expansions=2, eval_batch_size=4, alpha=0.6, seed=11,
            eval_repeats=spec["eval_repeats"], snapshot_tree=True,
            block_selection_strategy=spec["block_selection_strategy"], finalize_top_k=2,
        )
        if spec["implementation"]:
            kwargs["implementation_strategy_selection_strategy"] = spec["implementation"]
        manager = registry.get("manager", spec["manager"])(**kwargs)
        editor = _GoldenEditor(tmp, fail_call=3)
        suggester = _GoldenSuggester() if spec.get("suggester", True) else None
        manager.evolve(
            editor=editor, evaluator=_Crc32Evaluator(), gatherer=DefaultFeedbackGatherer(),
            seed_dir=seed, benchmark_dir=tmp / "bench", experiment_dir=exp,
            max_rounds=40, score_target=None,
            train_case_ids=[f"c{i}" for i in range(12)], eval_case_ids=None,
            block_suggester=suggester,
        )
        tree = []
        for nid, node in sorted(manager._tree.nodes.items()):
            fb = manager._feedback.get(nid)
            tree.append({
                "node_id": nid, "parent_id": node.parent_id, "edit_failed": node.edit_failed,
                "n_evals": node.n_evals, "mean": round(node.mean_utility, 9),
                "block": fb.strategy.block if fb is not None else None,
                "implementation_strategy": fb.strategy.implementation_strategy if fb is not None else None,
            })
        sidecar_keys = sorted({
            k for p in sorted(exp.glob("round_*/hgm_node.json"))
            for k in json.loads(p.read_text(encoding="utf-8"))
        })
        snaps = [json.loads(line) for line in
                 (exp / "snapshots" / "tree_snapshots.jsonl").read_text(encoding="utf-8").splitlines()]
        round_files = sorted({
            p.relative_to(p.parent.parent).as_posix().split("/", 1)[1]
            for p in exp.glob("round_*/*") if p.is_file()
        })
        return {
            "editor_calls": editor.calls,
            "suggester_calls": suggester.calls if suggester is not None else None,
            "tree": tree,
            "budget_spent": manager._budget_spent,
            "node_evals_spent": manager._node_evals_spent,
            "sidecar_keys": sidecar_keys,
            "round_files": round_files,
            "snapshot_events": [s["event"] for s in snaps],
            "snapshot_keys": sorted({k for s in snaps for k in s}),
            "snapshot_node_keys": sorted({k for s in snaps for n in s["nodes"] for k in n}),
            "last_snapshot": snaps[-1],
        }
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def render_all() -> dict:
    return {name: _run(spec) for name, spec in SCENARIOS.items()}


class HGMDefaultPathGoldenTests(unittest.TestCase):
    def test_default_path_matches_fixture(self) -> None:
        expected = json.loads(FIXTURE.read_text(encoding="utf-8"))
        actual = json.loads(json.dumps(render_all(), sort_keys=True))
        self.assertEqual(sorted(actual), sorted(expected))
        for name in expected:
            for field in expected[name]:
                with self.subTest(scenario=name, field=field):
                    self.assertEqual(actual[name][field], expected[name][field])


if __name__ == "__main__":
    if "--regen" in sys.argv:
        FIXTURE.parent.mkdir(parents=True, exist_ok=True)
        FIXTURE.write_text(json.dumps(render_all(), indent=1, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote {FIXTURE}")
    else:
        unittest.main()
