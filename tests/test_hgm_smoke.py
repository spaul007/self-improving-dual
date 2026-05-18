"""Smoke tests for the HGM (Huxley-Gödel Machine) manager.

No API key required: the pure tree/bandit math is tested directly, and the
end-to-end evolve() loop runs against stub editor/evaluator components plus
the real DefaultFeedbackGatherer.

    PYTHONPATH=. python3 -m unittest tests.test_hgm_smoke
"""
from __future__ import annotations

import random
import shutil
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
# Pure tree / bandit math
# --------------------------------------------------------------------------- #


class HGMTreeTests(unittest.TestCase):
    def _node(self, node_id, parent_id, scores):
        from meta_agent.managers.hgm_tree import HGMNode
        from meta_agent.models import CaseResult

        n = HGMNode(node_id=node_id, parent_id=parent_id, round_dir=Path("."))
        for i, s in enumerate(scores):
            n.record(CaseResult(case_id=f"{node_id}-{i}", passed=s >= 1.0, score=s))
        return n

    def test_clade_evals_include_self(self) -> None:
        from meta_agent.managers.hgm_tree import HGMTree

        tree = HGMTree(clade_pseudo_count=10, rng=random.Random(0))
        tree.add(self._node(0, None, [1.0]))
        tree.add(self._node(1, 0, [0.0]))
        # clade(0) = pseudo(0) + pseudo(1) = [1.0] + [0.0]
        self.assertEqual(sorted(tree.clade_evals(0)), [0.0, 1.0])
        self.assertEqual(tree.clade_evals(1), [0.0])
        self.assertAlmostEqual(tree.cmp(0), 0.5)
        self.assertAlmostEqual(tree.cmp(1), 0.0)

    def test_pseudo_count_caps_a_heavily_evaluated_node(self) -> None:
        from meta_agent.managers.hgm_tree import HGMTree

        tree = HGMTree(clade_pseudo_count=3, rng=random.Random(0))
        tree.add(self._node(0, None, [0.8] * 10))
        # n_evals (10) >= clade_pseudo_count (3) -> 3 copies of the mean.
        self.assertEqual(tree.clade_evals(0), [0.8, 0.8, 0.8])

    def test_cmp_rewards_a_strong_descendant(self) -> None:
        """The metaproductivity-performance mismatch: two nodes with the
        same own mean (0.5), but the one with a strong child has the
        higher Clade Metaproductivity."""
        from meta_agent.managers.hgm_tree import HGMTree

        tree = HGMTree(clade_pseudo_count=10, rng=random.Random(0))
        tree.add(self._node(0, None, []))            # root
        tree.add(self._node(1, 0, [0.5] * 5))        # mediocre, has a strong kid
        tree.add(self._node(2, 0, [0.5] * 5))        # mediocre, no kids
        tree.add(self._node(3, 1, [1.0] * 5))        # strong child of node 1
        self.assertGreater(tree.cmp(1), tree.cmp(2))
        self.assertAlmostEqual(tree.cmp(2), 0.5)

    def test_argmax_is_deterministic_under_fixed_seed(self) -> None:
        from meta_agent.managers.hgm_tree import HGMTree

        def build(seed):
            t = HGMTree(rng=random.Random(seed))
            t.add(self._node(0, None, [0.2] * 8))
            t.add(self._node(1, 0, [0.9] * 8))
            t.add(self._node(2, 0, [0.5] * 8))
            return t

        a = build(123).argmax_evaluate(2.0, [0, 1, 2])
        b = build(123).argmax_evaluate(2.0, [0, 1, 2])
        self.assertEqual(a, b)

    def test_lcb_prefers_more_evidence_at_equal_mean(self) -> None:
        from meta_agent.managers.hgm_tree import HGMTree

        tree = HGMTree(rng=random.Random(0))
        tree.add(self._node(0, None, [1.0]))         # 1 perfect eval
        tree.add(self._node(1, 0, [1.0] * 20))       # 20 perfect evals
        self.assertEqual(tree.lcb_select(0.25), 1)

    def test_lcb_excludes_edit_failed_nodes(self) -> None:
        from meta_agent.managers.hgm_tree import HGMNode, HGMTree

        tree = HGMTree(rng=random.Random(0))
        tree.add(self._node(0, None, [0.4] * 10))
        failed = HGMNode(node_id=1, parent_id=0, round_dir=Path("."))
        failed.edit_failed = True
        tree.add(failed)
        self.assertEqual(tree.lcb_select(0.25), 0)

    def test_lcb_select_restrict_to_limits_candidates(self) -> None:
        """restrict_to confines the final pick to the listed nodes (the
        fully-train-evaluated ones), so a thin-evidence fluke is excluded."""
        from meta_agent.managers.hgm_tree import HGMTree

        tree = HGMTree(rng=random.Random(0))
        tree.add(self._node(0, None, [0.6] * 30))   # root, full evidence
        tree.add(self._node(1, 0, [1.0] * 2))       # 2-sample fluke
        tree.add(self._node(2, 0, [0.7] * 30))      # full evidence, best
        # Restricted to the two fully-evaluated nodes — the fluke is out.
        self.assertEqual(tree.lcb_select(0.25, restrict_to={0, 2}), 2)
        # An empty restriction falls back to all real nodes (no crash).
        self.assertIn(tree.lcb_select(0.25, restrict_to=set()), {0, 1, 2})

    def test_tau_off_by_default_and_rises_with_cool_down(self) -> None:
        from meta_agent.managers.hgm_tree import HGMTree

        tree = HGMTree()
        # cool_down off (reference default): tau is a no-op.
        self.assertEqual(tree.tau(100, 100), 1.0)
        self.assertEqual(tree.tau(10, 100), 1.0)
        # cool_down on: tau = (B/b)**beta, rises as the budget drains.
        self.assertAlmostEqual(tree.tau(100, 100, cool_down=True), 1.0)
        self.assertAlmostEqual(tree.tau(10, 100, cool_down=True), 10.0)
        self.assertLess(
            tree.tau(50, 100, cool_down=True),
            tree.tau(10, 100, cool_down=True),
        )

    def test_scheduler_boundary_flips_with_budget_spent(self) -> None:
        from meta_agent.managers.hgm_tree import HGMTree

        tree = HGMTree(rng=random.Random(0))
        for nid in range(5):
            tree.add(self._node(nid, None if nid == 0 else 0, []))
        # 5 real nodes -> threshold = 5 - 1 = 4.
        # budget_spent=10: 10**0.6 ~= 3.98 < 4 -> evaluate.
        self.assertFalse(tree.schedule_favors_expand(0.6, budget_spent=10))
        # budget_spent=20: 20**0.6 ~= 6.03 >= 4 -> expand.
        self.assertTrue(tree.schedule_favors_expand(0.6, budget_spent=20))


# --------------------------------------------------------------------------- #
# End-to-end evolve() with stub editor + evaluator
# --------------------------------------------------------------------------- #


class _StubEditor:
    """Mimics AgentEditor.apply: copies base_dir/task_agent -> out_dir, no
    LLM. Optionally fails a chosen call (1-indexed) to exercise the
    edit-failed path."""

    def __init__(self, fail_call: int | None = None) -> None:
        self.calls = 0
        self.fail_call = fail_call

    def apply(self, feedback, base_dir, out_dir, *, context=None):
        from meta_agent.models import EditResult, EvolutionStrategy

        self.calls += 1
        src = Path(base_dir) / "task_agent"
        dst = Path(out_dir) / "task_agent"
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        strategy = EvolutionStrategy(
            target_files=["workflow.py"],
            optimization_goal="stub self-improvement",
            proposed_changes="stub",
            rationale="",
        )
        if self.fail_call is not None and self.calls == self.fail_call:
            return EditResult(
                success=False,
                errors=["stub editor forced failure"],
                strategy=strategy,
            )
        return EditResult(
            success=True, edited_files=["workflow.py"], strategy=strategy
        )


class _StubEvaluator:
    """Deterministic continuous per-case scores keyed by case id."""

    def __init__(self) -> None:
        self.run_calls = 0

    @staticmethod
    def _score(case_id: str) -> float:
        return (hash(case_id) % 1000) / 1000.0

    def run(self, round_dir, benchmark_dir, *, case_ids=None):
        from meta_agent.models import CaseResult, EvaluationResult

        self.run_calls += 1
        per_case = [
            CaseResult(
                case_id=cid,
                passed=self._score(cid) >= 0.5,
                score=self._score(cid),
            )
            for cid in (case_ids or [])
        ]
        passed = sum(1 for c in per_case if c.passed)
        score = sum(c.score for c in per_case) / max(len(per_case), 1)
        return EvaluationResult(
            score=score,
            passed=passed,
            failed=len(per_case) - passed,
            per_case=per_case,
        )


class HGMEvolveTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="hgm_evolve_"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        # A minimal seed agent directory (the stub editor only copies it).
        self.seed = self.tmp / "seed"
        (self.seed).mkdir()
        (self.seed / "workflow.py").write_text(
            "def run_task(task):\n    return None\n", encoding="utf-8"
        )
        self.experiment = self.tmp / "exp"
        self.experiment.mkdir()

    def _run(self, *, fail_call=None, eval_budget=40, init_expansions=2):
        from meta_agent.feedback_gatherer import DefaultFeedbackGatherer
        from meta_agent.managers.hgm import HGMManager

        manager = HGMManager(
            eval_budget=eval_budget,
            init_expansions=init_expansions,
            eval_batch_size=4,
            alpha=0.6,
            seed=7,
        )
        editor = _StubEditor(fail_call=fail_call)
        evaluator = _StubEvaluator()
        # The stub editor does all the "editing" — the HGM manager makes no
        # LLM call of its own, so no call_llm patch is needed.
        outcome = manager.evolve(
            editor=editor,
            evaluator=evaluator,
            gatherer=DefaultFeedbackGatherer(),
            seed_dir=self.seed,
            benchmark_dir=self.tmp / "bench",  # unused — stub evaluator
            experiment_dir=self.experiment,
            max_rounds=30,
            score_target=None,
            train_case_ids=[f"c{i}" for i in range(20)],
            eval_case_ids=None,
        )
        return manager, outcome

    def test_evolve_consumes_budget_and_builds_a_valid_tree(self) -> None:
        manager, outcome = self._run()

        # Loop budget lands exactly (batch capped at the remainder); the
        # root pre-eval and the finalist re-eval are NOT charged to it.
        self.assertEqual(manager._budget_spent, 40)
        self.assertEqual(manager._tree[0].n_evals, 20)  # root pre-evaluated
        # total_evals = root pre-eval (20) + loop (40) + finalist re-evals.
        self.assertGreaterEqual(manager._tree.total_evals, 60)

        # Tree has at least the root + init expansions.
        self.assertGreaterEqual(len(outcome.rounds), 3)
        self.assertEqual(len(outcome.rounds), len(manager._tree))

        # base_round values form a valid acyclic tree rooted at node 0.
        ids = {fb.round_number for fb in outcome.rounds}
        for fb in outcome.rounds:
            if fb.round_number == 0:
                continue
            self.assertIn(fb.base_round, ids)
            # walk to the root without looping
            seen, cur = set(), fb.round_number
            while cur != 0:
                self.assertNotIn(cur, seen)
                seen.add(cur)
                cur = manager._tree[cur].parent_id
                self.assertIsNotNone(cur)

        # best_round is a real node.
        self.assertIn(outcome.best_round, ids)

    def test_finalize_re_evaluates_finalists_on_full_train(self) -> None:
        """The finalist re-evaluation pass brings the top-k nodes to a
        full-train estimate before selection, without charging the loop
        budget — so the selected best has full-train evidence."""
        manager, outcome = self._run()
        n_train = 20
        # Loop budget untouched by finalization.
        self.assertEqual(manager._budget_spent, 40)
        # Root + finalists reach the full train-split size.
        full = [
            n for n in manager._tree.nodes.values() if n.n_evals == n_train
        ]
        self.assertGreaterEqual(len(full), 2)
        # The selected best is one of the fully-evaluated finalists.
        self.assertEqual(manager._tree[outcome.best_round].n_evals, n_train)

    def test_every_round_dir_has_artifacts(self) -> None:
        manager, _ = self._run()
        for nid, node in manager._tree.nodes.items():
            self.assertTrue((node.round_dir / "task_agent").is_dir(), nid)
            self.assertTrue((node.round_dir / "feedback.json").is_file(), nid)
            self.assertTrue((node.round_dir / "hgm_node.json").is_file(), nid)

    def test_expandable_excludes_unevaluated_and_zero_mean_nodes(self) -> None:
        """A node is expandable only once evaluated with a positive mean —
        faithful to hgm.py::expand()'s isfinite/positive filter."""
        from meta_agent.managers.hgm import HGMManager
        from meta_agent.managers.hgm_tree import HGMNode, HGMTree
        from meta_agent.models import CaseResult

        m = HGMManager()
        m._tree = HGMTree()
        n0 = HGMNode(0, None, Path("."))           # evaluated, positive
        n0.record(CaseResult(case_id="a", passed=True, score=0.7))
        m._tree.add(n0)
        m._tree.add(HGMNode(1, 0, Path(".")))      # unevaluated
        n2 = HGMNode(2, 0, Path("."))              # evaluated, zero mean
        n2.record(CaseResult(case_id="b", passed=False, score=0.0))
        m._tree.add(n2)
        self.assertEqual(m._expandable(), [0])

    def test_ancestor_goals_walks_the_lineage_root_to_node(self) -> None:
        """The strategy prompt's edit lineage must list optimization goals
        from the root down to the parent, in order."""
        from meta_agent.managers.hgm import HGMManager
        from meta_agent.managers.hgm_tree import HGMNode, HGMTree
        from meta_agent.models import (
            AgentFeedback,
            EvaluationResult,
            EvolutionStrategy,
        )

        m = HGMManager()
        m._tree = HGMTree()
        m._feedback = {}
        for nid, parent in [(0, None), (1, 0), (2, 1)]:
            m._tree.add(HGMNode(nid, parent, Path(".")))
            m._feedback[nid] = AgentFeedback(
                round_number=nid,
                base_round=parent or 0,
                strategy=EvolutionStrategy(
                    target_files=[],
                    optimization_goal=f"goal-{nid}",
                    proposed_changes="x",
                ),
                eval_result=EvaluationResult(score=0.0),
            )
        goals = m._ancestor_goals(2)
        self.assertEqual([d for d, _ in goals], [0, 1, 2])
        self.assertEqual([g for _, g in goals], ["goal-0", "goal-1", "goal-2"])

    def test_failed_edit_node_is_recorded_and_run_survives(self) -> None:
        manager, outcome = self._run(fail_call=3)

        failed = [fb for fb in outcome.rounds if fb.edit_errors]
        self.assertTrue(failed, "expected one edit-failed node")
        failed_ids = {fb.round_number for fb in failed}
        for nid in failed_ids:
            self.assertTrue(manager._tree[nid].edit_failed)
            # excluded from both selectable sets
            self.assertNotIn(nid, manager._expandable())
            self.assertNotIn(nid, manager._evaluable())
        # the failed node is never picked as best
        self.assertNotIn(outcome.best_round, failed_ids)


# --------------------------------------------------------------------------- #
# Config wiring
# --------------------------------------------------------------------------- #


class HGMConfigTests(unittest.TestCase):
    def test_registry_exposes_hgm(self) -> None:
        import meta_agent.managers  # noqa: F401 — triggers @register
        from meta_agent import registry

        self.assertIn("hgm", registry.available("manager"))

    def test_hgm_math_config_builds_hgm_manager(self) -> None:
        from meta_agent.config import build_components, load

        cfg = load(REPO_ROOT / "configs" / "hgm_math.yaml")
        fw = build_components(cfg)
        self.assertEqual(type(fw.manager).__name__, "HGMManager")
        self.assertEqual(fw.manager.eval_budget, 24)
        self.assertEqual(fw.manager.eval_batch_size, 4)

    def test_hgm_travel_and_shopping_configs_load(self) -> None:
        from meta_agent.config import load

        for name in ("hgm_travel.yaml", "hgm_shopping.yaml"):
            cfg = load(REPO_ROOT / "configs" / name)
            self.assertEqual(cfg.manager.type, "hgm")
            self.assertEqual(cfg.manager.config["eval_budget"], 400)
            self.assertIsNotNone(cfg.split)


if __name__ == "__main__":
    unittest.main()
