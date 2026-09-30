"""The with/without-memory arm choice is independent of block selection.

``choose_arm`` sees no block; its posteriors pool nodes of every block; it
draws from the layer's own RNGs, never the manager's block RNG; and the
block bandit never reads the arm. End to end: attaching a live memory layer
(real arm draws) leaves the block sequence and the tree exactly as they are
without one.

    PYTHONPATH=. python3 -m unittest tests.test_memory_arm_block_independence
"""
from __future__ import annotations

import inspect
import random
import shutil
import tempfile
import unittest
from pathlib import Path

from meta_agent.edit_memory.layer import EditMemoryLayer
from meta_agent.managers.hgm_tree import HGMNode, HGMTree
from meta_agent.models import AgentFeedback, CaseResult, EvaluationResult, EvolutionStrategy


def _node(nid, parent, scores, arm="none"):
    n = HGMNode(node_id=nid, parent_id=parent, round_dir=Path("."), memory_arm=arm)
    for i, s in enumerate(scores):
        n.record(CaseResult(case_id=f"{nid}-{i}", passed=s >= 0.5, score=s))
    return n


def _fb(nid, block):
    return AgentFeedback(round_number=nid, base_round=0,
                         strategy=EvolutionStrategy(optimization_goal="g", proposed_changes="p",
                                                    block=block),
                         eval_result=EvaluationResult(score=0.0))


class UnitTests(unittest.TestCase):
    def test_choose_arm_takes_no_block(self) -> None:
        self.assertEqual(list(inspect.signature(EditMemoryLayer.choose_arm).parameters),
                         ["self", "tree"])

    def test_tallies_pool_nodes_of_every_block(self) -> None:
        tree = HGMTree(rng=random.Random(0))
        tree.add(_node(0, None, [0.5]))
        tree.add(_node(1, 0, [1.0, 1.0], arm="with"))        # e.g. verifiers
        tree.add(_node(2, 0, [0.0], arm="with"))             # e.g. mixed
        tree.add(_node(3, 0, [0.5], arm="without"))
        t = EditMemoryLayer(lambda **kw: None).arm_tallies(tree)
        self.assertEqual(t["with"], {"S": 2.0, "F": 1.0, "n_nodes": 2})
        self.assertEqual(t["without"], {"S": 0.5, "F": 0.5, "n_nodes": 1})

    def test_block_bandit_ignores_the_arm(self) -> None:
        from meta_agent.block_bandit import BlockBandit

        def posteriors(arms):
            tree = HGMTree(rng=random.Random(0))
            tree.add(_node(0, None, [0.5]))
            for nid, arm in zip((1, 2, 3), arms):
                tree.add(_node(nid, 0, [0.2 * nid, 0.9], arm=arm))
            fb = {1: _fb(1, "verifiers"), 2: _fb(2, "mixed"), 3: _fb(3, "verifiers")}
            sel = BlockBandit(rng=random.Random(9)).select(tree, fb)
            return sel.block, {b: (p.n_success, p.n_failure, p.n_evals) for b, p in sel.posteriors.items()}

        self.assertEqual(posteriors(["with", "without", "none"]),
                         posteriors(["none", "with", "with"]))


class EndToEndTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="arm_block_"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.seed = self.tmp / "seed"
        (self.seed / "mutable_tools").mkdir(parents=True)
        (self.seed / "workflow.py").write_text("def run_task(task):\n    return None\n")
        (self.seed / "tool_wrapper.py").write_text("")
        (self.seed / "tools_schema.json").write_text("[]")

    def run_hgm(self, strategy: str, with_layer: bool):
        from meta_agent.feedback_gatherer import DefaultFeedbackGatherer
        from meta_agent.managers.hgm import HGMManager
        from tests.test_edit_memory_layer import _ScriptedLLM, _memory_stub_editor
        from tests.test_hgm_smoke import _StubEvaluator

        exp = self.tmp / f"exp_{strategy}_{with_layer}"
        exp.mkdir()
        (exp / "config.snapshot.yaml").write_text("")
        manager = HGMManager(eval_budget=48, init_expansions=2, eval_batch_size=4, alpha=0.6,
                             seed=7, finalize_top_k=0, expand_eval_size=4,
                             block_selection_strategy=strategy)
        layer = None
        if with_layer:
            layer = EditMemoryLayer(_ScriptedLLM(), repo_root=self.tmp / "repo", window_size=2,
                                    instruction_every=5, arm_min_pulls=1, seed=3,
                                    curator={"sandbox": "none", "max_llm_calls": 6, "timeout_s": 60})
        kwargs = {"edit_memory": layer} if layer is not None else {}
        manager.evolve(editor=_memory_stub_editor(), evaluator=_StubEvaluator(),
                       gatherer=DefaultFeedbackGatherer(), seed_dir=self.seed,
                       benchmark_dir=self.tmp / "b", experiment_dir=exp, max_rounds=30,
                       score_target=None, train_case_ids=[f"c{i}" for i in range(20)],
                       eval_case_ids=None, **kwargs)
        blocks = [manager._feedback[nid].strategy.block for nid in sorted(manager._feedback)]
        shape = [(nid, n.parent_id, n.n_evals) for nid, n in sorted(manager._tree.nodes.items())]
        arms = [n.memory_arm for _, n in sorted(manager._tree.nodes.items())]
        return blocks, shape, arms

    def test_live_layer_leaves_block_selection_and_search_unchanged(self) -> None:
        for strategy in ("non_adaptive", "adaptive"):
            with self.subTest(strategy=strategy):
                blocks0, shape0, _ = self.run_hgm(strategy, with_layer=False)
                blocks1, shape1, arms = self.run_hgm(strategy, with_layer=True)
                self.assertIn("with", arms)
                self.assertIn("without", arms)
                self.assertEqual(blocks1, blocks0)
                self.assertEqual(shape1, shape0)


if __name__ == "__main__":
    unittest.main()
