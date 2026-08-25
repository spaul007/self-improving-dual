"""Offline tests for the per-block Thompson-sampling bandit
(meta_agent/block_bandit.py), the "adaptive" block_selection_strategy.

No API key required -- pure math against hand-built HGMTree/feedback state.

    PYTHONPATH=. python3 -m unittest tests.test_block_bandit
"""
from __future__ import annotations

import random
import unittest
from pathlib import Path

from meta_agent.block_bandit import AdaptiveStrategy, BlockBandit, BlockPosterior
from meta_agent.managers.hgm_tree import HGMNode, HGMTree
from meta_agent.models import CaseResult, EvolutionStrategy


def _node(node_id, parent_id, scores, *, edit_failed=False):
    n = HGMNode(node_id=node_id, parent_id=parent_id, round_dir=Path("."))
    n.edit_failed = edit_failed
    for i, s in enumerate(scores):
        n.record(CaseResult(case_id=f"{node_id}-{i}", passed=s >= 1.0, score=s))
    return n


class _FakeFeedback:
    """Minimal stand-in for AgentFeedback -- BlockBandit only ever reads
    ``.strategy.block``."""

    def __init__(self, block):
        self.strategy = EvolutionStrategy(
            optimization_goal="test", proposed_changes="test", block=block
        )


def _tree_and_feedback(spec):
    """``spec``: list of (node_id, parent_id, block_or_None, scores,
    edit_failed) tuples. Returns (tree, feedback dict)."""
    tree = HGMTree(rng=random.Random(0))
    feedback = {}
    for node_id, parent_id, block, scores, edit_failed in spec:
        tree.add(_node(node_id, parent_id, scores, edit_failed=edit_failed))
        feedback[node_id] = _FakeFeedback(block)
    return tree, feedback


class BlockBanditTests(unittest.TestCase):
    BLOCKS = (
        "collaboration_workflow",
        "foundation_capability",
        "individual_subagent",
        "verifiers",
    )

    def test_seed_node_excluded_from_every_block(self) -> None:
        # Seed node has block=None (unedited) and a high score -- must not
        # be folded into any block's tally.
        tree, feedback = _tree_and_feedback(
            [(0, None, None, [1.0] * 10, False)]
        )
        bandit = BlockBandit(blocks=self.BLOCKS, rng=random.Random(1))
        result = bandit.select(tree, feedback)
        for block in self.BLOCKS:
            post = result.posteriors[block]
            self.assertEqual(post.n_success, 0.0)
            self.assertEqual(post.n_failure, 0.0)
            self.assertEqual(post.n_evals, 0)

    def test_failed_and_unevaluated_nodes_excluded(self) -> None:
        tree, feedback = _tree_and_feedback(
            [
                (0, None, None, [], False),
                (1, 0, "individual_subagent", [], False),  # n_evals == 0
                (2, 0, "individual_subagent", [0.9], True),  # edit_failed
            ]
        )
        bandit = BlockBandit(blocks=self.BLOCKS, rng=random.Random(1))
        result = bandit.select(tree, feedback)
        post = result.posteriors["individual_subagent"]
        self.assertEqual(post.n_evals, 0)
        self.assertEqual(post.n_success, 0.0)

    def test_all_candidate_blocks_always_present(self) -> None:
        tree, feedback = _tree_and_feedback(
            [(0, None, None, [], False), (1, 0, "verifiers", [0.5, 0.5], False)]
        )
        bandit = BlockBandit(blocks=self.BLOCKS, rng=random.Random(1))
        result = bandit.select(tree, feedback)
        self.assertEqual(set(result.posteriors), set(self.BLOCKS))
        # The untried blocks still get a full BlockPosterior, just at 0 evals.
        self.assertEqual(result.posteriors["collaboration_workflow"].n_evals, 0)

    def test_default_blocks_come_from_block_suggester(self) -> None:
        from meta_agent.block_suggester import _BLOCK_BODIES

        bandit = BlockBandit(rng=random.Random(0))
        self.assertEqual(bandit.blocks, tuple(sorted(_BLOCK_BODIES)))

    def test_higher_reward_block_is_picked_far_more_often(self) -> None:
        # individual_subagent: consistently strong (0.9 across 4 nodes x 5
        # cases). collaboration_workflow: consistently weak (0.1). Over many
        # independent selections (rebuilding the tree fresh each time so
        # each draw is an independent Thompson sample of the same
        # posterior, not a sequentially-updating one), the strong block
        # should dominate empirically -- this is a statistical smoke test,
        # not exact equality, since Thompson sampling is stochastic by
        # design.
        spec = [(0, None, None, [], False)]
        nid = 1
        for _ in range(4):
            spec.append((nid, 0, "individual_subagent", [0.9] * 5, False))
            nid += 1
        for _ in range(4):
            spec.append((nid, 0, "collaboration_workflow", [0.1] * 5, False))
            nid += 1
        tree, feedback = _tree_and_feedback(spec)

        rng = random.Random(42)
        bandit = BlockBandit(blocks=self.BLOCKS, rng=rng)
        counts = {b: 0 for b in self.BLOCKS}
        trials = 500
        for _ in range(trials):
            counts[bandit.select(tree, feedback).block] += 1

        self.assertGreater(
            counts["individual_subagent"], counts["collaboration_workflow"]
        )
        # Strong dominance expected given the wide score gap (0.9 vs 0.1),
        # but leave real headroom for sampling noise.
        self.assertGreater(counts["individual_subagent"], trials * 0.7)

    def test_never_tried_block_still_gets_picked_sometimes(self) -> None:
        # Cold-start fairness: an unexplored block (Beta(prior,prior)) must
        # not be starved to zero just because others have some history.
        spec = [(0, None, None, [], False)]
        nid = 1
        for _ in range(3):
            spec.append((nid, 0, "individual_subagent", [0.6] * 4, False))
            nid += 1
        # "verifiers" has zero evals -- never touched.
        tree, feedback = _tree_and_feedback(spec)

        rng = random.Random(7)
        bandit = BlockBandit(blocks=self.BLOCKS, rng=rng)
        counts = {b: 0 for b in self.BLOCKS}
        for _ in range(500):
            counts[bandit.select(tree, feedback).block] += 1

        self.assertGreater(counts["verifiers"], 0)

    def test_reproducible_under_fixed_seed(self) -> None:
        tree, feedback = _tree_and_feedback(
            [
                (0, None, None, [], False),
                (1, 0, "individual_subagent", [0.7, 0.8], False),
                (2, 0, "verifiers", [0.2, 0.3], False),
            ]
        )

        def run(seed):
            bandit = BlockBandit(blocks=self.BLOCKS, rng=random.Random(seed))
            return [bandit.select(tree, feedback).block for _ in range(20)]

        self.assertEqual(run(123), run(123))


if __name__ == "__main__":
    unittest.main()
