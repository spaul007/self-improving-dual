"""Offline tests for the per-implementation-strategy Thompson-sampling
bandit (meta_agent/implementation_strategy_bandit.py), the "adaptive"
implementation_strategy_selection_strategy.

No API key required -- pure math against hand-built HGMTree/feedback state.
Mirrors tests/test_block_bandit.py's structure exactly (same underlying
Beta-Bernoulli math, different axis).

    PYTHONPATH=. python3 -m unittest tests.test_implementation_strategy_bandit
"""
from __future__ import annotations

import random
import unittest
from pathlib import Path

from meta_agent.implementation_strategy_bandit import (
    AdaptiveImplementationStrategy,
    ImplementationStrategyBandit,
    ImplementationStrategyPosterior,
)
from meta_agent.managers.hgm_tree import HGMNode, HGMTree
from meta_agent.models import CaseResult, EvolutionStrategy


def _node(node_id, parent_id, scores, *, edit_failed=False):
    n = HGMNode(node_id=node_id, parent_id=parent_id, round_dir=Path("."))
    n.edit_failed = edit_failed
    for i, s in enumerate(scores):
        n.record(CaseResult(case_id=f"{node_id}-{i}", passed=s >= 1.0, score=s))
    return n


class _FakeFeedback:
    """Minimal stand-in for AgentFeedback -- ImplementationStrategyBandit
    only ever reads ``.strategy.implementation_strategy``."""

    def __init__(self, implementation_strategy):
        self.strategy = EvolutionStrategy(
            optimization_goal="test", proposed_changes="test",
            implementation_strategy=implementation_strategy,
        )


def _tree_and_feedback(spec):
    """``spec``: list of (node_id, parent_id, strategy_or_None, scores,
    edit_failed) tuples. Returns (tree, feedback dict)."""
    tree = HGMTree(rng=random.Random(0))
    feedback = {}
    for node_id, parent_id, strategy, scores, edit_failed in spec:
        tree.add(_node(node_id, parent_id, scores, edit_failed=edit_failed))
        feedback[node_id] = _FakeFeedback(strategy)
    return tree, feedback


class ImplementationStrategyBanditTests(unittest.TestCase):
    STRATEGIES = ("harness_heavy", "llm_heavy", "mixed")

    def test_seed_node_excluded_from_every_strategy(self) -> None:
        tree, feedback = _tree_and_feedback(
            [(0, None, None, [1.0] * 10, False)]
        )
        bandit = ImplementationStrategyBandit(strategies=self.STRATEGIES, rng=random.Random(1))
        result = bandit.select(tree, feedback)
        for s in self.STRATEGIES:
            post = result.posteriors[s]
            self.assertEqual(post.n_success, 0.0)
            self.assertEqual(post.n_failure, 0.0)
            self.assertEqual(post.n_evals, 0)

    def test_failed_and_unevaluated_nodes_excluded(self) -> None:
        tree, feedback = _tree_and_feedback(
            [
                (0, None, None, [], False),
                (1, 0, "llm_heavy", [], False),  # n_evals == 0
                (2, 0, "llm_heavy", [0.9], True),  # edit_failed
            ]
        )
        bandit = ImplementationStrategyBandit(strategies=self.STRATEGIES, rng=random.Random(1))
        result = bandit.select(tree, feedback)
        post = result.posteriors["llm_heavy"]
        self.assertEqual(post.n_evals, 0)
        self.assertEqual(post.n_success, 0.0)

    def test_all_candidates_always_present(self) -> None:
        tree, feedback = _tree_and_feedback(
            [(0, None, None, [], False), (1, 0, "mixed", [0.5, 0.5], False)]
        )
        bandit = ImplementationStrategyBandit(strategies=self.STRATEGIES, rng=random.Random(1))
        result = bandit.select(tree, feedback)
        self.assertEqual(set(result.posteriors), set(self.STRATEGIES))
        self.assertEqual(result.posteriors["harness_heavy"].n_evals, 0)

    def test_default_strategies_come_from_implementation_strategy_module(self) -> None:
        from meta_agent.implementation_strategy import _IMPLEMENTATION_STRATEGY_BODIES

        bandit = ImplementationStrategyBandit(rng=random.Random(0))
        self.assertEqual(bandit.strategies, tuple(sorted(_IMPLEMENTATION_STRATEGY_BODIES)))

    def test_higher_reward_strategy_is_picked_far_more_often(self) -> None:
        spec = [(0, None, None, [], False)]
        nid = 1
        for _ in range(4):
            spec.append((nid, 0, "llm_heavy", [0.9] * 5, False))
            nid += 1
        for _ in range(4):
            spec.append((nid, 0, "harness_heavy", [0.1] * 5, False))
            nid += 1
        tree, feedback = _tree_and_feedback(spec)

        rng = random.Random(42)
        bandit = ImplementationStrategyBandit(strategies=self.STRATEGIES, rng=rng)
        counts = {s: 0 for s in self.STRATEGIES}
        trials = 500
        for _ in range(trials):
            counts[bandit.select(tree, feedback).implementation_strategy] += 1

        self.assertGreater(counts["llm_heavy"], counts["harness_heavy"])
        self.assertGreater(counts["llm_heavy"], trials * 0.7)

    def test_never_tried_strategy_still_gets_picked_sometimes(self) -> None:
        spec = [(0, None, None, [], False)]
        nid = 1
        for _ in range(3):
            spec.append((nid, 0, "llm_heavy", [0.6] * 4, False))
            nid += 1
        # "mixed" has zero evals -- never touched.
        tree, feedback = _tree_and_feedback(spec)

        rng = random.Random(7)
        bandit = ImplementationStrategyBandit(strategies=self.STRATEGIES, rng=rng)
        counts = {s: 0 for s in self.STRATEGIES}
        for _ in range(500):
            counts[bandit.select(tree, feedback).implementation_strategy] += 1

        self.assertGreater(counts["mixed"], 0)

    def test_default_reward_metric_is_fractional_score(self) -> None:
        bandit = ImplementationStrategyBandit(rng=random.Random(0))
        self.assertEqual(bandit.reward_metric, "fractional_score")

    def test_invalid_reward_metric_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ImplementationStrategyBandit(rng=random.Random(0), reward_metric="not_a_real_metric")

    def test_boolean_increase_success_when_at_or_above_parent(self) -> None:
        tree, feedback = _tree_and_feedback(
            [
                (0, None, None, [], False),
                (1, 0, "llm_heavy", [0.5, 0.5], False),  # parent, mean=0.5
                (2, 1, "llm_heavy", [0.6], False),  # above -> success
                (3, 1, "llm_heavy", [0.4], False),  # below -> failure
                (4, 1, "llm_heavy", [0.5], False),  # tie -> success
            ]
        )
        bandit = ImplementationStrategyBandit(
            strategies=self.STRATEGIES, rng=random.Random(1), reward_metric="boolean_increase",
        )
        result = bandit.select(tree, feedback)
        post = result.posteriors["llm_heavy"]
        self.assertEqual(post.n_evals, 3)
        self.assertEqual(post.n_success, 2.0)
        self.assertEqual(post.n_failure, 1.0)

    def test_reproducible_under_fixed_seed(self) -> None:
        tree, feedback = _tree_and_feedback(
            [
                (0, None, None, [], False),
                (1, 0, "llm_heavy", [0.7, 0.8], False),
                (2, 0, "mixed", [0.2, 0.3], False),
            ]
        )

        def run(seed):
            bandit = ImplementationStrategyBandit(strategies=self.STRATEGIES, rng=random.Random(seed))
            return [bandit.select(tree, feedback).implementation_strategy for _ in range(20)]

        self.assertEqual(run(123), run(123))


class InitialRankingTests(unittest.TestCase):
    STRATEGIES = ("harness_heavy", "llm_heavy", "mixed")
    RANKING = ("llm_heavy", "mixed", "harness_heavy")

    def test_default_none_gives_every_strategy_zero_bonus(self) -> None:
        bandit = ImplementationStrategyBandit(strategies=self.STRATEGIES, rng=random.Random(0))
        self.assertEqual(
            bandit._initial_success_bonus, dict.fromkeys(self.STRATEGIES, 0.0)
        )

    def test_ranking_assigns_decreasing_bonus_by_rank(self) -> None:
        bandit = ImplementationStrategyBandit(
            strategies=self.STRATEGIES, rng=random.Random(0),
            initial_ranking=self.RANKING, initial_rank_strength=2.0,
        )
        bonus = bandit._initial_success_bonus
        self.assertEqual(bonus["llm_heavy"], 4.0)
        self.assertEqual(bonus["mixed"], 2.0)
        self.assertEqual(bonus["harness_heavy"], 0.0)

    def test_mismatched_ranking_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ImplementationStrategyBandit(
                strategies=self.STRATEGIES, rng=random.Random(0),
                initial_ranking=("llm_heavy", "mixed"),
            )
        with self.assertRaises(ValueError):
            ImplementationStrategyBandit(
                strategies=self.STRATEGIES, rng=random.Random(0),
                initial_ranking=self.RANKING + ("nonexistent",),
            )


if __name__ == "__main__":
    unittest.main()
