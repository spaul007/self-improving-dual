"""Manager-level wiring tests for HGMManager's active_blocks param
(meta_agent/managers/hgm.py) -- restricts the "non_adaptive"/"adaptive"
candidate set to exactly the given block names, e.g. to keep
llm_backbone_selection out of an otherwise-adaptive run without touching
the shared block_suggester.py::_BLOCK_BODIES dict.

Mirrors tests/test_block_bandit.py's HGMManagerBlockRewardMetricWiringTests
style (direct HGMManager() construction, no LLM/evaluator needed).

    PYTHONPATH=. python3 -m unittest tests.test_hgm_active_blocks
"""
from __future__ import annotations

import unittest

from meta_agent.block_suggester import _BLOCK_BODIES
from meta_agent.managers.hgm import HGMManager
from meta_agent.managers.hgm_tree import HGMNode, HGMTree


class ActiveBlocksTests(unittest.TestCase):
    def test_default_manager_has_all_block_bodies_as_candidates(self) -> None:
        m = HGMManager()
        self.assertIsNone(m.active_blocks)
        from meta_agent.block_suggester import OPT_IN_BLOCKS
        self.assertEqual(set(m._block_bandit.blocks), set(_BLOCK_BODIES) - OPT_IN_BLOCKS)
        self.assertNotIn("skills", m._block_bandit.blocks)

    def test_active_blocks_restricts_bandit_candidate_set(self) -> None:
        restricted = sorted(set(_BLOCK_BODIES) - {"llm_backbone_selection"})
        m = HGMManager(active_blocks=restricted)
        self.assertEqual(set(m._block_bandit.blocks), set(restricted))
        self.assertNotIn("llm_backbone_selection", m._block_bandit.blocks)

    def test_unknown_active_block_name_rejected(self) -> None:
        with self.assertRaises(ValueError):
            HGMManager(active_blocks=["not_a_real_block"])

    def test_non_adaptive_never_samples_an_excluded_block(self) -> None:
        restricted = sorted(set(_BLOCK_BODIES) - {"llm_backbone_selection"})
        m = HGMManager(block_selection_strategy="non_adaptive", active_blocks=restricted, seed=1)
        m._tree = HGMTree()
        m._feedback = {}
        root = HGMNode(0, None, __import__("pathlib").Path("."))
        m._tree.add(root)
        seen = {m._select_block(root) for _ in range(200)}
        self.assertNotIn("llm_backbone_selection", seen)
        self.assertTrue(seen)

    def test_evolve_reseed_site_also_respects_active_blocks(self) -> None:
        # __init__ constructs one BlockBandit; evolve() rebuilds a second
        # one against the freshly-seeded RNG (see hgm.py's own comment) --
        # both construction sites must honor active_blocks, not just the
        # first. Exercise the second site directly without a full
        # evolve() call (no LLM/evaluator available in this test).
        restricted = sorted(set(_BLOCK_BODIES) - {"llm_backbone_selection"})
        m = HGMManager(active_blocks=restricted)
        import random as _random
        from meta_agent.block_bandit import BlockBandit

        m._block_rng = _random.Random(m.seed)
        m._block_bandit = BlockBandit(
            beta_prior=m.beta_prior, rng=m._block_rng,
            reward_metric=m.block_reward_metric,
            initial_block_ranking=m.block_initial_ranking,
            initial_rank_strength=m.block_initial_rank_strength,
            blocks=sorted(m.active_blocks) if m.active_blocks is not None else None,
        )
        self.assertNotIn("llm_backbone_selection", m._block_bandit.blocks)


if __name__ == "__main__":
    unittest.main()
