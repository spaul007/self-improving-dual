"""Manager-level wiring tests for HGMManager's implementation_strategy axis
(meta_agent/managers/hgm.py) -- a new, orthogonal-to-block steering axis
(llm_heavy/mixed/harness_heavy) telling the editor how much of an EXPAND's
fix should be prompt/LLM work vs. deterministic code.

Mirrors tests/test_hgm_active_blocks.py's style (direct HGMManager()
construction, no LLM/evaluator needed) plus a couple of
_render_expand_context content-assertion tests.

    PYTHONPATH=. python3 -m unittest tests.test_hgm_implementation_strategy
"""
from __future__ import annotations

import unittest
from pathlib import Path

from meta_agent.block_suggester import _BLOCK_BODIES
from meta_agent.implementation_strategy import _IMPLEMENTATION_STRATEGY_BODIES
from meta_agent.managers.hgm import HGMManager
from meta_agent.managers.hgm_tree import HGMNode, HGMTree


def _fresh_manager_with_root(**kwargs) -> tuple[HGMManager, HGMNode]:
    m = HGMManager(**kwargs)
    m._tree = HGMTree()
    m._feedback = {}
    root = HGMNode(0, None, Path("."))
    m._tree.add(root)
    return m, root


class DefaultOffTests(unittest.TestCase):
    def test_default_manager_has_axis_off(self) -> None:
        m = HGMManager()
        self.assertIsNone(m.implementation_strategy_selection_strategy)

    def test_select_returns_none_when_axis_off(self) -> None:
        m, root = _fresh_manager_with_root()
        self.assertIsNone(m._select_implementation_strategy(root))
        self.assertIsNone(m._last_implementation_strategy_selection)

    def test_context_has_no_implementation_strategy_section_when_off(self) -> None:
        m, root = _fresh_manager_with_root()
        ctx_without = m._render_expand_context(root, "collaboration_workflow", Path("."), 1)
        ctx_with_none = m._render_expand_context(
            root, "collaboration_workflow", Path("."), 1, implementation_strategy=None,
        )
        # Regression guard: omitting the new kwarg entirely, and passing it
        # explicitly as None, must both be byte-identical to each other and
        # contain no implementation-strategy heading -- i.e. this feature
        # adds nothing to the context for a manager that hasn't opted in.
        self.assertEqual(ctx_without, ctx_with_none)
        self.assertNotIn("Implementation strategy for this EXPAND", ctx_without)

    def test_unknown_selection_strategy_rejected_at_construction(self) -> None:
        with self.assertRaises(ValueError):
            HGMManager(implementation_strategy_selection_strategy="not_a_real_strategy")


class FixedValueTests(unittest.TestCase):
    def test_each_fixed_value_returns_itself(self) -> None:
        for value in ("llm_heavy", "mixed", "harness_heavy"):
            m, root = _fresh_manager_with_root(implementation_strategy_selection_strategy=value)
            self.assertEqual(m._select_implementation_strategy(root), value)
            self.assertIsNone(m._last_implementation_strategy_selection)


class NonAdaptiveTests(unittest.TestCase):
    def test_samples_only_from_the_three_values(self) -> None:
        m, root = _fresh_manager_with_root(
            implementation_strategy_selection_strategy="non_adaptive", seed=1,
        )
        seen = {m._select_implementation_strategy(root) for _ in range(200)}
        self.assertEqual(seen, set(_IMPLEMENTATION_STRATEGY_BODIES))

    def test_reproducible_for_a_fixed_seed(self) -> None:
        def draws(seed):
            m, root = _fresh_manager_with_root(
                implementation_strategy_selection_strategy="non_adaptive", seed=seed,
            )
            return [m._select_implementation_strategy(root) for _ in range(20)]

        self.assertEqual(draws(7), draws(7))


class AdaptiveTests(unittest.TestCase):
    def test_adaptive_thompson_samples_via_the_bandit(self) -> None:
        m, root = _fresh_manager_with_root(
            implementation_strategy_selection_strategy="adaptive", seed=3,
        )
        value = m._select_implementation_strategy(root)
        self.assertIn(value, _IMPLEMENTATION_STRATEGY_BODIES)
        self.assertIsNotNone(m._last_implementation_strategy_selection)
        self.assertEqual(m._last_implementation_strategy_selection.implementation_strategy, value)


class DynamicBlockExclusionTests(unittest.TestCase):
    """harness_heavy -> llm_backbone_selection excluded from block
    candidates; llm_heavy/mixed/off -> block selection fully unaffected."""

    def test_harness_heavy_excludes_llm_backbone_selection_non_adaptive_block(self) -> None:
        m, root = _fresh_manager_with_root(
            implementation_strategy_selection_strategy="harness_heavy",
            block_selection_strategy="non_adaptive", seed=1,
        )
        for _ in range(200):
            impl = m._select_implementation_strategy(root)
            self.assertEqual(impl, "harness_heavy")
            block_exclude = {"llm_backbone_selection"} if impl == "harness_heavy" else None
            block = m._select_block(root, exclude=block_exclude)
            self.assertNotEqual(block, "llm_backbone_selection")

    def test_harness_heavy_excludes_llm_backbone_selection_adaptive_block(self) -> None:
        m, root = _fresh_manager_with_root(
            implementation_strategy_selection_strategy="harness_heavy",
            block_selection_strategy="adaptive", seed=1,
        )
        for _ in range(50):
            impl = m._select_implementation_strategy(root)
            block_exclude = {"llm_backbone_selection"} if impl == "harness_heavy" else None
            block = m._select_block(root, exclude=block_exclude)
            self.assertNotEqual(block, "llm_backbone_selection")

    def test_llm_heavy_leaves_block_candidates_unaffected(self) -> None:
        m, root = _fresh_manager_with_root(
            implementation_strategy_selection_strategy="llm_heavy",
            block_selection_strategy="non_adaptive", seed=1,
        )
        seen = set()
        for _ in range(300):
            impl = m._select_implementation_strategy(root)
            self.assertEqual(impl, "llm_heavy")
            block_exclude = {"llm_backbone_selection"} if impl == "harness_heavy" else None
            seen.add(m._select_block(root, exclude=block_exclude))
        # llm_backbone_selection must still be reachable -- exclusion never triggers for llm_heavy.
        self.assertIn("llm_backbone_selection", seen)

    def test_axis_off_leaves_block_candidates_unaffected(self) -> None:
        m, root = _fresh_manager_with_root(
            block_selection_strategy="non_adaptive", seed=1,
        )
        seen = set()
        for _ in range(300):
            impl = m._select_implementation_strategy(root)
            self.assertIsNone(impl)
            block_exclude = {"llm_backbone_selection"} if impl == "harness_heavy" else None
            seen.add(m._select_block(root, exclude=block_exclude))
        self.assertIn("llm_backbone_selection", seen)


class ContextSplicingTests(unittest.TestCase):
    def test_both_block_and_implementation_strategy_text_appear_together(self) -> None:
        m, root = _fresh_manager_with_root()
        ctx = m._render_expand_context(
            root, "foundation_capability", Path("."), 1,
            implementation_strategy="harness_heavy",
        )
        self.assertIn("Selected block for this EXPAND: foundation_capability", ctx)
        self.assertIn("Implementation strategy for this EXPAND: harness_heavy", ctx)
        self.assertIn(_IMPLEMENTATION_STRATEGY_BODIES["harness_heavy"], ctx)

    def test_each_implementation_strategy_body_appears_verbatim(self) -> None:
        m, root = _fresh_manager_with_root()
        for value, body in _IMPLEMENTATION_STRATEGY_BODIES.items():
            ctx = m._render_expand_context(
                root, "verifiers", Path("."), 1, implementation_strategy=value,
            )
            self.assertIn(body, ctx)


if __name__ == "__main__":
    unittest.main()
