"""Tests for meta_agent/managers/hgm_block_tagged.py's BlockTaggedHGMManager
and the _lineage_memory_label hook it overrides (meta_agent/managers/hgm.py).

    PYTHONPATH=. python3 -m unittest tests.test_hgm_block_tagged_memory
"""
from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from meta_agent import registry
from meta_agent.managers import hgm_block_tagged  # noqa: F401 -- registers the type
from meta_agent.managers.hgm import HGMManager
from meta_agent.managers.hgm_block_tagged import BlockTaggedHGMManager
from meta_agent.managers.hgm_tree import HGMNode, HGMTree
from meta_agent.models import EvolutionStrategy


class _FakeFeedback:
    """Minimal AgentFeedback stand-in -- _lineage_memory_label only ever
    reads .strategy.block (same pattern as test_block_bandit.py's)."""

    def __init__(self, block):
        self.strategy = EvolutionStrategy(
            optimization_goal="test", proposed_changes="test", block=block
        )


class RegistrationTests(unittest.TestCase):
    def test_registered_as_hgm_block_tagged(self) -> None:
        cls = registry.get("manager", "hgm_block_tagged")
        self.assertIs(cls, BlockTaggedHGMManager)

    def test_is_an_hgm_manager_subclass(self) -> None:
        self.assertTrue(issubclass(BlockTaggedHGMManager, HGMManager))


class LineageMemoryLabelTests(unittest.TestCase):
    def test_base_class_label_is_untagged(self) -> None:
        m = HGMManager()
        m._feedback[5] = _FakeFeedback("verifiers")
        self.assertEqual(m._lineage_memory_label(5), "round 5")

    def test_subclass_tags_with_block_when_known(self) -> None:
        m = BlockTaggedHGMManager()
        m._feedback[5] = _FakeFeedback("verifiers")
        self.assertEqual(m._lineage_memory_label(5), "round 5 (verifiers)")

    def test_subclass_falls_back_when_block_is_none(self) -> None:
        # Seed/root node: strategy present but block=None (unedited).
        m = BlockTaggedHGMManager()
        m._feedback[0] = _FakeFeedback(None)
        self.assertEqual(m._lineage_memory_label(0), "round 0")

    def test_subclass_falls_back_when_no_feedback_at_all(self) -> None:
        m = BlockTaggedHGMManager()
        self.assertEqual(m._lineage_memory_label(42), "round 42")


class RenderLineageMemoryIntegrationTests(unittest.TestCase):
    """Full walk through _render_lineage_memory with real behavior_memory.md
    files on disk, confirming the tag actually reaches the rendered output."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="lineage_memory_test_"))
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def _build_manager(self, cls):
        m = cls()
        m._tree = HGMTree()
        for node_id, parent_id, block, memory_text in [
            (0, None, None, "seed round, no memory written"),
            (1, 0, "individual_subagent", "fixed empty sightseeing output"),
            (2, 1, "verifiers", "added a transfer-time check"),
        ]:
            round_dir = self.tmp / f"round_{node_id}"
            round_dir.mkdir()
            if node_id != 0:
                (round_dir / "behavior_memory.md").write_text(
                    memory_text, encoding="utf-8"
                )
            node = HGMNode(node_id=node_id, parent_id=parent_id, round_dir=round_dir)
            m._tree.add(node)
            m._feedback[node_id] = _FakeFeedback(block)
        return m

    def test_base_manager_renders_untagged_labels(self) -> None:
        m = self._build_manager(HGMManager)
        rendered = m._render_lineage_memory(m._tree.nodes[2])
        self.assertIn("round 1", rendered)
        self.assertIn("round 2", rendered)
        self.assertNotIn("individual_subagent", rendered)
        self.assertNotIn("verifiers", rendered)

    def test_tagged_manager_renders_block_tagged_labels(self) -> None:
        m = self._build_manager(BlockTaggedHGMManager)
        rendered = m._render_lineage_memory(m._tree.nodes[2])
        self.assertIn("round 1 (individual_subagent)", rendered)
        self.assertIn("round 2 (verifiers)", rendered)


if __name__ == "__main__":
    unittest.main()
