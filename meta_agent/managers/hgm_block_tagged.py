"""HGMManager variant whose lineage behavior-memory entries are tagged
with the block their creating edit targeted.

Context: ``HGMManager._render_lineage_memory`` (hgm.py) injects each
ancestor's ``behavior_memory.md`` into the editor's steering context,
labeled only as "round N" -- with no indication of which block (see
block_suggester.py) that round's edit targeted. Since the editor is also
told "you're scoped to block X" (via the selected-block line + any
block-scoped suggestion) for the CURRENT EXPAND, an untagged lineage memory
gives it no way to tell "this ancestor's history is about my block" from
"unrelated block" without re-reading the memory text itself. Sibling
differentiation (the immediate-parent level) already carries block
information (see ``siblings=`` in ``_render_expand_context``); this
extends the same idea further up the lineage.

Opt-in via ``manager.type: "hgm_block_tagged"`` -- registered separately
from "hgm" so this changes nothing for any existing config.
"""
from __future__ import annotations

from ..registry import register
from .hgm import HGMManager


@register("manager", "hgm_block_tagged")
class BlockTaggedHGMManager(HGMManager):
    """Identical to ``HGMManager``, except each lineage memory entry is
    labeled ``"round N (block_name)"`` instead of just ``"round N"`` when
    that round's creating edit targeted a known block (e.g. the seed/root
    node, whose edit has no block, keeps the plain "round N" label)."""

    def _lineage_memory_label(self, node_id: int) -> str:
        fb = self._feedback.get(node_id)
        block = fb.strategy.block if fb is not None and fb.strategy is not None else None
        if block:
            return f"round {node_id} ({block})"
        return super()._lineage_memory_label(node_id)
