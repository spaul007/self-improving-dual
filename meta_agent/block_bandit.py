"""Per-block Thompson sampling for Block-Tiered HGM ("adaptive" strategy).

Reuses hgm_tree.py's own Beta-Bernoulli convention (see
``HGMTree._beta_sample``, ``HGMNode.record``) applied to a new axis: instead
of sampling which NODE to expand, sample which BLOCK (see block_suggester.py)
the next EXPAND should target, based on the accumulated (fractional) success
/ failure mass of every node whose creating edit targeted that block so far.

Deliberately self-contained ("blackbox"): ``BlockBandit.select`` takes the
tree + feedback map and returns a fresh ``AdaptiveStrategy`` snapshot every
call, with no state persisted on the bandit itself between calls. A future
different adaptive algorithm only needs to provide the same
``select(tree, feedback) -> AdaptiveStrategy``-shaped object; nothing in
hgm.py/hgm_dual.py depends on Thompson sampling specifically.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, Sequence

if TYPE_CHECKING:
    from .managers.hgm_tree import HGMTree
    from .models import AgentFeedback


@dataclass
class BlockPosterior:
    """One block's reward posterior at the moment of a single selection."""

    n_success: float
    n_failure: float
    n_evals: int  # count of qualifying (evaluated, edit-succeeded) nodes folded in
    mean: float  # posterior mean = (prior+success) / (2*prior+success+failure)
    sampled_value: float  # this round's Thompson draw for this block


@dataclass
class AdaptiveStrategy:
    """Returned by ``BlockBandit.select`` -- the block chosen this round,
    plus every candidate block's full posterior snapshot (always present,
    even for a block with zero evals so far), for persistence/inspection."""

    block: str
    beta_prior: float
    posteriors: dict[str, BlockPosterior] = field(default_factory=dict)


class BlockBandit:
    """Thompson-samples a block to target, from the accumulated per-block
    success/failure mass across every node evaluated so far. Stateless
    per call -- recomputes tallies fresh from ``tree``/``feedback`` each
    time rather than keeping a running total, so it can never drift out of
    sync with the tree it's reading."""

    def __init__(
        self,
        *,
        blocks: Optional[Sequence[str]] = None,
        beta_prior: float = 1.0,
        tau: float = 1.0,
        rng: random.Random,
    ) -> None:
        if blocks is None:
            # Canonical block-name source, same convention as the
            # "non_adaptive" strategy in hgm.py::_select_block -- never a
            # second hardcoded list that could drift out of sync.
            from .block_suggester import _BLOCK_BODIES

            blocks = sorted(_BLOCK_BODIES)
        self.blocks: tuple[str, ...] = tuple(blocks)
        self.beta_prior = beta_prior
        self.tau = tau
        self._rng = rng

    def _beta_sample(self, success: float, failure: float) -> float:
        # Same formula as HGMTree._beta_sample (hgm_tree.py) -- duplicated
        # rather than imported since that method is bound to one HGMTree
        # instance, not a free function. Keep the two in sync by hand if
        # either changes.
        a = self.tau * (self.beta_prior + success)
        b = self.tau * (self.beta_prior + failure)
        return self._rng.betavariate(a, b)

    def select(
        self, tree: "HGMTree", feedback: dict[int, "AgentFeedback"]
    ) -> AdaptiveStrategy:
        tallies: dict[str, tuple[float, float, int]] = {
            b: (0.0, 0.0, 0) for b in self.blocks
        }
        for node_id, node in tree.nodes.items():
            if node.edit_failed or node.n_evals == 0:
                continue
            fb = feedback.get(node_id)
            if fb is None or fb.strategy is None:
                continue
            block = fb.strategy.block
            if block not in tallies:
                # Seed node (block is None) or any block not in self.blocks.
                continue
            success, failure, n = tallies[block]
            tallies[block] = (
                success + node.n_success,
                failure + node.n_failure,
                n + 1,
            )

        posteriors: dict[str, BlockPosterior] = {}
        for block in self.blocks:
            success, failure, n = tallies[block]
            sampled_value = self._beta_sample(success, failure)
            mean = (self.beta_prior + success) / (
                2 * self.beta_prior + success + failure
            )
            posteriors[block] = BlockPosterior(
                n_success=success,
                n_failure=failure,
                n_evals=n,
                mean=mean,
                sampled_value=sampled_value,
            )

        # argmax over sampled values; ties broken by self.blocks order (the
        # order max() encounters them in) for determinism given a fixed seed.
        chosen = max(self.blocks, key=lambda b: posteriors[b].sampled_value)
        return AdaptiveStrategy(
            block=chosen, beta_prior=self.beta_prior, posteriors=posteriors
        )
