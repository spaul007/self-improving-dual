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


_ALLOWED_REWARD_METRICS = ("fractional_score", "boolean_increase")


class BlockBandit:
    """Thompson-samples a block to target, from the accumulated per-block
    success/failure mass across every node evaluated so far. Stateless
    per call -- recomputes tallies fresh from ``tree``/``feedback`` each
    time rather than keeping a running total, so it can never drift out of
    sync with the tree it's reading.

    ``reward_metric`` selects how one qualifying node contributes to its
    block's success/failure tally:

    - ``"fractional_score"`` (default -- today's exact behavior, zero
      change for every existing config): a node's own accumulated
      ``n_success``/``n_failure`` (continuous score mass, summed across
      every eval of that node -- see ``HGMNode.record``) is added straight
      into its block's tally. Nodes with more evals contribute more mass.
    - ``"boolean_increase"``: each qualifying node contributes exactly ONE
      Bernoulli trial -- success (1.0) if its ``mean_utility`` is >= its
      *direct parent's* ``mean_utility`` (a tie counts as success: "didn't
      regress"), failure (0.0) otherwise -- regardless of how many evals
      backed that node's mean. A node whose parent hasn't been evaluated
      yet (``n_evals == 0``, e.g. the root) has nothing to compare against
      and is excluded from the tally entirely, same as any other
      not-yet-qualifying node.
    """

    def __init__(
        self,
        *,
        blocks: Optional[Sequence[str]] = None,
        beta_prior: float = 1.0,
        tau: float = 1.0,
        rng: random.Random,
        reward_metric: str = "fractional_score",
        # Opt-in initial preference order among blocks, most-preferred
        # first (e.g. ["foundation_capability", "individual_subagent",
        # "verifiers", "collaboration_workflow"]) -- must be a permutation
        # of ``blocks``. None (default -- zero behavior change): every
        # block starts from the same symmetric Beta(beta_prior,
        # beta_prior), exactly today's behavior. When given, each block
        # gets a one-time prior SUCCESS pseudo-count bonus based on its
        # rank (highest for rank 0, zero for the last-ranked block,
        # spaced by ``initial_rank_strength``) -- folded into the same
        # Beta-Bernoulli posterior as real evidence, so it biases early
        # selection while real per-block evals accumulate, and gets
        # washed out by that real evidence over time (the same dynamic
        # ``beta_prior`` itself already has) rather than being a
        # permanent, hardcoded preference.
        initial_block_ranking: Optional[Sequence[str]] = None,
        # Prior-success-count gap between adjacent ranks. Only meaningful
        # when ``initial_block_ranking`` is set.
        initial_rank_strength: float = 2.0,
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
        if reward_metric not in _ALLOWED_REWARD_METRICS:
            raise ValueError(
                f"reward_metric must be one of {_ALLOWED_REWARD_METRICS}, "
                f"got {reward_metric!r}"
            )
        self.reward_metric = reward_metric
        self.initial_block_ranking = (
            tuple(initial_block_ranking) if initial_block_ranking is not None else None
        )
        self.initial_rank_strength = initial_rank_strength
        self._initial_success_bonus: dict[str, float] = dict.fromkeys(self.blocks, 0.0)
        if self.initial_block_ranking is not None:
            if set(self.initial_block_ranking) != set(self.blocks):
                raise ValueError(
                    "initial_block_ranking must be a permutation of "
                    f"blocks {sorted(self.blocks)!r}, got "
                    f"{list(self.initial_block_ranking)!r}"
                )
            n = len(self.initial_block_ranking)
            for rank, block in enumerate(self.initial_block_ranking):
                self._initial_success_bonus[block] = (
                    (n - 1 - rank) * self.initial_rank_strength
                )

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

            if self.reward_metric == "boolean_increase":
                parent = (
                    tree.nodes.get(node.parent_id)
                    if node.parent_id is not None else None
                )
                if parent is None or parent.n_evals == 0:
                    # Nothing to compare against yet (root, or a parent not
                    # evaluated) -- excluded from the tally entirely, same
                    # as any other not-yet-qualifying node.
                    continue
                increase = 1.0 if node.mean_utility >= parent.mean_utility else 0.0
                node_success, node_failure = increase, 1.0 - increase
            else:
                node_success, node_failure = node.n_success, node.n_failure

            success, failure, n = tallies[block]
            tallies[block] = (
                success + node_success,
                failure + node_failure,
                n + 1,
            )

        posteriors: dict[str, BlockPosterior] = {}
        for block in self.blocks:
            success, failure, n = tallies[block]
            # Fold in the one-time initial-ranking bonus (0.0 for every
            # block when initial_block_ranking is unset) as if it were
            # prior pseudo-successes -- same treatment as beta_prior
            # itself, so it biases early selection but is progressively
            # outweighed as real success/failure mass accumulates.
            success += self._initial_success_bonus[block]
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
