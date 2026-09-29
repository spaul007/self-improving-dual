"""Per-implementation-strategy Thompson sampling ("adaptive" strategy).

Structural clone of block_bandit.py::BlockBandit applied to the
implementation_strategy axis (llm_heavy / mixed / harness_heavy) instead of
block. Deliberately a SEPARATE class rather than a shared base with
BlockBandit: the two axes have different candidate-set sizes (3 vs. 4-6)
and different exclusion semantics (only BlockBandit needs the dynamic
`exclude` param, for the harness_heavy -> no llm_backbone_selection rule);
this codebase's own precedent (`_beta_sample`'s documented duplication
below) favors small, independently-editable per-axis bandits over a shared
abstraction that would need branching flags almost immediately.

Deliberately self-contained ("blackbox"), same convention as BlockBandit:
``ImplementationStrategyBandit.select`` takes the tree + feedback map and
returns a fresh ``AdaptiveImplementationStrategy`` snapshot every call, with
no state persisted on the bandit itself between calls.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, Sequence

if TYPE_CHECKING:
    from .managers.hgm_tree import HGMTree
    from .models import AgentFeedback


@dataclass
class ImplementationStrategyPosterior:
    """One implementation-strategy value's reward posterior at the moment
    of a single selection."""

    n_success: float
    n_failure: float
    n_evals: int  # count of qualifying (evaluated, edit-succeeded) nodes folded in
    mean: float  # posterior mean = (prior+success) / (2*prior+success+failure)
    sampled_value: float  # this round's Thompson draw for this value


@dataclass
class AdaptiveImplementationStrategy:
    """Returned by ``ImplementationStrategyBandit.select`` -- the value
    chosen this round, plus every candidate's full posterior snapshot
    (always present, even for a value with zero evals so far), for
    persistence/inspection."""

    implementation_strategy: str
    beta_prior: float
    posteriors: dict[str, ImplementationStrategyPosterior] = field(default_factory=dict)


_ALLOWED_REWARD_METRICS = ("fractional_score", "boolean_increase")


class ImplementationStrategyBandit:
    """Thompson-samples an implementation-strategy value to target, from
    the accumulated per-value success/failure mass across every node
    evaluated so far. Stateless per call -- recomputes tallies fresh from
    ``tree``/``feedback`` each time rather than keeping a running total.

    ``reward_metric`` -- identical semantics to BlockBandit's own (see
    that class's docstring): ``"fractional_score"`` (default) sums a
    qualifying node's own accumulated success/failure mass into its
    strategy's tally; ``"boolean_increase"`` contributes one Bernoulli
    trial per qualifying node based on whether it beat its direct
    parent's mean_utility.
    """

    def __init__(
        self,
        *,
        strategies: Optional[Sequence[str]] = None,
        beta_prior: float = 1.0,
        tau: float = 1.0,
        rng: random.Random,
        reward_metric: str = "fractional_score",
        # Opt-in initial preference order among strategies, most-preferred
        # first -- must be a permutation of ``strategies``. None (default,
        # zero behavior change): every value starts from the same
        # symmetric Beta(beta_prior, beta_prior). See BlockBandit's own
        # ``initial_block_ranking`` for the exact same mechanism/rationale.
        initial_ranking: Optional[Sequence[str]] = None,
        # Prior-success-count gap between adjacent ranks. Only meaningful
        # when ``initial_ranking`` is set.
        initial_rank_strength: float = 2.0,
    ) -> None:
        if strategies is None:
            # Canonical value source, same convention as BlockBandit reading
            # _BLOCK_BODIES -- never a second hardcoded list that could
            # drift out of sync.
            from .implementation_strategy import _IMPLEMENTATION_STRATEGY_BODIES

            strategies = sorted(_IMPLEMENTATION_STRATEGY_BODIES)
        self.strategies: tuple[str, ...] = tuple(strategies)
        self.beta_prior = beta_prior
        self.tau = tau
        self._rng = rng
        if reward_metric not in _ALLOWED_REWARD_METRICS:
            raise ValueError(
                f"reward_metric must be one of {_ALLOWED_REWARD_METRICS}, "
                f"got {reward_metric!r}"
            )
        self.reward_metric = reward_metric
        self.initial_ranking = (
            tuple(initial_ranking) if initial_ranking is not None else None
        )
        self.initial_rank_strength = initial_rank_strength
        self._initial_success_bonus: dict[str, float] = dict.fromkeys(self.strategies, 0.0)
        if self.initial_ranking is not None:
            if set(self.initial_ranking) != set(self.strategies):
                raise ValueError(
                    "initial_ranking must be a permutation of "
                    f"strategies {sorted(self.strategies)!r}, got "
                    f"{list(self.initial_ranking)!r}"
                )
            n = len(self.initial_ranking)
            for rank, value in enumerate(self.initial_ranking):
                self._initial_success_bonus[value] = (
                    (n - 1 - rank) * self.initial_rank_strength
                )

    def _beta_sample(self, success: float, failure: float) -> float:
        # Same formula as HGMTree._beta_sample / BlockBandit._beta_sample --
        # duplicated rather than imported, keep the three in sync by hand
        # if any changes.
        a = self.tau * (self.beta_prior + success)
        b = self.tau * (self.beta_prior + failure)
        return self._rng.betavariate(a, b)

    def select(
        self, tree: "HGMTree", feedback: dict[int, "AgentFeedback"]
    ) -> AdaptiveImplementationStrategy:
        tallies: dict[str, tuple[float, float, int]] = {
            s: (0.0, 0.0, 0) for s in self.strategies
        }
        for node_id, node in tree.nodes.items():
            if node.edit_failed or node.n_evals == 0:
                continue
            fb = feedback.get(node_id)
            if fb is None or fb.strategy is None:
                continue
            value = fb.strategy.implementation_strategy
            if value not in tallies:
                # Seed node (implementation_strategy is None) or a value
                # not in self.strategies.
                continue

            if self.reward_metric == "boolean_increase":
                parent = (
                    tree.nodes.get(node.parent_id)
                    if node.parent_id is not None else None
                )
                if parent is None or parent.n_evals == 0:
                    continue
                increase = 1.0 if node.mean_utility >= parent.mean_utility else 0.0
                node_success, node_failure = increase, 1.0 - increase
            else:
                node_success, node_failure = node.n_success, node.n_failure

            success, failure, n = tallies[value]
            tallies[value] = (
                success + node_success,
                failure + node_failure,
                n + 1,
            )

        posteriors: dict[str, ImplementationStrategyPosterior] = {}
        for value in self.strategies:
            success, failure, n = tallies[value]
            success += self._initial_success_bonus[value]
            sampled_value = self._beta_sample(success, failure)
            mean = (self.beta_prior + success) / (
                2 * self.beta_prior + success + failure
            )
            posteriors[value] = ImplementationStrategyPosterior(
                n_success=success,
                n_failure=failure,
                n_evals=n,
                mean=mean,
                sampled_value=sampled_value,
            )

        # argmax over sampled values; ties broken by self.strategies order
        # (the order max() encounters them in) for determinism given a
        # fixed seed.
        chosen = max(self.strategies, key=lambda s: posteriors[s].sampled_value)
        return AdaptiveImplementationStrategy(
            implementation_strategy=chosen, beta_prior=self.beta_prior, posteriors=posteriors
        )
