"""Thompson sampling over the focus axis (``default`` / ``reliability``; meta_agent/focus.py).

A small per-axis clone of implementation_strategy_bandit.py (this codebase prefers independent
per-axis bandits over a shared base). Stateless per call: tallies are recomputed from the tree
and the feedback map every time, so resume only has to restore the RNG.

``reward_metric``: ``"boolean_increase"`` (default here) gives a node one Bernoulli trial --
did it match or beat its parent's mean utility -- which is what a focus should be credited
for; ``"fractional_score"`` folds the node's own success/failure mass, as the other axes do.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, Sequence

from .focus import FOCUS_VALUES

if TYPE_CHECKING:
    from .managers.hgm_tree import HGMTree
    from .models import AgentFeedback

_ALLOWED_REWARD_METRICS = ("fractional_score", "boolean_increase")


@dataclass
class FocusPosterior:
    n_success: float
    n_failure: float
    n_evals: int
    mean: float
    sampled_value: float


@dataclass
class AdaptiveFocus:
    focus: str
    beta_prior: float
    posteriors: dict[str, FocusPosterior] = field(default_factory=dict)


class FocusBandit:
    def __init__(self, *, values: Optional[Sequence[str]] = None, beta_prior: float = 1.0,
                 rng: random.Random, reward_metric: str = "boolean_increase") -> None:
        self.values: tuple[str, ...] = tuple(values or FOCUS_VALUES)
        if reward_metric not in _ALLOWED_REWARD_METRICS:
            raise ValueError(f"reward_metric must be one of {_ALLOWED_REWARD_METRICS}, got {reward_metric!r}")
        self.beta_prior, self.reward_metric, self._rng = beta_prior, reward_metric, rng

    def select(self, tree: "HGMTree", feedback: dict[int, "AgentFeedback"],
               allowed: Optional[Sequence[str]] = None) -> AdaptiveFocus:
        """Sample a focus among ``allowed`` (default: every value). Nodes are credited to the
        focus stamped on their strategy; nodes without one (seed, focus off) are ignored."""
        tallies = {v: [0.0, 0.0, 0] for v in self.values}
        for node_id, node in tree.nodes.items():
            if node.edit_failed or node.n_evals == 0:
                continue
            fb = feedback.get(node_id)
            value = getattr(getattr(fb, "strategy", None), "focus", None)
            if value not in tallies:
                continue
            if self.reward_metric == "boolean_increase":
                parent = tree.nodes.get(node.parent_id) if node.parent_id is not None else None
                if parent is None or parent.n_evals == 0:
                    continue
                inc = 1.0 if node.mean_utility >= parent.mean_utility else 0.0
                s, f = inc, 1.0 - inc
            else:
                s, f = node.n_success, node.n_failure
            tallies[value][0] += s
            tallies[value][1] += f
            tallies[value][2] += 1
        candidates = [v for v in self.values if allowed is None or v in allowed] or list(self.values)
        posteriors: dict[str, FocusPosterior] = {}
        for v in self.values:
            s, f, n = tallies[v]
            draw = self._rng.betavariate(self.beta_prior + s, self.beta_prior + f)
            posteriors[v] = FocusPosterior(n_success=s, n_failure=f, n_evals=n,
                                           mean=(self.beta_prior + s) / (2 * self.beta_prior + s + f),
                                           sampled_value=draw)
        chosen = max(candidates, key=lambda v: posteriors[v].sampled_value)
        return AdaptiveFocus(focus=chosen, beta_prior=self.beta_prior, posteriors=posteriors)
