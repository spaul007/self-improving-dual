"""Tree, Beta-sampling bandit, scheduler, and final selection for the
Huxley-Gödel Machine manager (``hgm.py``).

This module is deliberately **pure** — no LLM calls, no subprocesses, no
filesystem writes — so the entire search logic is unit-testable without an
API key. The manager (``hgm.py``) owns all I/O; this module owns the math.

Two scores live here, and only one spans multiple nodes:

* **Own score** — a node's own (agent, task) evaluations. Drives the
  EVALUATE bandit and the final selection.
* **Clade Metaproductivity (CMP)** — aggregated over a node *and its whole
  subtree*. Drives the EXPAND bandit: a mediocre node that spawned a strong
  descendant gets a high CMP. This is the cross-node calculation.

CMP is not a raw sum of descendant tallies. Faithful to the reference
``tree.py::get_pseudo_decendant_evals``, each descendant's contribution is
pseudo-count normalized (``clade_pseudo_count`` mean-votes once it is
heavily evaluated) so one deep branch cannot swamp the clade signal.

Continuous scores: the meta-agent's scorers return ``score in [0, 1]``, not
binary pass/fail. Each case folds in as fractional mass — ``n_success += s``,
``n_failure += 1 - s`` — and Beta sampling / clade ratios work unchanged.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from ..models import CaseResult


@dataclass
class HGMNode:
    """One agent in the tree. Stores **only its own** evaluation tallies;
    clade (cross-node) quantities are derived on demand by ``HGMTree``."""

    node_id: int
    parent_id: Optional[int]
    round_dir: Path
    # Own tallies — fractional mass folded from continuous case scores.
    n_success: float = 0.0
    n_failure: float = 0.0
    utility_measures: list[float] = field(default_factory=list)
    # Cumulative per-case results, kept so the manager can rebuild a full
    # EvaluationResult for feedback after each lazy-eval batch.
    case_results: list[CaseResult] = field(default_factory=list)
    evaluated_case_ids: set[str] = field(default_factory=set)
    children: list[int] = field(default_factory=list)
    # True when this node was born from a failed editor.apply — it occupies
    # a node id for a complete run dir but is neither expandable nor
    # evaluable (its task_agent is just the parent's code reset).
    edit_failed: bool = False

    @property
    def n_evals(self) -> int:
        return len(self.utility_measures)

    @property
    def mean_utility(self) -> float:
        if not self.utility_measures:
            return 0.0
        return sum(self.utility_measures) / len(self.utility_measures)

    def record(self, case: CaseResult) -> None:
        """Fold one (agent, task) evaluation into this node's own tallies."""
        s = min(1.0, max(0.0, float(case.score)))
        self.n_success += s
        self.n_failure += 1.0 - s
        self.utility_measures.append(s)
        self.case_results.append(case)
        self.evaluated_case_ids.add(case.case_id)


class HGMTree:
    """The agent tree plus the HGM bandit primitives."""

    def __init__(
        self,
        *,
        beta_prior: float = 1.0,
        clade_pseudo_count: int = 10000,
        rng: Optional[random.Random] = None,
    ) -> None:
        # Additive Beta prior (the paper's ``1 +``); applied to both arms.
        self.beta_prior = beta_prior
        # num_pseudo in tree.py::get_decendant_evals. The reference's
        # config.yaml ships 10000 — so large the cap never triggers and the
        # clade aggregate is the raw union of the subtree's scores. Kept as
        # a knob; lower it only to deliberately down-weight deep branches.
        self.clade_pseudo_count = clade_pseudo_count
        self._rng = rng or random.Random(0)
        self.nodes: dict[int, HGMNode] = {}

    def add(self, node: HGMNode) -> None:
        self.nodes[node.node_id] = node
        if node.parent_id is not None and node.parent_id in self.nodes:
            parent = self.nodes[node.parent_id]
            if node.node_id not in parent.children:
                parent.children.append(node.node_id)

    def __len__(self) -> int:
        return len(self.nodes)

    def __getitem__(self, node_id: int) -> HGMNode:
        return self.nodes[node_id]

    def __contains__(self, node_id: int) -> bool:
        return node_id in self.nodes

    @property
    def total_evals(self) -> int:
        """Total (agent, task) pairs evaluated across the whole tree."""
        return sum(n.n_evals for n in self.nodes.values())

    # ------------------------------------------------------------------ #
    # Clade aggregation — the cross-node (Clade Metaproductivity) math
    # ------------------------------------------------------------------ #

    def _pseudo_evals(self, node: HGMNode) -> list[float]:
        """One node's contribution to a clade aggregate, pseudo-count
        normalized. Mirrors ``tree.py::get_pseudo_decendant_evals``: raw
        per-case scores while the node is sparsely evaluated, else
        ``clade_pseudo_count`` copies of its mean so a heavily-evaluated
        branch cannot dominate the clade signal."""
        if node.n_evals < self.clade_pseudo_count:
            return list(node.utility_measures)
        return [node.mean_utility] * self.clade_pseudo_count

    def _descendants(self, node_id: int) -> list[HGMNode]:
        """Flat list of every node in ``subtree(node_id)`` EXCLUDING the
        node itself."""
        out: list[HGMNode] = []
        for child_id in self.nodes[node_id].children:
            out.append(self.nodes[child_id])
            out.extend(self._descendants(child_id))
        return out

    def clade_evals(self, node_id: int) -> list[float]:
        """Score list for the clade rooted at ``node_id``. Faithful to
        ``tree.py::get_decendant_evals``: the clade ROOT is pseudo-count
        normalized, but every descendant contributes its RAW
        ``utility_measures`` (the pseudo-cap applies to the queried node
        only, not recursively). With the default ``clade_pseudo_count``
        the cap never fires, so the clade is just the raw union of the
        subtree's scores."""
        evals = list(self._pseudo_evals(self.nodes[node_id]))
        for descendant in self._descendants(node_id):
            evals.extend(descendant.utility_measures)
        return evals

    def clade_success(self, node_id: int) -> float:
        return float(sum(self.clade_evals(node_id)))

    def clade_failure(self, node_id: int) -> float:
        return float(sum(1.0 - s for s in self.clade_evals(node_id)))

    def cmp(self, node_id: int) -> float:
        """Clade Metaproductivity estimate — the clade's success rate."""
        evals = self.clade_evals(node_id)
        if not evals:
            return 0.0
        return sum(evals) / len(evals)

    # ------------------------------------------------------------------ #
    # Thompson-sampling bandit
    # ------------------------------------------------------------------ #

    def tau(
        self,
        remaining: int,
        budget: int,
        *,
        cool_down: bool = False,
        beta: float = 1.0,
    ) -> float:
        """Beta-concentration multiplier applied to both Beta arms.

        Mirrors ``hgm.py::TS_sample``'s cooling factor. With ``cool_down``
        off — the reference's committed ``config.yaml`` default — this is a
        no-op (1.0): plain stationary Thompson sampling. With ``cool_down``
        on it is the exploration→exploitation schedule ``(B / b) ** beta``,
        rising as the budget drains; the reference uses 10000 once the
        budget is exhausted."""
        if not cool_down:
            return 1.0
        if remaining <= 0:
            return 1e4
        return (budget / remaining) ** beta

    def _beta_sample(self, success: float, failure: float, tau: float) -> float:
        a = tau * (self.beta_prior + success)
        b = tau * (self.beta_prior + failure)
        return self._rng.betavariate(a, b)

    def sample_self_score(self, node_id: int, tau: float) -> float:
        """Thompson sample over a node's **own** tallies (EVALUATE bandit)."""
        n = self.nodes[node_id]
        return self._beta_sample(n.n_success, n.n_failure, tau)

    def sample_clade_score(self, node_id: int, tau: float) -> float:
        """Thompson sample over a node's **clade** tallies (EXPAND bandit)."""
        return self._beta_sample(
            self.clade_success(node_id), self.clade_failure(node_id), tau
        )

    def argmax_expand(self, tau: float, expandable: list[int]) -> int:
        return max(expandable, key=lambda nid: self.sample_clade_score(nid, tau))

    def argmax_evaluate(self, tau: float, evaluable: list[int]) -> int:
        return max(evaluable, key=lambda nid: self.sample_self_score(nid, tau))

    # ------------------------------------------------------------------ #
    # Scheduler + final selection
    # ------------------------------------------------------------------ #

    def n_real_nodes(self) -> int:
        """Nodes that exist as real agents — excludes edit-failed
        placeholders. The reference never creates a node for a failed
        expansion, so these must not count toward the schedule."""
        return sum(1 for n in self.nodes.values() if not n.edit_failed)

    def schedule_favors_expand(self, alpha: float, budget_spent: int) -> bool:
        """UCB-Air adaptive widening, faithful to ``hgm.py::sample``:
        expand while ``budget_spent ** alpha >= (real node count - 1)``.

        The ``- 1`` excludes the root; ``budget_spent`` counts only
        loop evaluations (the root's free pre-evaluation is excluded). With
        ``budget_spent == 0`` this is False once the tree has any non-root
        node, so a fresh tree evaluates before it widens further."""
        return budget_spent ** alpha >= (self.n_real_nodes() - 1)

    def lcb_select(
        self,
        epsilon: float,
        *,
        restrict_to: Optional[set[int]] = None,
        samples: int = 8192,
    ) -> int:
        """Final pick: the node with the highest ε-quantile (lower-
        confidence bound) of its own ``Beta(1 + n_success, 1 + n_failure)``.
        Penalizes nodes whose good mean rests on thin evidence. ``scipy`` is
        not installed, so the quantile is a fixed-seed numpy Monte-Carlo
        estimate (reproducible).

        Edit-failed nodes are always excluded. When ``restrict_to`` is given,
        only those node ids are eligible — the caller passes the set of
        fully-train-evaluated nodes so the pick compares candidates on equal
        evidence (a thinly-evaluated node's optimistic estimate cannot win).
        Falls back to all real nodes if the restriction empties the set."""
        candidates = [
            (nid, n)
            for nid, n in sorted(self.nodes.items())
            if not n.edit_failed and (restrict_to is None or nid in restrict_to)
        ]
        if not candidates:
            candidates = [
                (nid, n)
                for nid, n in sorted(self.nodes.items())
                if not n.edit_failed
            ]
        if not candidates:
            return min(self.nodes) if self.nodes else 0
        gen = np.random.default_rng(0)
        best_id = candidates[0][0]
        best_lcb = -1.0
        for nid, n in candidates:
            draws = gen.beta(1.0 + n.n_success, 1.0 + n.n_failure, size=samples)
            lcb = float(np.quantile(draws, epsilon))
            if lcb > best_lcb:
                best_lcb, best_id = lcb, nid
        return best_id
