"""Generic, project-agnostic time-series snapshots of an evolutionary search.

During a run, every evolution manager (HGM, HGM-dual, hill climbing, …) mutates
its population continuously: nodes are created and re-scored, so "which node is
currently best" is a moving target only known at the end. This module records a
*time series* of the whole node roster, keyed by evaluation budget, so an analyst
can later pick the best agent at any budget level, re-evaluate it (see
``snapshot_eval.py``), and compare methods head-to-head at equal budget.

This mirrors the original HGM's ``update_metadata`` / ``hgm_metadata.jsonl``
mechanism (github.com/metauto-ai/HGM, ``hgm.py::update_metadata`` +
``tree.py::Node.save_as_dict``): one record appended per EXPAND/EVALUATE holding
the full node list plus the budget spent. Our record is a *superset* of the
reference fields — ``budget_spent`` maps to ``n_task_evals`` and each node carries
``node_id``/``parent_id``/``mean_utility``/``n_evals`` (== reference
``id``/``parent_id``/``mean_utility``/``num_evals``) plus extras — so a
reference-style loader still works.

Deliberately pure I/O: no LLM calls, no subprocesses. Managers own the search;
this module only serializes a snapshot it is handed.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass
class NodeSnapshot:
    """One node's state at a single point in time. Project-agnostic — holds
    only generic search-tree quantities, never task/domain specifics."""

    node_id: int
    parent_id: Optional[int]
    # Round directory name (relative to the experiment dir), e.g. "round_001".
    # The runnable agent lives at "<round_dir>/task_agent/" for re-evaluation.
    round_dir: str
    edit_failed: bool
    n_evals: int
    mean_utility: float
    n_success: float
    n_failure: float
    # Clade metaproductivity, when the manager exposes a clade aggregate
    # (HGM family). ``None`` for managers without one (e.g. hill climbing).
    cmp: Optional[float] = None


class TreeSnapshotWriter:
    """Appends one JSON line per step to
    ``<experiment_dir>/snapshots/<filename>``. A no-op when ``enabled`` is
    False, so callers can wire it in unconditionally and gate via config."""

    def __init__(
        self,
        experiment_dir: Path,
        *,
        enabled: bool = True,
        filename: str = "tree_snapshots.jsonl",
    ) -> None:
        self.enabled = enabled
        self._idx = 0
        self._path = Path(experiment_dir) / "snapshots" / filename
        if self.enabled:
            self._path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def path(self) -> Path:
        return self._path

    def record(
        self,
        *,
        event: str,
        manager: str,
        budget_spent: int,
        node_evals_spent: int,
        nodes: list[NodeSnapshot],
        best_node_id: Optional[int],
        best_mean_utility: Optional[float],
        best_round_dir: Optional[str],
        extra: Optional[dict[str, Any]] = None,
    ) -> None:
        """Append a single snapshot line. Captures the full node roster (not a
        delta) so each line is self-contained — durable even on crash."""
        if not self.enabled:
            return
        record: dict[str, Any] = {
            "snapshot_idx": self._idx,
            "event": event,
            "manager": manager,
            # Budget axis — analog of the reference's ``n_task_evals``.
            "budget_spent": budget_spent,
            "node_evals_spent": node_evals_spent,
            "best_node_id": best_node_id,
            "best_mean_utility": best_mean_utility,
            "best_round_dir": best_round_dir,
            "n_nodes": len(nodes),
            "nodes": [asdict(n) for n in nodes],
        }
        if extra:
            record.update(extra)
        with self._path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        self._idx += 1
