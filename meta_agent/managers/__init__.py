"""Manager implementations. Importing this package triggers each module's
``@register("manager", ...)`` decorator so the registry is populated.

Each round folder a manager produces must contain:
    task_agent/      — the runnable task agent for this round
    logs/            — trace.jsonl plus any stderr/stdout captures
    strategy.json    — the EvolutionStrategy applied (or null for round 0)
    eval_result.json
    feedback.json
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from ..agent_editor import AgentEditor
from ..evaluator import Evaluator
from ..feedback_gatherer import FeedbackGatherer
from ..models import EvolutionOutcome


class EvolutionManager(Protocol):
    def evolve(
        self,
        editor: AgentEditor,
        evaluator: Evaluator,
        gatherer: FeedbackGatherer,
        seed_dir: Path,
        benchmark_dir: Path,
        experiment_dir: Path,
        max_rounds: int,
        score_target: float | None,
        train_case_ids: list[str] | None = None,
        eval_case_ids: list[str] | None = None,
        edit_memory: Any = None,
    ) -> EvolutionOutcome: ...


from . import hgm  # noqa: F401,E402

__all__ = ["EvolutionManager", "EvolutionOutcome", "hgm"]
