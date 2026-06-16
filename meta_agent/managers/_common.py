"""Shared helpers used by both ``HillClimbingManager`` and ``HGMManager``.

Minimum useful extraction — only the bits that are line-for-line duplicates
in both managers live here. Each manager keeps its own ``_run_round_zero`` /
``_run_seed`` wiring because the surrounding instance state (``_history`` vs
``_tree``) is too tightly coupled to share.
"""
from __future__ import annotations

import shutil
from pathlib import Path

from ..feedback_gatherer import persist_round_artifacts
from ..models import (
    AgentFeedback,
    EvaluationResult,
    EvolutionStrategy,
)


def copy_seed_to(seed_dir: Path, out_dir: Path) -> Path:
    """Copy ``seed_dir`` → ``out_dir/task_agent`` (clobbering any prior
    contents) and create the round's ``logs/`` directory. Returns
    ``out_dir`` for convenience."""
    agent_dst = out_dir / "task_agent"
    if agent_dst.exists():
        shutil.rmtree(agent_dst)
    agent_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(seed_dir, agent_dst)
    (out_dir / "logs").mkdir(exist_ok=True)
    return out_dir


def make_seed_strategy(
    *, goal: str = "Establish baseline (seed agent).", rationale: str = "Round 0."
) -> EvolutionStrategy:
    """Zero-edit ``EvolutionStrategy`` used for round 0 / the HGM root."""
    return EvolutionStrategy(
        target_files=[],
        optimization_goal=goal,
        proposed_changes="(none — seed used as-is)",
        rationale=rationale,
    )


def synth_failed_edit_feedback(
    *,
    round_num: int,
    base_round: int,
    strategy: EvolutionStrategy,
    errors: list[str],
    out_dir: Path,
) -> AgentFeedback:
    """Build the canonical ``AgentFeedback`` for a round whose editor call
    failed validation, persist it, and return it. The round folder is left
    in a reviewable state (feedback.json / strategy.json / empty
    eval_result.json) so a downstream consumer doesn't trip over a half-
    built directory."""
    zero_eval = EvaluationResult(
        score=0.0,
        metrics={},
        passed=0,
        failed=0,
        per_case=[],
        wall_time_s=0.0,
        crashed=False,
    )
    feedback = AgentFeedback(
        round_number=round_num,
        base_round=base_round,
        strategy=strategy,
        eval_result=zero_eval,
        tool_usage={},
        llm_calls=0,
        runtime_exceptions=[],
        log_excerpt="",
        edit_errors=errors,
    )
    persist_round_artifacts(out_dir, feedback)
    return feedback
