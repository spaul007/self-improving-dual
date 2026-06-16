"""Default EvolutionManager — single-trajectory hill climbing.

Round 0 evaluates the seed verbatim. Each subsequent round branches from the
best round so far (or the latest, configurable), hands the base round to the
editor for one self-improvement, evaluates the result, gathers feedback,
persists everything, and loops until the score target or the round budget is
hit.

The manager no longer makes its own "strategy proposal" LLM call. It builds
a cheap steering *context* string (recent-round history + best round) and
passes it to ``editor.apply``; the editor's single self-improvement call
diagnoses and edits in one step and emits the ``EvolutionStrategy`` summary
on ``EditResult.strategy``.

When ``train_case_ids`` / ``eval_case_ids`` are supplied, optimization runs
only against the train half and a held-out eval composite score is computed
and persisted (``eval_score.json``) per round. The eval score does not feed
the context or "best round" selection — it's a sidecar metric.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Literal, Optional

from ..agent_editor import AgentEditor, fallback_strategy
from ..evaluator import Evaluator
from ..feedback_gatherer import (
    FeedbackGatherer,
    persist_round_artifacts,
    render_metrics,
)
from ..models import (
    AgentFeedback,
    EvaluationResult,
    EvolutionOutcome,
    EvolutionStrategy,
)
from ..registry import register
from ..tree_snapshot import NodeSnapshot, TreeSnapshotWriter


@register("manager", "hill_climbing")
class HillClimbingManager:
    def __init__(
        self,
        *,
        branch_policy: Literal["best", "latest"] = "best",
        strategy_history_window: int = 5,
        snapshot_tree: bool = False,
    ) -> None:
        self.branch_policy = branch_policy
        self.strategy_history_window = strategy_history_window
        # Opt-in time-series snapshots of the round history, written after
        # every round, so the best agent at any budget level can be recovered
        # and re-evaluated later. See meta_agent/tree_snapshot.py and
        # snapshot_eval.py. Off by default — zero behavior change.
        self.snapshot_tree = snapshot_tree
        self._history: list[AgentFeedback] = []
        self._train_case_ids: Optional[list[str]] = None
        self._eval_case_ids: Optional[list[str]] = None
        self._snapshotter: Optional[TreeSnapshotWriter] = None

    # ------------------------------------------------------------------ #
    # Public API (EvolutionManager protocol)
    # ------------------------------------------------------------------ #

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
        train_case_ids: Optional[list[str]] = None,
        eval_case_ids: Optional[list[str]] = None,
    ) -> EvolutionOutcome:
        self._history = []
        self._train_case_ids = train_case_ids
        self._eval_case_ids = eval_case_ids
        self._snapshotter = TreeSnapshotWriter(
            experiment_dir, enabled=self.snapshot_tree
        )

        # Round 0 — copy the seed verbatim and evaluate it.
        self._run_round_zero(seed_dir, evaluator, gatherer, benchmark_dir, experiment_dir)
        self._snapshot("seed")
        if self._target_reached(score_target):
            return self._outcome()

        for round_num in range(1, max_rounds + 1):
            base_round = self._base_round_for_next()
            base_dir = experiment_dir / f"round_{base_round:03d}"
            out_dir = experiment_dir / f"round_{round_num:03d}"
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "logs").mkdir(exist_ok=True)

            recent = self._history[-self.strategy_history_window :]
            best = max(self._history, key=lambda f: f.eval_result.score)
            context = self._render_change_context(recent, best)
            edit_result = editor.apply(
                self._history[-1], base_dir, out_dir, context=context
            )
            strategy = edit_result.strategy or fallback_strategy()

            if not edit_result.success:
                feedback = self._synth_failed_edit_feedback(
                    round_num, base_round, strategy, edit_result.errors, out_dir
                )
            else:
                eval_result = evaluator.run(
                    out_dir, benchmark_dir, case_ids=self._train_case_ids
                )
                feedback = gatherer.compile(
                    round_num, base_round, strategy, eval_result, out_dir
                )

            self._history.append(feedback)
            self._run_eval_split(round_num, out_dir, evaluator, benchmark_dir, feedback)
            self._snapshot("evaluate")
            if self._target_reached(score_target):
                break

        return self._outcome()

    # ------------------------------------------------------------------ #
    # Steering context — recent-round history handed to the editor
    # ------------------------------------------------------------------ #

    def _render_change_context(
        self,
        recent: list[AgentFeedback],
        best: Optional[AgentFeedback],
    ) -> str:
        """Build the manager's steering context for the editor: the best
        round so far and a window of recent rounds (scores, tool errors,
        project metrics, exceptions, prior goals). Pure string assembly —
        the editor reads the agent's actual code itself, so no source is
        embedded here."""
        parts: list[str] = []
        if best is not None:
            parts.append(
                f"## Best round so far\n"
                f"round={best.round_number}  score={best.eval_result.score:.3f}\n"
            )
        if recent:
            parts.append("## Recent rounds")
            for fb in recent:
                ev = fb.eval_result
                parts.append(
                    f"- round={fb.round_number} score={ev.score:.3f} "
                    f"passed={ev.passed} failed={ev.failed} "
                    f"crashed={ev.crashed} llm_calls={fb.llm_calls} "
                    f"tool_usage={fb.tool_usage}"
                )
                if fb.tool_error_rate:
                    ranked = [
                        (n, r) for n, r in fb.tool_error_rate.items() if r > 0
                    ]
                    ranked.sort(key=lambda kv: -kv[1])
                    if ranked:
                        rendered = ", ".join(f"{n}={r:.2f}" for n, r in ranked[:3])
                        parts.append(f"    tool error rates: {rendered}")
                if fb.project_metrics:
                    parts.extend(render_metrics(fb.project_metrics, cap=3, indent="    "))
                if fb.runtime_exceptions:
                    for exc in fb.runtime_exceptions[:3]:
                        parts.append(f"    error: {exc[:200]}")
                if fb.edit_errors:
                    parts.append(f"    edit_errors (this round did not run):")
                    for err in fb.edit_errors[:5]:
                        parts.append(f"      - {err[:300]}")
                if fb.strategy.optimization_goal:
                    parts.append(f"    goal: {fb.strategy.optimization_goal[:200]}")
        else:
            parts.append("## No history yet — this is the first edit.")
        parts.append(
            "\nUse this history, the feedback, and the current code to make "
            "one focused improvement. Keep the scope small enough that it "
            "can be applied correctly in one pass."
        )
        return "\n".join(parts)

    # ------------------------------------------------------------------ #
    # Round-loop internals
    # ------------------------------------------------------------------ #

    def _run_round_zero(
        self,
        seed_dir: Path,
        evaluator: Evaluator,
        gatherer: FeedbackGatherer,
        benchmark_dir: Path,
        experiment_dir: Path,
    ) -> None:
        out_dir = experiment_dir / "round_000"
        agent_dst = out_dir / "task_agent"
        if agent_dst.exists():
            shutil.rmtree(agent_dst)
        agent_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(seed_dir, agent_dst)
        (out_dir / "logs").mkdir(exist_ok=True)

        zero_strategy = EvolutionStrategy(
            target_files=[],
            optimization_goal="Establish baseline (round 0).",
            proposed_changes="(none — seed used as-is)",
            rationale="Round 0 is the seed.",
        )
        eval_result = evaluator.run(
            out_dir, benchmark_dir, case_ids=self._train_case_ids
        )
        feedback = gatherer.compile(0, 0, zero_strategy, eval_result, out_dir)
        self._history.append(feedback)
        self._run_eval_split(0, out_dir, evaluator, benchmark_dir, feedback)

    def _base_round_for_next(self) -> int:
        if not self._history:
            return 0
        if self.branch_policy == "latest":
            return self._history[-1].round_number
        best = max(self._history, key=lambda f: f.eval_result.score)
        return best.round_number

    def _best_round(self) -> AgentFeedback:
        return max(self._history, key=lambda f: f.eval_result.score)

    def _snapshot(self, event: str) -> None:
        """Append a full-history snapshot keyed by cumulative cases evaluated
        (so the budget axis is comparable to HGM's). A no-op unless
        ``snapshot_tree`` is enabled. Records every round as a node and a
        pointer to the current best-by-score round."""
        if self._snapshotter is None or not self._snapshotter.enabled:
            return
        nodes: list[NodeSnapshot] = []
        budget_spent = 0
        for fb in self._history:
            ev = fb.eval_result
            n_evals = ev.passed + ev.failed
            budget_spent += n_evals
            nodes.append(
                NodeSnapshot(
                    node_id=fb.round_number,
                    parent_id=fb.base_round,
                    round_dir=f"round_{fb.round_number:03d}",
                    edit_failed=bool(fb.edit_errors),
                    n_evals=n_evals,
                    mean_utility=ev.score,
                    n_success=float(ev.passed),
                    n_failure=float(ev.failed),
                    cmp=None,
                )
            )
        best = self._best_round() if self._history else None
        best_id = best.round_number if best is not None else None
        self._snapshotter.record(
            event=event,
            manager=type(self).__name__,
            budget_spent=budget_spent,
            node_evals_spent=budget_spent,
            nodes=nodes,
            best_node_id=best_id,
            best_mean_utility=best.eval_result.score if best is not None else None,
            best_round_dir=(
                f"round_{best_id:03d}" if best_id is not None else None
            ),
        )

    def _target_reached(self, score_target: float | None) -> bool:
        if score_target is None or not self._history:
            return False
        return self._history[-1].eval_result.score >= score_target

    def _outcome(self) -> EvolutionOutcome:
        if not self._history:
            return EvolutionOutcome(rounds=[], best_round=0, final_score=0.0)
        best = self._best_round()
        return EvolutionOutcome(
            rounds=list(self._history),
            best_round=best.round_number,
            final_score=best.eval_result.score,
        )

    def _synth_failed_edit_feedback(
        self,
        round_num: int,
        base_round: int,
        strategy: EvolutionStrategy,
        errors: list[str],
        out_dir: Path,
    ) -> AgentFeedback:
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
        # Persist artifacts so the round folder is complete and reviewable.
        persist_round_artifacts(out_dir, feedback)
        return feedback

    def _run_eval_split(
        self,
        round_num: int,
        out_dir: Path,
        evaluator: Evaluator,
        benchmark_dir: Path,
        feedback: AgentFeedback,
    ) -> None:
        """Compute the held-out eval composite and persist alongside.

        No-op when no eval split is configured. The eval score is *not*
        added to ``self._history`` — best-round selection and the steering
        context continue to read only the train score.
        """
        train_score = feedback.eval_result.score
        if self._eval_case_ids is None:
            print(f"round {round_num}: train={train_score:.3f}", flush=True)
            return
        # Skip eval when the editor failed validation. ``feedback.edit_errors``
        # is the signal — set by ``_synth_failed_edit_feedback`` whenever
        # EditResult came back unsuccessful.
        if feedback.edit_errors:
            (out_dir / "eval_score.json").write_text(
                json.dumps(
                    {
                        "skipped": True,
                        "reason": "edit failed validation",
                        "edit_errors": feedback.edit_errors,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            first = feedback.edit_errors[0][:80] if feedback.edit_errors else ""
            print(
                f"round {round_num}: train={train_score:.3f} eval=skipped "
                f"(edit failed: {first})",
                flush=True,
            )
            return
        eval_result = evaluator.run(
            out_dir, benchmark_dir, case_ids=self._eval_case_ids
        )
        (out_dir / "eval_score.json").write_text(
            json.dumps(
                {
                    "composite_score": eval_result.score,
                    "passed": eval_result.passed,
                    "failed": eval_result.failed,
                    "wall_time_s": eval_result.wall_time_s,
                    "crashed": eval_result.crashed,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(
            f"round {round_num}: train={train_score:.3f} eval={eval_result.score:.3f}",
            flush=True,
        )
