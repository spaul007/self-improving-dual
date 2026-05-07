"""Default EvolutionManager — single-trajectory hill climbing.

Round 0 evaluates the seed verbatim. Each subsequent round branches from the
best round so far (or the latest, configurable), proposes one edit directive
via an LLM call, hands it to the editor, evaluates the result, gathers
feedback, persists everything, and loops until the score target or the round
budget is hit.

The strategy used to live in its own module behind a ``Strategy`` protocol
with one implementation. Since that implementation was always coupled to
this manager, the proposal logic now lives here as ``_propose_strategy``.

When ``train_case_ids`` / ``eval_case_ids`` are supplied, optimization runs
only against the train half and a held-out eval composite score is computed
and persisted (``eval_score.json``) per round. The eval score does not feed
the proposal prompt or "best round" selection — it's a sidecar metric.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Literal, Optional

from .. import verbose_log
from ..agent_editor import AgentEditor
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


# Tool schema for the LLM's structured strategy proposal. Lifted out of
# ``_propose_strategy`` so it's reviewable in isolation and reusable
# from tests / docs.
PROPOSE_EDIT_TOOL: dict = {
    "name": "propose_edit",
    "description": "Propose the next edit for the agent editor to apply.",
    "input_schema": {
        "type": "object",
        "properties": {
            "target_files": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": ["workflow.py", "tool_wrapper.py", "tools_schema.json"],
                },
            },
            "optimization_goal": {"type": "string"},
            "proposed_changes": {"type": "string"},
            "rationale": {"type": "string"},
        },
        "required": ["target_files", "optimization_goal", "proposed_changes"],
    },
}


@register("manager", "hill_climbing")
class HillClimbingManager:
    def __init__(
        self,
        *,
        branch_policy: Literal["best", "latest"] = "best",
        strategy_model: Optional[str] = None,
        strategy_reasoning_effort: Optional[str] = None,
        strategy_history_window: int = 5,
        strategy_temperature: float = 0.4,
    ) -> None:
        self.branch_policy = branch_policy
        self.strategy_model = strategy_model
        self.strategy_reasoning_effort = strategy_reasoning_effort
        self.strategy_history_window = strategy_history_window
        self.strategy_temperature = strategy_temperature
        self._history: list[AgentFeedback] = []
        self._train_case_ids: Optional[list[str]] = None
        self._eval_case_ids: Optional[list[str]] = None

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
        seed_files = self._read_seed(seed_dir)

        # Round 0 — copy the seed verbatim and evaluate it.
        self._run_round_zero(seed_dir, evaluator, gatherer, benchmark_dir, experiment_dir)
        if self._target_reached(score_target):
            return self._outcome()

        for round_num in range(1, max_rounds + 1):
            base_round = self._base_round_for_next()
            base_dir = experiment_dir / f"round_{base_round:03d}"
            out_dir = experiment_dir / f"round_{round_num:03d}"
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "logs").mkdir(exist_ok=True)

            strategy = self._propose_strategy(seed_files, out_dir)
            edit_result = editor.apply(strategy, self._history[-1], base_dir, out_dir)

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
            if self._target_reached(score_target):
                break

        return self._outcome()

    # ------------------------------------------------------------------ #
    # Strategy: ask the LLM for the next edit directive
    # ------------------------------------------------------------------ #

    def _propose_strategy(
        self, seed_files: dict[str, str], out_dir: Optional[Path] = None
    ) -> EvolutionStrategy:
        # Imported lazily so unit tests don't require the SDK.
        from platform_core.llm_wrapper import call_llm

        recent = self._history[-self.strategy_history_window :]
        best = (
            max(self._history, key=lambda f: f.eval_result.score)
            if self._history
            else None
        )

        system = (
            "You are the strategy module of a self-evolving agent framework. "
            "Your job is to propose ONE focused edit to the task agent's "
            "code that should improve its benchmark score. Reason about "
            "which file to change, what concretely to change, and why."
        )
        user = self._render_strategy_prompt(recent, best, seed_files)

        llm_kwargs: dict = {
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "tools": [PROPOSE_EDIT_TOOL],
        }
        if self.strategy_model:
            llm_kwargs["model"] = self.strategy_model
        if self.strategy_reasoning_effort:
            llm_kwargs["reasoning_effort"] = self.strategy_reasoning_effort
        else:
            llm_kwargs["temperature"] = self.strategy_temperature
        response = call_llm(**llm_kwargs)

        if out_dir is not None and verbose_log.is_enabled():
            verbose_log.write_text(
                out_dir,
                "manager_strategy_prompt.txt",
                system + "\n\n---\n\n" + user,
            )
            verbose_log.write_json(
                out_dir,
                "manager_strategy_response.json",
                {
                    "content": getattr(response, "content", None),
                    "tool_calls": [
                        {"name": c.name, "arguments": c.arguments}
                        for c in (getattr(response, "tool_calls", []) or [])
                    ],
                },
            )

        for call in getattr(response, "tool_calls", []) or []:
            if call.name == "propose_edit":
                args = call.arguments
                return EvolutionStrategy(
                    target_files=args.get("target_files") or ["workflow.py"],
                    optimization_goal=args.get("optimization_goal", ""),
                    proposed_changes=args.get("proposed_changes", ""),
                    rationale=args.get("rationale", ""),
                )

        # Fallback when the LLM didn't tool-call: keep the round productive.
        return EvolutionStrategy(
            target_files=["workflow.py"],
            optimization_goal="Improve task-agent reliability.",
            proposed_changes=response.content or "(no proposal returned)",
            rationale="LLM did not produce a structured proposal.",
        )

    def _render_strategy_prompt(
        self,
        recent: list[AgentFeedback],
        best: Optional[AgentFeedback],
        seed_files: dict[str, str],
    ) -> str:
        parts: list[str] = []
        parts.append("## Seed (round 0) sources")
        for fname, body in seed_files.items():
            parts.append(f"### {fname}\n```\n{body[:2000]}\n```")
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
            "\nCall the propose_edit tool with one focused change. Keep the "
            "scope small enough that the editor can apply it correctly in one "
            "pass."
        )
        return "\n".join(parts)

    # ------------------------------------------------------------------ #
    # Round-loop internals
    # ------------------------------------------------------------------ #

    def _read_seed(self, seed_dir: Path) -> dict[str, str]:
        sources: dict[str, str] = {}
        for path in seed_dir.rglob("*"):
            if path.is_dir() or "__pycache__" in path.parts:
                continue
            rel = path.relative_to(seed_dir)
            try:
                sources[str(rel)] = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
        return sources

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
        added to ``self._history`` — best-round selection and the
        strategy prompt continue to read only the train score.
        """
        train_score = feedback.eval_result.score
        if self._eval_case_ids is None:
            print(f"round {round_num}: train={train_score:.3f}", flush=True)
            return
        # Skip eval when the editor failed validation. The editor's retry
        # loop always leaves out_dir/task_agent populated (reset to base
        # on the last failure), so a filesystem check would never fire.
        # ``feedback.edit_errors`` is the actual signal — set by the
        # manager's ``_synth_failed_edit_feedback`` whenever EditResult
        # came back unsuccessful.
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
