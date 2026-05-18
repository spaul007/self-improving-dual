"""FeedbackGatherer — turns raw round artifacts (trace.jsonl + EvaluationResult)
into a digest the strategy can reason over.

The framework gatherer is **project-agnostic**. It reads only what's
universally available: trace events emitted by ``platform_core.tools.call_tool``
plus per-case errors from ``EvaluationResult``. It produces:

- ``tool_usage`` (call counts by name)
- ``tool_error_rate`` (errors / calls by name)
- ``llm_calls``
- ``runtime_exceptions`` (trace ``error`` events + per-case ``case.error``)
- ``log_excerpt`` (tail of trace + every ``error`` event)

For project-specific roll-ups, the scorer is the owner: a project's
scorer class may define an optional
``aggregate(per_case, trace_events) -> dict`` method. The framework
gatherer detects that method and calls it; the result lands in
``AgentFeedback.project_metrics``. Co-locating per-case ``score()`` and
round-level ``aggregate()`` on the scorer means the emit-shape and
consume-shape are next to each other in one project file. The
framework neither knows nor cares which keys a project chooses — the
prompt renderers walk the dict generically.

Also hosts ``persist_round_artifacts`` (the gatherer is the natural single
writer of ``feedback.json`` / ``eval_result.json`` / ``strategy.json``);
the manager calls it on the failed-edit synth path.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional, Protocol

from . import verbose_log
from .models import AgentFeedback, EvaluationResult, EvolutionStrategy
from .registry import register


def persist_round_artifacts(round_dir: Path, feedback: AgentFeedback) -> None:
    """Write the three canonical round artifacts — ``feedback.json``,
    ``eval_result.json``, ``strategy.json`` — into ``round_dir``. The sole
    writer of these files; called by ``DefaultFeedbackGatherer.compile`` and
    by the managers' failed-edit synth path."""
    round_dir.mkdir(parents=True, exist_ok=True)
    (round_dir / "feedback.json").write_text(
        feedback.model_dump_json(indent=2), encoding="utf-8"
    )
    (round_dir / "eval_result.json").write_text(
        feedback.eval_result.model_dump_json(indent=2), encoding="utf-8"
    )
    (round_dir / "strategy.json").write_text(
        feedback.strategy.model_dump_json(indent=2), encoding="utf-8"
    )


_ERROR_PREVIEW_RE = re.compile(r'^Error\b|"error"\s*:', re.IGNORECASE)


def render_metrics(metrics: dict[str, Any], *, cap: int = 5, indent: str = "  ") -> list[str]:
    """Render ``project_metrics`` (and similarly-shaped dicts) as prompt
    lines.

    Walks the dict by value type:

    - **scalars** (int/float/str/bool) — single inline line per key.
    - **list of (name, count)** — top ``cap`` entries, one per line.
    - **dict of name → number** — top ``cap`` entries by value (low to
      high — surfaces "weakest" first), comma-joined.
    - **anything else** — repr'd inline.

    The returned list is suitable for ``"\\n".join(...)`` into a prompt.
    Empty inputs produce an empty list.
    """
    lines: list[str] = []
    for key, val in metrics.items():
        if isinstance(val, bool):
            lines.append(f"{indent}{key}: {val}")
        elif isinstance(val, (int, float)):
            lines.append(f"{indent}{key}: {val:.3f}" if isinstance(val, float) else f"{indent}{key}: {val}")
        elif isinstance(val, str):
            lines.append(f"{indent}{key}: {val[:200]}")
        elif isinstance(val, list) and val and isinstance(val[0], (list, tuple)) and len(val[0]) >= 2:
            top = val[:cap]
            lines.append(f"{indent}{key} (top {len(top)}):")
            for entry in top:
                name, count = entry[0], entry[1]
                lines.append(f"{indent}  - {name}: {count}")
        elif isinstance(val, dict):
            try:
                ranked = sorted(val.items(), key=lambda kv: kv[1])[:cap]
                rendered = ", ".join(f"{n}={v:.2f}" for n, v in ranked)
                lines.append(f"{indent}{key} (low {len(ranked)}): {rendered}")
            except TypeError:
                lines.append(f"{indent}{key}: {val!r}")
        elif isinstance(val, list):
            top = val[:cap]
            rendered = ", ".join(str(x) for x in top)
            lines.append(f"{indent}{key}: [{rendered}]")
        else:
            lines.append(f"{indent}{key}: {val!r}")
    return lines


class FeedbackGatherer(Protocol):
    def compile(
        self,
        round_number: int,
        base_round: int,
        strategy: EvolutionStrategy,
        eval_result: EvaluationResult,
        round_dir: Path,
    ) -> AgentFeedback: ...


@register("gatherer", "default")
class DefaultFeedbackGatherer:
    def __init__(
        self,
        *,
        log_tail: int = 30,
        exception_limit: int = 20,
        scorer: Any = None,
    ) -> None:
        self.log_tail = log_tail
        self.exception_limit = exception_limit
        # The scorer instance — used to source project-specific roll-ups
        # via its optional ``aggregate(per_case, trace_events)`` method.
        # Injected by ``meta_agent.config.build_components`` when the
        # configured scorer is a registered class. ``None`` (or a scorer
        # without ``aggregate``) leaves ``project_metrics`` empty.
        self.scorer = scorer

    def compile(
        self,
        round_number: int,
        base_round: int,
        strategy: EvolutionStrategy,
        eval_result: EvaluationResult,
        round_dir: Path,
    ) -> AgentFeedback:
        events = self._read_trace(round_dir / "logs" / "trace.jsonl")

        tool_usage: dict[str, int] = {}
        tool_errors: dict[str, int] = {}
        llm_calls = 0
        runtime_exceptions: list[str] = []
        # Track tool_call name by id so we can attribute the error to the
        # right tool when its tool_result event arrives.
        call_name_by_id: dict[str, str] = {}

        for ev in events:
            kind = ev.get("kind")
            payload = ev.get("payload") or {}
            if kind == "llm_call":
                llm_calls += 1
            elif kind == "tool_call":
                name = payload.get("name", "?")
                tool_usage[name] = tool_usage.get(name, 0) + 1
                call_id = payload.get("id")
                if call_id:
                    call_name_by_id[call_id] = name
            elif kind == "tool_result":
                preview = payload.get("result_preview") or ""
                if isinstance(preview, str) and _ERROR_PREVIEW_RE.search(preview):
                    name = (
                        payload.get("name")
                        or call_name_by_id.get(payload.get("id") or "")
                        or "?"
                    )
                    tool_errors[name] = tool_errors.get(name, 0) + 1
            elif kind == "error":
                msg = f"{payload.get('where', '?')}: {payload.get('exception', '?')}"
                runtime_exceptions.append(msg)

        # Also surface per-case errors from the evaluator (subprocess crashes,
        # scorer exceptions) — those don't appear in trace.jsonl.
        for case in eval_result.per_case:
            if case.error:
                runtime_exceptions.append(f"case {case.case_id}: {case.error}")

        runtime_exceptions = runtime_exceptions[: self.exception_limit]
        log_excerpt = self._build_excerpt(events)
        tool_error_rate = self._tool_error_rate(tool_usage, tool_errors)
        project_metrics = self._project_metrics(eval_result, events)

        feedback = AgentFeedback(
            round_number=round_number,
            base_round=base_round,
            strategy=strategy,
            eval_result=eval_result,
            tool_usage=tool_usage,
            llm_calls=llm_calls,
            tool_error_rate=tool_error_rate,
            runtime_exceptions=runtime_exceptions,
            log_excerpt=log_excerpt,
            project_metrics=project_metrics,
        )
        persist_round_artifacts(round_dir, feedback)

        if verbose_log.is_enabled():
            rendered_metrics = (
                "\n".join(render_metrics(project_metrics, cap=20, indent="  "))
                if project_metrics
                else "(none)"
            )
            verbose_log.write_text(
                round_dir, "gatherer_project_metrics.txt", rendered_metrics
            )
            verbose_log.write_text(
                round_dir, "gatherer_log_excerpt.txt", log_excerpt or "(empty)"
            )

        return feedback

    # ------------------------------------------------------------------ #
    # Hooks for project-specific subclasses
    # ------------------------------------------------------------------ #

    def _project_metrics(
        self,
        eval_result: EvaluationResult,
        trace_events: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Dispatch to the scorer's optional ``aggregate(per_case,
        trace_events)`` method. Returns ``{}`` when the scorer doesn't
        define ``aggregate`` (the typical case for simple benchmarks
        like math).

        The framework's prompt renderers iterate the returned dict
        generically: floats are rendered inline, lists of ``(name,
        count)`` tuples as a top-N list, and dicts of name → score as a
        weakest-N summary. Project scorers should pick whichever
        shapes match those rendering rules.
        """
        aggregate = getattr(self.scorer, "aggregate", None) if self.scorer else None
        if aggregate is None:
            return {}
        try:
            result = aggregate(eval_result.per_case, trace_events)
        except Exception as exc:  # noqa: BLE001
            # Aggregation must never crash the round; the scorer's per-case
            # results stand on their own. Warn so a broken aggregate() is
            # visible rather than silently yielding empty project_metrics.
            print(
                f"[gatherer] warning: scorer.aggregate() raised {exc!r}; "
                "project_metrics left empty",
                flush=True,
            )
            return {}
        return dict(result or {})

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _tool_error_rate(
        self, tool_usage: dict[str, int], tool_errors: dict[str, int]
    ) -> dict[str, float]:
        rates: dict[str, float] = {}
        for name, calls in tool_usage.items():
            if calls <= 0:
                continue
            errs = tool_errors.get(name, 0)
            rates[name] = errs / calls
        return rates

    def _read_trace(self, path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        events: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return events

    def _build_excerpt(self, events: list[dict[str, Any]]) -> str:
        if not events:
            return ""
        tail = events[-self.log_tail :]
        errors = [e for e in events if e.get("kind") == "error"]
        included = errors + [e for e in tail if e not in errors]
        return "\n".join(json.dumps(ev, ensure_ascii=False) for ev in included)
