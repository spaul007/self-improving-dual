from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


MutableFileName = Literal["workflow.py", "tool_wrapper.py", "tools_schema.json"]


class EvolutionStrategy(BaseModel):
    target_files: list[MutableFileName] = Field(default_factory=list)
    optimization_goal: str
    proposed_changes: str
    rationale: str = ""


class TraceEvent(BaseModel):
    timestamp: float
    kind: Literal["llm_call", "llm_response", "tool_call", "tool_result", "error"]
    payload: dict[str, Any]


class CaseResult(BaseModel):
    case_id: str
    passed: bool
    score: float
    error: Optional[str] = None
    details: dict[str, Any] = Field(default_factory=dict)


class EvaluationResult(BaseModel):
    score: float
    metrics: dict[str, float] = Field(default_factory=dict)
    passed: int = 0
    failed: int = 0
    per_case: list[CaseResult] = Field(default_factory=list)
    wall_time_s: float = 0.0
    crashed: bool = False


class AgentFeedback(BaseModel):
    round_number: int
    base_round: int
    strategy: EvolutionStrategy
    eval_result: EvaluationResult
    tool_usage: dict[str, int] = Field(default_factory=dict)
    llm_calls: int = 0
    # Per-tool errors / calls. Generic: derived from trace events emitted by
    # platform_core.tools.call_tool (an "Error" or '"error":' substring at
    # the start of a tool_result.result_preview marks a failed call).
    tool_error_rate: dict[str, float] = Field(default_factory=dict)
    runtime_exceptions: list[str] = Field(default_factory=list)
    log_excerpt: str = ""
    edit_errors: list[str] = Field(default_factory=list)

    # Project-defined per-case roll-ups produced by the project's scorer
    # (via its optional ``aggregate(per_case, trace_events)`` method, which
    # ``DefaultFeedbackGatherer`` calls). Format is a flat dict of named
    # scalars, lists of ``(name, count)`` tuples, or small dicts of
    # name → score. The framework's prompt renderers iterate this
    # generically — projects pick whichever keys make sense for their
    # benchmark. Default ``{}`` for projects whose scorer doesn't define
    # ``aggregate``.
    project_metrics: dict[str, Any] = Field(default_factory=dict)


class EditResult(BaseModel):
    success: bool
    errors: list[str] = Field(default_factory=list)
    edited_files: list[str] = Field(default_factory=list)


class EvolutionOutcome(BaseModel):
    rounds: list[AgentFeedback]
    best_round: int
    final_score: float
