from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class EvolutionStrategy(BaseModel):
    # Informational only (logging/rendering) -- not a gate; see
    # agent_editor.py's _is_path_allowed/_write_edits for the actual
    # write-time enforcement. Plain str rather than a fixed Literal because
    # projects using an exclude-list mutable surface (see
    # config.FrameworkConfig.mutable_exclude) can report arbitrary edited
    # paths, not just the legacy {workflow.py, tool_wrapper.py,
    # tools_schema.json} include-list.
    target_files: list[str] = Field(default_factory=list)
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

    # How many distinct cases the TRACE-derived stats above (``tool_usage``,
    # ``tool_error_rate``, ``llm_calls``, ``log_excerpt``) actually cover — i.e.
    # the cases present in the parsed ``trace.jsonl`` (the latest evaluation
    # batch, since the evaluator truncates the trace each batch). The per-case
    # parts (``eval_result``, ``project_metrics``, ``failure_report``) are
    # cumulative over every case the node has seen, which may be more. The
    # editor renders the gap from these counts (data-driven), rather than a
    # hardcoded "last batch" claim. 0 when no trace events carried a case id.
    trace_n_cases: int = 0

    # Project-defined per-case roll-ups produced by the project's scorer
    # (via its optional ``aggregate(per_case, trace_events)`` method, which
    # ``DefaultFeedbackGatherer`` calls). Format is a flat dict of named
    # scalars, lists of ``(name, count)`` tuples, or small dicts of
    # name → score. The framework's prompt renderers iterate this
    # generically — projects pick whichever keys make sense for their
    # benchmark. Default ``{}`` for projects whose scorer doesn't define
    # ``aggregate``.
    project_metrics: dict[str, Any] = Field(default_factory=dict)

    # Generic, example-driven failure digest built by ``DefaultFeedbackGatherer``
    # from the per-case contract (score/passed/error + ``details["query"]`` +
    # ``details["raw_result"]``) and the project's ``categorize_errors`` output.
    # Shape (all framework-agnostic — no domain keys):
    #   {"summary": {...}, "categories": [...],
    #    "examples": [{category_id, case_id, score, query, plan, failed}],
    #    "hard_cases": [{case_id, score, query, plan, failed}]}
    # Default ``{}`` when disabled or no failures. See meta_agent/failure_report.py.
    failure_report: dict[str, Any] = Field(default_factory=dict)


class EditResult(BaseModel):
    success: bool
    errors: list[str] = Field(default_factory=list)
    edited_files: list[str] = Field(default_factory=list)
    # The editor's self-diagnosed summary of the change it made. Populated
    # by AgentEditor (the single self-improvement call emits it alongside
    # the file edits); the manager forwards it to the gatherer for
    # strategy.json. On a failed edit it carries the last attempt's summary.
    strategy: Optional[EvolutionStrategy] = None


class EvolutionOutcome(BaseModel):
    rounds: list[AgentFeedback]
    best_round: int
    final_score: float
