"""Framework-mandated entry point: `platform_core.runner._invoke_workflow`
does a bare `import workflow`, relying on this exact module name/location.
Bridges this MAS's own async `mas_workflow.run_task` to the framework's
synchronous contract via `asyncio.run(...)` -- same pattern as
projects/math_mas/math_mas/workflow.py, confirmed by direct read of
`platform_core/runner.py::_invoke_workflow` that the framework never runs an
event loop of its own, so each subprocess case gets a fresh one here.

Interface contract (preserved across rounds):
    def run_task(task: Task) -> AgentOutput

This file is excluded from the editor's mutable surface -- it's fixed
framework-integration glue, not the MAS's own behavior.
"""
from __future__ import annotations

import asyncio
from typing import Any

import mas_workflow
from platform_core.runner import AgentOutput, Task


def _to_math_item(task: Task) -> dict[str, Any]:
    """`task.description` carries the problem's prose text (the benchmark's
    `input` field, see `benchmark/generate_cases.py`). Ground truth (`answer`)
    lives only in the case's `meta_info` and never reaches `Task` -- so
    `"answer"` is deliberately never set here. mas_workflow.run_task copies
    `item.get("answer", "")` straight into its output as bookkeeping only
    (never used for any behavioral decision), so leaving it unset just means
    that field comes back empty; the framework's scorer compares
    independently against the real gold answer."""
    return {"unique_id": task.case_id, "problem": task.description}


def run_task(task: Task) -> AgentOutput:
    result = asyncio.run(mas_workflow.run_task(_to_math_item(task)))
    return AgentOutput(
        result={
            "prediction": result["prediction"],
            "final_raw": result["final_raw"],
        },
        metadata={
            "predictor_answer": result["predictor_answer"],
            "verifier_answer": result["verifier_answer"],
            "trajectory": result["trajectory"],
            "first_draft": result["first_draft"],
            "verifier_final_context": result["verifier_final_context"],
            "context_used_by_reflector": result["context_used_by_reflector"],
            "pathology_flags": result["pathology_flags"],
            "elapsed_s": result["elapsed_s"],
            "error": result["error"],
        },
    )
