"""Framework-mandated entry point: `platform_core.runner._invoke_workflow`
does a bare `import workflow`, relying on this exact module name/location.

wikihop_mas's own `mas_workflow.run_task` is fully synchronous (a plain
`def`, not `async def`) so, unlike math_mas/db_mas, no `asyncio.run(...)`
bridging is needed here. The framework's own invocation
(`platform_core.runner._invoke_workflow`) is purely synchronous too, so a
direct call is the correct shape.

Fully self-contained (no `adapter/` dependency) -- mirrors math_mas/
workflow.py's design: the Task/AgentOutput translation is small enough
(question + context paragraphs in, prediction + trajectory out) to live
directly here rather than behind a separate glue module. Imports only
`mas_workflow` and `run_inference`, both siblings of this file in every
copy the editor/HGM makes, so no path bootstrap is needed (contrast
adapter/wikihopmas_path.py, which resolves a path for benchmark/scorer.py --
a file that is NEVER copied per-round).

Interface contract (preserved across rounds):
    def run_task(task: Task) -> AgentOutput

This file is excluded from the editor's mutable surface (see
configs/hgm_wikihop_mas_sanity.yaml's `mutable_exclude`) -- it's fixed
framework-integration glue, not the MAS's own behavior.
"""
from __future__ import annotations

from typing import Any

import mas_workflow
import run_inference
from platform_core.runner import AgentOutput, Task


def _to_wikihop_item(task: Task) -> dict[str, Any]:
    """`task.description` carries the question text (the benchmark's `input`
    field, see `benchmark/generate_cases.py`). `task.context["paragraphs"]`
    carries the raw `[[title, [sentence, ...]], ...]` context list the
    Retriever's BM25 index is built over -- REQUIRED agent input, not a
    label. Reuses `run_inference.normalize_context` verbatim -- the vendored
    project's own [title, [sentences]] -> list[Paragraph] conversion; never
    reimplemented here.

    Ground truth (`answer`/`type`/`supporting_facts`/`evidences`) lives only
    in the case's `meta_info` and never reaches `Task` -- so those keys are
    deliberately never set here. `mas_workflow.build_state()` treats every
    one of them as optional (`item.get(key, default)`), so leaving them
    unset just means those fields come back blank bookkeeping (see
    mas_state.py's MASState.gold_answer/gold_type/gold_supporting_facts/
    evidences); the framework's scorer compares independently against the
    real gold values in `meta_info`. `.get(..., [])` (not a bare
    `task.context["paragraphs"]`) so a malformed/empty context degrades to
    a (bad, but non-crashing) empty-corpus run instead of losing the whole
    case to a KeyError.
    """
    paragraphs = run_inference.normalize_context(task.context.get("paragraphs", []))
    return {
        "unique_id": task.case_id,
        "question": task.description,
        "context_paragraphs": paragraphs,
    }


def run_task(task: Task) -> AgentOutput:
    result = mas_workflow.run_task(_to_wikihop_item(task))
    return AgentOutput(
        result={
            "prediction": result["prediction"],
        },
        metadata={
            "predicted_type": result["predicted_type"],
            "final_answer_pre_retry": result["final_answer_pre_retry"],
            "predicted_supporting_facts": result["predicted_supporting_facts"],
            "concluder_rounds": result["concluder_rounds"],
            "trajectory": result["trajectory"],
            "elapsed_s": result["elapsed_s"],
            "error": result["error"],
        },
    )
