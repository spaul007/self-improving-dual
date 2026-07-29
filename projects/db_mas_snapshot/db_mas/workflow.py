"""Framework-mandated entry point: `platform_core.runner._invoke_workflow`
does a bare `import workflow`, relying on this exact module name/location.
Bridges this project's own async `mas_workflow.run_task` to the framework's
synchronous contract via `asyncio.run(...)` -- confirmed by direct read of
`platform_core/runner.py::_invoke_workflow` that the framework never runs an
event loop of its own, so each subprocess case gets a fresh one here.

Fully self-contained (no `adapter/` dependency), matching math_mas's/
wikihop_mas's workflow.py design -- the Task/AgentOutput translation is small
enough to live directly here.

Interface contract (preserved across rounds):
    def run_task(task: Task) -> AgentOutput

This file is excluded from the editor's mutable surface -- it's fixed
framework-integration glue, not the MAS's own behavior.
"""
from __future__ import annotations

import asyncio
from typing import Any

import config
import mas_workflow
from platform_core.runner import AgentOutput, Task


def _to_db_item(task: Task) -> dict[str, Any]:
    """`task.case_id` must be the plain-integer string matching a
    `data/marble-db/db_cache/<unique_id>.json` snapshot filename exactly --
    `mas_workflow.run_task` uses it verbatim to resolve which recorded DB
    snapshot every `query_db` tool call in this task replays from
    (`_snapshot_path(unique_id)`).

    `task.context["labels"]`/`["number_of_labels_pred"]` are legitimate
    non-gold task input (the 5 candidate labels are byte-identical across
    every task and already embedded verbatim in the problem prose itself;
    `number_of_labels_pred` only reveals a *count*, never *which* labels are
    correct) -- safe to pass through.

    Ground truth (`root_causes`, config.GOLD_KEY) is deliberately NEVER set
    here. `mas_workflow.run_task` does `item.get(config.GOLD_KEY, [])`
    purely as bookkeeping in its returned record (never used to influence
    any behavior) -- leaving it unset just means that field comes back
    empty; the framework's scorer compares independently against the real
    gold answer in the case's `meta_info`.
    """
    return {
        "unique_id": task.case_id,
        "problem": task.description,
        "labels": task.context.get("labels", config.LABELS),
        "number_of_labels_pred": task.context.get("number_of_labels_pred"),
    }


def run_task(task: Task) -> AgentOutput:
    result = asyncio.run(mas_workflow.run_task(_to_db_item(task)))
    return AgentOutput(
        result={
            "prediction": result["prediction"],
            "predicted_labels": result["predicted_labels"],
        },
        metadata={
            "unique_id": result["unique_id"],
            # A structural "did my environment even load" signal, unlike
            # math_mas/wikihop_mas which have no analogous check: False
            # means `unique_id` never matched a db_cache/*.json snapshot
            # file, independent of model quality -- surfaced so the
            # scorer/summarizer can distinguish a wiring bug from a
            # genuine agent-quality failure.
            "snapshot_found": result["snapshot_found"],
            "trajectory": result["trajectory"],
            "elapsed_s": result["elapsed_s"],
            "error": result["error"],
        },
    )
