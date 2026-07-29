"""Adapter: translates the framework's Task/AgentOutput contract into
db-mas's own task-dict/TaskResult shape and calls its `mas_workflow.run_task`
completely unmodified. The full db-mas implementation (Coordinator +
5-Specialist Postgres root-cause-diagnosis system) is vendored verbatim at
`../db-mas` (a straight copy of `/groups/AIC-MV/v.kulkarni1/db-mas`, kept for
project self-containment) and is imported here as a real dependency --
nothing about its code is duplicated or flattened into this file.
`DBMAS_ROOT` overrides the path (e.g. to point at the original development
checkout instead of this vendored copy) when set (see `dbmas_path.py`).

Interface contract (preserved across rounds):
    def run_task(task: Task) -> AgentOutput

`seed/workflow.py` is a thin shim that re-exports `run_task` from here --
this is the framework-mandated entry point's actual implementation.
"""
from __future__ import annotations

import os
import re
from typing import Any

from . import dbmas_path

dbmas_path.ensure_on_path()

import mas_workflow  # noqa: E402 -- db-mas's own module, imported unmodified

from platform_core.runner import AgentOutput, Task  # noqa: E402

_SANITIZE_RE = re.compile(r"[^a-z0-9_-]+")


def _project_safe_id(case_id: str) -> str:
    """Docker Compose project names (`-p`, see db-mas's docker_lifecycle.py)
    must be lowercase alnum/dash/underscore. `case_id` is framework-controlled
    and not guaranteed to already look like that, so sanitize it rather than
    trust it -- this only affects the container/project name, not scoring."""
    safe = _SANITIZE_RE.sub("_", case_id.lower()).strip("_-")
    return safe or "case"


def _to_dbmas_task(task: Task) -> dict[str, Any]:
    """`task.context` carries the benchmark record's `environment`
    (init_sql, anomalies) and `task` (content, labels, number_of_labels_pred)
    blocks verbatim -- everything db-mas's own `environment/task_setup.py`
    and `mas_workflow.run_task` already expect. Ground truth (`root_causes`)
    is scorer-only (kept in the case's `meta_info`, see benchmark/cases.jsonl)
    and never reaches `Task.context`, so it's filled with a placeholder here:
    db-mas threads `root_causes` through `TaskResult` purely for its own
    bookkeeping/transcript, and the framework's scorer compares the returned
    `predicted_root_causes` against the real ground truth independently."""
    ctx = task.context
    task_content = dict(ctx["task"])
    task_content.setdefault("root_causes", [])
    return {
        "task_id": _project_safe_id(task.case_id),
        "environment": ctx["environment"],
        "task": task_content,
    }


def run_task(task: Task) -> AgentOutput:
    port = os.environ.get("DBMAS_PORT")
    result = mas_workflow.run_task(
        _to_dbmas_task(task),
        port=int(port) if port else None,
    )
    return AgentOutput(
        result={
            "predicted_root_causes": result.predicted_root_causes,
            "reasoning": result.reasoning,
        },
        metadata={
            "task_id": result.task_id,
            "number_of_labels_pred": result.number_of_labels_pred,
            "forced_fallback": result.forced_fallback,
            "validation_error": result.validation_error,
            "error": result.error,
            "token_usage": result.token_usage,
            "timing": result.timing,
            "transcript": result.transcript,
        },
    )
