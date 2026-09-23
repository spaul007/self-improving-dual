"""Framework entry point for the seedling coding agent (FROZEN: listed in mutable_exclude).

The runner imports this file from ``<round>/task_agent`` and calls ``run_task(task)``.
Everything that evolves lives in ``seedling/`` (the Pier agent package); this file only
hands the task to the project adapter, which runs ``pier run`` for ONE DeepSWE task with
this directory on PYTHONPATH (so pier imports the candidate's ``seedling`` package).

Must never raise and never print: the runner parses the last stdout line.
"""
from __future__ import annotations

from pathlib import Path

from platform_core.runner import AgentOutput


def run_task(task) -> AgentOutput:
    try:
        from projects.deepswe_seedling.adapter.pier_case import run_case
    except Exception as exc:  # noqa: BLE001
        return AgentOutput(
            result=f"INFRA-EXCLUDED (adapter import failed: {exc!r})"[:1000],
            metadata={"status": "adapter_import_error"},
        )
    return run_case(task, Path(__file__).resolve().parent)
