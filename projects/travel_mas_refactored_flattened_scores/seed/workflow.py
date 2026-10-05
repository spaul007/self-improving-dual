"""Framework-mandated entry point: ``platform_core.runner._invoke_workflow``
does a bare ``import workflow``, relying on this exact module name/location.
Fully self-contained, single-line delegation to ``mas_workflow.run_task`` --
travel_mas_refactored's own orchestration is already synchronous with the
exact ``Task -> AgentOutput`` signature the framework wants, so no
async-bridging or Task/dict translation is needed here (contrast
math_mas/workflow.py, which does need both).

Interface contract (preserved across rounds):
    def run_task(task: Task) -> AgentOutput

This file is excluded from the editor's mutable surface (see
``mutable_exclude`` in the project's config) -- it's fixed framework-
integration glue, not the MAS's own behavior. All real orchestration lives
in ``mas_workflow.py`` and ``agents/*.py``, both editable.
"""
from __future__ import annotations

import mas_workflow
from platform_core.runner import AgentOutput, Task


def run_task(task: Task) -> AgentOutput:
    return mas_workflow.run_task(task)
