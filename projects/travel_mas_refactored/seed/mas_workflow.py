"""Orchestration of the 4-role sequential travel MAS: Flight -> Train ->
Sightseeing -> Accounting. Each role owns a distinct slice of the task and
(except Accounting) a distinct subset of the 9 tools; each role's final
text becomes a "note" handed to the next role. There is no step that
re-reads the finished plan and audits/patches it against the scoring
rubric -- Accounting does the itemized budget tally its role would do
anyway, nothing more.

This module is the mutable orchestration layer -- it and every file under
``agents/`` are HGM's editable surface (see ``mutable_exclude`` in the
project's config). ``workflow.py`` (a sibling, frozen) is the only file
excluded from editing; it exists solely to satisfy the framework's
Task/AgentOutput entry-point contract and delegates straight here.
"""
from __future__ import annotations

from platform_core.runner import AgentOutput, Task

from agents.accounting import run_accounting_stage
from agents.flight import run_flight_stage
from agents.sightseeing import run_sightseeing_stage
from agents.train import run_train_stage
from tool_wrapper import ToolWrapper


def run_task(task: Task) -> AgentOutput:
    wrapper = ToolWrapper()
    full_schema = wrapper.get_schema()

    flight_note, flight_iters, flight_exhausted = run_flight_stage(
        task.description, wrapper, full_schema
    )
    train_note, train_iters, train_exhausted = run_train_stage(
        task.description, wrapper, full_schema
    )
    sightseeing_body, sightseeing_iters, sightseeing_exhausted = run_sightseeing_stage(
        task.description, flight_note, train_note, wrapper, full_schema
    )

    metadata = {
        "stage_iterations": {
            "flight": flight_iters,
            "train": train_iters,
            "sightseeing": sightseeing_iters,
        },
        "budget_exhausted": (
            flight_exhausted or train_exhausted or sightseeing_exhausted
        ),
    }

    if not sightseeing_body:
        # Sightseeing never produced a real <itinerary> block, even after
        # its retry nudge. Do not hand this to Accounting -- it has no
        # real itinerary to compute a budget from and would fabricate
        # numbers (confirmed live: it invented a plausible-looking budget
        # summary from a Sightseeing stage's leftover reasoning prose).
        # Mirrors the single agent's own "no lenient fallback" contract:
        # an honest empty result, not a fabricated one.
        metadata["sightseeing_failed"] = True
        return AgentOutput(result="", metadata=metadata)

    plan = run_accounting_stage(task.description, sightseeing_body)
    return AgentOutput(result=plan, metadata=metadata)
