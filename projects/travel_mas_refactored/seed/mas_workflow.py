"""Orchestration of the 4-role sequential travel MAS: Flight -> Train ->
Sightseeing -> Accounting. Each role owns a distinct slice of the task and
(except Accounting) a distinct subset of the 9 tools. There is no step
that re-reads the finished plan and audits/patches it against the scoring
rubric -- Accounting does the itemized budget tally its role would do
anyway, nothing more.

Every stage function takes a standard `(task: Task, inbox:
list[AgentMessage], ...)` signature and returns a single `AgentMessage`
(see ``agents/immutable/message.py``) -- collaboration between agents is
just "which prior AgentMessages does this call's `inbox` list contain,"
visible directly at each call site below.

This module is the mutable orchestration layer -- it and every file under
``agents/`` (except ``agents/immutable/``, the frozen AgentMessage
contract) are HGM's editable surface (see ``mutable_exclude`` in the
project's config). ``workflow.py`` (a sibling, frozen) is the only other
file excluded from editing; it exists solely to satisfy the framework's
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

    flight_msg = run_flight_stage(task, [], wrapper, full_schema)
    train_msg = run_train_stage(task, [], wrapper, full_schema)
    sightseeing_msg = run_sightseeing_stage(task, [flight_msg, train_msg], wrapper, full_schema)

    metadata = {
        "stage_iterations": {
            "flight": flight_msg.iterations,
            "train": train_msg.iterations,
            "sightseeing": sightseeing_msg.iterations,
        },
        "budget_exhausted": (
            flight_msg.budget_exhausted
            or train_msg.budget_exhausted
            or sightseeing_msg.budget_exhausted
        ),
    }

    if not sightseeing_msg.ok:
        # Sightseeing never produced a real <itinerary> block, even after
        # its retry nudge. Do not hand this to Accounting -- it has no
        # real itinerary to compute a budget from and would fabricate
        # numbers (confirmed live: it invented a plausible-looking budget
        # summary from a Sightseeing stage's leftover reasoning prose).
        # Mirrors the single agent's own "no lenient fallback" contract:
        # an honest empty result, not a fabricated one.
        metadata["sightseeing_failed"] = True
        return AgentOutput(result="", metadata=metadata)

    accounting_msg = run_accounting_stage(task, [sightseeing_msg])
    return AgentOutput(result=accounting_msg.content, metadata=metadata)
