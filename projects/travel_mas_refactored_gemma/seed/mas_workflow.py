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

from platform_core import trace
from platform_core.runner import AgentOutput, Task

from agents.accounting import run_accounting_stage
from agents.flight import run_flight_stage
from agents.immutable.message import AgentMessage
from agents.sightseeing import run_sightseeing_stage
from agents.train import run_train_stage
from tool_wrapper import ToolWrapper


def _record_stage_outcome(metadata: dict, msg: AgentMessage) -> None:
    """Fold one stage's AgentMessage into `metadata` and, if there's
    anything noteworthy, emit a structured trace event -- generic over
    EVERY stage (flight/train/sightseeing/accounting), not special-cased
    to sightseeing the way this used to be.

    Two INDEPENDENT per-agent flags, not one derived/gated one:
    - `{sender}_output_failure`: this stage did not produce usable output
      (`msg.ok` is False), for whatever reason.
    - `{sender}_budget_exhausted`: this stage hit its own tool-calling
      iteration cap.
    These can co-occur or not -- whether budget_exhausted is also set
    alongside output_failure is itself the signal for whether running out
    of iterations was likely the CAUSE, without needing a third flag to
    encode that relationship. (The previous design's `task_failure` was
    exactly `output_failure AND NOT budget_exhausted` under a different,
    sightseeing-only name -- redundant with `sightseeing_failed` in
    practice since budget_exhausted essentially never fires; the two
    independent flags here make that relationship visible by inspection
    instead of baking it into a name.)
    """
    if msg.budget_exhausted:
        metadata[f"{msg.sender}_budget_exhausted"] = True
    if not msg.ok:
        metadata[f"{msg.sender}_output_failure"] = True
        if msg.output_truncated:
            metadata[f"{msg.sender}_output_truncated"] = True
        if msg.error:
            metadata[f"{msg.sender}_output_failure_reason"] = msg.error

    if msg.budget_exhausted or not msg.ok:
        # So a diagnosis reading logs/trace.jsonl directly (e.g. an
        # agentic block_suggester/editor call) can find "which agent
        # failed and why" without reconstructing it from a case's final
        # AgentOutput.metadata after the fact.
        trace.emit(
            "error",
            {
                "agent": msg.sender,
                "output_failure": not msg.ok,
                "budget_exhausted": msg.budget_exhausted,
                "output_truncated": msg.output_truncated,
                "iterations": msg.iterations,
                "message": msg.error,
            },
        )


def run_task(task: Task) -> AgentOutput:
    wrapper = ToolWrapper()
    full_schema = wrapper.get_schema()

    flight_msg = run_flight_stage(task, [], wrapper, full_schema)
    train_msg = run_train_stage(task, [], wrapper, full_schema)
    sightseeing_msg = run_sightseeing_stage(task, [flight_msg, train_msg], wrapper, full_schema)

    metadata: dict = {
        "stage_iterations": {
            "flight": flight_msg.iterations,
            "train": train_msg.iterations,
            "sightseeing": sightseeing_msg.iterations,
        },
    }
    for msg in (flight_msg, train_msg, sightseeing_msg):
        _record_stage_outcome(metadata, msg)

    if not sightseeing_msg.ok:
        # Sightseeing never produced a real <itinerary> block, even after
        # its retry nudge. Do not hand this to Accounting -- it has no
        # real itinerary to compute a budget from and would fabricate
        # numbers (confirmed live: it invented a plausible-looking budget
        # summary from a Sightseeing stage's leftover reasoning prose).
        # Mirrors the single agent's own "no lenient fallback" contract:
        # an honest empty result, not a fabricated one.
        return AgentOutput(result="", metadata=metadata)

    accounting_msg = run_accounting_stage(task, [sightseeing_msg])
    _record_stage_outcome(metadata, accounting_msg)
    return AgentOutput(result=accounting_msg.content, metadata=metadata)
