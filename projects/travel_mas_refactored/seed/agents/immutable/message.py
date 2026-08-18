"""The fixed inter-agent collaboration contract for travel_mas_refactored.

Every stage function in ``agents/*.py`` returns exactly one
``AgentMessage`` and takes an ``inbox: list[AgentMessage]`` of whichever
upstream stages' outputs it depends on. This module defines that contract
and nothing else -- it lives under ``agents/immutable/`` and is excluded
from HGM's editable surface (see ``mutable_exclude`` in
configs/travel_mas_refactored_qwen35b_implicit.yaml), the same way
workflow.py is excluded, so the contract itself can never be silently
redefined by an edit. Everything HGM should actually be free to tune --
prompts, which senders a stage reads from, iteration caps, retry logic --
stays in agents/common.py and agents/*.py, fully editable.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AgentMessage:
    """The single standard interface for one stage to hand its output to
    another: every stage function returns exactly one of these, and takes
    an `inbox: list[AgentMessage]` of every upstream stage's output it
    depends on. Structural wrapper around the pipeline's existing
    behavior, not a new mechanism.

    Immutable by design (frozen=True): once a stage constructs its
    AgentMessage, nothing downstream can mutate it. This keeps HGM's
    editable surface focused on *what gets constructed and passed at each
    call site* (prompts, which senders a stage reads from, how content is
    built) rather than opening a second edit surface of reaching into and
    modifying an already-returned message.
    """
    sender: str
    content: str
    ok: bool = True
    iterations: int = 0
    budget_exhausted: bool = False


def from_sender(inbox: list[AgentMessage], sender: str) -> AgentMessage:
    """Look up one upstream agent's message in an inbox by its sender
    name. Raises if absent -- a stage's inbox should always contain every
    sender it depends on."""
    for msg in inbox:
        if msg.sender == sender:
            return msg
    raise KeyError(f"no message from {sender!r} in inbox (have: {[m.sender for m in inbox]})")
