"""The fixed inter-agent collaboration contract for shopping_mas_refactored.

Every stage function in agents/*/workflow.py returns exactly one
AgentMessage and reads whichever upstream stages' outputs it depends on
via `from_sender(inbox, "<sender>")`, rather than touching a shared
mutable object. Adapted from travel_mas_refactored's own
agents/immutable/message.py: same shape, `content: str` generalized to
`content: Any` because this MAS's stages hand off structured data
(parsed line items, per-item product candidates, a cart plan, execution
issues) rather than travel's free-text note.

One generic type, not a subclass per stage: each stage's own (fully
mutable) workflow.py documents what shape it puts in `content` for its
sender name in its own docstring -- exactly how travel documents what
"content" means for flight/train/sightseeing/accounting messages. Adding
a subclass per stage would mean editing this frozen, HGM-excluded file
every time a stage's output shape changes, which defeats the point of
keeping the frozen surface minimal and the editable surface (what gets
constructed/read at each call site) maximal.

Per-sender `content` shape used by this project's own mas_workflow.py:
  "requirement_parser" -> {"line_items": [LineItem], "budget": dict|None, "raw": dict}
  "product_scout"      -> {item_id: {"candidates": [product dict], "note": str}}
                           (ONE aggregate message built by the orchestrator after
                           the ThreadPoolExecutor fan-in completes, not one message
                           per scout -- from_sender is an exact-name lookup
                           returning a single message, and nothing downstream ever
                           wants exactly one scout's output in isolation)
  "cart_optimizer"      -> the plan dict, or None
  "cart_executor"       -> {"status": str, "issues": [dict], "raw": dict}

`budget_exhausted`/`iterations`/`output_truncated` are populated less
richly than travel's equivalents: this MAS enforces one case-wide
LLM-call budget (llm_client.CallCounter / LLMBudgetExceeded), not
travel's per-stage tool-round cap, so these fields are best-effort
diagnostics here, not a control-flow mechanism. Exceptions
(LLMBudgetExceeded, any crash) are never rerouted through
`AgentMessage.ok=False` -- they keep their existing short-circuit
behavior in mas_workflow.py and abort the case before any further
AgentMessage is even constructed, the same split travel itself uses
(its own budget_exhausted/ok flags are post-hoc diagnostics folded into
metadata, never the actual abort mechanism).

Lives under agents/immutable/ and is excluded from HGM's editable
surface (see mutable_exclude in
configs/hgm_dual_shopping_mas_refactored_sanity.yaml), the same way
workflow.py is excluded.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AgentMessage:
    sender: str
    content: Any
    ok: bool = True
    iterations: int = 0
    budget_exhausted: bool = False
    error: str | None = None
    output_truncated: bool = False


def from_sender(inbox: list[AgentMessage], sender: str) -> AgentMessage:
    for msg in inbox:
        if msg.sender == sender:
            return msg
    raise KeyError(
        f"no message from {sender!r} in inbox (have: {[m.sender for m in inbox]})"
    )
