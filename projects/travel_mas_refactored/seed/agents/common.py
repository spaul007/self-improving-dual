"""Shared mechanics used by every agent's tool-calling loop: prompt-shared
rule text, the LLM/tool call plumbing, and the plan-tag extraction helper
kept from the single-agent seed (unused by any current stage here, carried
over unchanged from the pre-split ``workflow.py`` for fidelity).

Every stage module (``agents/flight.py``, ``agents/train.py``,
``agents/sightseeing.py``, ``agents/accounting.py``) imports from here
rather than duplicating this plumbing.
"""
from __future__ import annotations

import json
import os
import re

from platform_core.llm_wrapper import call_llm
from tool_wrapper import ToolWrapper

MAX_ITERATIONS_PER_STAGE = 25
# Sightseeing is the heaviest role (hotel + N attractions + N restaurants +
# road-route lookups, then composing the full multi-day body) -- 25 was
# sometimes too tight: live evaluation showed a case where it ran out of
# budget mid-tool-loop and its trailing "final" text was just leftover
# reasoning prose ("Let me pick another attraction for Day 2...") instead
# of a real itinerary, which the Accounting stage then tried to compute a
# budget from anyway, fabricating numbers. Higher cap here reduces how
# often that happens; the <itinerary> tag + retry (in agents/sightseeing.py)
# is the actual safety net.

# ================================================================
# Shared format/pricing rules, quoted from the reference plan
# format so all four roles emit numbers and lines the scorer
# actually accepts. Split per-role in each agent's own module
# rather than repeating the single agent's full monolithic prompt.
# ================================================================

COMMON_RULES = """
All information must come exclusively from tool query results. Do not
fabricate, guess, or use any data outside of tool results. Names must
match tool query results exactly -- do not abbreviate, rename, or add
extra descriptions.
"""

INTERCITY_LINE_SPEC = """
Output one line per intercity leg you book, in this exact format:
  HH:MM-HH:MM | travel_intercity_public | <flight/train> <No.>, <Departure Stop> - <Arrival Stop>, ¥<price>/person
Example: 07:00-09:00 | travel_intercity_public | flight CA1234, Shanghai Hongqiao International Airport - Beijing Capital International Airport, ¥650/person
Times/duration must match the tool result exactly, without adjustment.
Price shown is per person; do not multiply it yourself -- state the raw
per-person price from the tool result verbatim (the Accounting stage
will apply passenger-count multiplication).
"""

_THINK_END_RE = re.compile(r"</think>", re.IGNORECASE)
_PLAN_RE = re.compile(r"<plan>(.*?)</plan>", re.DOTALL | re.IGNORECASE)


def _extract_plan(text: str) -> str:
    """Same extraction rule as the single-agent seed: strip everything up
    to the last ``</think>``, join all ``<plan>...</plan>`` blocks, return
    ``""`` if none are present (no lenient fallback)."""
    if not text:
        return ""
    think_ends = list(_THINK_END_RE.finditer(text))
    if think_ends:
        text = text[think_ends[-1].end():]
    matches = _PLAN_RE.findall(text)
    cleaned = [m.strip() for m in matches if m.strip()]
    return "\n\n".join(cleaned) if cleaned else ""


def _item_type(item) -> str:
    t = getattr(item, "type", None)
    if t is None and isinstance(item, dict):
        t = item.get("type")
    return t or ""


def _strip_reasoning(raw_output):
    """See projects/travel/seed/workflow.py for the full rationale: local
    vLLM-served models misread echoed `reasoning` items as a wrap-up signal.
    Controlled by the same META_AGENT_STRIP_REASONING env var/convention."""
    items = list(raw_output or [])
    if os.environ.get("META_AGENT_STRIP_REASONING") != "1":
        return items
    return [item for item in items if _item_type(item) != "reasoning"]


def filter_schema(schema: list[dict], allowed_names: set[str]) -> list[dict]:
    return [t for t in schema if t.get("function", {}).get("name") in allowed_names]


def run_tool_stage(
    system_prompt: str,
    user_content: str,
    schema: list[dict],
    wrapper: ToolWrapper,
    max_iterations: int = MAX_ITERATIONS_PER_STAGE,
) -> tuple[str, int, bool, list]:
    """Run one role's bounded tool-calling loop. Returns
    (final_text, iterations_used, budget_exhausted, messages) -- the
    accumulated ``messages`` lets a caller do a follow-up nudge call with
    full context (see ``agents/sightseeing.py``'s retry) instead of
    starting over."""
    messages: list = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    last_text = ""
    for i in range(max_iterations):
        response = call_llm(messages=messages, tools=schema)
        last_text = response.content or last_text

        raw = getattr(response, "raw", None)
        raw_output = getattr(raw, "output", None) or []
        messages.extend(_strip_reasoning(raw_output))

        if not response.tool_calls:
            return response.content or "", i + 1, False, messages

        for tc in response.tool_calls:
            try:
                result = wrapper.execute(tc.name, tc.arguments)
            except Exception as e:  # noqa: BLE001 - surface tool errors to the model
                result = json.dumps({"error": str(e)}, ensure_ascii=False)
            messages.append({
                "type": "function_call_output",
                "call_id": tc.id,
                "output": result,
            })

    return last_text, max_iterations, True, messages


def run_notool_stage(system_prompt: str, user_content: str) -> str:
    response = call_llm(messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ])
    return response.content or ""
