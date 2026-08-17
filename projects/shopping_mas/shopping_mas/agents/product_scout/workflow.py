"""Product-Scout agent workflow.  (E)

Two LLM turns per line item, both tool-calling and both agent-driven:

  1. SEARCH  — chains the read-only catalog tools to find every product
     satisfying the item's constraints, self-verifying each candidate with
     get_product_details / calculate_transport_time.
  2. VERIFY  — an independent pass over the first turn's candidates: it
     re-checks them field by field, hunts for products the search missed,
     and drops assumptions the first turn invented.

No programmatic constraint checking, sweeping, sorting or profile
filtering happens here. Python only checks that returned ids exist in the
catalog and caps the pool size (a context-size bound, trusting the
agent's cheapest-first ordering).
"""

import json

from agents import base
from agents.product_scout import prompt
from tools import loader


def run(llm, cfg, item, state, toolset, products_map, trace=None, counter=None):
    """-> (candidates: [product dict], note: str)."""
    handlers = loader.make_handlers(toolset, loader.CATALOG_TOOLS)
    tools_schema = loader.openai_tools(loader.CATALOG_TOOLS)
    profile = {
        "gender": state.user_info.get("demographics", {}).get("gender"),
        "standard_sizes": state.user_info.get("body_profile", {}).get("standard_sizes"),
    }

    def extract(raw):
        ids = []
        for c in raw.get("candidates") or []:
            pid = c.get("product_id") if isinstance(c, dict) else None
            if pid and pid in products_map and pid not in ids:
                ids.append(pid)
        return ids

    payload = {
        "line_item": item.constraints | {"quantity": item.quantity},
        "destination_province": state.destination,
        "user_profile_hint": profile,
        "catalog_size": len(products_map),
    }
    raw = base.call_agent(llm, prompt, cfg.level,
                          json.dumps(payload, ensure_ascii=False, indent=2),
                          tools=tools_schema, tool_handlers=handlers,
                          trace=trace, counter=counter)
    ids = extract(raw) if "_error" not in raw else []
    note = str(raw.get("notes", "")) if "_error" not in raw else raw["_error"]

    verify_payload = {
        "line_item": item.constraints | {"quantity": item.quantity},
        "destination_province": state.destination,
        "user_profile_hint": profile,
        "catalog_size": len(products_map),
        "search_candidates": ids,
        "search_notes": note,
    }
    raw2 = base.call_agent(llm, prompt, cfg.level,
                           json.dumps(verify_payload, ensure_ascii=False, indent=2),
                           tools=tools_schema, tool_handlers=handlers,
                           trace=trace, counter=counter,
                           task_override=prompt.VERIFY_TASK_INSTRUCTION)
    if "_error" not in raw2 and (raw2.get("candidates") or not ids):
        ids = extract(raw2)
        note = f"{note} | verify: {raw2.get('notes', '')}"
    candidates = [products_map[pid] for pid in ids]
    return candidates[: max(cfg.top_k_candidates, 8)], note
