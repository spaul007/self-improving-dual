"""Requirement-Parser agent workflow.  (E)

One stateless LLM call (no tools): query + user profile -> structured line
items + budget. Output is normalized/validated here so downstream code can
trust field types; malformed entries are dropped with a note rather than
crashing the pipeline.
"""

import json

from agents import base
from agents.requirement_parser import prompt

_NUMERIC_KEYS = {
    "price", "stock_quantity", "sales_volume.monthly", "sales_volume.total",
    "rating.average_score", "rating.total_reviews",
    "rating.distribution.5_star", "rating.distribution.4_star",
    "rating.distribution.3_star", "rating.distribution.2_star",
    "rating.distribution.1_star",
}
_OPS = {">", ">=", "<", "<=", "=="}
_CAT_FIELDS = ("product_name", "brand", "color", "size",
               "target_demographic", "suitable_season")


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def normalize(raw: dict) -> tuple[list, dict | None, list[str]]:
    """-> ([LineItem], budget|None, notes)."""
    notes = []
    items = []
    for i, li in enumerate(raw.get("line_items") or [], start=1):
        if not isinstance(li, dict):
            notes.append(f"dropped non-dict line item #{i}")
            continue
        c = {"description": str(li.get("description", ""))}
        for f in _CAT_FIELDS:
            val = li.get(f)
            if val is not None and str(val).strip() and str(val).lower() != "null":
                c[f] = str(val).strip()
        ncs = []
        for nc in li.get("numeric_constraints") or []:
            key, op, value = nc.get("key"), nc.get("op"), _num(nc.get("value"))
            if key in _NUMERIC_KEYS and op in _OPS and value is not None:
                ncs.append({"key": key, "op": op, "value": value})
            else:
                notes.append(f"item {i}: dropped invalid numeric constraint {nc}")
        c["numeric_constraints"] = ncs
        dd = li.get("max_delivery_days")
        if isinstance(dd, dict) and _num(dd.get("value")) is not None:
            c["max_delivery_days"] = {
                "op": dd.get("op") if dd.get("op") in ("<", "<=") else "<=",
                "value": _num(dd.get("value")),
            }
        qty = li.get("quantity")
        qty = int(qty) if isinstance(qty, (int, float)) and qty >= 1 else 1
        items.append(base.LineItem(item_id=len(items) + 1, constraints=c, quantity=qty))

    b = raw.get("budget") or {}
    budget = {"min": _num(b.get("min")), "max": _num(b.get("max"))}
    if budget["min"] is None and budget["max"] is None:
        budget = None
    return items, budget, notes


def run(llm, level: int, query: str, user_info: dict, trace=None, counter=None):
    """-> (line_items, budget, raw_output)."""
    user_msg = json.dumps({
        "shopping_request": query,
        "user_profile": user_info,
    }, ensure_ascii=False, indent=2)
    raw = base.call_agent(llm, prompt, level, user_msg, trace=trace, counter=counter)
    if "_error" in raw:
        return [], None, raw
    items, budget, notes = normalize(raw)
    if notes:
        raw["_normalization_notes"] = notes
    return items, budget, raw
