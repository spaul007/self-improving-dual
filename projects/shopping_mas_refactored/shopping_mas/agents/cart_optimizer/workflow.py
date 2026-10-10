"""Cart-Optimizer agent workflow.  (E)

The LLM is the solver. There is no combinatorial search and no coupon
calculator in Python — the agent performs the optimization itself under an
explicit procedure (constraint re-check, profile narrowing, cheapest
distinct assignment, budget window, coupon arithmetic), and a second
independent audit turn recomputes the plan and may correct it.

If both turns fail to produce a parseable plan (usually because the long
reply hit the token budget), one last retry asks for the decision alone
under a minimal schema. That is retry plumbing, not a guard: every choice
is still the model's.

Python here only materializes the model's answer — id existence in the
catalog and type coercion. It never checks constraints, computes prices,
or overrides a choice.
"""

import json

from agents import base
from agents.cart_optimizer import prompt
from agents.immutable.message import from_sender


def _brief(p):
    return {
        "product_id": p.get("product_id"), "name": p.get("name"),
        "price": p.get("price"), "brand": p.get("brand"), "color": p.get("color"),
        "size": p.get("size"), "target_demographic": p.get("target_demographic"),
        "suitable_season": p.get("suitable_season"),
    }


def _plan_from_llm(raw, items, products_map):
    """Materialize a plan dict from LLM output, or None if unusable."""
    if not isinstance(raw, dict) or not isinstance(raw.get("assignments"), dict):
        return None
    assignments, skipped = {}, []
    for it in items:
        pid = raw["assignments"].get(str(it.item_id), raw["assignments"].get(it.item_id))
        if isinstance(pid, str) and pid in products_map:
            assignments[it.item_id] = products_map[pid]
        else:
            skipped.append(it.item_id)
    if not assignments:
        return None
    coupons = {}
    for name, q in (raw.get("coupons") or {}).items():
        if isinstance(q, (int, float)) and q >= 1:
            coupons[name] = int(q)

    def num(x):
        try:
            return round(float(x), 2)
        except (TypeError, ValueError):
            return None

    return {
        "assignments": assignments,
        "coupons": coupons,
        "skipped_items": skipped,
        "base_total": num(raw.get("base_total")),
        "discount": num(raw.get("discount")),
        "final_price": num(raw.get("final_price")),
        "budget_feasible": raw.get("budget_feasible", True),
        "reasoning": str(raw.get("reasoning", "")),
        "source": "optimize",
    }


def run(llm, cfg, items, query, user_info, inbox, products_map, repair_notes=None,
        trace=None, counter=None):
    """-> plan dict, or None if no turn produced a usable plan. Plan shape:
    {"assignments": {item_id: product}, "coupons": {name: qty},
     "base_total", "discount", "final_price", "source", "skipped_items"}

    `query`/`user_info` are passed directly (per-case inputs the
    orchestrator fetches once, not another stage's output); `budget`
    (requirement_parser's own output) and each item's candidates/notes
    (product_scout's own output) are read from `inbox` via
    `from_sender` -- see agents/immutable/message.py."""
    parser_content = from_sender(inbox, "requirement_parser").content
    scout_content = from_sender(inbox, "product_scout").content
    budget = parser_content["budget"]
    payload = {
        "shopping_request": query,
        "line_items": [it.constraints | {"item_id": it.item_id, "quantity": it.quantity}
                       for it in items],
        "candidates_per_item": {
            it.item_id: [_brief(p) for p in
                         scout_content.get(it.item_id, {}).get("candidates", [])]
            for it in items},
        # The profile is what lets the optimizer resolve demographic/size the
        # query left unstated — the benchmark's ground truth applies that rule.
        "user_profile": {
            "gender": user_info.get("demographics", {}).get("gender"),
            "standard_sizes": user_info.get("body_profile", {}).get("standard_sizes"),
            "destination_province": base.destination(user_info),
        },
        "scout_notes_per_item": {it.item_id: scout_content.get(it.item_id, {}).get("note", "")
                                 for it in items},
        "budget": budget,
        "owned_coupons": base.owned_coupons(user_info),
        "is_vip": base.is_vip(user_info),
    }
    if repair_notes:
        payload["execution_failures_to_avoid"] = repair_notes

    raw = base.call_agent(llm, prompt, cfg.level,
                          json.dumps(payload, ensure_ascii=False, indent=2),
                          agent_name="cart_optimizer",
                          trace=trace, counter=counter)
    plan = _plan_from_llm(raw, items, products_map)

    audit_payload = dict(payload)
    audit_payload["proposed_plan"] = {
        k: raw.get(k) for k in ("assignments", "coupons", "base_total",
                                "discount", "final_price", "budget_feasible")
    } if isinstance(raw, dict) else None
    raw2 = base.call_agent(llm, prompt, cfg.level,
                           json.dumps(audit_payload, ensure_ascii=False, indent=2),
                           agent_name="cart_optimizer",
                           trace=trace, counter=counter,
                           task_override=prompt.CHECK_TASK_INSTRUCTION)
    plan2 = _plan_from_llm(raw2, items, products_map)
    plan = plan2 or plan
    if plan is not None:
        if plan is plan2:
            plan["source"] = "optimize+audit"
        return plan

    # Neither turn parsed. Losing the whole case here is the worst outcome,
    # so re-ask once for the decision alone with a minimal schema.
    raw3 = base.call_agent(llm, prompt, cfg.level,
                           json.dumps(payload, ensure_ascii=False, indent=2),
                           agent_name="cart_optimizer",
                           trace=trace, counter=counter,
                           task_override=prompt.MINIMAL_TASK_INSTRUCTION,
                           schema_override=prompt.MINIMAL_OUTPUT_SCHEMA,
                           thinking=False)
    plan3 = _plan_from_llm(raw3, items, products_map)
    if plan3 is not None:
        plan3["source"] = "minimal_retry"
    return plan3
