"""Cart-Executor agent workflow.  (E)

A tool-calling LLM turn that drives the benchmark's cart tools to realize
the approved plan, then verifies the result itself with get_cart_info and
repairs any discrepancy. Nothing checks the cart after this agent — its
own verification is final, and the status/issues it reports are what the
workflow's repair loop acts on.

The reasoning pass is disabled for this turn: executing a given plan is
mechanical, and reasoning here only costs latency.
"""

import json

from agents import base
from agents.cart_executor import prompt
from tools import loader


def run(llm, cfg, plan, inbox, toolset, trace=None, counter=None):
    """-> (status: str, issues: [dict], raw agent output).

    `inbox` is accepted for contract symmetry with the other stages but
    not read from today -- this function's body never used the old
    `state` parameter either; everything it needs (`plan`) is already
    its own explicit argument."""
    handlers = loader.make_handlers(toolset, loader.CART_TOOLS)
    tools_schema = loader.openai_tools(loader.CART_TOOLS)
    payload = {
        "plan": {
            "products": [
                {"item_id": iid, "product_id": p["product_id"], "name": p.get("name"),
                 "price": p.get("price"),
                 "quantity": plan.get("quantities", {}).get(iid, 1)}
                for iid, p in plan["assignments"].items()],
            "coupons": plan.get("coupons") or {},
            "expected_base_total": plan.get("base_total"),
            "expected_final_price": plan.get("final_price"),
        },
        "reasoning": plan.get("reasoning", ""),
    }
    raw = base.call_agent(llm, prompt, cfg.level,
                          json.dumps(payload, ensure_ascii=False, indent=2),
                          agent_name="cart_executor",
                          tools=tools_schema, tool_handlers=handlers,
                          trace=trace, counter=counter, thinking=False)
    if "_error" in raw:
        return "failed", [{"kind": "pipeline", "id": "executor", "error": raw["_error"]}], raw
    status = "ok" if raw.get("status") == "ok" else "failed"
    issues = [i for i in (raw.get("issues") or []) if isinstance(i, dict)]
    return status, issues, raw
