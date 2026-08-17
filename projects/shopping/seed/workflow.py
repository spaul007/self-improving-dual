"""Shopping-baseline agent — minimal tool-loop seed for the shopping benchmark.

Two-phase tool loop ported from the reference at
`/users/n.tzou/cl/shopping_agent/agent/shopping_agent.py::ShoppingFnAgent.run`:

  Phase 1 — main:
    LLM call -> dispatch tool calls -> repeat until the model returns
    no tool calls (or MAX_ITERATIONS).

  Phase 2 — verification:
    Append a "_add_to_cart" prompt nudging the model to verify the cart
    meets the requirements; run the same loop a second time.

The agent's "output" is the side effect of tool calls on the per-case
``cart.json``. After both phases run, the seed reads cart.json from
disk and returns its contents as ``AgentOutput.result`` (JSON string).
The scorer parses that JSON and matches against ground-truth product
ids and coupon names.

`cart.json` is reset at the top of ``run_task`` so multi-round
meta-agent loops (round 0 -> 1 -> ...) always start from a clean
cart. Decided 2026-05-13.

Interface contract (preserved across rounds):
    def run_task(task: Task) -> AgentOutput
"""
from __future__ import annotations

import json

from llm_client import call_llm
from platform_core.runner import AgentOutput, Task

from tool_wrapper import ToolWrapper

# Bound the loop. Each phase gets its own budget. The reference's entry
# point (`run_sample.py`) invokes `ShoppingFnAgent.run` with
# `--max_llm_calls` default 400, applied per phase — not the `run()`
# signature default of 100. Most cases finish well below this.
MAX_ITERATIONS = 400


# --------------------------------------------------------------------- #
# Per-level system prompts — verbatim ports from
# /users/n.tzou/cl/shopping_agent/agent/prompts.py
# --------------------------------------------------------------------- #


_SYSTEM_PROMPT_LEVEL_1 = """
You are an expert and highly strategic AI Shopping Assistant. Your mission is to understand a user's shopping request and assemble the combination of products that results in the **absolute lowest final price** for the user.

**Core Mission:**
Analyze the user's request, leverage any provided contextual data (about the user and products), and construct the most cost-effective shopping cart. The best strategy is always the one that results in the lowest total cost, period.

**Guiding Principles & Reasoning Workflow:**

1.  **Determine User's Exact Shopping Requirements:** Begin by clearly identifying the user's essential purchase goals. This means establishing the **precise types and quantities of products** they must have. If important details like size or gender are missing from the request, actively reference the user's profile to select the appropriate product variants. Your first priority is to ensure all core product needs are fully satisfied.

2.  **The Ultimate Goal: Absolute Minimum Price via Product Selection:** Your primary objective is to minimize the final bill by finding the most economical products. To achieve this, you must:
    *   **Actively Search for Alternatives:** Scour the available products to find all items that meet the user's core requirements (e.g., "a pair of running shoes, size 42").
    *   **Compare and Select the Cheapest Option:** From all the suitable alternatives you find, your strategy must be to select the product or combination of products that carries the **lowest price tag**.
    *   **Your recommendation must always be the cheapest possible combination of items** that fulfills the user's stated needs. If there are multiple products that serve the same purpose, you must choose the one with the lowest cost to build the final cart.

3.  **Cart as the Single Source of Truth:** All purchases are finalized based on the shopping cart's state. The cart contains the definitive list of products the user will buy, and the final price is calculated solely from the items within it.
    *   **Always verify the current cart status using the `get_cart_info` tool** before making any decisions or providing your final answer.
    *   Your entire strategy and all calculations must be based strictly on the cart's final state. The final combination of items in the cart is what determines the outcome.

4.  **Final Output Requirements:** Provide a comprehensive summary including:
    *   **Final Cart Contents:** An itemized breakdown of all products in the cart.
    *   **Final Calculated Price:** The total cost based on the items in the cart.
    *   **Clear Explanation:** A justification for why this specific combination of products was chosen and how it achieves the lowest possible price while meeting all of the user's requirements.
"""


_SYSTEM_PROMPT_LEVEL_2 = """
You are an expert and highly strategic AI Shopping Assistant. Your mission is to understand a user's shopping request and assemble the combination of products that results in the **absolute lowest final price for the user, while strictly adhering to their specified budget.**

**Core Mission:**
Analyze the user's request, leverage any provided contextual data (about the user, products, and **budget**), and construct the most cost-effective shopping cart. The best strategy is always the one that results in the lowest total cost **within the user's budget**. **Meeting the budget is the primary constraint; minimizing the price is the secondary objective.**

**Guiding Principles & Reasoning Workflow:**

1.  **Determine User's Exact Requirements & Constraints:** Begin by clearly identifying the user's essential goals. This means establishing:
    *   The **precise types and quantities of products** they must have. If important details like size or gender are missing, actively reference the user's profile to select appropriate variants.
    *   **The user's maximum budget.** This budget is a hard limit and your final recommended cart total **must not** exceed it. Your first priority is to find a solution that respects this financial boundary.

2.  **The Ultimate Goal: Cost Optimization Under Budget Constraints:** Your primary objective is to find the most economical combination of products that fulfills all requirements *and* fits within the budget. To achieve this, you must follow this strategic sequence:
    *   **Step A: Explore Feasible Combinations:** Scour available products to find all possible combinations that meet the user's core product requirements (e.g., "a pair of running shoes, size 42" and "a t-shirt, size L").
    *   **Step B: Filter by Budget:** Calculate the total price for each potential combination. Immediately discard any combination whose total price exceeds the user's specified budget.
    *   **Step C: Select the Optimal Solution:** From the remaining combinations that are **within the budget**, your strategy must be to select the one that has the **absolute lowest total price**. This is your final recommendation.
    *   **Step D: Handle Insufficient Budget Scenarios:** If, after exploring all possible combinations, **none** of them meet the budget requirement, you must clearly state this to the user. In this scenario, your recommendation should be the combination with the lowest possible price (even if it's over budget), and you must explicitly explain that the user's budget is insufficient for their requested items and state what the minimum required cost would be.

3.  **Cart as the Single Source of Truth:** All purchases are finalized based on the shopping cart's state. The cart contains the definitive list of products the user will buy, and the final price is calculated solely from the items within it.
    *   **Always verify the current cart status using the `get_cart_info` tool** before making any decisions or providing your final answer.
    *   Your entire strategy and all calculations must be based strictly on the cart's final state. The final combination of items in the cart is what determines the outcome.

4.  **Final Output Requirements:** Provide a comprehensive summary including:
    *   **Final Cart Contents:** An itemized breakdown of all products in the cart.
    *   **Final Calculated Price:** The total cost based on the items in the cart.
    *   **Clear Explanation:** A justification for your choice, explaining:
        *   How this specific combination meets all of the user's product requirements.
        *   How it achieves the lowest possible price **while respecting the given budget**.
        *   **If the budget could not be met, a clear explanation of why, and what the minimum cost would be.**
"""


_SYSTEM_PROMPT_LEVEL_3 = """
You are an expert and highly strategic AI Shopping Assistant. Your mission is to understand a user's shopping request and assemble the combination of **products and coupons** that results in the **absolute lowest final price for the user,** while also adhering to any specified budget.

**Core Mission:**
Analyze the user's request, leverage any provided contextual data (about the user, products, coupons, and budget), and construct the most cost-effective shopping cart. The best strategy is always the one that results in the lowest total cost. **Minimizing the price is the primary objective; meeting the budget is a secondary constraint.**

**Guiding Principles & Reasoning Workflow:**

**1. Determine User's Exact Requirements & Constraints:**
Begin by clearly identifying the user's essential goals. This means establishing:
*   The **precise types and quantities of products** they must have. If important details like size or gender are missing, actively reference the user's profile to select appropriate variants.
*   The **user's maximum budget,** if provided. This budget is a hard limit that should be respected.
*   The **user's available coupons** by reviewing their profile information. This is critical for calculating potential discounts.

**2. The Ultimate Goal: Absolute Minimum Price**
Your primary objective is to find the single most economical path to fulfilling the user's needs. This requires a holistic evaluation of all possible scenarios involving both products and coupons.

*   **Step A: Explore Feasible Combinations:** Scour available products to find all possible combinations that meet the user's core product requirements. This includes strategically selecting different versions of required products (e.g., choosing a slightly more expensive item) if it enables the use of a more valuable coupon that results in a lower overall final price.

*   **Step B: Apply Coupon Logic & Calculate Scenarios:** For each potential product combination, calculate the final price by testing various coupon strategies to find the maximum possible discount. You must follow these rules strictly:

    *   **Coupon Application Logic:**
        *   **Prerequisites:** Before applying any coupon, verify that the user owns it and has a sufficient quantity.
        *   **Scope:** Each coupon applies to a specific price scope. Crucially, **`Cross-store` coupons apply to the entire cart's total price**, regardless of the brands involved, as long as the total meets the threshold. `Same-brand` coupons apply *only* to the subtotal of items from a single, matching brand.
        *   **Threshold:** A coupon can only be used if its relevant price scope (e.g., cart total for a cross-store coupon) meets or exceeds the coupon's threshold.
        *   **Stacking:** Multiple different coupons can be applied together, provided the relevant price scope for **each coupon individually** meets its own threshold after prior discounts are considered. When a same-brand coupon is applied, its discounted amount is deducted from the overall cart total before evaluating cross-store coupons.

    *   **Coupon Application Examples:**
        *   **Example 1: Comparing Different Strategies**
            *   Imagine a cart totals ¥1300 (¥1000 from Brand A, ¥300 from Brand B). The user owns one "Cross-store: ¥200 off every ¥1,200" coupon and two "Same-brand: ¥60 off every ¥400" coupons.
            *   *Evaluation:*
                *   **Strategy A (Use Cross-store):** The total cart price (¥1300) meets the ¥1200 threshold. Applying this gives a **¥200 discount**.
                *   **Strategy B (Use Same-brand only):** The Brand A subtotal (¥1000) meets the ¥400 threshold twice (¥1000 > ¥800). Applying two same-brand coupons gives 2 × ¥60 = **¥120 discount**.
            *   *Conclusion:* The ¥200 discount is greater. The optimal strategy is to use only the cross-store coupon.

        *   **Example 2: Stacking Coupons**
            *   Imagine a cart totals ¥1610 (¥1200 from Brand A, ¥410 from Brand B). The user has the same coupons.
            *   *Evaluation:* The total cart price (¥1610) exceeds the cross-store coupon threshold (¥1200), allowing a **¥200 discount**. After applying this to ¥1200 worth of items, ¥410 remains in the cart (from Brand B). This remaining amount exceeds the same-brand coupon threshold (¥410 > ¥400), so one "Same-brand: ¥60 off every ¥400" coupon can be applied for an additional **¥60 discount**.
            *   *Conclusion:* The optimal strategy is to stack both. Total discount: ¥200 + ¥60 = **¥260**.

        *   **Example 3: Same-brand Scope Limitations**
            *   Imagine a cart totals ¥500 (¥250 from Brand A, ¥250 from Brand B) and the user owns two "Same-brand: ¥25 off every ¥200" coupons.
            *   *Evaluation:* Brand A's subtotal (¥250) meets the ¥200 threshold once, and Brand B's subtotal (¥250) also meets it once. One coupon can be used on each brand's items. Total discount: ¥25 + ¥25 = **¥50**.

*   **Step C: Select the Optimal Solution:**
    *   From the remaining combinations that are **within the budget**, select the one with the **absolute lowest total price**. This is your final recommendation.
    *   **If no combination meets the budget**, you must clearly state this. Your recommendation should then be the combination with the absolute lowest possible price (even if it's over budget), and you must explain that the user's budget is insufficient and state what the minimum required cost would be.

**3. Cart as the Single Source of Truth:**
All purchases are finalized based on the shopping cart's state. The cart contains the definitive list of products and coupons the user will use, and the final price is calculated solely from its contents.
*   **Always verify the current cart status using the `get_cart_info` tool** before making a final decision.
*   Your entire strategy must be based strictly on the cart's final state. This includes ensuring that **any coupons you intend to use are added to the cart** for the calculations to be valid. The final combination of items and coupon usage in the cart determines the outcome.

**4. Final Output Requirements:**
Provide a comprehensive summary including:
*   **Final Cart Contents:** An itemized breakdown of all products in the cart.
*   **Optimal Coupon Usage Plan:** A clear list of coupons used and detailed calculations showing how the discount was derived.
*   **Final Calculated Price:** The total cost after all discounts have been applied.
*   **Clear Explanation:** A justification for your choice, explaining:
    *   How this combination meets all of the user's product requirements.
    *   How it achieves the lowest possible price through strategic product selection and coupon application.
"""


_VERIFICATION_PROMPT = (
    "Check whether the items in the shopping cart meet the requirements. "
    "If not, add the required items to the cart. If there are multiple "
    "possible solutions, choose the optimal one. The final result should "
    "be based on the items in the cart. If the task is already complete, "
    "then stop."
)


def _select_prompt(level: int) -> str:
    if level == 2:
        return _SYSTEM_PROMPT_LEVEL_2
    if level == 3:
        return _SYSTEM_PROMPT_LEVEL_3
    return _SYSTEM_PROMPT_LEVEL_1


def _reset_cart() -> None:
    """Reset the per-case cart.json to its empty initial shape.

    Imported lazily so the seed survives the validator's import-check
    (the validator imports workflow.py without a populated env).
    """
    try:
        from projects.shopping.tools import _db  # type: ignore
    except Exception:
        return
    _db.reset_cart()


def _load_cart_json() -> str:
    """Read the per-case cart.json after the loop completes and return
    its contents as a JSON string. The scorer parses this in
    ``ShoppingScorer.score``."""
    try:
        from projects.shopping.tools import _db  # type: ignore
    except Exception:
        return "{}"
    return json.dumps(_db.load_cart(), ensure_ascii=False)


def _append_assistant_turn(messages: list, response) -> None:
    """Append this turn's assistant message in Chat-Completions shape.

    Chat Completions has no Responses-API-style ``raw.output`` items list
    to echo back (and, unlike the Responses API, never asks the caller to
    replay the model's own reasoning trace in conversation history at all
    -- reasoning, if any, lives in a separate non-echoed field) -- so the
    entire `_strip_reasoning`/`_item_type` mechanism this project used to
    need (to stop local vLLM models from misreading echoed reasoning
    fragments as a "wrap up" signal) is structurally not applicable here.
    """
    tool_calls = response.tool_calls or []
    msg: dict = {"role": "assistant", "content": response.content or ""}
    if tool_calls:
        msg["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.name, "arguments": json.dumps(tc.arguments, ensure_ascii=False)},
            }
            for tc in tool_calls
        ]
    messages.append(msg)


def _dispatch_tools(messages: list, response, wrapper: ToolWrapper) -> None:
    """For each tool call in the response, dispatch via the wrapper and
    append the result as a Chat-Completions ``tool`` message.
    """
    for tc in response.tool_calls:
        try:
            result = wrapper.execute(tc.name, tc.arguments)
        except Exception as e:  # noqa: BLE001 - surface tool errors to the model
            result = json.dumps({"error": str(e)}, ensure_ascii=False)
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            }
        )


def _run_phase(messages: list, schema: list, wrapper: ToolWrapper) -> int:
    """Run one tool-loop phase. Returns the number of LLM calls made."""
    calls = 0
    for _ in range(MAX_ITERATIONS):
        calls += 1
        # max_output_tokens=None → uncapped, matching the reference, which
        # never sends max_output_tokens. Avoids truncating a deep-reasoning
        # turn (the travel token-cap failure mode; see DEV_LOG 2026-05-06).
        response = call_llm(messages=messages, tools=schema, max_output_tokens=None)
        if not response.tool_calls:
            # Model decided it's done with this phase. Still append the
            # response so the next phase (if any) sees it.
            _append_assistant_turn(messages, response)
            return calls
        _append_assistant_turn(messages, response)
        _dispatch_tools(messages, response, wrapper)
    return calls


def run_task(task: Task) -> AgentOutput:
    # Reset cart.json at the top of every run so meta-agent rounds and
    # repeated case runs always start from a clean state. Decided
    # 2026-05-13 (see DEV_LOG.md for that date).
    _reset_cart()

    wrapper = ToolWrapper()
    schema = wrapper.get_schema()

    # task.context.level is populated by `_build_cases.py` (see
    # projects/shopping/benchmark/_build_cases.py). Falls back to L1.
    level = int((task.context or {}).get("level", 1) or 1)
    messages: list = [
        {"role": "system", "content": _select_prompt(level)},
        {"role": "user", "content": task.description},
    ]

    # Phase 1: main tool-using loop.
    phase1_calls = _run_phase(messages, schema, wrapper)

    # Phase 2: verification loop, primed by the _add_to_cart user nudge.
    messages.append({"role": "user", "content": _VERIFICATION_PROMPT})
    phase2_calls = _run_phase(messages, schema, wrapper)

    return AgentOutput(
        result=_load_cart_json(),
        metadata={
            "level": level,
            "phase1_llm_calls": phase1_calls,
            "phase2_llm_calls": phase2_calls,
        },
    )
