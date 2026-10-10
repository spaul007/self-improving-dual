"""Global prompts + per-agent prompt config.  (E)

Global prompt elements shared by every agent live here; per-agent pieces
live in agents/<name>/prompt.py and are registered in agent_prompts().
Editable pieces: GLOBAL_PREAMBLE, LEVEL_NOTES, COUPON_RULES, JSON_PROTOCOL
and each agent's TASK_INSTRUCTION. Each agent's ROLE_INSTRUCTION states its
frozen role in the pipeline and should not be edited.
"""

GLOBAL_PREAMBLE = (
    "You are one of four cooperating agents (Requirement-Parser, Product-Scout, "
    "Cart-Optimizer, Cart-Executor) in a multi-agent system solving the DeepPlanning "
    "shopping benchmark. Each task gives a user's natural-language shopping request; "
    "the goal is to end with the shopping cart holding EXACTLY the cheapest combination "
    "of catalog products (and, when applicable, coupons) satisfying every stated "
    "requirement, after resolving any detail the request leaves unstated (notably "
    "gender/demographic and size) from the user's own profile. The cart file is the "
    "answer — nothing you write matters unless the "
    "cart ends up correct. Scoring compares cart product IDs and coupon name+quantity "
    "against a hidden ground truth, so precision matters: exactly one product per "
    "requirement, no substitutes when a product is named, no extra items."
)

LEVEL_NOTES = {
    1: "Level 1: no budget, no coupons. Objective: for each requested item, the "
       "cheapest catalog product meeting all stated constraints.",
    2: "Level 2: the query states a budget, usually as a range 'between A and B'. "
       "Landing INSIDE that window is a hard requirement, not a preference — it "
       "outranks cheapness. A cart cheaper than A is WRONG, exactly as wrong as "
       "one above B. So do not stop at the per-item cheapest combination: total it "
       "up, and if the total is below A, upgrade items (cheapest upgrade step "
       "first) until the total enters [A, B]; if it is above B, downgrade the same "
       "way. Only among carts already inside the window does the cheapest win. "
       "State the running total explicitly as you add each item. If, after trying, "
       "no combination can land inside the window, take the cheapest overall and "
       "flag the budget as infeasible.",
    3: "Level 3: the user owns coupons (see their profile). The objective is the "
       "lowest FINAL price after coupon discounts — sometimes a pricier product "
       "unlocks a coupon worth more than the price difference. The chosen coupons "
       "must be added to the cart with exact quantities.",
}

COUPON_RULES = (
    "COUPON MECHANICS — exactly what the environment enforces when a coupon is added:\n"
    "- A coupon is named '<Kind>: ¥D off every ¥T': D = discount per use, T = threshold "
    "per use. Its name is a key of the owned_coupons object you were given — copy that "
    "key character for character, including the ¥ signs and the thousands comma. A name "
    "differing by one character is rejected and the discount is lost silently.\n"
    "- BASE TOTAL = sum over cart items of price x quantity, BEFORE any discount. Every "
    "threshold test is against the BASE TOTAL; coupons never shrink the total that other "
    "coupons (or further uses of the same coupon) are measured against, so never subtract "
    "discounts as you go.\n"
    "- Picture each use of a coupon as RESERVING ¥T of the BASE TOTAL, and reservations may "
    "not overlap. A plan 'coupon i used q_i times' is valid if and only if (a) q_i does not "
    "exceed the number owned; (b) VIP-kind coupons are used only when is_vip is true; and "
    "(c) the SUM over all applied coupons of (T_i x q_i) is <= BASE TOTAL.\n"
    "- Test (c) is the whole threshold story and it also caps one coupon at "
    "floor(BASE TOTAL / T) uses, so the SAME coupon may legitimately be used two or three "
    "times when the user owns that many copies.\n"
    "- TOTAL DISCOUNT = sum of (D_i x q_i); FINAL PRICE = BASE TOTAL - TOTAL DISCOUNT. Among "
    "plans passing (a)-(c), choose the one with the largest TOTAL DISCOUNT: enumerate the "
    "quantity combinations rather than guessing, and write the check "
    "'SUM(T_i x q_i) = ... <= BASE TOTAL = ...' out with real numbers before committing."
)

JSON_PROTOCOL = (
    "Communication protocol: finish with a single valid JSON object matching the "
    "schema below, and nothing else — no markdown fences, no commentary outside "
    "the JSON. (If you have tools, call them first as needed; the JSON is your "
    "final message once you stop calling tools.)"
)


def build_system_prompt(prompt_module, level: int, task_override: str | None = None,
                        schema_override: str | None = None) -> str:
    """Assemble an agent's system prompt: global preamble + level note +
    frozen role instruction + editable task instruction + schema + protocol.
    task_override/schema_override swap in alternative task instructions and
    output schemas (used by the scout verify / optimizer audit turns); the role
    instruction stays frozen."""
    parts = [
        GLOBAL_PREAMBLE,
        LEVEL_NOTES.get(level, ""),
    ]
    if level == 3:
        parts.append(COUPON_RULES)
    parts += [
        prompt_module.ROLE_INSTRUCTION,
        task_override or prompt_module.TASK_INSTRUCTION,
        "Output JSON schema:\n" + (schema_override or prompt_module.OUTPUT_SCHEMA),
        JSON_PROTOCOL,
    ]
    return "\n\n".join(p for p in parts if p)


def agent_prompts():
    """Registry of per-agent prompt modules (imported lazily to avoid
    circular imports)."""
    from agents.requirement_parser import prompt as requirement_parser
    from agents.product_scout import prompt as product_scout
    from agents.cart_optimizer import prompt as cart_optimizer
    from agents.cart_executor import prompt as cart_executor
    return {
        "requirement_parser": requirement_parser,
        "product_scout": product_scout,
        "cart_optimizer": cart_optimizer,
        "cart_executor": cart_executor,
    }
