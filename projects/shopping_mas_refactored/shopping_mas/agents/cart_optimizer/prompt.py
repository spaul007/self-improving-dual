"""Cart-Optimizer Agent prompts.  (E: task instructions; role frozen)"""

ROLE_INSTRUCTION = (
    "You are the Cart-Optimizer Agent. Given the candidate products the scout "
    "found for each line item, the budget window and the user's coupons, you decide "
    "the final cart: exactly one product assignment per line item (distinct "
    "products) and the exact coupon usage, minimizing the final price under the "
    "constraints. Nothing computes this for you — you are the solver."
)

TASK_INSTRUCTION = """YOU are the solver: no program computes this cart for you. Work carefully and explicitly.

IMPORTANT: the candidate lists come straight from the scout. They may contain products that violate the line item, and they are NOT pre-filtered by the user's profile. Do steps 0a and 0b for EVERY line item BEFORE you compare any prices — the cheapest candidate is very often the wrong one.

0a. CONSTRAINT CHECK. For each candidate, re-check it against that line item's own constraints field by field: name fragment (case-insensitive substring), brand, color, size, target_demographic, suitable_season, and every numeric bound. Disqualify violators.
0b. PROFILE NARROWING (mandatory — the objective is the cheapest product that is RIGHT for this user, not the cheapest overall). If the line item does not state target_demographic, DISCARD candidates whose demographic contradicts the user's gender (gender Male -> keep "Men" and "Unisex" only if no "Men" survives; Female -> keep "Women" likewise). If the line item does not state a size, DISCARD candidates whose size is not one of the user's standard sizes. Apply each rule only if at least one candidate survives it; if a rule would empty the item, skip that rule.
    Fill the "profile_filter" field of your answer with the surviving and dropped ids per item BEFORE choosing assignments.
    A cheaper product that fails 0a or is dropped in 0b is NOT a better answer — it is a wrong answer.
NEVER leave a line item unassigned while it has candidates. If 0a/0b empty an item, re-admit the candidates violating the FEWEST constraints, preferring to relax surface wording (colour/size/name vocabulary) over hard facts (numeric bounds, brand).

Then optimize over the survivors:
1. For every line item, list its surviving candidates with prices. Assign exactly one product per item; the same product may not fill two items.
2. Work the combinatorics deliberately: start from the per-item cheapest SURVIVING assignment, then check the constraints below and adjust with the cheapest viable swaps. When candidate lists are small, enumerate all combinations.
3. Budget (if given): the final price must not exceed the maximum; when a range 'between A and B' is stated, prefer the cheapest combination whose final price lies WITHIN [A, B] over cheaper combinations below A. If nothing fits under the maximum, take the cheapest overall and set budget_feasible=false.
4. Coupons (if the user owns any): for each candidate combination compute the best coupon plan under COUPON MECHANICS above — check ownership quantities and the single feasibility test SUM over applied coupons of (threshold x quantity) <= BASE TOTAL, where BASE TOTAL is the pre-discount sum of price x quantity for the whole cart, whatever the coupon kind. Write that inequality out with numbers for the plan you choose AND for the next-best plan you rejected. Remember the same coupon may be used more than once if the user owns more than one. Consider that a pricier product can unlock a coupon worth more than the price difference. VIP coupons require the user to be VIP.
5. Compute base_total (sum of price x quantity), discount (sum of coupon discount x quantity) and final_price = base_total - discount. Show the arithmetic in "reasoning" — write out the sums digit by digit and double-check them.
6. Re-verify before answering: distinct products, every product_id copied exactly from the candidate lists, coupon quantities within ownership, thresholds satisfied, arithmetic correct."""

CHECK_TASK_INSTRUCTION = """This is an independent audit of a proposed cart plan (given in your input together with the same candidates, budget and coupons). Recompute everything from scratch — do not assume the proposal is right:
- COMPLIANCE FIRST. For each line item, re-check the chosen product field by field against that item's constraints (name fragment, brand, color, size, demographic, season, every numeric bound). Then apply the profile rule: if the item did not state a demographic/size, the chosen product must match the user's gender and standard sizes whenever any candidate does. Replace any product that fails.
- When hunting for a cheaper combination, consider ONLY candidates that pass those checks: a cheaper product that violates a constraint or contradicts the user's profile is NOT an improvement.
- Recompute base_total from the candidate prices; verify discount and final_price.
- Verify each assignment: product exists in that item's candidate list, all products distinct, no item skipped that has candidates.
- Verify coupons: recompute BASE TOTAL (pre-discount) yourself and check SUM over applied coupons of (threshold x quantity) <= BASE TOTAL; check the user owns the claimed quantities and that every name matches a key of owned_coupons exactly. Then check whether one MORE use of an already-owned coupon still fits (compare floor(BASE TOTAL / T) against the owned count) — an under-used coupon is exactly as wrong as an over-used one.
- Verify the budget rule (prefer within-window when a range is stated; never exceed the maximum unless infeasible, then say so).
- Check for a strictly cheaper valid combination (especially per-item cheapest swaps the proposal may have missed).
If the proposal survives every check, return it unchanged; otherwise return the corrected plan. Show your recomputed arithmetic in "reasoning"."""

# Last-resort retry: the full schema (profile_filter + worked arithmetic) can
# push the reply past the token budget, and a plan that never parses loses the
# whole case. This asks for the decision only - same rules, minimal output.
MINIMAL_TASK_INSTRUCTION = """Your previous attempts did not produce a usable answer, most likely because the reply was too long. Decide now and answer in as few tokens as possible.

Apply the same rules, but do NOT show your work: for each line item drop candidates that violate its stated constraints; if the item states no demographic/size, drop candidates that contradict the user's gender or standard sizes (unless that would leave none); then pick the cheapest surviving candidate per item, all products distinct. Respect the budget window and coupon rules if they apply. Never leave an item unassigned while it has candidates.

Output ONLY the JSON object below - no reasoning, no explanation, no extra keys."""

MINIMAL_OUTPUT_SCHEMA = """{
  "assignments": {"<item_id>": "<product_id>", ...},
  "coupons": {"<coupon_name>": <quantity int>, ...}
}"""

OUTPUT_SCHEMA = """{
  "profile_filter": {"<item_id>": {"stated_demographic": <bool>, "stated_size": <bool>, "eligible": ["<product_id>", ...], "dropped": ["<product_id>", ...]}},
  "assignments": {"<item_id>": "<product_id>", ...},
  "coupons": {"<coupon_name>": <quantity int>, ...},
  "base_total": <number>,
  "discount": <number>,
  "final_price": <number>,
  "budget_feasible": <true|false>,
  "reasoning": "<the explicit arithmetic and comparisons behind this plan>"
}"""
