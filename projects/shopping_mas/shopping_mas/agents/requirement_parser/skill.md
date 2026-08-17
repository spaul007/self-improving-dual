# Requirement-Parser Agent

**Role (frozen):** first agent in the pipeline; converts the natural-language
shopping request + user profile into structured line items and a budget window.

**Input:** `{shopping_request, user_profile}` (profile includes address,
body sizes, VIP flag, owned coupons).

**Output:** `line_items` (one per requested item, with categorical fields,
`numeric_constraints` as dot-path key/op/value triples, and delivery
constraints), `budget {min,max}`, `parsing_notes`.

**Workflow (`workflow.py`):** single stateless LLM call, no tools. The raw
JSON is normalized (`normalize()`): unknown numeric keys/ops dropped with a
note, categorical nulls stripped, quantities coerced to ints ≥ 1, item ids
re-numbered sequentially. The normalized structure is the contract the
the Product-Scout's own verification turn
checks against, so the key vocabulary in the prompt must stay in sync with
`_NUMERIC_KEYS` here.

**Design notes / deviations:** the parser is deliberately forbidden from
inferring `target_demographic` or sizes from the user's own profile unless
the query defers to it — over-constraining silently empties the candidate
pool, which is much worse than under-constraining (the scout + optimizer
still pick the cheapest qualifying product).
