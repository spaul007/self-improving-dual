# Cart-Optimizer Agent

**Role (frozen):** given the scout's candidates per line item, the budget
window and the user's coupons, decide the final cart — exactly one product
per line item (distinct across items) and the exact coupon quantities —
minimizing the final price. **The agent is the solver**; no program
computes the answer.

**Tools:** none. It reasons over the payload it is given: the query, the
line items, the candidates, the user profile (gender + standard sizes),
the scout's notes, the budget window, owned coupons and VIP flag.

**Workflow (`workflow.py`):** up to three turns.

1. **Optimize.** An explicit procedure: (0a) re-check every candidate
   against the item's constraints and disqualify violators; (0b) narrow by
   the user's profile when the item leaves demographic/size unstated —
   applied only while at least one candidate survives; then the cheapest
   distinct assignment, the budget window, and the coupon arithmetic. The
   output schema forces a `profile_filter` field listing eligible and
   dropped ids per item *before* the assignments, which makes the rule
   executed rather than merely acknowledged.
2. **Audit.** An independent turn that recomputes everything from scratch,
   checks compliance first, and may correct the plan.
3. **Minimal retry** (only if neither turn produced parseable JSON): asks
   for the decision alone under a tiny schema with reasoning disabled.
   This is retry plumbing — the model still makes every choice.

**Level specifics.** The budget window (level 2) is stated as a hard
constraint that outranks cheapness, with an upgrade-until-inside procedure.
Coupon mechanics (level 3) are given as the environment's actual rule:
names copied verbatim from `owned_coupons`, BASE TOTAL is pre-discount, and
each use reserves its threshold, so the plan is feasible iff
`SUM(threshold x quantity) <= BASE TOTAL`.

**Python does not decide anything here** — `_plan_from_llm` only
materializes the model's answer (id existence, type coercion).
