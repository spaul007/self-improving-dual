# Cart-Executor Agent

**Role (frozen):** the last agent. Make the on-disk cart match the
approved plan exactly using the cart tools, verify it, and report. The
final `cart.json` is the benchmark's answer; nothing else counts.

**Tools:** the 5 benchmark cart tools — `add/delete_product_from_cart`,
`add/delete_coupon_to_cart`, `get_cart_info`.

**Workflow (`workflow.py`):** one tool-calling turn.

1. `get_cart_info` to see the current cart.
2. Remove anything not in the plan.
3. Add the planned products, then the planned coupons — products first,
   because `add_coupon_to_cart` validates thresholds against the cart total.
4. `get_cart_info` again and compare item by item against the plan; fix any
   discrepancy and re-check until it matches or a tool failure makes it
   impossible.
5. Report `status: "ok"` only when the final cart matches exactly;
   otherwise `failed` with precise issues. Tool failures (out of stock,
   coupon threshold) are reported verbatim, never patched with substitutes
   — the workflow's repair loop excludes the failed product and re-plans.

Nothing checks the cart after this agent: its own verification is final,
and the status it reports drives the repair loop. The reasoning pass is
disabled for this turn — executing a given plan is mechanical, and
reasoning here only added latency.

Its final message is plain JSON with no tool calls, which also satisfies
the benchmark's trace-completion rule.
