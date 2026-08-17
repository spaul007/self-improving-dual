"""shopping_mas's named behavior-summarizer extension point.

Like the sibling `projects/shopping` project, `meta_agent.behavior_summarizer
.BehaviorSummarizer._extract_failure_hint`'s generic hint-sniffing doesn't
recognize this scorer's key names, so the base class's hint would come out
empty for every case without an override.

Beyond that baseline need, shopping_mas can do something the single-agent
project has no equivalent for: attribute each failure to a specific one of
its 4 pipeline stages (Requirement-Parser / Product-Scout / Cart-Optimizer /
Cart-Executor), using the copied scorer's `details["failure_causes"]`
buckets cross-referenced with this MAS's own self-reported telemetry
(`metadata["issues"]`/`metadata["agent_log"]`, injected automatically into
`case.details["agent_metadata"]` by the evaluator's generic merge -- the
same mechanism math_mas/math_mas_pathology's summarizers already rely on).
"""
from __future__ import annotations

from typing import Any, Optional

from meta_agent.behavior_summarizer import BehaviorSummarizer
from meta_agent.models import CaseResult
from meta_agent.registry import register

_HINT_CHAR_CAP = 2200

# failure_causes bucket -> the pipeline stage most likely responsible.
_CAUSE_TO_STAGE = {
    "feature_mismatch": "product_scout",
    "user_info_mismatch": "requirement_parser",
    "not_cheapest": "cart_optimizer",
    "missing_product": "product_scout",
    "ambiguous": "cart_optimizer",
}


def _stage_hints(failure_causes: dict[str, Any]) -> list[str]:
    hints: list[str] = []
    for cause, stage in _CAUSE_TO_STAGE.items():
        bucket = failure_causes.get(cause)
        if not bucket:
            continue
        # bucket is either a flat slot ({sub_queries,fields,predicates,product_ids})
        # or, for user_info_mismatch, a {gender:slot, size:slot} map.
        if cause == "user_info_mismatch":
            violated = [k for k, v in bucket.items() if v.get("product_ids")]
            if violated:
                hints.append(f"[{stage}] user_info_mismatch: {'/'.join(violated)}")
            continue
        n = len(bucket.get("product_ids") or [])
        fields = bucket.get("fields") or []
        detail = f"fields={fields[:3]}" if fields else ""
        hints.append(f"[{stage}] {cause}: {n} product(s) {detail}".strip())
    return hints


@register("summarizer", "shopping_mas_default")
class ShoppingMasBehaviorSummarizer(BehaviorSummarizer):
    @staticmethod
    def _extract_failure_hint(case: CaseResult) -> str:
        parts: list[str] = []

        if case.error:
            parts.append(f"runtime_error: {case.error}")

        details: dict[str, Any] = case.details or {}

        parts.append(
            f"case_score={details.get('case_score')} "
            f"composite={details.get('composite_score')} "
            f"matched={details.get('matched_count')}/{details.get('expected_count')}"
        )

        failure_causes = details.get("failure_causes") or {}
        stage_hints = _stage_hints(failure_causes)
        if stage_hints:
            parts.append(" | ".join(stage_hints))

        budget_check = details.get("budget_check") or {}
        if budget_check.get("status") not in (None, "ok"):
            parts.append(f"[cart_optimizer] budget_check: {budget_check}")

        coupon_ownership = details.get("coupon_ownership") or {}
        if coupon_ownership.get("applied_not_owned") or coupon_ownership.get("over_owned_qty"):
            parts.append(f"[cart_executor] coupon_ownership violation: {coupon_ownership}")

        # This MAS's own self-reported telemetry, injected by the evaluator's
        # generic agent_metadata merge -- surfaces problems even when scoring
        # alone wouldn't explain them (e.g. the executor gave up on a product
        # the scout never found, or the pipeline hit its LLM-call budget).
        agent_meta = details.get("agent_metadata") or {}
        issues = agent_meta.get("issues") or []
        if issues:
            parts.append(f"mas_issues={issues[:3]}")

        if not parts:
            return ""

        text = " | ".join(str(p) for p in parts).replace("\n", " ")
        if len(text) > _HINT_CHAR_CAP:
            text = text[: _HINT_CHAR_CAP - 1].rstrip() + "…"
        return text

    def _build_prompt(
        self, aggregate: dict[str, Any], prior_memory: Optional[str] = None
    ) -> tuple[str, str]:
        user, system = super()._build_prompt(aggregate, prior_memory=prior_memory)
        system += (
            "\n\nADDITIONALLY, shopping_mas has FOUR sub-agents in a linear "
            "pipeline: REQUIREMENT-PARSER (query+profile -> line items+budget), "
            "PRODUCT-SCOUT (one instance per line item, searches the catalog for "
            "candidates), CART-OPTIMIZER (chooses the cheapest valid combination "
            "+ coupons), CART-EXECUTOR (adds the chosen items/coupons to the cart, "
            "self-verifies). Per-case hints above are tagged with the pipeline "
            "stage most likely responsible for that failure (derived from which "
            "failure_causes bucket fired). Using ONLY those tags and the per-case "
            "table, add one more section:\n"
            "  ## Sub-agent reasoning issues\n"
            "  - For each stage that shows up as responsible for multiple cases, "
            "describe the recurring pattern you can actually support from the "
            "hints shown (e.g. Product-Scout consistently missing a specific "
            "feature category, Cart-Optimizer picking a non-cheapest valid "
            "product) -- cite case_ids.\n"
            "  - If a stage never shows up as responsible, do not speculate about "
            "it.\n"
            "  Do not invent a pattern beyond what the hints actually show."
        )
        return user, system
