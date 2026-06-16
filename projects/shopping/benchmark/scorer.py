"""Shopping scorer — set-intersection match on product ids + coupons.

No LLM call at scoring time (the agent's effect is recorded in the
per-case ``cart.json``; the scorer compares directly against
``validation_cases.json``). Ports
`/users/n.tzou/cl/shopping_agent/evaluation/evaluation_pipeline.py::
evaluate_single_case` verbatim, just adapted to read the cart from
``agent_output.result`` (a JSON string) instead of disk.

Registered as ``shopping_default`` via the meta-agent's component
registry. Same shape contract as travel's scorer:

  ``score(case, agent_output) -> {"score", "passed", "details"}``
  ``aggregate(per_case, trace_events) -> dict``

``aggregate`` lands on ``AgentFeedback.project_metrics`` and reports
the per-level breakdown that the optimizer/strategy can use.
"""
from __future__ import annotations

import json
from typing import Any

from meta_agent.registry import register


# Map a requirement-feature ``field`` (as it appears in validation_cases.json's
# meta_info[*].features) to a readable error category. Defined from the field
# names present in the shopping data (brand/color/rating.*/sales_volume.*/...),
# NOT copied from any external categorizer. Used only to enrich diagnostics in
# ``details["missing_feature_categories"]`` — it never affects the score.
_FIELD_TO_CATEGORY = {
    "brand": "brand",
    "color": "color",
    "name": "name",
    "size": "size",
    "stock_quantity": "stock",
    "suitable_season": "season",
    "target_demographic": "demographic",
    "transport_time": "delivery_time",
    "price": "price",
    "rating.average_score": "rating_score",
    "rating.total_reviews": "review_count",
    "sales_volume.monthly": "sales_volume",
    "sales_volume.total": "sales_volume",
}


def _field_to_category(field: str) -> str:
    """Readable category for a requirement-feature field. Prefix rules cover
    the star-distribution / rating / sales families; unknown fields degrade to
    a slugified field name so nothing is silently dropped."""
    if not field:
        return "unknown"
    if field in _FIELD_TO_CATEGORY:
        return _FIELD_TO_CATEGORY[field]
    if field.startswith("rating.distribution."):
        return "review_distribution"
    if field.startswith("sales_volume."):
        return "sales_volume"
    if field.startswith("rating."):
        return "rating_score"
    return field.replace(".", "_")


def _parse_cart(agent_output: Any) -> dict[str, Any]:
    """``agent_output`` may be an ``AgentOutput`` (with a JSON-string
    ``result``), a raw JSON string, or a dict. Return a dict."""
    raw = getattr(agent_output, "result", agent_output)
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        return {}
    raw = raw.strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _load_validation(case: dict[str, Any]) -> dict[str, Any]:
    """Read the case's validation_cases.json off disk. The scorer runs
    in the parent process (so we don't go through ``_db.case_dir``,
    which reads env vars only set in the evaluator's child)."""
    env = case.get("env") or {}
    level = env.get("SHOPPING_LEVEL")
    sample_id = env.get("SHOPPING_SAMPLE_ID")
    if not level or not sample_id:
        return {}
    import os
    from pathlib import Path

    root_env = os.environ.get("SHOPPING_DATABASE_ROOT")
    if root_env:
        root = Path(root_env)
    else:
        # Project-relative default. scorer.py at projects/shopping/benchmark/scorer.py
        # -> repo root is three parents up; data lives at projects/shopping/data.
        root = Path(__file__).resolve().parents[3] / "projects" / "shopping" / "data"
    validation = root / f"database_level{level}" / f"case_{sample_id}" / "validation_cases.json"
    if not validation.exists():
        return {}
    try:
        return json.loads(validation.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


@register("scorer", "shopping_default")
class ShoppingScorer:
    """Shopping scorer + round-level aggregator.

    Matches the reference's scoring (`evaluation_pipeline.py:85`):
    score = (matched_products + matched_coupons) / (expected_products + expected_coupons).
    A product is matched when its id appears in both the cart and
    ground truth. A coupon is matched when the (name, quantity) pair
    appears identically. ``case_score = 1.0`` only when ``matched ==
    expected``; otherwise 0.0 (kept as a separate bool inside details).
    """

    def score(self, case: dict[str, Any], agent_output: Any) -> dict[str, Any]:
        cart = _parse_cart(agent_output)
        validation = _load_validation(case)

        cart_items = cart.get("items") or []
        gt_products = validation.get("ground_truth_products") or []
        gt_coupons = validation.get("ground_truth_coupons") or {}
        cart_coupons = cart.get("used_coupons") or []

        if not validation:
            # Without a ground-truth file we can't score anything.
            return {
                "score": 0.0,
                "passed": False,
                "details": {
                    "error": "validation_cases.json not found for this case",
                    "level": (case.get("meta_info") or {}).get("level"),
                    "composite_score": 0.0,
                    "matched_count": 0,
                    "expected_count": 0,
                },
            }

        cart_pids = {
            it.get("product_id") for it in cart_items if it.get("product_id")
        }
        gt_pids = {
            p.get("product_id") for p in gt_products if p.get("product_id")
        }
        matched_products = cart_pids & gt_pids

        cart_coupon_names: set[str] = set()
        matched_coupons = 0
        matched_coupon_names: set[str] = set()
        coupon_details: list[dict[str, Any]] = []
        for c in cart_coupons:
            name = c.get("coupon_name", "")
            qty = int(c.get("quantity", 0))
            cart_coupon_names.add(name)
            expected_qty = int(gt_coupons.get(name, 0))
            ok = name in gt_coupons and qty == expected_qty
            if ok:
                matched_coupons += 1
                matched_coupon_names.add(name)
            coupon_details.append(
                {
                    "coupon_name": name,
                    "quantity": qty,
                    "expected_quantity": expected_qty,
                    "match": ok,
                }
            )

        matched_count = len(matched_products) + matched_coupons
        expected_count = len(gt_pids) + len(gt_coupons)
        composite = matched_count / expected_count if expected_count else 0.0
        passed = matched_count == expected_count and expected_count > 0
        coupon_score = (
            matched_coupons / len(gt_coupons) if gt_coupons else 0.0
        )

        gt_coupon_names = set(gt_coupons.keys())
        extra_products = list(cart_pids - gt_pids)
        missing_products = list(gt_pids - cart_pids)
        extra_coupons = list(cart_coupon_names - gt_coupon_names)
        missing_coupons = list(gt_coupon_names - matched_coupon_names)

        # Feature-level categorization of misses — diagnostics only, does NOT
        # change the score. meta_info[idx].features (index-aligned with
        # ground_truth_products) tells us which *kind* of requirement each gold
        # product encodes. We emit two views, both read only from the project's
        # own validation data (degrades to "unknown" if meta_info is missing):
        #   * gold_feature_categories: {category: [all gold pids needing it]} —
        #     the per-category trial set used by category_significance.
        #   * missing_feature_categories: {category: {sub_queries, fields,
        #     predicates, product_ids}} for the *missing* gold products — the
        #     rich steering signal the HGM-dual categorizer reports on.
        meta_info = validation.get("meta_info") or []
        pid_to_idx = {
            p.get("product_id"): i
            for i, p in enumerate(gt_products)
            if p.get("product_id")
        }

        def _pid_categories(pid: str) -> tuple[str, dict[str, list[dict[str, Any]]]]:
            """(sub_query, {category: [predicate dicts]}) for one gold pid."""
            idx = pid_to_idx.get(pid)
            req = (
                meta_info[idx]
                if isinstance(idx, int)
                and idx < len(meta_info)
                and isinstance(meta_info[idx], dict)
                else {}
            )
            by_cat: dict[str, list[dict[str, Any]]] = {}
            for ft in req.get("features") or []:
                if not isinstance(ft, dict):
                    continue
                field = str(ft.get("field") or "")
                if not field:
                    continue
                by_cat.setdefault(_field_to_category(field), []).append(
                    {
                        "field": field,
                        "operator": ft.get("operator"),
                        "operator_value": ft.get("operator_value"),
                    }
                )
            if not by_cat:
                by_cat = {"unknown": []}
            return str(req.get("sub_query") or ""), by_cat

        gold_feature_categories: dict[str, list[str]] = {}
        for pid in gt_pids:
            _, by_cat = _pid_categories(pid)
            for cat in by_cat:
                gold_feature_categories.setdefault(cat, []).append(pid)

        missing_feature_categories: dict[str, dict[str, Any]] = {}
        for pid in missing_products:
            sub_q, by_cat = _pid_categories(pid)
            for cat, preds in by_cat.items():
                slot = missing_feature_categories.setdefault(
                    cat,
                    {"sub_queries": [], "fields": [], "predicates": [], "product_ids": []},
                )
                if sub_q and sub_q not in slot["sub_queries"]:
                    slot["sub_queries"].append(sub_q)
                if pid not in slot["product_ids"]:
                    slot["product_ids"].append(pid)
                for p in preds:
                    if p["field"] not in slot["fields"]:
                        slot["fields"].append(p["field"])
                    if p not in slot["predicates"]:
                        slot["predicates"].append(p)

        details: dict[str, Any] = {
            "composite_score": composite,
            "case_score": 1.0 if passed else 0.0,
            "matched_count": matched_count,
            "expected_count": expected_count,
            "coupon_score": coupon_score,
            "matched_products": list(matched_products),
            "missing_products": missing_products,
            "extra_products": extra_products,
            "coupon_details": coupon_details,
            "extra_coupons": extra_coupons,
            "missing_coupons": missing_coupons,
            "missing_feature_categories": missing_feature_categories,
            "gold_feature_categories": gold_feature_categories,
            "level": (case.get("meta_info") or {}).get("level"),
        }
        return {"score": composite, "passed": passed, "details": details}

    # ----- round-level aggregate ----- #

    def aggregate(
        self, per_case: list[Any], trace_events: list[Any]
    ) -> dict[str, Any]:
        """Per-level breakdown for the round-level project_metrics.

        The gatherer hands ``per_case`` a list of ``CaseResult`` objects.
        Each carries ``.score`` and a ``.details`` dict — the same dict
        ``score()`` emitted (``level``, ``expected_count``, ...). We bucket
        by level and report mean composite + match count per level so the
        strategy prompt sees signal about where the optimizer is improving.
        """
        levels: dict[int, list[float]] = {1: [], 2: [], 3: []}
        all_scores: list[float] = []
        completed = 0
        for r in per_case or []:
            score = float(getattr(r, "score", 0.0) or 0.0)
            # CaseResult exposes the scorer's emitted dict as ``.details``
            # (there is no ``.metrics`` attribute — reading that left
            # per_level / level_n / cases_with_ground_truth silently empty).
            details = getattr(r, "details", {}) or {}
            lvl = details.get("level")
            all_scores.append(score)
            if lvl in levels:
                levels[lvl].append(score)
            if details.get("expected_count"):
                completed += 1
        return {
            "score_overall": (sum(all_scores) / len(all_scores)) if all_scores else 0.0,
            "per_level": {
                lvl: (sum(scores) / len(scores)) if scores else 0.0
                for lvl, scores in levels.items()
            },
            "level_n": {lvl: len(scores) for lvl, scores in levels.items()},
            "cases_with_ground_truth": completed,
        }


# Default instance for the evaluator's "load scorer.py" fallback path
# (i.e. when nothing is passed to SubprocessEvaluator(scorer=...)). The
# in-config path runs `build_components` which constructs a fresh
# `ShoppingScorer()` and injects it; this module-level instance is what
# the standalone `evaluate.py` invocation uses.
_DEFAULT_SCORER = ShoppingScorer()


def score(case: dict[str, Any], agent_output: Any) -> dict[str, Any]:
    """Module-level scorer entry point used by ``SubprocessEvaluator``
    when no registered scorer instance is provided."""
    return _DEFAULT_SCORER.score(case, agent_output)
