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
            "level": (case.get("meta_info") or {}).get("level"),
        }
        return {"score": composite, "passed": passed, "details": details}

    # ----- round-level aggregate ----- #

    def aggregate(
        self, per_case: list[Any], trace_events: list[Any]
    ) -> dict[str, Any]:
        """Per-level breakdown for the round-level project_metrics.

        Travel's gatherer hands ``per_case`` a list of ``CaseResult``
        objects; each has ``.score`` and ``.metrics`` mirrors the
        ``details`` dict the scorer emitted. We bucket by level and
        report mean composite + match count per level so the strategy
        prompt sees signal about where the optimizer is improving.
        """
        levels: dict[int, list[float]] = {1: [], 2: [], 3: []}
        all_scores: list[float] = []
        completed = 0
        for r in per_case or []:
            score = float(getattr(r, "score", 0.0) or 0.0)
            metrics = getattr(r, "metrics", {}) or {}
            lvl = metrics.get("level")
            all_scores.append(score)
            if lvl in levels:
                levels[lvl].append(score)
            if metrics.get("expected_count"):
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
