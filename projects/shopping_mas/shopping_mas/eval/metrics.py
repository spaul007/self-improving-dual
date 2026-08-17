"""Benchmark scoring, ported from the official shoppingplanning
evaluation_pipeline.py (same arithmetic, minus the report plumbing).

Per case:  score = (matched products + matched coupons) / (expected
products + expected coupons), matching cart product-id set against ground
truth and coupon name+exact-quantity. case_score is all-or-nothing.
Summary:  overall_match_rate and average_case_score are the paper metrics;
a run is valid only if <=10% of cases end on a pending tool call.
"""

import json
from pathlib import Path


def _load(path: Path, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            return json.loads(content) if content else default
    except Exception:
        return default


def check_completion(messages_path: Path) -> bool:
    """Official rule: incomplete if the trace is missing/empty or its last
    message is a tool message or an assistant message with tool_calls."""
    data = _load(messages_path, None)
    if not data:
        return False
    messages = data.get("messages", [])
    if not messages:
        return False
    last = messages[-1]
    if last.get("role") == "tool":
        return False
    if last.get("role") == "assistant" and last.get("tool_calls"):
        return False
    return last.get("role") == "assistant"


def score_case(case_dir: Path) -> dict:
    cart = _load(case_dir / "cart.json", {})
    validation = _load(case_dir / "validation_cases.json", {})
    is_completed = check_completion(case_dir / "messages.json")

    if not cart or not validation:
        return {"case_name": case_dir.name, "success": False,
                "error": "Missing required files", "score": 0.0,
                "case_score": 0.0, "matched_count": 0, "expected_count": 0,
                "extra_products_count": 0, "is_completed": False}

    cart_product_ids = {i.get("product_id") for i in cart.get("items", []) if i.get("product_id")}
    gt_products = validation.get("ground_truth_products", [])
    gt_product_ids = {p.get("product_id") for p in gt_products if p.get("product_id")}
    matched_product_ids = cart_product_ids & gt_product_ids

    gt_coupons = validation.get("ground_truth_coupons", {}) or {}
    matched_coupons = 0
    cart_coupon_names = set()
    coupon_details = []
    for c in cart.get("used_coupons", []):
        name, q = c.get("coupon_name", ""), int(c.get("quantity", 0))
        cart_coupon_names.add(name)
        match = name in gt_coupons and q == gt_coupons[name]
        matched_coupons += int(match)
        coupon_details.append({"coupon_name": name, "quantity": q,
                               "expected_quantity": gt_coupons.get(name, 0),
                               "match": match})

    matched_count = len(matched_product_ids) + matched_coupons
    expected_count = len(gt_product_ids) + len(gt_coupons)
    score = matched_count / expected_count if expected_count else 0.0
    extra_products = sorted(cart_product_ids - gt_product_ids)
    extra_coupons = sorted(cart_coupon_names - set(gt_coupons))

    return {
        "case_name": case_dir.name,
        "success": True,
        "score": score,
        "case_score": 1.0 if matched_count == expected_count else 0.0,
        "matched_count": matched_count,
        "expected_count": expected_count,
        "matched_products": sorted(matched_product_ids),
        "unmatched_ground_truth_products": [
            {"product_id": p.get("product_id"), "name": p.get("name", ""),
             "price": p.get("price")}
            for p in gt_products if p.get("product_id") not in matched_product_ids],
        "extra_products": extra_products,
        "extra_products_count": len(extra_products) + len(extra_coupons),
        "coupon_details": coupon_details,
        "ground_truth_coupons": gt_coupons,
        "coupon_score": matched_coupons / len(gt_coupons) if gt_coupons else 0.0,
        "is_completed": is_completed,
    }


def summarize(case_results: list[dict]) -> dict:
    total = len(case_results)
    successful = sum(1 for r in case_results if r.get("case_score", 0.0) == 1.0)
    total_matched = sum(r.get("matched_count", 0) for r in case_results)
    total_expected = sum(r.get("expected_count", 0) for r in case_results)
    incomplete = sum(1 for r in case_results if not r.get("is_completed", True))
    incomplete_rate = incomplete / total if total else 0.0
    return {
        "total_cases": total,
        "successful_cases": successful,
        "failed_cases": total - successful,
        "average_case_score": successful / total if total else 0.0,
        "average_score": (sum(r.get("score", 0.0) for r in case_results) / total
                          if total else 0.0),
        "total_matched_products": total_matched,
        "total_expected_products": total_expected,
        "total_extra_products": sum(r.get("extra_products_count", 0) for r in case_results),
        "overall_match_rate": total_matched / total_expected if total_expected else 0.0,
        "incomplete_cases": incomplete,
        "incomplete_rate": incomplete_rate,
        "valid": incomplete_rate <= 0.1,
    }
