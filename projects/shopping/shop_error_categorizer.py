"""Shopping-specific error categorizer for HGM sequential optimization.

The shopping scorer intentionally keeps scoring simple: it reports product-id
and coupon overlap against the gold cart. This module turns those coarse
per-case details into domain categories that ``hgm_seq_v2`` can use for
category-conditioned edits.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Iterable

from meta_agent.models import CaseResult


_FIELD_TO_CATEGORY = {
    "brand": "brand_filter",
    "color": "color_filter",
    "name": "name_match",
    "suitable_season": "season_filter",
    "target_demographic": "demographic_filter",
    "transport_time": "delivery_time",
    "rating.total_reviews": "review_count",
    "rating.average_score": "rating_score",
    "rating.distribution.1_star": "review_threshold",
    "rating.distribution.2_star": "review_threshold",
    "rating.distribution.3_star": "review_threshold",
    "rating.distribution.4_star": "review_threshold",
    "rating.distribution.5_star": "review_threshold",
    "sales_volume.monthly": "sales_volume",
    "sales_volume.total": "sales_volume",
    "price": "price_constraint",
    "size": "size_filter",
    "stock_quantity": "stock_filter",
}

_CATEGORY_DESC = {
    "brand_filter": "a required product BRAND is not satisfied by the cart",
    "color_filter": "a required product COLOR is not satisfied by the cart",
    "name_match": "a required product NAME / keyword is not satisfied by the cart",
    "season_filter": "a required SUITABLE-SEASON is not satisfied by the cart",
    "demographic_filter": "a required TARGET-DEMOGRAPHIC is not satisfied by the cart",
    "delivery_time": "a required DELIVERY/transport-time is not satisfied by the cart",
    "review_count": "a required total-REVIEW-COUNT threshold is not satisfied",
    "rating_score": "a required average RATING-SCORE is not satisfied",
    "review_threshold": "a required star-DISTRIBUTION review threshold is not satisfied",
    "sales_volume": "a required SALES-VOLUME threshold is not satisfied",
    "price_constraint": "a required PRICE constraint is not satisfied",
    "size_filter": "a required SIZE is not satisfied by the cart",
    "stock_filter": "a required STOCK-quantity constraint is not satisfied",
}

_CART_ERROR_DESC = {
    "wrong_coupon_quantity": "an applied coupon has the WRONG quantity vs the gold coupon",
    "missing_coupon": "an expected coupon was NOT applied",
    "incomplete_case": "the agent did not finish (cart left incomplete)",
}


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "unnamed"


def _parse_case_id(case_id: str, details: dict[str, Any]) -> tuple[int | None, str | None]:
    match = re.fullmatch(r"L(\d+)-(.+)", str(case_id))
    if match:
        return int(match.group(1)), match.group(2)
    level = details.get("level")
    sample_id = details.get("sample_id") or details.get("case") or case_id
    try:
        return int(level), str(sample_id).split("-", 1)[-1]
    except (TypeError, ValueError):
        return None, None


def _validation_path(level: int, sample_id: str) -> Path:
    root_env = os.environ.get("SHOPPING_DATABASE_ROOT")
    root = (
        Path(root_env)
        if root_env
        else Path(__file__).resolve().parents[2] / "projects" / "shopping" / "data"
    )
    return root / f"database_level{level}" / f"case_{sample_id}" / "validation_cases.json"


def _load_validation(case_id: str, details: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
    level, sample_id = _parse_case_id(case_id, details)
    if level is None or sample_id is None:
        return {}, f"could not parse shopping case id {case_id!r}"
    path = _validation_path(level, sample_id)
    if not path.exists():
        return {}, f"validation_cases.json not found at {path}"
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {}, f"could not read validation_cases.json: {exc!r}"
    return parsed if isinstance(parsed, dict) else {}, None


def _feature_categories(features: list[dict[str, Any]]) -> set[str]:
    cats: set[str] = set()
    for feat in features or []:
        if not isinstance(feat, dict):
            continue
        field = str(feat.get("field") or "")
        if field:
            cats.add(_FIELD_TO_CATEGORY.get(field, field.replace(".", "_")))
    return cats


def _product_label(product: dict[str, Any]) -> str:
    pid = product.get("product_id") or "?"
    name = product.get("name")
    return f"{pid} ({name})" if name else str(pid)


def categorize_errors(
    per_case: Iterable[CaseResult],
    *,
    max_representative_errors: int = 5,
) -> list[dict[str, Any]]:
    """Group shopping failures into prioritized categories for HGM prompts."""
    cases = list(per_case)
    total = len(cases)
    if total == 0:
        return []

    groups: dict[str, dict[str, Any]] = {}

    def _ensure(cat_id: str, cat_type: str, cat_name: str, slug: str) -> dict[str, Any]:
        group = groups.get(cat_id)
        if group is None:
            group = {
                "category_id": cat_id,
                "category_type": cat_type,
                "category_name": cat_name,
                "slug": slug,
                "failing_sample_ids": set(),
                "examples": [],
            }
            groups[cat_id] = group
        return group

    def _add_example(
        group: dict[str, Any],
        sample_id: str,
        checks_failed: list[str],
        messages: list[str],
        **extra: Any,
    ) -> None:
        group["failing_sample_ids"].add(sample_id)
        if len(group["examples"]) >= max_representative_errors:
            return
        example = {
            "sample_id": sample_id,
            "checks_failed": checks_failed,
            "messages": [str(m)[:300] for m in messages if m],
        }
        for key, value in extra.items():
            if value:
                example[key] = value
        group["examples"].append(example)

    def _add_product_category(
        sample_id: str,
        category: str,
        product: dict[str, Any] | None,
        message: str | None = None,
    ) -> None:
        slug = _slug(category)
        group = _ensure(
            f"missing_product__{slug}",
            "product_constraint",
            f"Missing product constraint ({category})",
            slug,
        )
        msg = message or _CATEGORY_DESC.get(category, f"missing product category {category}")
        product_label = _product_label(product or {})
        _add_example(
            group,
            sample_id,
            [category],
            [f"{msg}: missing gold product {product_label}"],
            missing_products=[product_label] if product else None,
        )

    def _add_cart_category(
        sample_id: str,
        category: str,
        checks_failed: list[str],
        messages: list[str],
        **extra: Any,
    ) -> None:
        slug = _slug(category)
        group = _ensure(
            f"cart__{slug}",
            "cart",
            f"Cart error ({category})",
            slug,
        )
        _add_example(group, sample_id, checks_failed, messages, **extra)

    def _add_generic(sample_id: str, slug: str, name: str, message: str) -> None:
        group = _ensure(f"generic__{slug}", "generic", name, slug)
        _add_example(group, sample_id, [slug], [message])

    for case in cases:
        sample_id = str(case.case_id)
        details = dict(case.details or {})

        if case.passed:
            continue

        if case.error and not details:
            _add_generic(sample_id, "runtime_error", "Runtime / subprocess crash", str(case.error))
            continue

        if case.error:
            _add_generic(sample_id, "runtime_error", "Runtime / subprocess crash", str(case.error))

        details_error = details.get("error")
        if details_error:
            text = str(details_error)
            slug = "validation_missing" if "validation_cases.json not found" in text else "scorer_error"
            name = "Validation data missing" if slug == "validation_missing" else "Scorer error"
            _add_generic(sample_id, slug, name, text)

        missing_products = [str(p) for p in (details.get("missing_products") or []) if p]
        validation: dict[str, Any] = {}
        validation_error: str | None = None
        if missing_products:
            validation, validation_error = _load_validation(sample_id, details)
            if validation_error:
                _add_generic(sample_id, "validation_missing", "Validation data missing", validation_error)

        if missing_products and validation:
            gt_products = validation.get("ground_truth_products") or []
            meta_info = validation.get("meta_info") or []
            pid_to_idx = {
                str(product.get("product_id")): i
                for i, product in enumerate(gt_products)
                if isinstance(product, dict) and product.get("product_id")
            }
            for pid in missing_products:
                idx = pid_to_idx.get(pid)
                product = gt_products[idx] if idx is not None and idx < len(gt_products) else {"product_id": pid}
                req = meta_info[idx] if idx is not None and idx < len(meta_info) else {}
                cats = _feature_categories(req.get("features", []) if isinstance(req, dict) else [])
                if not cats:
                    cats = {"unknown_product_requirement"}
                for category in sorted(cats):
                    _add_product_category(sample_id, category, product if isinstance(product, dict) else None)
        elif missing_products:
            group = _ensure(
                "missing_product__unknown",
                "product_constraint",
                "Missing product constraint (unknown)",
                "unknown",
            )
            _add_example(
                group,
                sample_id,
                ["missing_product"],
                [f"missing gold product(s): {', '.join(missing_products)}"],
                missing_products=missing_products,
            )

        missing_coupons = [str(c) for c in (details.get("missing_coupons") or []) if c]
        if missing_coupons:
            _add_cart_category(
                sample_id,
                "missing_coupon",
                ["missing_coupon"],
                [f"{_CART_ERROR_DESC['missing_coupon']}: {missing_coupons}"],
                missing_coupons=missing_coupons,
            )

        wrong_qty: list[str] = []
        for coupon in details.get("coupon_details") or []:
            if not isinstance(coupon, dict) or coupon.get("match") is not False:
                continue
            name = str(coupon.get("coupon_name") or "?")
            qty = coupon.get("quantity")
            expected = coupon.get("expected_quantity")
            if expected:
                wrong_qty.append(f"{name}: quantity {qty}, expected {expected}")
        if wrong_qty:
            _add_cart_category(
                sample_id,
                "wrong_coupon_quantity",
                ["wrong_coupon_quantity"],
                [f"{_CART_ERROR_DESC['wrong_coupon_quantity']}: {wrong_qty}"],
                wrong_coupon_quantities=wrong_qty,
            )

        metadata = details.get("agent_metadata") or {}
        if isinstance(metadata, dict) and metadata.get("budget_exhausted"):
            _add_cart_category(
                sample_id,
                "incomplete_case",
                ["incomplete_case"],
                [_CART_ERROR_DESC["incomplete_case"]],
            )

    out: list[dict[str, Any]] = []
    for cat_id, group in groups.items():
        num_failing = len(group["failing_sample_ids"])
        out.append(
            {
                "category_id": cat_id,
                "category_type": group["category_type"],
                "category_name": group["category_name"],
                "slug": group["slug"],
                "num_failing_samples": num_failing,
                "total_samples": total,
                "failure_rate": num_failing / total,
                "representative_errors": group["examples"],
            }
        )
    out.sort(key=lambda c: (-c["failure_rate"], c["category_id"]))
    return out
