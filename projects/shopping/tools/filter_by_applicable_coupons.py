"""filter_by_applicable_coupons — keep products whose applicable_coupons
contains ALL of the requested coupon names (set-superset semantics).
"""
from __future__ import annotations

import json

from platform_core.tools import register_tool
from . import _db

NAME = "filter_by_applicable_coupons"

VALID_COUPONS = [
    "Cross-store: ¥30 off every ¥300",
    "Cross-store: ¥60 off every ¥500",
    "Cross-store: ¥120 off every ¥900",
    "Cross-store: ¥200 off every ¥1,200",
    "Cross-store: ¥300 off every ¥1,500",
    "Same-brand: ¥25 off every ¥200",
    "Same-brand: ¥60 off every ¥400",
    "Same-brand: ¥180 off every ¥1,000",
    "Same-brand: ¥300 off every ¥1,200",
    "VIP: ¥200 off every ¥1,000",
]

SCHEMA = {
    "name": NAME,
    "description": (
        "Filters product_ids by applicable coupons. Returns products whose "
        "`applicable_coupons` is a superset of the requested coupon_names. "
        "If product_ids is omitted, filters from the entire per-case catalog."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "product_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional candidate product_ids.",
            },
            "coupon_names": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Coupon names; products must have ALL of them applicable.",
            },
        },
        "required": ["coupon_names"],
    },
}


def run(coupon_names=None, product_ids=None) -> str:
    products = _db.load_products()
    if not products:
        return _db.db_missing_sentinel(NAME)
    coupon_names = list(coupon_names or [])
    if not coupon_names:
        return json.dumps({"filtered_products_ids": []}, ensure_ascii=False)
    invalid = [c for c in coupon_names if c not in VALID_COUPONS]
    if invalid:
        return json.dumps(
            {
                "error": (
                    f"Invalid coupon names: {invalid}. "
                    f"Valid coupons are: {VALID_COUPONS}"
                )
            },
            ensure_ascii=False,
        )
    if product_ids:
        by_id = _db.products_by_id()
        search_space = [by_id[pid] for pid in product_ids if pid in by_id]
    else:
        search_space = products
    requested = set(coupon_names)
    out = [
        p.get("product_id")
        for p in search_space
        if requested.issubset(set(p.get("applicable_coupons", [])))
        and p.get("product_id")
    ]
    return json.dumps({"filtered_products_ids": out}, ensure_ascii=False)


register_tool(NAME, SCHEMA, run)
