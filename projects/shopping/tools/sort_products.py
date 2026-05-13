"""sort_products — sort product_ids by a numeric or string field."""
from __future__ import annotations

import json
from functools import reduce
from typing import Any

from platform_core.tools import register_tool
from . import _db

NAME = "sort_products"

SCHEMA = {
    "name": NAME,
    "description": (
        "Sorts a list of product_ids by a specified dimension. If product_ids "
        "is omitted, sorts the entire per-case catalog."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "product_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional candidate product_ids.",
            },
            "sort_by": {
                "type": "string",
                "description": "Field to sort by (dot-paths allowed).",
                "enum": [
                    "price",
                    "sales_volume.monthly",
                    "sales_volume.total",
                    "rating.average_score",
                    "stock_quantity",
                    "name",
                ],
            },
            "order": {
                "type": "string",
                "description": "asc or desc (default desc).",
                "enum": ["asc", "desc"],
            },
        },
        "required": ["sort_by"],
    },
}


def _nested(obj: dict, key_path: str) -> Any:
    try:
        return reduce(
            lambda d, k: d.get(k) if isinstance(d, dict) else None,
            key_path.split("."),
            obj,
        )
    except (TypeError, AttributeError):
        return None


def run(sort_by=None, order="desc", product_ids=None) -> str:
    products = _db.load_products()
    if not products:
        return _db.db_missing_sentinel(NAME)
    if not sort_by:
        return json.dumps(
            {"error": "Missing required parameter: sort_by."}, ensure_ascii=False
        )
    order = (order or "desc").lower()
    if order not in ("asc", "desc"):
        return json.dumps(
            {"error": f"Invalid order {order!r}. Must be 'asc' or 'desc'."},
            ensure_ascii=False,
        )
    if product_ids:
        by_id = _db.products_by_id()
        search_space = [by_id[pid] for pid in product_ids if pid in by_id]
    else:
        search_space = products
    if not search_space:
        return json.dumps({"sorted_product_ids": []}, ensure_ascii=False)

    def key(p):
        v = _nested(p, sort_by)
        if isinstance(v, str):
            return v.lower()
        if isinstance(v, (int, float)):
            return v
        # Push Nones to the end regardless of order.
        return -float("inf") if order == "desc" else float("inf")

    try:
        ordered = sorted(search_space, key=key, reverse=(order == "desc"))
    except Exception as e:  # noqa: BLE001
        return json.dumps(
            {"error": f"Sort failed on key {sort_by!r}: {e}"}, ensure_ascii=False
        )
    return json.dumps(
        {"sorted_product_ids": [p.get("product_id") for p in ordered if p.get("product_id")]},
        ensure_ascii=False,
    )


register_tool(NAME, SCHEMA, run)
