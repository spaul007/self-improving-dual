"""filter_by_range — numeric comparison filter over a product field."""
from __future__ import annotations

import json
from functools import reduce
from typing import Any

from platform_core.tools import register_tool
from . import _db

NAME = "filter_by_range"

SCHEMA = {
    "name": NAME,
    "description": (
        "Filters a list of product_ids by a numerical feature, a comparison "
        "operator, and a threshold value. condition_key can address a nested "
        "field with dot notation (e.g. 'rating.average_score'). If product_ids "
        "is omitted, filters from the entire per-case catalog."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "product_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional candidate product_ids.",
            },
            "condition_key": {
                "type": "string",
                "description": "Numerical feature to filter on.",
                "enum": [
                    "price",
                    "stock_quantity",
                    "sales_volume.monthly",
                    "sales_volume.total",
                    "rating.average_score",
                    "rating.total_reviews",
                ],
            },
            "operator": {
                "type": "string",
                "description": "Comparison operator.",
                "enum": [">", ">=", "<", "<=", "=="],
            },
            "value": {
                "type": "number",
                "description": "Numerical value to compare against.",
            },
        },
        "required": ["condition_key", "operator", "value"],
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


def run(condition_key=None, operator=None, value=None, product_ids=None) -> str:
    products = _db.load_products()
    if not products:
        return _db.db_missing_sentinel(NAME)
    if not condition_key or not operator or value is None:
        return json.dumps(
            {"error": "Missing required parameters: condition_key, operator, or value."},
            ensure_ascii=False,
        )
    if product_ids:
        by_id = _db.products_by_id()
        search_space = [by_id[pid] for pid in product_ids if pid in by_id]
    else:
        search_space = products
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return json.dumps(
            {"error": f"value {value!r} is not numeric"}, ensure_ascii=False
        )
    out: list[str] = []
    for p in search_space:
        pv = _nested(p, condition_key)
        if pv is None:
            continue
        try:
            pv_f = float(pv)
        except (TypeError, ValueError):
            continue
        match = (
            (operator == ">" and pv_f > value_f)
            or (operator == ">=" and pv_f >= value_f)
            or (operator == "<" and pv_f < value_f)
            or (operator == "<=" and pv_f <= value_f)
            or (operator == "==" and pv_f == value_f)
        )
        if match and p.get("product_id"):
            out.append(p["product_id"])
    return json.dumps({"filtered_products_ids": out}, ensure_ascii=False)


register_tool(NAME, SCHEMA, run)
