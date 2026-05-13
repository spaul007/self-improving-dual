"""filter_by_size — case-insensitive size match (OR across multiple sizes)."""
from __future__ import annotations

import json

from platform_core.tools import register_tool
from . import _db

NAME = "filter_by_size"

SCHEMA = {
    "name": NAME,
    "description": (
        "Filters a list of product_ids by one or more sizes (case-insensitive, "
        "OR logic). If product_ids is omitted, filters from the entire per-case "
        "catalog."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "product_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional candidate product_ids. Defaults to the full catalog.",
            },
            "sizes": {
                "type": "array",
                "items": {"type": "string"},
                "description": 'Sizes to match, e.g. ["M", "L", "42"].',
            },
        },
        "required": ["sizes"],
    },
}


def run(sizes=None, product_ids=None) -> str:
    products = _db.load_products()
    if not products:
        return _db.db_missing_sentinel(NAME)
    size_set = {(s or "").lower() for s in (sizes or []) if s}
    if not size_set:
        return json.dumps({"filtered_products_ids": []}, ensure_ascii=False)
    if product_ids:
        by_id = _db.products_by_id()
        search_space = [by_id[pid] for pid in product_ids if pid in by_id]
    else:
        search_space = products
    filtered = [
        p.get("product_id")
        for p in search_space
        if str(p.get("size", "")).lower() in size_set and p.get("product_id")
    ]
    return json.dumps({"filtered_products_ids": filtered}, ensure_ascii=False)


register_tool(NAME, SCHEMA, run)
