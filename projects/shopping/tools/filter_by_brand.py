"""filter_by_brand — case-insensitive brand match (OR across multiple names)."""
from __future__ import annotations

import json

from platform_core.tools import register_tool
from . import _db

NAME = "filter_by_brand"

SCHEMA = {
    "name": NAME,
    "description": (
        "Filters a list of product_ids by one or more brand names "
        "(case-insensitive, OR logic). If product_ids is omitted, "
        "filters from the entire per-case catalog."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "product_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional candidate product_ids. Defaults to the full catalog.",
            },
            "brand_names": {
                "type": "array",
                "items": {"type": "string"},
                "description": 'Brand names to match, e.g. ["Nike", "ZARA"].',
            },
        },
        "required": ["brand_names"],
    },
}


def run(brand_names=None, product_ids=None) -> str:
    products = _db.load_products()
    if not products:
        return _db.db_missing_sentinel(NAME)
    brands = {(b or "").lower() for b in (brand_names or []) if b}
    if not brands:
        return json.dumps({"filtered_products_ids": []}, ensure_ascii=False)
    if product_ids:
        by_id = _db.products_by_id()
        search_space = [by_id[pid] for pid in product_ids if pid in by_id]
    else:
        search_space = products
    filtered = [
        p.get("product_id")
        for p in search_space
        if str(p.get("brand", "")).lower() in brands and p.get("product_id")
    ]
    return json.dumps({"filtered_products_ids": filtered}, ensure_ascii=False)


register_tool(NAME, SCHEMA, run)
