"""get_product_details — return full records for the given product_ids."""
from __future__ import annotations

import json

from platform_core.tools import register_tool
from . import _db

NAME = "get_product_details"

SCHEMA = {
    "name": NAME,
    "description": (
        "Returns the full product record for each of the given product_ids. "
        "Missing ids are silently skipped."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "product_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Product ids to look up.",
            },
        },
        "required": ["product_ids"],
    },
}


def run(product_ids=None) -> str:
    products = _db.load_products()
    if not products:
        return _db.db_missing_sentinel(NAME)
    if not product_ids:
        return json.dumps({"products": []}, ensure_ascii=False)
    by_id = _db.products_by_id()
    out = [by_id[pid] for pid in product_ids if pid in by_id]
    return json.dumps({"products": out}, ensure_ascii=False)


register_tool(NAME, SCHEMA, run)
