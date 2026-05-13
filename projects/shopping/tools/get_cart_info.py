"""get_cart_info — read the per-case cart.json, returning items + summary."""
from __future__ import annotations

import json

from platform_core.tools import register_tool
from . import _db

NAME = "get_cart_info"

SCHEMA = {
    "name": NAME,
    "description": (
        "Returns the current shopping cart: items (each with product_id, "
        "name, quantity, price), used_coupons, and a summary object with "
        "total_items_count and total_price."
    ),
    "input_schema": {
        "type": "object",
        "properties": {},
        "required": [],
    },
}


def run(**_kwargs) -> str:
    return json.dumps(_db.load_cart(), ensure_ascii=False)


register_tool(NAME, SCHEMA, run)
