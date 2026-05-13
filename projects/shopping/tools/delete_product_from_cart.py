"""delete_product_from_cart — reduce or remove a cart line item."""
from __future__ import annotations

import json

from platform_core.tools import register_tool
from . import _db

NAME = "delete_product_from_cart"

SCHEMA = {
    "name": NAME,
    "description": (
        "Removes a product from the cart or reduces its quantity. If the "
        "quantity to remove equals or exceeds the cart quantity, the item is "
        "fully removed. Returns the updated cart."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "product_id": {
                "type": "string",
                "description": "Product id (must exist and currently be in the cart).",
            },
            "quantity": {
                "type": "integer",
                "description": "Optional. Positive integer; default 1.",
            },
        },
        "required": ["product_id"],
    },
}


def run(product_id=None, quantity=1) -> str:
    products = _db.products_by_id()
    if not products:
        return _db.db_missing_sentinel(NAME)
    if not product_id:
        return json.dumps({"error": "product_id is required"}, ensure_ascii=False)
    if not isinstance(quantity, (int, float)) or quantity <= 0:
        return json.dumps(
            {"error": "quantity must be a positive number"}, ensure_ascii=False
        )
    quantity = int(quantity)
    if product_id not in products:
        return json.dumps(
            {"error": f"Product not found: product_id {product_id!r}"},
            ensure_ascii=False,
        )

    cart = _db.load_cart()
    items = cart.setdefault("items", [])
    existing_idx = -1
    for idx, it in enumerate(items):
        if it.get("product_id") == product_id:
            existing_idx = idx
            break
    if existing_idx < 0:
        return json.dumps(
            {"error": f"Product not in cart: product_id {product_id!r}"},
            ensure_ascii=False,
        )
    existing_qty = int(items[existing_idx].get("quantity", 0))
    new_qty = max(0, existing_qty - quantity)
    if new_qty == 0:
        items.pop(existing_idx)
    else:
        items[existing_idx]["quantity"] = new_qty
    cart["items"] = [it for it in items if int(it.get("quantity", 0)) > 0]
    _db.update_cart_summary(cart)
    _db.write_cart(cart)
    return json.dumps(cart, ensure_ascii=False)


register_tool(NAME, SCHEMA, run)
