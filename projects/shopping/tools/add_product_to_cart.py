"""add_product_to_cart — validates the product + stock and mutates cart.json."""
from __future__ import annotations

import json

from platform_core.tools import register_tool
from . import _db

NAME = "add_product_to_cart"

SCHEMA = {
    "name": NAME,
    "description": (
        "Adds a product to the cart. Validates product existence and stock "
        "availability. If the product is already in the cart, increases its "
        "quantity. Returns the updated cart."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "product_id": {
                "type": "string",
                "description": "Product id (must exist in the per-case catalog).",
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

    product = products[product_id]
    product_name = product.get("name", "")
    product_price = float(product.get("price", 0.0))
    stock = int(product.get("stock_quantity", 0))

    cart = _db.load_cart()
    items = cart.setdefault("items", [])
    existing_qty = 0
    existing_idx = -1
    for idx, it in enumerate(items):
        if it.get("product_id") == product_id:
            existing_qty = int(it.get("quantity", 0))
            existing_idx = idx
            break
    needed = existing_qty + quantity
    if needed > stock:
        return json.dumps(
            {
                "error": (
                    f"Insufficient stock: Product '{product_name}' (ID: {product_id}) "
                    f"has {stock} in stock; cart already has {existing_qty}; "
                    f"cannot add {quantity} more"
                )
            },
            ensure_ascii=False,
        )
    if existing_idx >= 0:
        items[existing_idx]["quantity"] = needed
        items[existing_idx]["price"] = product_price
    else:
        items.append(
            {
                "product_id": product_id,
                "name": product_name,
                "quantity": quantity,
                "price": product_price,
            }
        )
    _db.update_cart_summary(cart)
    _db.write_cart(cart)
    return json.dumps(cart, ensure_ascii=False)


register_tool(NAME, SCHEMA, run)
