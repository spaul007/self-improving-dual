"""delete_coupon_from_cart — reduce or remove a used-coupon line item."""
from __future__ import annotations

import json
import re

from platform_core.tools import register_tool
from . import _db
from .add_coupon_to_cart import VALID_COUPONS, _COUPON_RE  # share definitions

NAME = "delete_coupon_from_cart"

SCHEMA = {
    "name": NAME,
    "description": (
        "Removes a coupon from the cart or reduces its quantity. If the "
        "quantity to remove equals or exceeds the current cart quantity, "
        "the coupon is fully removed. Returns the updated cart with "
        "recomputed `summary.total_price`."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "coupon_name": {
                "type": "string",
                "description": "Coupon name (must currently be in the cart).",
            },
            "quantity": {
                "type": "integer",
                "description": "Optional. Positive integer; default 1.",
            },
        },
        "required": ["coupon_name"],
    },
}


def _parse_coupon(coupon_name: str):
    m = _COUPON_RE.search(coupon_name or "")
    if not m:
        return None, None
    try:
        return float(m.group(1).replace(",", "")), float(m.group(2).replace(",", ""))
    except ValueError:
        return None, None


def _base_total(cart: dict) -> float:
    total = 0.0
    for it in cart.get("items", []):
        total += float(it.get("price", 0.0)) * int(it.get("quantity", 0))
    return round(total, 2)


def _total_discount(used_coupons) -> float:
    total = 0.0
    for c in used_coupons or []:
        d, _ = _parse_coupon(c.get("coupon_name", ""))
        if d is not None:
            total += d * int(c.get("quantity", 0))
    return round(total, 2)


def _update_summary(cart: dict) -> None:
    items = cart.get("items", [])
    total_qty = sum(int(it.get("quantity", 0)) for it in items)
    base = _base_total(cart)
    final = max(0.0, base - _total_discount(cart.get("used_coupons")))
    cart.setdefault("summary", {})
    cart["summary"]["total_items_count"] = total_qty
    cart["summary"]["total_price"] = round(final, 2)


def run(coupon_name=None, quantity=1) -> str:
    if _db.case_dir() is None:
        return _db.db_missing_sentinel(NAME)
    if not coupon_name:
        return json.dumps({"error": "coupon_name is required"}, ensure_ascii=False)
    if not isinstance(quantity, (int, float)) or quantity <= 0:
        return json.dumps(
            {"error": "quantity must be a positive number"}, ensure_ascii=False
        )
    quantity = int(quantity)
    if coupon_name not in VALID_COUPONS:
        return json.dumps(
            {
                "error": (
                    f"Coupon not found: {coupon_name!r}. "
                    f"Valid coupons: {', '.join(VALID_COUPONS)}"
                )
            },
            ensure_ascii=False,
        )

    cart = _db.load_cart()
    used = cart.setdefault("used_coupons", [])
    idx = -1
    for i, c in enumerate(used):
        if c.get("coupon_name") == coupon_name:
            idx = i
            break
    if idx < 0:
        return json.dumps(
            {"error": f"Coupon not in cart: {coupon_name!r}"},
            ensure_ascii=False,
        )
    cur_qty = int(used[idx].get("quantity", 0))
    new_qty = max(0, cur_qty - quantity)
    if new_qty == 0:
        used.pop(idx)
    else:
        used[idx]["quantity"] = new_qty
    cart["used_coupons"] = used
    _update_summary(cart)
    _db.write_cart(cart)
    return json.dumps(cart, ensure_ascii=False)


register_tool(NAME, SCHEMA, run)
