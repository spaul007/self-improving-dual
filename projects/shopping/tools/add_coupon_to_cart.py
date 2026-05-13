"""add_coupon_to_cart — validate user ownership + threshold + VIP status,
mutate `used_coupons`, and recompute the post-discount summary price."""
from __future__ import annotations

import json
import re
from typing import Optional, Tuple

from platform_core.tools import register_tool
from . import _db

NAME = "add_coupon_to_cart"

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
        "Adds a coupon to the cart. Validates: coupon name is recognized; "
        "user owns enough copies of it; VIP coupons require VIP status; the "
        "combined threshold across all used coupons is met by the cart "
        "total. Returns the updated cart with `used_coupons` and a post-"
        "discount `summary.total_price`."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "coupon_name": {
                "type": "string",
                "description": (
                    "Coupon name. Examples: 'Cross-store: ¥30 off every ¥300', "
                    "'Same-brand: ¥25 off every ¥200', 'VIP: ¥200 off every ¥1,000'."
                ),
            },
            "quantity": {
                "type": "integer",
                "description": "Optional. Positive integer; default 1.",
            },
        },
        "required": ["coupon_name"],
    },
}

_COUPON_RE = re.compile(r"¥([\d,]+)\s+off\s+every\s+¥([\d,]+)")


def _parse_coupon(coupon_name: str) -> Tuple[Optional[float], Optional[float]]:
    m = _COUPON_RE.search(coupon_name or "")
    if not m:
        return None, None
    try:
        return float(m.group(1).replace(",", "")), float(m.group(2).replace(",", ""))
    except ValueError:
        return None, None


def _base_total(cart: dict) -> float:
    items = cart.get("items", [])
    total = 0.0
    for it in items:
        price = float(it.get("price", 0.0))
        qty = int(it.get("quantity", 0))
        total += price * qty
    return round(total, 2)


def _total_discount(used_coupons) -> float:
    total = 0.0
    for c in used_coupons or []:
        d, _ = _parse_coupon(c.get("coupon_name", ""))
        if d is not None:
            total += d * int(c.get("quantity", 0))
    return round(total, 2)


def _validate_combination(base_total: float, used_coupons) -> Tuple[bool, str]:
    # Sum threshold * quantity across all DISTINCT coupon entries; the cart
    # total must cover the combined threshold.
    usage: dict[str, int] = {}
    for c in used_coupons or []:
        name = c.get("coupon_name", "")
        usage[name] = usage.get(name, 0) + int(c.get("quantity", 0))
    required = 0.0
    for name, qty in usage.items():
        d, thr = _parse_coupon(name)
        if d is None or thr is None:
            return False, f"Invalid coupon format: {name}"
        required += thr * qty
    if base_total < required:
        return False, (
            f"Cart total {base_total} is insufficient for this combination of "
            f"coupons (requires at least {required})"
        )
    return True, ""


def _update_summary(cart: dict) -> None:
    items = cart.get("items", [])
    total_qty = sum(int(it.get("quantity", 0)) for it in items)
    base = _base_total(cart)
    final = max(0.0, base - _total_discount(cart.get("used_coupons")))
    cart.setdefault("summary", {})
    cart["summary"]["total_items_count"] = total_qty
    cart["summary"]["total_price"] = round(final, 2)


def run(coupon_name=None, quantity=1) -> str:
    user = _db.load_user()
    if not user and _db.case_dir() is None:
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
    user_coupons = user.get("coupons", {}) or {}
    owned = int(user_coupons.get(coupon_name, 0))
    cart = _db.load_cart()
    used = cart.setdefault("used_coupons", [])
    currently_used = sum(
        int(c.get("quantity", 0)) for c in used if c.get("coupon_name") == coupon_name
    )
    if currently_used + quantity > owned:
        return json.dumps(
            {
                "error": (
                    f"Insufficient coupon quantity: User owns {owned} of "
                    f"{coupon_name!r}, cart already uses {currently_used}, "
                    f"cannot add {quantity} more"
                )
            },
            ensure_ascii=False,
        )
    if coupon_name.startswith("VIP:") and not user.get("is_vip", False):
        return json.dumps(
            {
                "error": (
                    f"VIP coupon {coupon_name!r} requires VIP status, but "
                    "user is not a VIP"
                )
            },
            ensure_ascii=False,
        )

    # Tentatively update used coupons and validate the combined threshold.
    rolled_back_from = None
    found = False
    for c in used:
        if c.get("coupon_name") == coupon_name:
            rolled_back_from = c["quantity"]
            c["quantity"] = currently_used + quantity
            found = True
            break
    if not found:
        used.append({"coupon_name": coupon_name, "quantity": quantity})

    ok, err = _validate_combination(_base_total(cart), used)
    if not ok:
        # Roll back.
        if found:
            for c in used:
                if c.get("coupon_name") == coupon_name:
                    c["quantity"] = rolled_back_from
                    break
        else:
            used.pop()
        cart["used_coupons"] = used
        return json.dumps({"error": err}, ensure_ascii=False)

    _update_summary(cart)
    _db.write_cart(cart)
    return json.dumps(cart, ensure_ascii=False)


register_tool(NAME, SCHEMA, run)
