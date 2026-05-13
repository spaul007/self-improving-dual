"""Immutable shopping tools.

Each sub-module exports ``NAME``, ``SCHEMA``, ``run(**kwargs) -> str`` and
calls :func:`platform_core.tools.register_tool` at import time. This
package's ``__init__`` triggers the auto-discovery side effect by
importing every sub-module.

Per-case data access uses three environment variables (set per-case
by the benchmark's ``env:`` block):

* ``SHOPPING_DATABASE_ROOT`` — root of the data tree. Defaults to
  ``projects/shopping/data/``.
* ``SHOPPING_LEVEL`` — difficulty level (1 / 2 / 3).
* ``SHOPPING_SAMPLE_ID`` — case id within the level.

Tools resolve their per-case directory via :mod:`._db`. When the env
is incomplete, helpers return a "database not loaded" sentinel JSON
so the agent loop still terminates.
"""
from __future__ import annotations

# Importing the modules registers each tool via register_tool() side effects.
from . import (  # noqa: F401
    search_products,
    filter_by_brand,
    filter_by_color,
    filter_by_size,
    filter_by_range,
    filter_by_applicable_coupons,
    sort_products,
    get_product_details,
    calculate_transport_time,
    get_user_info,
    add_product_to_cart,
    delete_product_from_cart,
    get_cart_info,
    add_coupon_to_cart,
    delete_coupon_from_cart,
)
