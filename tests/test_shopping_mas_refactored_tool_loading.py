"""Tests for projects/shopping_mas_refactored/shopping_mas/tools/loader.py's
load-bearing assertions, added on request after a confusing but harmless
"[tools] skipped ... ModuleNotFoundError" log (from platform_core.tools'
own, unrelated generic tool-discovery scanner -- see tools/loader.py's own
ALL_TOOL_NAMES comment) raised the question of whether this project's REAL
tool-loading path (this file) was actually working.

Real gap this closes: tool_registry()/make_handlers()/openai_tools()
previously degraded SILENTLY on a missing tool (a dict-skip, not an
error) -- a tool module that imports cleanly but never calls
@register_tool (a bug in the vendored class body), or a typo in
CATALOG_TOOLS/CART_TOOLS, would hand the agent an incomplete tool set
with no error anywhere, surfacing only much later as a confusing
"unknown tool" deep inside llm_client.py's dispatch loop.

    PYTHONPATH=. python3 -m unittest tests.test_shopping_mas_refactored_tool_loading
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SEED_DIR = REPO_ROOT / "projects" / "shopping_mas_refactored" / "shopping_mas"

_SEED_MODULE_NAMES = ["tools", "tools.loader", "base_shopping_tool"] + [
    f"{m}" for m in [
        "search_products_tool", "filter_by_brand_tool", "filter_by_color_tool",
        "filter_by_size_tool", "filter_by_applicable_coupons_tool",
        "filter_by_range_tool", "sort_product_tool", "get_product_details_tool",
        "calculate_transport_time_tool", "get_user_info", "get_cart_info",
        "add_product_to_cart", "delete_product_from_cart", "add_coupon_to_cart",
        "delete_coupon_from_cart",
    ]
]


class ToolLoadingTests(unittest.TestCase):
    def setUp(self) -> None:
        for name in _SEED_MODULE_NAMES:
            sys.modules.pop(name, None)
        sys.path.insert(0, str(SEED_DIR))
        self.addCleanup(lambda: sys.path.remove(str(SEED_DIR)))
        self.addCleanup(lambda: [sys.modules.pop(n, None) for n in _SEED_MODULE_NAMES])

        from tools import loader
        self.loader = loader
        self.loader._registry = None  # fresh load per test, not the process cache

    def test_all_fifteen_tools_actually_register(self) -> None:
        registry = self.loader.tool_registry()
        self.assertEqual(set(registry.keys()), self.loader.ALL_TOOL_NAMES)

    def test_make_toolset_instantiates_every_tool(self) -> None:
        toolset = self.loader.make_toolset(tempfile.mkdtemp())
        self.assertEqual(set(toolset.keys()), self.loader.ALL_TOOL_NAMES)
        self.assertEqual(len(self.loader.make_handlers(toolset, self.loader.CATALOG_TOOLS)), 9)
        self.assertEqual(len(self.loader.make_handlers(toolset, self.loader.CART_TOOLS)), 5)

    def test_make_handlers_asserts_on_a_missing_tool_name(self) -> None:
        toolset = self.loader.make_toolset(tempfile.mkdtemp())
        with self.assertRaises(AssertionError) as ctx:
            self.loader.make_handlers(toolset, ["not_a_real_tool"])
        self.assertIn("not_a_real_tool", str(ctx.exception))

    def test_openai_tools_asserts_on_a_missing_schema_name(self) -> None:
        with self.assertRaises(AssertionError) as ctx:
            self.loader.openai_tools(["not_a_real_tool"])
        self.assertIn("not_a_real_tool", str(ctx.exception))

    def test_openai_tools_returns_a_schema_per_requested_catalog_tool(self) -> None:
        schemas = self.loader.openai_tools(self.loader.CATALOG_TOOLS)
        self.assertEqual(len(schemas), 9)
        self.assertEqual(
            {s["function"]["name"] for s in schemas}, set(self.loader.CATALOG_TOOLS)
        )


if __name__ == "__main__":
    unittest.main()
