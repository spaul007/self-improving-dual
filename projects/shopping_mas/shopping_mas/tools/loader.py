"""Loader for the vendored benchmark tools in tools/immutable/.

The benchmark tool modules use flat imports (`from base_shopping_tool import
...`) and register themselves into base_shopping_tool.TOOL_REGISTRY via the
@register_tool decorator at import time, so tools/immutable/ is put on
sys.path and the modules are imported once per process. Tool *instances* are
per-case: each is constructed with cfg={"database_path": <case dir>} and
caches that case's products/cart/user files.

This file is MAS infrastructure, not part of the benchmark contract; the
files in tools/immutable/ are vendored verbatim and must not be modified.
"""

import importlib
import json
import sys
import threading
from pathlib import Path

IMMUTABLE_DIR = Path(__file__).resolve().parent / "immutable"
SCHEMA_PATH = IMMUTABLE_DIR / "shopping_tool_schema.json"

_TOOL_MODULES = [
    "search_products_tool",
    "filter_by_brand_tool",
    "filter_by_color_tool",
    "filter_by_size_tool",
    "filter_by_applicable_coupons_tool",
    "filter_by_range_tool",
    "sort_product_tool",
    "get_product_details_tool",
    "calculate_transport_time_tool",
    "get_user_info",
    "get_cart_info",
    "add_product_to_cart",
    "delete_product_from_cart",
    "add_coupon_to_cart",
    "delete_coupon_from_cart",
]

# Read-only catalog tools offered to the product scout agent.
CATALOG_TOOLS = [
    "search_products",
    "filter_by_brand",
    "filter_by_color",
    "filter_by_size",
    "filter_by_applicable_coupons",
    "filter_by_range",
    "sort_products",
    "get_product_details",
    "calculate_transport_time",
]

# Cart-mutating tools offered to the cart executor agent.
CART_TOOLS = [
    "add_product_to_cart",
    "delete_product_from_cart",
    "add_coupon_to_cart",
    "delete_coupon_from_cart",
    "get_cart_info",
]

_import_lock = threading.Lock()
_registry = None


def tool_registry() -> dict:
    """Import the benchmark tool modules once and return TOOL_REGISTRY."""
    global _registry
    with _import_lock:
        if _registry is None:
            if str(IMMUTABLE_DIR) not in sys.path:
                sys.path.insert(0, str(IMMUTABLE_DIR))
            base = importlib.import_module("base_shopping_tool")
            for mod in _TOOL_MODULES:
                importlib.import_module(mod)
            _registry = base.TOOL_REGISTRY
    return _registry


def make_toolset(database_path: str | Path) -> dict:
    """Instantiate every benchmark tool against one case's database dir."""
    cfg = {"database_path": str(database_path)}
    return {name: cls(cfg=cfg) for name, cls in tool_registry().items()}


def make_handlers(toolset: dict, names: list[str]) -> dict:
    """Handlers for the LLM tool-dispatch loop: name -> fn(raw_json) -> str."""
    def bind(inst):
        return lambda raw_args: inst.call(raw_args)
    return {name: bind(toolset[name]) for name in names if name in toolset}


def openai_tools(names: list[str]) -> list[dict]:
    """OpenAI function-calling schemas for a subset of tools."""
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        schemas = json.load(f)
    by_name = {s["function"]["name"]: s for s in schemas
               if isinstance(s, dict) and s.get("type") == "function"}
    return [by_name[n] for n in names if n in by_name]
