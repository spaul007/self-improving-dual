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

# Every tool name this project ever requests by name, anywhere -- CATALOG_TOOLS
# + CART_TOOLS, plus "get_user_info" (read directly via toolset["get_user_info"]
# in mas_workflow.py::_fetch_user_info, never through make_handlers). Used
# below as the hard floor tool_registry() must clear -- confirmed live this
# session: platform_core.tools' own generic tool-discovery scanner (a
# DIFFERENT, unrelated mechanism -- see tool_source_dirs in this project's
# configs) logs a "[tools] skipped ... ModuleNotFoundError" per file for
# every one of these modules, because it tries to import each tools/immutable/
# *.py file in isolation rather than through this loader's own sys.path
# setup below -- that warning is harmless noise from a path this project
# doesn't use, but it looks alarming enough that an explicit, loud assertion
# on the path that actually matters (this one) is worth having rather than
# trusting a human to tell the two apart from a log.
ALL_TOOL_NAMES = frozenset(CATALOG_TOOLS) | frozenset(CART_TOOLS) | {"get_user_info"}

_import_lock = threading.Lock()
_registry = None


def tool_registry() -> dict:
    """Import the benchmark tool modules once and return TOOL_REGISTRY.

    Asserts every name in ALL_TOOL_NAMES actually registered -- a tool
    module can import cleanly (no ImportError) while still failing to
    call @register_tool (e.g. a bug in the vendored class body), which
    would otherwise surface only much later as a silently-missing tool
    handler (see make_handlers/openai_tools below)."""
    global _registry
    with _import_lock:
        if _registry is None:
            if str(IMMUTABLE_DIR) not in sys.path:
                sys.path.insert(0, str(IMMUTABLE_DIR))
            base = importlib.import_module("base_shopping_tool")
            for mod in _TOOL_MODULES:
                importlib.import_module(mod)
            registry = base.TOOL_REGISTRY
            missing = ALL_TOOL_NAMES - set(registry.keys())
            assert not missing, (
                f"tools/loader.py: {sorted(missing)} never registered into "
                f"base_shopping_tool.TOOL_REGISTRY after importing all of "
                f"{_TOOL_MODULES} -- a tool module imported without error "
                f"but its @register_tool call never ran (registry has: "
                f"{sorted(registry.keys())})"
            )
            _registry = registry
    return _registry


def make_toolset(database_path: str | Path) -> dict:
    """Instantiate every benchmark tool against one case's database dir."""
    cfg = {"database_path": str(database_path)}
    registry = tool_registry()
    toolset = {name: cls(cfg=cfg) for name, cls in registry.items()}
    missing = ALL_TOOL_NAMES - set(toolset.keys())
    assert not missing, (
        f"tools/loader.py: {sorted(missing)} missing from the instantiated "
        f"toolset (registry had: {sorted(registry.keys())})"
    )
    return toolset


def make_handlers(toolset: dict, names: list[str]) -> dict:
    """Handlers for the LLM tool-dispatch loop: name -> fn(raw_json) -> str.

    Asserts every requested name is present in `toolset` -- silently
    dropping one here would hand the agent an incomplete tool schema with
    no matching handler (or vice versa), a load-bearing mismatch that
    would otherwise only surface as a confusing "unknown tool" error deep
    inside llm_client.py's tool-dispatch loop."""
    missing = [n for n in names if n not in toolset]
    assert not missing, f"tools/loader.py: requested tool(s) {missing} not in toolset"

    def bind(inst):
        return lambda raw_args: inst.call(raw_args)
    return {name: bind(toolset[name]) for name in names}


def openai_tools(names: list[str]) -> list[dict]:
    """OpenAI function-calling schemas for a subset of tools.

    Asserts every requested name has a schema -- same reasoning as
    make_handlers above; a name missing here would silently offer the
    LLM fewer tools than the agent's own code assumes it has."""
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        schemas = json.load(f)
    by_name = {s["function"]["name"]: s for s in schemas
               if isinstance(s, dict) and s.get("type") == "function"}
    missing = [n for n in names if n not in by_name]
    assert not missing, (
        f"tools/loader.py: requested tool schema(s) {missing} not found in "
        f"{SCHEMA_PATH} (has: {sorted(by_name.keys())})"
    )
    return [by_name[n] for n in names]
