"""Shared per-case database access for shopping tools.

The shopping benchmark has 120 cases split across three difficulty
levels. Each case has its own directory under
``projects/shopping/data/database_level{1,2,3}/case_{N}/`` containing:

  - ``products.jsonl`` — product catalog (read-only)
  - ``user_info.json`` — user profile, including owned coupons (read-only)
  - ``validation_cases.json`` — ground truth (read by the scorer)
  - ``cart.json`` — the agent's shopping cart. **Mutated by tools.**

Tools resolve their per-case directory via three environment variables
set by the evaluator's case-env merge (see ``meta_agent/evaluator.py``):

  - ``SHOPPING_DATABASE_ROOT`` — root of the data tree. Defaults to
    ``projects/shopping/data/`` (project-relative, same convention
    travel uses).
  - ``SHOPPING_LEVEL`` — 1 / 2 / 3.
  - ``SHOPPING_SAMPLE_ID`` — the case's numeric id within its level.

When any of these is missing the helpers return safe empty values so
the agent loop still terminates with a "no data" sentinel rather than
crashing.
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any, Optional

DB_ROOT_ENV = "SHOPPING_DATABASE_ROOT"
LEVEL_ENV = "SHOPPING_LEVEL"
SAMPLE_ID_ENV = "SHOPPING_SAMPLE_ID"

# Module-level caches: products and user_info are read-only and
# expensive to deserialize; cart.json is mutated so it's never cached.
_PRODUCTS_CACHE: dict[str, list[dict]] = {}
_USER_CACHE: dict[str, dict] = {}
_VALIDATION_CACHE: dict[str, dict] = {}
_LOCK = threading.Lock()


def database_root() -> Optional[Path]:
    """Return the shopping data root.

    Order of precedence:
      1. ``$SHOPPING_DATABASE_ROOT`` if set (YAML ``env:`` override).
      2. The project-relative default ``projects/shopping/data``
         when that directory exists.
      3. ``None`` — tools surface a "database not loaded" sentinel.
    """
    val = os.environ.get(DB_ROOT_ENV)
    if val:
        return Path(val)
    # ``_db.py`` lives at projects/shopping/tools/_db.py; repo root is
    # three parents up (matching travel's _csv.py pattern).
    candidate = Path(__file__).resolve().parents[3] / "projects" / "shopping" / "data"
    return candidate if candidate.exists() else None


def level() -> Optional[str]:
    return os.environ.get(LEVEL_ENV)


def sample_id() -> Optional[str]:
    return os.environ.get(SAMPLE_ID_ENV)


def case_dir() -> Optional[Path]:
    """Resolve the per-case directory or return None when the env is
    incomplete."""
    root = database_root()
    lvl = level()
    sid = sample_id()
    if root is None or not lvl or not sid:
        return None
    return root / f"database_level{lvl}" / f"case_{sid}"


# --------------------------------------------------------------------- #
# Reads (cached)
# --------------------------------------------------------------------- #


def load_products() -> list[dict]:
    """Read ``products.jsonl`` from the per-case dir. Cached per-process."""
    cd = case_dir()
    if cd is None:
        return []
    key = str(cd / "products.jsonl")
    with _LOCK:
        cached = _PRODUCTS_CACHE.get(key)
        if cached is not None:
            return cached
    products: list[dict] = []
    path = cd / "products.jsonl"
    if not path.exists():
        with _LOCK:
            _PRODUCTS_CACHE[key] = products
        return products
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                products.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    with _LOCK:
        _PRODUCTS_CACHE[key] = products
    return products


def products_by_id() -> dict[str, dict]:
    """Return a {product_id: product} index over the per-case catalog."""
    return {p.get("product_id"): p for p in load_products() if p.get("product_id")}


def load_user() -> dict:
    cd = case_dir()
    if cd is None:
        return {}
    key = str(cd / "user_info.json")
    with _LOCK:
        cached = _USER_CACHE.get(key)
        if cached is not None:
            return cached
    path = cd / "user_info.json"
    if not path.exists():
        with _LOCK:
            _USER_CACHE[key] = {}
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        data = {}
    with _LOCK:
        _USER_CACHE[key] = data
    return data


def load_validation() -> dict:
    """Read ``validation_cases.json`` — only the scorer should call this."""
    cd = case_dir()
    if cd is None:
        return {}
    key = str(cd / "validation_cases.json")
    with _LOCK:
        cached = _VALIDATION_CACHE.get(key)
        if cached is not None:
            return cached
    path = cd / "validation_cases.json"
    if not path.exists():
        with _LOCK:
            _VALIDATION_CACHE[key] = {}
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        data = {}
    with _LOCK:
        _VALIDATION_CACHE[key] = data
    return data


# --------------------------------------------------------------------- #
# Cart (read-write, never cached)
# --------------------------------------------------------------------- #


def _empty_cart(user: Optional[dict] = None) -> dict:
    user = user or {}
    return {
        "user_id": user.get("user_id", ""),
        "user_name": user.get("username") or user.get("user_name", ""),
        "items": [],
        "used_coupons": [],
        "summary": {"total_items_count": 0, "total_price": 0.0},
    }


def load_cart() -> dict:
    cd = case_dir()
    if cd is None:
        return _empty_cart()
    path = cd / "cart.json"
    if not path.exists():
        return _empty_cart(load_user())
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return _empty_cart(load_user())
    if not isinstance(data, dict):
        return _empty_cart(load_user())
    # Defensive: backfill any missing fields.
    data.setdefault("items", [])
    data.setdefault("used_coupons", [])
    data.setdefault(
        "summary", {"total_items_count": 0, "total_price": 0.0}
    )
    return data


def write_cart(cart: dict) -> None:
    cd = case_dir()
    if cd is None:
        return
    cd.mkdir(parents=True, exist_ok=True)
    path = cd / "cart.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(cart, fh, ensure_ascii=False, indent=2)


def reset_cart() -> None:
    """Restore the cart to its initial empty shape (user info preserved).

    Called by the seed at the top of ``run_task`` so multi-round
    meta-agent loops start from a clean state. Decided 2026-05-13.
    """
    write_cart(_empty_cart(load_user()))


def update_cart_summary(cart: dict) -> dict:
    """Recompute total_items_count and total_price on the items list. The
    summary is what the scorer never reads (it inspects items + coupons
    directly), but the model uses it as a sanity check via
    ``get_cart_info``.
    """
    items = cart.get("items", [])
    total_qty = sum(int(it.get("quantity", 0)) for it in items)
    total_price = sum(
        float(it.get("price", 0.0)) * int(it.get("quantity", 0)) for it in items
    )
    summary = cart.setdefault("summary", {})
    summary["total_items_count"] = total_qty
    summary["total_price"] = round(total_price, 2)
    return cart


def db_missing_sentinel(tool_name: str) -> str:
    """Standard message tools return when the env isn't wired up."""
    return json.dumps(
        {
            "error": (
                f"Shopping database not loaded for tool {tool_name!r}. "
                f"Need {DB_ROOT_ENV}+{LEVEL_ENV}+{SAMPLE_ID_ENV} env vars "
                "OR projects/shopping/data/ populated."
            )
        },
        ensure_ascii=False,
    )
