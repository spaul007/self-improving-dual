"""BM25 semantic search over the per-case product catalog.

Ported from `/users/n.tzou/cl/shopping_agent/tools/search_products_tool.py`.
"""
from __future__ import annotations

import json
import threading
from typing import Any

from platform_core.tools import register_tool
from . import _db

NAME = "search_products"

SCHEMA = {
    "name": NAME,
    "description": (
        "Handles broad, open-ended natural language queries. Performs a "
        "semantic (BM25) search on key product information (brand, color, "
        "size, material, season, demographics, and name) and returns an "
        "initial set of relevant product_ids."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural language query, e.g. 'Nike running shoes' or 'orange Uniqlo T-shirt'.",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of product_ids to return (default 20).",
            },
        },
        "required": ["query"],
    },
}

_BM25_SCORE_THRESHOLD = 2
_INDEX_CACHE: dict[str, tuple[Any, list[dict]]] = {}
_INDEX_LOCK = threading.Lock()


def _searchable_text(product: dict) -> str:
    return " ".join(
        str(product.get(field, ""))
        for field in (
            "brand",
            "color",
            "size",
            "thickness",
            "elasticity",
            "version_type",
            "collar_type",
            "suitable_season",
            "target_demographic",
            "name",
        )
    )


def _get_index() -> tuple[Any, list[dict]]:
    """Return (BM25Okapi, products). Cached per case directory."""
    cd = _db.case_dir()
    key = str(cd) if cd else ""
    with _INDEX_LOCK:
        cached = _INDEX_CACHE.get(key)
        if cached is not None:
            return cached
    products = _db.load_products()
    if not products:
        with _INDEX_LOCK:
            _INDEX_CACHE[key] = (None, [])
        return None, []
    try:
        from rank_bm25 import BM25Okapi  # type: ignore
    except ImportError:
        BM25Okapi = None  # type: ignore
    if BM25Okapi is None:
        with _INDEX_LOCK:
            _INDEX_CACHE[key] = (None, products)
        return None, products
    corpus = [_searchable_text(p).lower().split() for p in products]
    index = BM25Okapi(corpus)
    with _INDEX_LOCK:
        _INDEX_CACHE[key] = (index, products)
    return index, products


def run(query: str = "", limit: int = 20) -> str:
    if not query:
        return json.dumps({"product_ids": []}, ensure_ascii=False)
    bm25, products = _get_index()
    if bm25 is None or not products:
        return _db.db_missing_sentinel(NAME)
    scores = bm25.get_scores(query.lower().split())
    hits = [
        (products[i].get("product_id"), float(scores[i]))
        for i in range(len(products))
        if scores[i] > _BM25_SCORE_THRESHOLD and products[i].get("product_id")
    ]
    hits.sort(key=lambda t: t[1], reverse=True)
    return json.dumps(
        {"product_ids": [pid for pid, _ in hits[: int(limit) if limit else 20]]},
        ensure_ascii=False,
    )


register_tool(NAME, SCHEMA, run)
