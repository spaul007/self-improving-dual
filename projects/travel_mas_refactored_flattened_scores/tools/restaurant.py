"""Restaurant tools — recommend_restaurants + query_restaurant_details.

Re-implementation of restaurant_query_tool.py against our framework's
interface.
"""
from __future__ import annotations

import json
from typing import Any

from platform_core.tools import register_tool
from . import _csv

# ---------------------------------------------------------------------------
# recommend_restaurants
# ---------------------------------------------------------------------------

RECOMMEND_NAME = "recommend_restaurants"

RECOMMEND_SCHEMA = {
    "name": RECOMMEND_NAME,
    "description": (
        "Recommend restaurants near a given coordinate. Latitude and longitude "
        "must be exact six-decimal values that came from another tool. Returns "
        "a JSON list of restaurants with name, coordinates, price-per-person, "
        "cuisine, opening hours, nearby attraction, and rating."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "latitude": {"type": "string", "description": "Latitude, 6 decimal places."},
            "longitude": {"type": "string", "description": "Longitude, 6 decimal places."},
        },
        "required": ["latitude", "longitude"],
    },
}


def _str(v: Any) -> str:
    return "" if v is None else str(v)


def recommend_run(latitude: str, longitude: str) -> str:
    rows, path = _csv.load_for_tool("restaurants/restaurants.csv")
    if not rows:
        if path is None:
            return _csv.db_not_loaded_message("Restaurant")
        return (
            f"No recommended restaurants found near coordinates "
            f"({latitude}, {longitude}), please check coordinate source"
        )

    if "query_latitude" in rows[0] and "query_longitude" in rows[0]:
        matched = [
            r for r in rows
            if str(r.get("query_latitude")) == str(latitude)
            and str(r.get("query_longitude")) == str(longitude)
        ]
    else:
        matched = []

    if not matched:
        return (
            f"No recommended restaurants found near coordinates "
            f"({latitude}, {longitude}), please check coordinate source"
        )

    results: list[dict[str, Any]] = []
    for row in matched:
        result: dict[str, Any] = {
            "name": _str(row.get("restaurant_name", "")),
            "latitude": _str(row.get("latitude", "0")),
            "longitude": _str(row.get("longitude", "0")),
            "price_per_person": _str(row.get("price_per_person", "0")),
            "cuisine": _str(row.get("cuisine", "")),
            "opening_time": _str(row.get("opening_time", "")),
            "closing_time": _str(row.get("closing_time", "")),
            "nearby_attraction_name": _str(row.get("nearby_attraction_name", "")),
            "rating": _str(row.get("rating", "4.5")),
        }
        tags = row.get("tags")
        if isinstance(tags, str) and tags.strip():
            result["tags"] = tags.split(";")
        results.append(result)

    return json.dumps(results, ensure_ascii=False, indent=2)


register_tool(RECOMMEND_NAME, RECOMMEND_SCHEMA, recommend_run)


# ---------------------------------------------------------------------------
# query_restaurant_details
# ---------------------------------------------------------------------------

DETAILS_NAME = "query_restaurant_details"

DETAILS_SCHEMA = {
    "name": DETAILS_NAME,
    "description": (
        "Look up a restaurant by exact name. Returns a JSON object with the "
        "restaurant's id, coordinates, price per person, cuisine, opening "
        "hours, nearby attraction, and rating."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "restaurant_name": {"type": "string"},
        },
        "required": ["restaurant_name"],
    },
}


def details_run(restaurant_name: str) -> str:
    # Reference restaurant_query_tool.py:153-169 returns error envelopes
    # via format_result_as_json (which uses ensure_ascii=False, indent=2).
    # Match that shape so error responses look identical to success ones.
    rows, path = _csv.load_for_tool("restaurants/restaurants.csv")
    if not rows:
        if path is None:
            return json.dumps(
                {
                    "message": _csv.db_not_loaded_message("Restaurant"),
                    "restaurant_name": restaurant_name,
                },
                ensure_ascii=False,
                indent=2,
            )
        return json.dumps(
            {
                "message": f"Detailed information not found for restaurant {restaurant_name}",
                "restaurant_name": restaurant_name,
            },
            ensure_ascii=False,
            indent=2,
        )

    matched = [r for r in rows if r.get("restaurant_name") == restaurant_name]
    if not matched:
        return json.dumps(
            {
                "message": f"Detailed information not found for restaurant {restaurant_name}",
                "restaurant_name": restaurant_name,
            },
            ensure_ascii=False,
            indent=2,
        )
    row = matched[0]
    result: dict[str, Any] = {
        "id": _str(row.get("restaurant_id", "")),
        "name": _str(row.get("restaurant_name", restaurant_name)),
        "latitude": _str(row.get("latitude", "0")),
        "longitude": _str(row.get("longitude", "0")),
        "price_per_person": _str(row.get("price_per_person", "100")),
        "cuisine": _str(row.get("cuisine", "")),
        "opening_time": _str(row.get("opening_time", "")),
        "closing_time": _str(row.get("closing_time", "")),
        "nearby_attraction_name": _str(row.get("nearby_attraction_name", "")),
        "rating": _str(row.get("rating", "4.0")),
    }
    tags = row.get("tags")
    if isinstance(tags, str) and tags.strip():
        result["tags"] = tags.split(";")
    return json.dumps(result, ensure_ascii=False, indent=2)


register_tool(DETAILS_NAME, DETAILS_SCHEMA, details_run)
