"""Attraction tools — query_attraction_details + recommend_attractions.

Re-implementation of attraction_query_tool.py against our framework's
interface. Two tools, one module — the registry treats them independently.
"""
from __future__ import annotations

from typing import Any

from platform_core.tools import register_tool
from . import _csv

# ---------------------------------------------------------------------------
# query_attraction_details
# ---------------------------------------------------------------------------

DETAILS_NAME = "query_attraction_details"

DETAILS_SCHEMA = {
    "name": DETAILS_NAME,
    "description": (
        "Query attraction details by name. Returns a formatted text block with "
        "id, name, city, address, coordinates, description, rating, opening "
        "hours, closing days, recommended visit duration, ticket price and "
        "attraction type."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "attraction_name": {"type": "string"},
        },
        "required": ["attraction_name"],
    },
}


def _to_num(v: Any) -> float | None:
    try:
        f = float(v)
        if f != f:  # NaN check
            return None
        return f
    except (TypeError, ValueError):
        return None


def details_run(attraction_name: str) -> str:
    rows, path = _csv.load_for_tool("attractions/attractions.csv")
    if not rows:
        if path is None:
            return _csv.db_not_loaded_message("Attraction")
        return f"Detailed information not found for attraction {attraction_name}"

    matches = [r for r in rows if r.get("attraction_name") == attraction_name]
    if not matches:
        return f"Detailed information not found for attraction {attraction_name}"
    row = matches[0]

    rating_val = _to_num(row.get("rating"))
    min_h = _to_num(row.get("min_visit_hours"))
    max_h = _to_num(row.get("max_visit_hours"))
    ticket = _to_num(row.get("ticket_price"))

    opening = row.get("opening_time", "") or ""
    closing = row.get("closing_time", "") or ""
    hours_line = (
        f"Opening Hours: {opening}"
        if opening == closing
        else f"Opening Hours: {opening} to {closing}"
    )

    lines = [
        f"Attraction ID: {row.get('attraction_id', '')}",
        f"Attraction Name: {row.get('attraction_name', attraction_name)}",
        f"City: {row.get('city', '')}",
        f"Address: {row.get('address', '')}",
        f"Coordinates: Latitude {row.get('latitude', '')}, Longitude {row.get('longitude', '')}",
        f"Description: {row.get('description', '')}",
        f"Rating: {rating_val if rating_val is not None else ''} (average visitor rating)",
        hours_line,
        f"Closed Dates: {row.get('closing_dates', '')}",
        f"Minimum Visit Duration: {min_h if min_h is not None else ''} hours",
        f"Maximum Visit Duration: {max_h if max_h is not None else ''} hours",
        f"Ticket Price: {ticket if ticket is not None else 0} RMB",
        f"Attraction Type: {row.get('attraction_type', '')}",
    ]
    return "\n".join(lines)


register_tool(DETAILS_NAME, DETAILS_SCHEMA, details_run)


# ---------------------------------------------------------------------------
# recommend_attractions
# ---------------------------------------------------------------------------

RECOMMEND_NAME = "recommend_attractions"

RECOMMEND_SCHEMA = {
    "name": RECOMMEND_NAME,
    "description": (
        "Recommend attractions for a given city, optionally filtered by type. "
        "Returns a multi-line text listing attraction names, descriptions and "
        "types. All attractions in the final plan must come from this tool."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "City name or keyword."},
            "attraction_type": {
                "type": "string",
                "description": (
                    "Optional filter. Options: 'Historical and Cultural', "
                    "'Natural Scenery', 'Art Exhibition', 'City Landmark', "
                    "'Theme Park', 'Leisure Experience'."
                ),
            },
        },
        "required": ["city"],
    },
}


def recommend_run(city: str, attraction_type: str = "") -> str:
    rows, path = _csv.load_for_tool("attractions/attractions.csv")
    if not rows:
        if path is None:
            return _csv.db_not_loaded_message("Attraction")
        return "No attraction recommendations found"

    if attraction_type:
        rows = [r for r in rows if r.get("attraction_type") == attraction_type]
    if not rows:
        return "No attraction recommendations found"

    out = ["Recommended attractions:"]
    for r in rows:
        name = r.get("attraction_name", "")
        desc = r.get("description", "")
        atype = r.get("attraction_type", "")
        out.append(f"{name}, {desc}. This is a {atype} type attraction")
    return "\n".join(out)


register_tool(RECOMMEND_NAME, RECOMMEND_SCHEMA, recommend_run)
