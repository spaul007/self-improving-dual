"""Location search tool — re-implementation of LocationSearchTool against
our framework's interface. Reads ``locations/locations_coords.csv``.
"""
from __future__ import annotations

import json

from platform_core.tools import register_tool
from . import _csv

NAME = "search_location"

SCHEMA = {
    "name": NAME,
    "description": (
        "Look up the latitude and longitude of a place by exact name. The "
        "place name must come from another tool's output verbatim — no "
        "abbreviations, no translation, no added words. Returns a JSON "
        "object with place_name, latitude, longitude (six decimal places)."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "place_name": {
                "type": "string",
                "description": "Place name, exactly as it appeared in another tool's output.",
            },
        },
        "required": ["place_name"],
    },
}


def run(place_name: str) -> str:
    rows, path = _csv.load_for_tool("locations/locations_coords.csv")
    if not rows:
        if path is None:
            return _csv.db_not_loaded_message("Location")
        return (
            f"Coordinate information not found for location {place_name}, "
            "please check: 1. Whether the place name comes from other tool "
            "results; 2. Whether the place name is exactly consistent with "
            "tool results, no abbreviation, renaming or additional "
            "description allowed"
        )

    # Reference falls back to 'place_name' if 'poi_name' is absent.
    col = "poi_name" if "poi_name" in rows[0] else "place_name"
    matched = [r for r in rows if r.get(col) == place_name]
    if not matched:
        return (
            f"Coordinate information not found for location {place_name}, "
            "please check: 1. Whether the place name comes from other tool "
            "results; 2. Whether the place name is exactly consistent with "
            "tool results, no abbreviation, renaming or additional "
            "description allowed"
        )

    row = matched[0]
    result = {
        "place_name": row.get(col, place_name),
        "latitude": str(row.get("latitude", "")),
        "longitude": str(row.get("longitude", "")),
    }
    return json.dumps(result, ensure_ascii=False, indent=2)


register_tool(NAME, SCHEMA, run)
