"""Road route query tool — re-implementation of RoadRouteInfoQueryTool.
Reads ``transportation/distance_matrix.csv`` and returns distance,
duration, and cost between two coordinate strings.

Coordinates must be exact "latitude,longitude" strings with six decimal
places, matching values produced by ``search_location`` or other tools.
The tool validates that both endpoints exist in the database before
returning a result, mirroring the reference's coord-precision check.
"""
from __future__ import annotations

import json

from platform_core.tools import register_tool
from . import _csv

NAME = "query_road_route_info"

SCHEMA = {
    "name": NAME,
    "description": (
        "Query distance, duration, and cost between two coordinates "
        "(format 'latitude,longitude' with six decimal places). Both "
        "coordinates must come from another tool's output — manual entry "
        "or rounded values will be rejected."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "origin": {
                "type": "string",
                "description": "Origin coordinate, format 'latitude,longitude'.",
            },
            "destination": {
                "type": "string",
                "description": "Destination coordinate, format 'latitude,longitude'.",
            },
        },
        "required": ["origin", "destination"],
    },
}


def _coord_not_in_range(coord: str) -> str:
    return (
        f"Coordinate {coord} is not in query range, please check:\n"
        "1. Whether coordinate comes from valid tool query result, not "
        "manual input or fabrication;\n"
        "2. Whether coordinate precision is exactly consistent with query "
        "result, 6 decimal places"
    )


def _to_int(v: object) -> int:
    try:
        return int(float(v))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0


def run(origin: str, destination: str) -> str:
    rows, path = _csv.load_for_tool("transportation/distance_matrix.csv")
    if not rows:
        if path is None:
            return _csv.db_not_loaded_message("Road route")
        return f"No transportation information found from {origin} to {destination}"

    # Build the union of all coordinates that appear as either an origin
    # or a destination in the matrix. The reference does this to verify
    # that the agent didn't hallucinate or round a coordinate.
    coords: set[str] = set()
    for r in rows:
        o = r.get("origin", "")
        d = r.get("destination", "")
        if o:
            coords.add(o)
        if d:
            coords.add(d)
    if origin not in coords:
        return _coord_not_in_range(origin)
    if destination not in coords:
        return _coord_not_in_range(destination)

    matched = [
        r for r in rows
        if r.get("origin") == origin and r.get("destination") == destination
    ]
    if not matched:
        return f"No transportation information found from {origin} to {destination}"

    row = matched[0]
    result = {
        "origin": row.get("origin", origin),
        "destination": row.get("destination", destination),
        "distance_in_meters": _to_int(row.get("distance_meters", 0)),
        "duration_in_minutes": _to_int(row.get("duration_minutes", 0)),
        "cost": _to_int(row.get("cost", 0)),
    }
    return json.dumps(result, ensure_ascii=False, indent=2)


register_tool(NAME, SCHEMA, run)
