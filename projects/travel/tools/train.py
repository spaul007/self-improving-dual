"""Train query tool — re-implementation of TrainQueryTool."""
from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

from platform_core.tools import register_tool
from . import _csv

NAME = "query_train_info"

SCHEMA = {
    "name": NAME,
    "description": (
        "Query train ticket information by origin city, destination city, "
        "and departure date (and optional seat class). Returns candidate "
        "train routes with segment details. Train-related details in the "
        "plan must come exclusively from this tool."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "origin": {"type": "string", "description": "Origin city name."},
            "destination": {"type": "string", "description": "Destination city name."},
            "depDate": {
                "type": "string",
                "description": "Departure date, format YYYY-MM-DD.",
            },
            "seatClassName": {
                "type": "string",
                "description": "Optional seat class: First Class Seat, Second Class Seat, Business Seat.",
            },
        },
        "required": ["origin", "destination", "depDate"],
    },
}


def _segment(
    idx: int,
    row: dict[str, str],
    prev_row: dict[str, str] | None,
) -> dict[str, Any]:
    seat_status = (row.get("seat_status") or "").strip()
    if not seat_status or seat_status.lower() == "nan":
        seat_status = "Available"
    try:
        duration = int(float(row.get("duration") or 0))
    except ValueError:
        duration = 0
    # Match reference train_query_tool.py:108 — for segment 2+, derive
    # depCityName from the previous segment's arr_station_name (with the
    # " Station" suffix stripped). The CSV's origin_city always reports
    # the route's starting city, so without this derivation segment 2+
    # would falsely claim to depart from the original origin.
    if idx == 1 or prev_row is None:
        dep_city = row.get("origin_city", "")
    else:
        prev_arr = prev_row.get("arr_station_name", "")
        dep_city = prev_arr.split(" Station")[0]
    return {
        f"Segment {idx}": {
            "arrCityName": row.get("destination_city", ""),
            "arrStationCode": row.get("arr_station_code", ""),
            "arrStationName": row.get("arr_station_name", ""),
            "depCityName": dep_city,
            "depStationCode": row.get("dep_station_code", ""),
            "depStationName": row.get("dep_station_name", ""),
            "duration": duration,
            "arrDateTime": row.get("arr_datetime", ""),
            "depDateTime": row.get("dep_datetime", ""),
            "marketingTransportName": row.get("train_type", ""),
            "marketingTransportNo": row.get("train_no", ""),
            "seatClassName": row.get("seat_class", ""),
            "Remaining Seats": seat_status,
        }
    }


def run(origin: str, destination: str, depDate: str, seatClassName: str = "") -> str:
    rows, path = _csv.load_for_tool("trains/trains.csv")
    if not rows:
        if path is None:
            return _csv.db_not_loaded_message("Train")
        return f"No train information found from {origin} to {destination} on {depDate}"

    matched = [
        r for r in rows
        if r.get("origin_city") == origin
        and r.get("destination_city") == destination
        and r.get("dep_date") == depDate
    ]
    if seatClassName:
        matched = [r for r in matched if r.get("seat_class") == seatClassName]
    if not matched:
        return f"No train information found from {origin} to {destination} on {depDate}"

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for r in matched:
        grouped[r.get("route_index", "")].append(r)

    routes: list[Any] = []
    for route_idx in sorted(grouped):
        segments = sorted(
            grouped[route_idx],
            key=lambda r: int(float(r.get("segment_index") or 0)),
        )
        route: dict[str, Any] = {}
        price: float | None = None
        prev_row: dict[str, str] | None = None
        for i, row in enumerate(segments, start=1):
            route.update(_segment(i, row, prev_row))
            if i == 1:
                try:
                    price = float(row.get("price") or 0)
                except ValueError:
                    price = None
            prev_row = row
        route["price"] = price if price is not None else 0
        # Reference returns `routes.append([route_data])` — wrap in a one-item list
        # to preserve the expected output shape.
        routes.append([route])

    return json.dumps(routes, ensure_ascii=False, indent=2)


register_tool(NAME, SCHEMA, run)
