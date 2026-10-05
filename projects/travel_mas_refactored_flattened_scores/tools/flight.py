"""Flight query tool — re-implementation of FlightQueryTool against our
framework's interface. Reads the same `flights/flights.csv` columns as the
reference."""
from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

from platform_core.tools import register_tool
from . import _csv

NAME = "query_flight_info"

SCHEMA = {
    "name": NAME,
    "description": (
        "Query flight information by origin city, destination city, and "
        "departure date (and optional seat class). Returns a JSON list of "
        "candidate flights or routes; each route may contain multiple "
        "segments. Flight-related details in the final plan must come "
        "exclusively from this tool's output."
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
                "description": "Optional seat class: Economy Class, Business Class, First Class.",
            },
        },
        "required": ["origin", "destination", "depDate"],
    },
}


def _segment(idx: int, row: dict[str, str]) -> dict[str, Any]:
    seat_status = (row.get("seat_status") or "").strip()
    if not seat_status or seat_status.lower() == "nan":
        seat_status = "Available"
    try:
        duration = int(float(row.get("duration") or 0))
    except ValueError:
        duration = 0
    return {
        f"Segment {idx}": {
            "arrCityName": row.get("destination_city", ""),
            "arrStationCode": row.get("arr_station_code", ""),
            "arrStationName": row.get("arr_station_name", ""),
            "depCityName": row.get("origin_city", ""),
            "depStationCode": row.get("dep_station_code", ""),
            "depStationName": row.get("dep_station_name", ""),
            "duration": duration,
            "arrDateTime": row.get("arr_datetime", ""),
            "depDateTime": row.get("dep_datetime", ""),
            "marketingTransportName": row.get("airline", ""),
            "marketingTransportNo": row.get("flight_no", ""),
            "seatClassName": row.get("seat_class", ""),
            "Remaining Seats": seat_status,
            "equipSize": row.get("equip_size", ""),
            "equipType": row.get("equip_type", ""),
            "manufacturer": row.get("manufacturer", ""),
        }
    }


def run(origin: str, destination: str, depDate: str, seatClassName: str = "") -> str:
    rows, path = _csv.load_for_tool("flights/flights.csv")
    if not rows:
        if path is None:
            return _csv.db_not_loaded_message("Flight")
        return f"No flight information found from {origin} to {destination} on {depDate}"

    matched = [
        r for r in rows
        if r.get("origin_city") == origin
        and r.get("destination_city") == destination
        and r.get("dep_date") == depDate
    ]
    if seatClassName:
        matched = [r for r in matched if r.get("seat_class") == seatClassName]
    if not matched:
        return f"No flight information found from {origin} to {destination} on {depDate}"

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for r in matched:
        grouped[r.get("route_index", "")].append(r)

    routes: list[dict[str, Any]] = []
    for route_idx in sorted(grouped):
        segments = sorted(
            grouped[route_idx],
            key=lambda r: int(float(r.get("segment_index") or 0)),
        )
        route: dict[str, Any] = {}
        price: float | None = None
        for i, row in enumerate(segments, start=1):
            route.update(_segment(i, row))
            if i == 1:
                try:
                    price = float(row.get("price") or 0)
                except ValueError:
                    price = None
        route["price"] = price if price is not None else 0
        routes.append(route)

    return json.dumps(routes, ensure_ascii=False, indent=2)


register_tool(NAME, SCHEMA, run)
