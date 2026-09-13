"""Hotel query tool — re-implementation of HotelQueryTool."""
from __future__ import annotations

import json
from typing import Any

from platform_core.tools import register_tool
from . import _csv

NAME = "query_hotel_info"

SCHEMA = {
    "name": NAME,
    "description": (
        "Query hotel information by destination, check-in/check-out dates, "
        "and optional star rating / brand. Returns a JSON list of matching "
        "hotels."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "destination": {"type": "string", "description": "Destination city or region."},
            "checkinDate": {
                "type": "string",
                "description": "Check-in date, format YYYY-MM-DD.",
            },
            "checkoutDate": {
                "type": "string",
                "description": "Check-out date, format YYYY-MM-DD.",
            },
            "hotelStar": {"type": "string", "description": "Star rating 1-5 (optional)."},
            "hotelBrands": {
                "type": "string",
                "description": (
                    "Optional brand name. Supported: Ji Hotel, Atour Hotel, Marriott, "
                    "Hilton, Home Inn, Jinjiang Inn, Hanting Hotel, Orange Hotel."
                ),
            },
        },
        "required": ["destination", "checkinDate", "checkoutDate"],
    },
}


def _to_str(val: Any) -> str:
    if val is None:
        return ""
    return str(val)


def run(
    destination: str,
    checkinDate: str,
    checkoutDate: str,
    hotelStar: str = "",
    hotelBrands: str = "",
) -> str:
    rows, path = _csv.load_for_tool("hotels/hotels.csv")
    if not rows:
        if path is None:
            return _csv.db_not_loaded_message("Hotel")
        return (
            f"No hotel information found in {destination} from {checkinDate} "
            f"to {checkoutDate}, please check parameters or reduce constraints"
        )

    matched = rows
    if hotelStar:
        matched = [r for r in matched if r.get("hotel_star") == hotelStar]
    if hotelBrands:
        matched = [r for r in matched if r.get("brand") == hotelBrands]
    if not matched:
        return (
            f"No hotel information found in {destination} from {checkinDate} "
            f"to {checkoutDate}, please check parameters or reduce constraints"
        )

    results: list[dict[str, Any]] = []
    for row in matched:
        result: dict[str, Any] = {
            "name": _to_str(row.get("name", "")),
            "address": _to_str(row.get("address", "")),
            "latitude": _to_str(row.get("latitude", "")),
            "longitude": _to_str(row.get("longitude", "")),
            "decorationTime": _to_str(row.get("decoration_time", "")),
            "hotelStar": _to_str(row.get("hotel_star", "")),
            "price": _to_str(row.get("price", "")),
            "score": _to_str(row.get("score", "")),
            "brand": _to_str(row.get("brand", "")),
        }
        services = row.get("services")
        if isinstance(services, str) and services.strip():
            result["services"] = services.split(";")
        results.append(result)

    return json.dumps(results, ensure_ascii=False, indent=2)


register_tool(NAME, SCHEMA, run)
