"""get_user_info — return the per-case user record (single user per case)."""
from __future__ import annotations

import json

from platform_core.tools import register_tool
from . import _db

NAME = "get_user_info"

SCHEMA = {
    "name": NAME,
    "description": (
        "Returns the per-case user record (user_id, address, body profile, "
        "owned coupons, vip status, etc.). The user_id argument is optional; "
        "if provided and matches the user_id stored in the per-case record, "
        "returns it. Otherwise returns the case's default user."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "user_id": {
                "type": "string",
                "description": "Optional. The unique identifier of the user to retrieve.",
            },
        },
        "required": [],
    },
}


def run(user_id=None) -> str:
    user = _db.load_user()
    if not user:
        return _db.db_missing_sentinel(NAME)
    if user_id and user_id != user.get("user_id"):
        return json.dumps(
            {"error": f"User with user_id {user_id!r} not found", "user": None},
            ensure_ascii=False,
        )
    return json.dumps(user, ensure_ascii=False)


register_tool(NAME, SCHEMA, run)
