"""Mutable tool router for the shopping seed. Loads tools_schema.json and
routes ``execute()`` calls to either ``platform_core.tools`` (immutable)
or this round's ``mutable_tools/*``.

The editor may modify this file across rounds — to add caching, retries,
arg coercion, composite-tool routing, etc. — but it must always go
through ``platform_core.tools.call_tool`` to reach a real capability.
Direct shape parity with ``projects/travel/seed/tool_wrapper.py``.
"""
from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

from platform_core import tools as immutable_tools


class ToolWrapper:
    def __init__(self, schema_path: str | Path | None = None) -> None:
        if schema_path is None:
            schema_path = Path(__file__).parent / "tools_schema.json"
        self.schema_path = Path(schema_path)
        with open(self.schema_path, "r", encoding="utf-8") as fh:
            self._schema = json.load(fh)

    def get_schema(self) -> list[dict[str, Any]]:
        return self._schema

    def execute(self, tool_name: str, kwargs: dict[str, Any]) -> str:
        if immutable_tools.is_immutable(tool_name):
            return immutable_tools.call_tool(tool_name, **kwargs)
        try:
            mod = importlib.import_module(f"mutable_tools.{tool_name}")
        except ModuleNotFoundError:
            return f"Error: tool {tool_name!r} not recognised."
        run = getattr(mod, "run", None)
        if run is None:
            return f"Error: mutable tool {tool_name!r} has no run() function."
        return run(**kwargs)
