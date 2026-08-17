"""Mutable tool router. Loads tools_schema.json and routes execute() calls
to either platform_core.tools (immutable) or this round's mutable_tools/*.

The editor may modify this file across rounds — to add caching, retries,
argument massaging, composite-tool routing, etc. — but it must always reach
capabilities through platform_core.tools: call_tool for immutable tools and
call_mutable_tool for mutable_tools/* (the latter keeps mutable-tool calls in
the trace, like immutable ones).

``import projects.travel.tools`` below registers the same 9 real tool
implementations projects/travel/seed uses (same benchmark, same per-sample
CSV data) via their real dotted package path, so their internal
``from . import _csv`` relative imports resolve. The framework's
``tool_source_dirs`` mechanism was tried first but doesn't work here: it
loads each file standalone by file path with no parent package, which
breaks exactly that kind of relative import (confirmed:
"ImportError: attempted relative import with no known parent package").
``platform_core.tools``'s own discovery (keyed off ``project: "travel_mas"``
in the YAML) then finds no ``projects.travel_mas.tools`` package, which is
an explicitly-swallowed no-op — the tools registered here via the real
``projects.travel.tools`` import are what's actually in the registry.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import projects.travel.tools  # noqa: F401 -- registers the real tool implementations
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
        # Mutable tools route through call_mutable_tool so their invocations are
        # recorded in trace.jsonl (tool_call/tool_result) just like immutable ones.
        return immutable_tools.call_mutable_tool(tool_name, **kwargs)
