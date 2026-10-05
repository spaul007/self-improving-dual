"""Mutable tool router. Loads tools_schema.json and routes execute() calls
to either platform_core.tools (immutable) or this round's mutable_tools/*.

The editor may modify this file across rounds — to add caching, retries,
argument massaging, composite-tool routing, etc. — but it must always reach
capabilities through platform_core.tools: call_tool for immutable tools and
call_mutable_tool for mutable_tools/* (the latter keeps mutable-tool calls in
the trace, like immutable ones).

The 9 real tool implementations live in this project's own
``projects/travel_mas_refactored/tools/`` package (copied from
``projects/travel/tools/`` so this project has zero import-time or
data-path dependency on ``projects/travel`` — see that package's own
``_csv.py`` for the project-relative database-path default, which points
at ``projects/travel_mas_refactored/data/database_en`` (a symlink to the
real per-sample CSVs, not a duplicated copy). ``platform_core.tools``'s
standard discovery (keyed off ``project: "travel_mas_refactored"`` in the
YAML, via ``importlib.import_module("projects.travel_mas_refactored.tools")``)
finds this project's own tools package directly -- no explicit import hack
needed here the way the original travel_mas required (it had to reach
across to ``projects.travel.tools`` since its own tools/ package didn't
exist yet).
"""
from __future__ import annotations

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
        # Per-tool required and valid parameter sets, parsed once from the
        # schema so execute() can validate arguments before calling the
        # actual tool — stops malformed LLM tool calls from raising
        # TypeError (which burns an iteration and gives the model a
        # cryptic traceback instead of actionable feedback).
        self._required_params: dict[str, set[str]] = {}
        self._valid_params: dict[str, set[str]] = {}
        self._tool_error_count: int = 0
        self._last_error: str | None = None
        self._parse_schema()

    def _parse_schema(self) -> None:
        for entry in self._schema:
            func = entry.get("function", {})
            name = func.get("name", "")
            if not name:
                continue
            params = func.get("parameters", {})
            required = set(params.get("required", []))
            properties = params.get("properties", {})
            valid = set(properties.keys())
            self._required_params[name] = required
            self._valid_params[name] = valid

    def get_schema(self) -> list[dict[str, Any]]:
        return self._schema

    @property
    def tool_error_count(self) -> int:
        return self._tool_error_count

    @property
    def last_tool_error(self) -> str | None:
        return self._last_error

    def get_and_reset_error_count(self) -> int:
        """Snapshot and reset the per-stage error counter so callers can
        attribute tool-call failures to the correct stage."""
        count = self._tool_error_count
        self._tool_error_count = 0
        return count

    def execute(self, tool_name: str, kwargs: dict[str, Any]) -> str:
        # ── Pre-execution schema validation ──────────────────────────
        # Check that every required parameter is present and that no
        # unknown/malformed parameter names are passed — the LLM
        # sometimes hallucinates argument names (e.g. '<parameter=origin'
        # instead of 'origin'), and catching those here with a clear
        # error message gives it a better chance of recovering on the
        # next turn instead of triggering a TypeError that burns an
        # iteration with a cryptic traceback.
        required = self._required_params.get(tool_name)
        valid = self._valid_params.get(tool_name)

        if required is not None:
            missing = required - set(kwargs.keys())
            if missing:
                error_msg = (
                    f"Missing required argument(s): {', '.join(sorted(missing))}. "
                    f"Required: {', '.join(sorted(required))}."
                )
                self._tool_error_count += 1
                self._last_error = error_msg
                return json.dumps({"error": error_msg}, ensure_ascii=False)

        if valid is not None:
            unexpected = set(kwargs.keys()) - valid
            if unexpected:
                error_msg = (
                    f"Unexpected argument(s): {', '.join(sorted(unexpected))}. "
                    f"Valid parameters: {', '.join(sorted(valid))}."
                )
                self._tool_error_count += 1
                self._last_error = error_msg
                return json.dumps({"error": error_msg}, ensure_ascii=False)

        # ── Execute ──────────────────────────────────────────────────
        # Still wrap in try/except for runtime errors that pass
        # validation but fail inside the tool (e.g. missing CSV data,
        # logic errors inside the tool implementation itself).
        try:
            if immutable_tools.is_immutable(tool_name):
                return immutable_tools.call_tool(tool_name, **kwargs)
            # Mutable tools route through call_mutable_tool so their
            # invocations are recorded in trace.jsonl (tool_call/tool_result)
            # just like immutable ones.
            return immutable_tools.call_mutable_tool(tool_name, **kwargs)
        except Exception as e:  # noqa: BLE001
            self._tool_error_count += 1
            self._last_error = str(e)
            return json.dumps({"error": str(e)}, ensure_ascii=False)
