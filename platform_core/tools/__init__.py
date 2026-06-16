"""Registry of immutable tools.

Tools live under ``projects/<name>/tools/``. Each tool module exports:
    NAME: str
    SCHEMA: dict   # OpenAI or Anthropic tool-schema fragment
    def run(**kwargs) -> str: ...

and calls :func:`register_tool` at import time. The :func:`call_tool`
function is the sole entry point for invoking an immutable tool — the
mutable wrapper and any mutable tools must route through it. This function
emits ``tool_call`` and ``tool_result`` trace events and is the hard
boundary the editor cannot cross.

Population:

* :func:`load_project` — import a project's tools package
  (``projects.<name>.tools``) so each sub-module's :func:`register_tool`
  call runs. Used by ``runtime_env.apply_project_tools`` in the parent and
  re-triggered in the evaluator subprocess via the ``META_AGENT_PROJECT``
  env var.
"""
from __future__ import annotations

import importlib
import json
import os
import time
import uuid
from typing import Any, Callable

from .. import trace

_TOOLS: dict[str, dict[str, Any]] = {}
_LOADED = False

_PROJECT_ENV = "META_AGENT_PROJECT"


def load_project(project_name: str) -> None:
    """Import ``projects.<project_name>.tools`` so each tool sub-module's
    :func:`register_tool` call runs. Subsequent :func:`_discover` calls
    short-circuit."""
    global _LOADED
    if not project_name:
        raise ValueError("load_project requires a non-empty project name")
    importlib.import_module(f"projects.{project_name}.tools")
    _LOADED = True


def _discover() -> None:
    """Populate the registry from the ``META_AGENT_PROJECT`` env var. The
    parent (``runtime_env.apply_project_tools``) sets this before spawning
    the evaluator subprocess."""
    global _LOADED
    if _LOADED:
        return
    project = os.environ.get(_PROJECT_ENV, "").strip()
    if project:
        load_project(project)


def register_tool(name: str, schema: dict[str, Any], run: Callable[..., str]) -> None:
    """Register an immutable tool. Called from each tool module at import time."""
    if name in _TOOLS:
        raise ValueError(f"Immutable tool {name!r} is already registered")
    _TOOLS[name] = {"schema": schema, "run": run}


def get_schema(name: str) -> dict[str, Any]:
    _discover()
    if name not in _TOOLS:
        raise KeyError(f"Unknown immutable tool: {name!r}. Available: {sorted(_TOOLS)}")
    return _TOOLS[name]["schema"]


def all_schemas() -> dict[str, dict[str, Any]]:
    _discover()
    return {name: entry["schema"] for name, entry in _TOOLS.items()}


def is_immutable(name: str) -> bool:
    _discover()
    return name in _TOOLS


def call_tool(name: str, **kwargs: Any) -> str:
    """Invoke an immutable tool. The only sanctioned path to a real capability."""
    _discover()
    if name not in _TOOLS:
        raise KeyError(f"Unknown immutable tool: {name!r}. Available: {sorted(_TOOLS)}")

    call_id = uuid.uuid4().hex[:12]
    trace.emit("tool_call", {"id": call_id, "name": name, "arguments": kwargs})

    started = time.time()
    try:
        result = _TOOLS[name]["run"](**kwargs)
    except Exception as exc:
        trace.emit(
            "error",
            {"id": call_id, "where": f"tool:{name}", "exception": repr(exc)},
        )
        raise
    elapsed = time.time() - started

    if not isinstance(result, str):
        result = json.dumps(result, ensure_ascii=False)

    trace.emit(
        "tool_result",
        {
            "id": call_id,
            "name": name,
            "elapsed_s": elapsed,
            "result_preview": result[:200],
            "result_len": len(result),
        },
    )
    return result


def call_mutable_tool(name: str, **kwargs: Any) -> str:
    """Invoke a round-mutable tool (``mutable_tools/<name>.py::run``) with the
    SAME ``tool_call`` / ``tool_result`` tracing as :func:`call_tool`.

    Mutable tools are dispatched directly (not via the immutable registry), so
    without this their invocations leave no trace event — invisible to the
    feedback gatherer's ``tool_usage`` and the behavior summarizer. Routing the
    mutable branch of ``tool_wrapper.execute`` through here makes mutable-tool
    usage show up in ``trace.jsonl`` exactly like immutable tools. Keeping the
    emission in immutable ``platform_core`` (not in the editable wrapper) means
    tracing survives editor rewrites of ``tool_wrapper.py``.
    """
    call_id = uuid.uuid4().hex[:12]
    # ``mutable: True`` lets consumers (behavior summarizer) tell editor-added
    # tools apart from immutable ones in the trace.
    trace.emit(
        "tool_call",
        {"id": call_id, "name": name, "arguments": kwargs, "mutable": True},
    )

    started = time.time()
    try:
        try:
            mod = importlib.import_module(f"mutable_tools.{name}")
        except ModuleNotFoundError:
            result: Any = f"Error: tool {name!r} not recognised."
        else:
            run = getattr(mod, "run", None)
            if run is None:
                result = f"Error: mutable tool {name!r} has no run() function."
            else:
                result = run(**kwargs)
    except Exception as exc:
        trace.emit(
            "error",
            {"id": call_id, "where": f"mutable_tool:{name}", "exception": repr(exc)},
        )
        raise
    elapsed = time.time() - started

    if not isinstance(result, str):
        result = json.dumps(result, ensure_ascii=False)

    trace.emit(
        "tool_result",
        {
            "id": call_id,
            "name": name,
            "elapsed_s": elapsed,
            "result_preview": result[:200],
            "result_len": len(result),
            "mutable": True,
        },
    )
    return result
