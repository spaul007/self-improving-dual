"""Registry of immutable tools.

Tools live under ``projects/<name>/tools/`` by default. Each tool module
exports:
    NAME: str
    SCHEMA: dict   # OpenAI or Anthropic tool-schema fragment
    def run(**kwargs) -> str: ...

and calls :func:`register_tool` at import time. The :func:`call_tool`
function is the sole entry point for invoking an immutable tool — the
mutable wrapper and any mutable tools must route through it. This function
emits ``tool_call`` and ``tool_result`` trace events and is the hard
boundary the editor cannot cross.

Population, two ways:

* :func:`load_project` (default) — import a project's tools package
  (``projects.<name>.tools``) so each sub-module's :func:`register_tool`
  call runs.
* ``source_dirs`` (see ``config.FrameworkConfig.tool_source_dirs``) — for a
  project whose tool-defining code is spread across multiple folders inside
  its own vendored implementation (e.g. a shared folder plus per-agent
  folders) rather than one dedicated ``tools/`` package: every ``*.py``
  file under each directory (recursively) is loaded directly by file path
  via :func:`load_paths`, so whichever ones call :func:`register_tool`
  register into the shared registry — this needs no dotted-import path, so
  it works regardless of where on disk the project's code lives.

Used by ``runtime_env.apply_project_tools`` in the parent and re-triggered
in the evaluator subprocess via the ``META_AGENT_PROJECT`` /
``META_AGENT_TOOL_SOURCE_DIRS`` env vars.
"""
from __future__ import annotations

import importlib
import importlib.util
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Optional

from .. import trace

_TOOLS: dict[str, dict[str, Any]] = {}
_LOADED = False

_PROJECT_ENV = "META_AGENT_PROJECT"
_TOOL_SOURCE_DIRS_ENV = "META_AGENT_TOOL_SOURCE_DIRS"
_TOOL_SOURCE_ROOT_ENV = "META_AGENT_TOOL_SOURCE_ROOT"


def load_paths(paths: list[Path]) -> None:
    """Load each ``.py`` file in ``paths`` directly by file path (not as a
    dotted import), so tool modules can live anywhere on disk without being
    reachable as ``projects.<name>.tools``. Each file that calls
    :func:`register_tool` at import time registers into the shared flat
    registry.

    Best-effort per file: these files weren't necessarily written with this
    discovery mechanism in mind (e.g. one that assumes its own package root
    is already on ``sys.path`` for a sibling top-level import — the common
    case is covered by ``load_project``'s ``sys_path_root``, but not every
    layout is predictable). A file that doesn't load cleanly, or doesn't
    call :func:`register_tool` at all, just registers nothing rather than
    aborting the rest of the scan."""
    for path in paths:
        if path.name == "__init__.py":
            continue
        spec = importlib.util.spec_from_file_location(
            f"_immutable_tool_{path.stem}_{id(path)}", path
        )
        if not (spec and spec.loader):
            continue
        mod = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(mod)
        except Exception as exc:  # noqa: BLE001 - best-effort tool discovery
            print(f"[tools] skipped {path} (failed to load): {exc!r}", flush=True)


def load_project(
    project_name: str,
    *,
    source_dirs: Optional[list[Path]] = None,
    sys_path_root: Optional[Path] = None,
) -> None:
    """Populate the immutable-tool registry for ``project_name``.
    Subsequent :func:`_discover` calls short-circuit.

    ``source_dirs`` (absolute paths) switches to directory-scan mode (see
    :func:`load_paths`) instead of the default ``projects.<project_name>.tools``
    package import — set via ``config.FrameworkConfig.tool_source_dirs``.
    ``sys_path_root`` (the project's seed dir) is inserted onto ``sys.path``
    first when given, so a scanned file's own top-level sibling imports
    (e.g. ``import config`` resolving to a sibling file in the same seed
    dir) work the same way they do when the real workflow runs.

    A project with neither pattern producing anything is valid too — not
    every task agent routes its capabilities through
    ``call_tool``/``register_tool`` (e.g. a vendored multi-agent system that
    calls its own tool functions directly). In default mode, only the case
    where ``projects.<project_name>.tools`` itself can't be found is
    swallowed; an ``ImportError`` raised *from inside* an existing tools
    package (a real bug) still surfaces.
    """
    global _LOADED
    if not project_name:
        raise ValueError("load_project requires a non-empty project name")
    if source_dirs:
        if sys_path_root is not None:
            root_str = str(sys_path_root)
            if root_str not in sys.path:
                sys.path.insert(0, root_str)
        paths: list[Path] = []
        for d in source_dirs:
            if d.is_dir():
                paths.extend(sorted(d.rglob("*.py")))
        load_paths(paths)
    else:
        mod_path = f"projects.{project_name}.tools"
        try:
            importlib.import_module(mod_path)
        except ModuleNotFoundError as exc:
            if exc.name != mod_path:
                raise
    _LOADED = True


def _discover() -> None:
    """Populate the registry from the ``META_AGENT_PROJECT`` (and optional
    ``META_AGENT_TOOL_SOURCE_DIRS``/``META_AGENT_TOOL_SOURCE_ROOT``) env
    vars. The parent (``runtime_env.apply_project_tools``) sets these before
    spawning the evaluator subprocess."""
    global _LOADED
    if _LOADED:
        return
    project = os.environ.get(_PROJECT_ENV, "").strip()
    dirs_raw = os.environ.get(_TOOL_SOURCE_DIRS_ENV, "").strip()
    source_dirs = [Path(p) for p in dirs_raw.split(os.pathsep) if p] or None
    root_raw = os.environ.get(_TOOL_SOURCE_ROOT_ENV, "").strip()
    sys_path_root = Path(root_raw) if root_raw else None
    if project:
        load_project(project, source_dirs=source_dirs, sys_path_root=sys_path_root)


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
