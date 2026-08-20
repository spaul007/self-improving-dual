"""Shared source-context helpers: reading a task_agent workspace's mutable
sources and rendering static project reference material for an LLM prompt.

Extracted from AgentEditor (meta_agent/agent_editor.py) so any component
that needs to ground itself in the agent's ACTUAL current code -- not just
AgentEditor -- shares one implementation instead of an independently
maintained copy. AgentEditor's own _read_mutable_sources /
_format_current_sources / _format_project_context now delegate here,
unchanged in behavior. See meta_agent/block_suggester.py for the other
caller.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from .editor_validators import MUTABLE_DIRS, MUTABLE_FILES, is_excluded

# Same noise set agent_editor.py's exclude-mode scan skips -- generated/
# scratch output and Python's own cache, never source to read or judge.
ALWAYS_IGNORE_DIRS = {"__pycache__", "results"}


def read_mutable_sources(
    agent_dir: Path, *, mutable_exclude: Optional[list[str]] = None
) -> dict[str, str]:
    """Every currently-editable source file under ``agent_dir``, path -> text.

    Exclude-list mode (``mutable_exclude`` is not None) walks the whole tree
    minus excluded paths; legacy include-list mode reads only
    ``MUTABLE_FILES`` plus ``mutable_tools/*.py``."""
    if mutable_exclude is not None:
        sources: dict[str, str] = {}
        for path in sorted(agent_dir.rglob("*")):
            if path.is_dir() or path.name == "__init__.py":
                continue
            if set(path.relative_to(agent_dir).parts) & ALWAYS_IGNORE_DIRS:
                continue
            rel = path.relative_to(agent_dir).as_posix()
            if is_excluded(rel, mutable_exclude):
                continue
            try:
                sources[rel] = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
        return sources

    sources = {}
    for fname in sorted(MUTABLE_FILES):
        path = agent_dir / fname
        if path.exists():
            sources[fname] = path.read_text(encoding="utf-8")
    mutable_dir = agent_dir / "mutable_tools"
    if mutable_dir.exists():
        for py in sorted(mutable_dir.glob("*.py")):
            if py.name == "__init__.py":
                continue
            sources[f"mutable_tools/{py.name}"] = py.read_text(encoding="utf-8")
    return sources


def format_current_sources(sources: dict[str, str]) -> str:
    if not sources:
        return "## Current sources\n(empty)\n"
    parts = ["## Current sources"]
    for path, body in sources.items():
        parts.append(f"### {path}\n```\n{body}\n```")
    return "\n".join(parts) + "\n"


def format_project_context(
    *,
    tools_source: Optional[str] = None,
    db_schema: Optional[str] = None,
    scorer_source: Optional[str] = None,
) -> list[str]:
    """Static project reference material: immutable tool implementations,
    the database schema, and (whitebox only) the evaluation scoring code.
    Each is injected by ``build_components`` from the project folder;
    absent ones are skipped."""
    parts: list[str] = []
    if tools_source:
        parts.append(
            "## Tool implementations (immutable — reached via "
            "platform_core.tools.call_tool; read to see what each tool "
            f"actually does)\n{tools_source}\n"
        )
    if db_schema:
        parts.append(f"## Database schema (what the tools query against)\n{db_schema}\n")
    if scorer_source:
        parts.append(
            "## Evaluation scoring code (read-only — exactly how your "
            "output is graded; ground-truth data is NOT accessible)\n"
            f"{scorer_source}\n"
        )
    return parts
