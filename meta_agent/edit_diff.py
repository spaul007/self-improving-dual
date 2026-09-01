"""Diff of a task agent's mutable surface between two rounds.

Pure I/O + ``difflib``: no LLM, no AST, no project knowledge. Lifted verbatim
out of ``BehaviorSummarizer`` so both it and ``EditMemory`` share one
implementation of "what changed between parent and child".

The only structural assumption is the framework's mutable-surface contract —
``MUTABLE_FILES`` / ``MUTABLE_DIRS`` in ``editor_validators`` — which is
imported rather than restated so the two can never drift apart.
"""
from __future__ import annotations

import difflib
from pathlib import Path
from typing import Optional

from .editor_validators import MUTABLE_DIRS, MUTABLE_FILES

# Default cap on the rendered diff handed to an LLM. Callers may override.
DIFF_CHAR_CAP = 3000


def truncate_middle(text: str, cap: int) -> str:
    """Keep the head and tail of ``text``, eliding the middle.

    Head-only truncation would hide the last file's changes entirely (the diff
    is ordered per file), so both ends are preserved and the gap is marked.
    """
    if len(text) <= cap:
        return text
    keep = (cap - 80) // 2
    head, tail = text[:keep], text[-keep:]
    elided = len(text) - 2 * keep
    return f"{head}\n<... {elided} chars elided ...>\n{tail}"


def changed_mutable_files(parent_round_dir: Path, round_dir: Path) -> list[str]:
    """Mutable file paths (relative to ``task_agent/``) that differ between
    parent and child — added, removed, or modified."""
    out: list[str] = []
    parent_root = Path(parent_round_dir) / "task_agent"
    child_root = Path(round_dir) / "task_agent"
    if not child_root.exists():
        return out

    candidates: set[str] = set(MUTABLE_FILES)
    for sub in MUTABLE_DIRS:
        for src in (parent_root / sub, child_root / sub):
            if src.exists():
                for p in src.glob("*.py"):
                    if p.name == "__init__.py":
                        continue
                    candidates.add(f"{sub}/{p.name}")

    for rel in sorted(candidates):
        p_path, c_path = parent_root / rel, child_root / rel
        p_exists, c_exists = p_path.exists(), c_path.exists()
        if not p_exists and not c_exists:
            continue
        if p_exists != c_exists:
            out.append(rel)
        else:
            try:
                if p_path.read_text(encoding="utf-8") != c_path.read_text(encoding="utf-8"):
                    out.append(rel)
            except OSError:
                out.append(rel)
    return out


def diff_mutable_files(
    parent_round_dir: Path, round_dir: Path, *, char_cap: int = DIFF_CHAR_CAP
) -> str:
    """Unified diff of the changed mutable files, capped via
    :func:`truncate_middle`."""
    parent_root = Path(parent_round_dir) / "task_agent"
    child_root = Path(round_dir) / "task_agent"

    parts: list[str] = []
    for rel in changed_mutable_files(parent_round_dir, round_dir):
        try:
            old_lines = (
                (parent_root / rel).read_text(encoding="utf-8").splitlines()
                if (parent_root / rel).exists() else []
            )
            new_lines = (
                (child_root / rel).read_text(encoding="utf-8").splitlines()
                if (child_root / rel).exists() else []
            )
        except OSError:
            continue
        parts.append("\n".join(difflib.unified_diff(
            old_lines, new_lines, fromfile=f"parent/{rel}", tofile=f"child/{rel}",
            n=3, lineterm="",
        )))

    return truncate_middle("\n\n".join(p for p in parts if p), char_cap)


def read_text(path: Path) -> Optional[str]:
    """``""`` when absent (the empty side of an add/delete), ``None`` when
    present but unreadable. Never raises."""
    path = Path(path)
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError, ValueError):
        return None
