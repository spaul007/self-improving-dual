"""Which mutable files of a task agent changed between two rounds.

Pure I/O: no LLM, no AST, no project knowledge. The agentic editor uses
``changed_mutable_files`` to detect an edit and derive a submission's
``target_files``.

The only structural assumption is the framework's mutable-surface contract —
``MUTABLE_FILES`` / ``MUTABLE_DIRS`` in ``editor_validators`` — which is
imported rather than restated so the two can never drift apart.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from .editor_validators import MUTABLE_DIRS, MUTABLE_FILES, is_excluded


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



# Never part of the agent's editable source (bytecode, generated output) --
# the same set as AgentEditor._ALWAYS_IGNORE_DIRS.
_IGNORE_DIRS = frozenset({"__pycache__", "results"})


def changed_agent_files(
    parent_round_dir: Path,
    round_dir: Path,
    *,
    mutable_exclude: Optional[list[str]] = None,
) -> list[str]:
    """Editable file paths (relative to ``task_agent/``) that differ between
    parent and child — added, removed, or modified (byte comparison).

    ``mutable_exclude`` None: the single-agent surface, exactly
    ``changed_mutable_files``. Otherwise the exclude-list surface of
    multi-agent projects: every file under ``task_agent/`` except the
    excluded prefixes and ``__pycache__`` / ``results`` output (``__init__.py``
    included — it is editable there)."""
    if mutable_exclude is None:
        return changed_mutable_files(parent_round_dir, round_dir)
    parent_root = Path(parent_round_dir) / "task_agent"
    child_root = Path(round_dir) / "task_agent"
    if not child_root.exists():
        return []

    def files(root: Path) -> dict[str, Path]:
        if not root.exists():
            return {}
        out: dict[str, Path] = {}
        for p in root.rglob("*"):
            rel = p.relative_to(root)
            if set(rel.parts) & _IGNORE_DIRS or not (p.is_file() or p.is_symlink()):
                continue
            rel_s = rel.as_posix()
            if is_excluded(rel_s, list(mutable_exclude)):
                continue
            out[rel_s] = p
        return out

    old, new = files(parent_root), files(child_root)
    changed: list[str] = []
    for rel in sorted(set(old) | set(new)):
        if rel not in old or rel not in new:
            changed.append(rel)
            continue
        try:
            if old[rel].read_bytes() != new[rel].read_bytes():
                changed.append(rel)
        except OSError:
            changed.append(rel)
    return changed
