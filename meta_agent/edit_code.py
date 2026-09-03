"""Per-node code record — what an edit's implementation actually looks like.

Complements ``edit_memory.md`` (what was changed and why, in prose) with the
code itself: the verbatim diff vs the parent at a higher cap than the tagger
sees, plus the final-state source of every top-level def/class the edit added
or modified. Written once per node at record time; NEVER injected into
steering — it is read on demand by the retrieval stage (``edit_archive``), so
the record file stays unbloated.

Deterministic by design: pure ``ast`` + ``edit_diff``, no LLM. The only
LLM-derived content is the sub-edit map header, which reuses the tagger's
already-produced sub-edit names.
"""
from __future__ import annotations

import ast
import os
import re
import tempfile
from pathlib import Path
from typing import Mapping, Optional, Sequence

from .edit_diff import changed_mutable_files, diff_mutable_files, truncate_middle

CODE_NAME = "edit_code.md"
# The tagger's diff cap is 6000; this record exists precisely to keep more.
CODE_DIFF_CHAR_CAP = 20000
CODE_DEFS_CHAR_CAP = 30000

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _atomic_write(path: Path, text: str) -> None:
    """tmp-in-same-dir -> fsync -> replace (same contract as edit_memory's)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall((text or "").lower()))


def extract_changed_defs(
    parent_text: str, child_text: str
) -> list[tuple[str, str, str, str]]:
    """Top-level defs/classes in ``child_text`` that the edit added or changed.

    Returns ``[(name, kind, status, source_segment)]`` in child file order;
    ``kind`` is ``function`` / ``async function`` / ``class``, ``status`` is
    ``added`` / ``changed``. A child that fails to parse yields ``[]`` (the
    diff section still carries the change); a parent that fails to parse
    degrades to treating every child def as ``added``.
    """
    try:
        child_tree = ast.parse(child_text)
    except (SyntaxError, ValueError):
        return []
    parent_segs: dict[str, str] = {}
    try:
        for node in ast.parse(parent_text).body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                                 ast.ClassDef)):
                seg = ast.get_source_segment(parent_text, node)
                if seg is not None:
                    parent_segs[node.name] = seg.strip()
    except (SyntaxError, ValueError):
        parent_segs = {}

    out: list[tuple[str, str, str, str]] = []
    for node in child_tree.body:
        if isinstance(node, ast.FunctionDef):
            kind = "function"
        elif isinstance(node, ast.AsyncFunctionDef):
            kind = "async function"
        elif isinstance(node, ast.ClassDef):
            kind = "class"
        else:
            continue
        seg = ast.get_source_segment(child_text, node)
        if seg is None:
            continue
        old = parent_segs.get(node.name)
        if old is None:
            out.append((node.name, kind, "added", seg))
        elif old != seg.strip():
            out.append((node.name, kind, "changed", seg))
    return out


def map_subedits(
    sub_edits: Optional[Sequence[Mapping[str, str]]],
    changed_by_file: Mapping[str, list[tuple[str, str, str, str]]],
    files: Sequence[str],
) -> list[str]:
    """Best-effort deterministic header lines mapping each sub-edit to the
    files/defs it plausibly touched (token overlap between the sub-edit's
    name+what and the def/file names). Defs no sub-edit claims are listed
    under ``(unattributed)``. With no sub-edits yet (the pre-tagger write),
    only the changed-files line is emitted.
    """
    lines = ["- changed files: " + (", ".join(sorted(files)) or "(none)")]
    all_defs: list[tuple[str, str]] = []  # (rel_path, def_name)
    for rel in sorted(changed_by_file):
        for name, _kind, _status, _src in changed_by_file[rel]:
            all_defs.append((rel, name))
    if not sub_edits:
        for rel, name in all_defs:
            lines.append(f"- (unattributed) {rel} :: {name}")
        return lines

    claimed: set[tuple[str, str]] = set()
    for i, e in enumerate(sub_edits, 1):
        want = _tokens(e.get("name", "")) | _tokens(e.get("what", ""))
        mine: list[str] = []
        for rel, name in all_defs:
            have = _tokens(name) | _tokens(rel)
            if want & have:
                mine.append(f"{rel} :: {name}")
                claimed.add((rel, name))
        target = "; ".join(mine) if mine else "(no matching def — see diff)"
        lines.append(f"- `{e.get('name', f'edit-{i}')}` (Edit {i}) -> {target}")
    for rel, name in all_defs:
        if (rel, name) not in claimed:
            lines.append(f"- (unattributed) {rel} :: {name}")
    return lines


def render_edit_code(
    parent_round_dir: Path,
    round_dir: Path,
    *,
    node_id: int,
    parent_id: int,
    sub_edits: Optional[Sequence[Mapping[str, str]]] = None,
    diff_char_cap: int = CODE_DIFF_CHAR_CAP,
    defs_char_cap: int = CODE_DEFS_CHAR_CAP,
) -> str:
    parent_root = Path(parent_round_dir) / "task_agent"
    child_root = Path(round_dir) / "task_agent"
    files = changed_mutable_files(parent_round_dir, round_dir)
    diff = diff_mutable_files(parent_round_dir, round_dir, char_cap=diff_char_cap)

    changed_by_file: dict[str, list[tuple[str, str, str, str]]] = {}
    for rel in files:
        if not rel.endswith(".py"):
            continue
        try:
            child_text = ((child_root / rel).read_text(encoding="utf-8")
                          if (child_root / rel).exists() else "")
            parent_text = ((parent_root / rel).read_text(encoding="utf-8")
                           if (parent_root / rel).exists() else "")
        except (OSError, UnicodeDecodeError):
            continue
        defs = extract_changed_defs(parent_text, child_text)
        if defs:
            changed_by_file[rel] = defs

    lines = ["---", f"node: {node_id}", f"parent: {parent_id}", "---", "",
             "## Sub-edit map"]
    lines += map_subedits(sub_edits, changed_by_file, files)
    lines += ["", f"## Diff vs parent (cap {diff_char_cap} chars)",
              "```diff", diff or "(no textual diff)", "```", "",
              "## Final-state definitions (added/changed, from child sources)"]
    used = 0
    truncated = False
    for rel in sorted(changed_by_file):
        for name, kind, status, src in changed_by_file[rel]:
            if used + len(src) > defs_char_cap:
                truncated = True
                break
            lines += [f"### {rel} :: {name} ({kind}, {status})",
                      "```python", src, "```", ""]
            used += len(src)
        if truncated:
            break
    if truncated:
        lines.append(f"<... remaining definitions elided at {defs_char_cap} "
                     "chars — see the diff above ...>")
    if not changed_by_file:
        lines.append("(no top-level def/class changes extracted — "
                     "see the diff above)")
    return "\n".join(lines).rstrip("\n") + "\n"


def write_edit_code(
    parent_round_dir: Path,
    round_dir: Path,
    *,
    node_id: int,
    parent_id: int,
    sub_edits: Optional[Sequence[Mapping[str, str]]] = None,
    diff_char_cap: int = CODE_DIFF_CHAR_CAP,
    defs_char_cap: int = CODE_DEFS_CHAR_CAP,
) -> Optional[Path]:
    """Render + atomically write ``edit_code.md``. Best-effort: returns the
    path, or ``None`` on any failure (printed, never raised)."""
    try:
        dest = Path(round_dir) / CODE_NAME
        _atomic_write(dest, render_edit_code(
            parent_round_dir, round_dir, node_id=node_id, parent_id=parent_id,
            sub_edits=sub_edits, diff_char_cap=diff_char_cap,
            defs_char_cap=defs_char_cap))
        return dest
    except Exception as exc:  # noqa: BLE001
        print(f"[edit_code] node {node_id}: write failed: {exc!r}", flush=True)
        return None
