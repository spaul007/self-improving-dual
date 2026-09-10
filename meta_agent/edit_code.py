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
import difflib
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from .edit_diff import changed_mutable_files, diff_mutable_files

CODE_NAME = "edit_code.md"
# The tagger's diff cap is 6000; this record exists precisely to keep more.
CODE_DIFF_CHAR_CAP = 20000
CODE_DEFS_CHAR_CAP = 30000

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
# File order for the implementation view: the workflow first, then the
# wrapper, then mutable tools (sorted), then the schema.
_FILE_ORDER = {"workflow.py": 0, "tool_wrapper.py": 1}
IMPLEMENTATION_HEADER = ("## Implementation (added lines vs parent, grouped by "
                         "top-level definition; removed lines counted)")
NEW_DEFS_HEADER = "## New definitions (full source)"


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


def top_level_spans(text: str) -> list[tuple[str, str, int, int]]:
    """``[(name, kind, first_line, last_line)]`` for every top-level def /
    class / named assignment in ``text`` (decorators included). Assignments
    matter because prompt-only edits change module-level string constants —
    naming the constant beats a "(module level)" bucket. ``[]`` when the
    text does not parse."""
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return []
    out: list[tuple[str, str, int, int]] = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            name, kind = node.name, "function"
        elif isinstance(node, ast.AsyncFunctionDef):
            name, kind = node.name, "async function"
        elif isinstance(node, ast.ClassDef):
            name, kind = node.name, "class"
        elif (isinstance(node, ast.Assign) and node.targets
              and all(isinstance(t, ast.Name) for t in node.targets)):
            name, kind = ", ".join(t.id for t in node.targets), "assignment"
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            name, kind = node.target.id, "assignment"
        else:
            continue
        start = min([d.lineno for d in getattr(node, "decorator_list", [])]
                    + [node.lineno])
        out.append((name, kind, start, node.end_lineno or node.lineno))
    return out


def hunks(old_text: str, new_text: str) -> list[dict[str, Any]]:
    """Zero-context unified-diff hunks: ``{new_start, new_len, added, removed}``
    with ``added`` the new lines verbatim and ``removed`` a count."""
    out: list[dict[str, Any]] = []
    cur: Optional[dict[str, Any]] = None
    for line in difflib.unified_diff(old_text.splitlines(), new_text.splitlines(),
                                     lineterm="", n=0):
        if line.startswith(("---", "+++")):
            continue
        m = _HUNK_RE.match(line)
        if m:
            cur = {"new_start": int(m.group(3)),
                   "new_len": int(m.group(4)) if m.group(4) is not None else 1,
                   "added": [], "removed": 0}
            out.append(cur)
            continue
        if cur is None:
            continue
        if line.startswith("+"):
            cur["added"].append(line[1:])
        elif line.startswith("-"):
            cur["removed"] += 1
    return out


def group_hunks_by_def(rel: str, child_text: str,
                       parent_text: str) -> list[dict[str, Any]]:
    """One unit per top-level definition the edit touched, in child source
    order: ``{file, def, kind, status, added, removed, hunks}``. Hunks outside
    any def form a ``(module level)`` unit; a non-Python file is one
    ``(file)`` unit; a child that does not parse degrades to module level."""
    hs = hunks(parent_text, child_text)
    if not hs:
        return []
    if not rel.endswith(".py"):
        return [{"file": rel, "def": "(file)", "kind": "", "status": "file",
                 "added": [l for h in hs for l in h["added"]],
                 "removed": sum(h["removed"] for h in hs), "hunks": len(hs)}]
    spans = top_level_spans(child_text)
    status_by_name = {name: status for name, _k, status, _s
                      in extract_changed_defs(parent_text, child_text)}
    parent_names = {n for n, _k, _s, _e in top_level_spans(parent_text)}

    def enclosing(line: int) -> tuple[str, str]:
        for n, k, s, e in spans:
            if s <= line <= e:
                return n, k
        return "(module level)", ""

    units: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    hunk_ids: dict[str, set[int]] = {}
    last_line: dict[str, int] = {}

    def unit_for(name: str, kind: str) -> dict[str, Any]:
        u = units.get(name)
        if u is None:
            if name == "(module level)":
                status = "module"
            else:
                status = status_by_name.get(name) or (
                    "added" if name not in parent_names else "changed")
            u = {"file": rel, "def": name, "kind": kind, "status": status,
                 "added": [], "removed": 0, "hunks": 0}
            units[name] = u
            order.append(name)
            hunk_ids[name] = set()
        return u

    # One zero-context hunk can span several new definitions, so every
    # ADDED line is attributed by its own child line number; removed lines
    # (which have no child line) go to the definition at the hunk's anchor.
    for hi, h in enumerate(hs):
        for i, text in enumerate(h["added"]):
            line = h["new_start"] + i
            name, kind = enclosing(line)
            if name == "(module level)" and not text.strip():
                continue  # blank lines between definitions carry nothing
            u = unit_for(name, kind)
            if u["added"] and name in last_line and line != last_line[name] + 1:
                u["added"].append("...")  # gap between non-adjacent lines
            u["added"].append(text)
            last_line[name] = line
            hunk_ids[name].add(hi)
        if h["removed"]:
            name, kind = enclosing(h["new_start"])
            u = unit_for(name, kind)
            u["removed"] += h["removed"]
            hunk_ids[name].add(hi)
    for name in order:
        units[name]["hunks"] = len(hunk_ids[name])
    return [units[n] for n in order]


def _file_key(rel: str) -> tuple[int, str]:
    if rel in _FILE_ORDER:
        return _FILE_ORDER[rel], rel
    if rel.endswith(".py"):
        return 2, rel
    return 3, rel


def render_implementation_view(parent_round_dir: Path, round_dir: Path, *,
                               char_budget: int) -> tuple[str, dict[str, int]]:
    """The retrieval-facing view of an edit's implementation, computed from
    the parent and child sources: per touched top-level definition, the
    ADDED lines (removed lines counted), then the full source of every
    definition the edit added. Whole units only — a unit that does not fit
    the budget is skipped and named on a trailing ``omitted`` line; nothing
    is ever cut mid-text. Returns ``(text, stats)``; ``("", stats)`` when
    the edit changed no mutable file."""
    parent_root = Path(parent_round_dir) / "task_agent"
    child_root = Path(round_dir) / "task_agent"
    stats = {"hunks_shown": 0, "hunks_omitted": 0, "defs_shown": 0,
             "defs_omitted": 0, "chars": 0}
    files = changed_mutable_files(parent_round_dir, round_dir)
    if not files:
        return "", stats

    units: list[dict[str, Any]] = []
    added_defs: list[tuple[str, str, str, str]] = []
    for rel in sorted(files, key=_file_key):
        try:
            child_text = ((child_root / rel).read_text(encoding="utf-8")
                          if (child_root / rel).exists() else "")
            parent_text = ((parent_root / rel).read_text(encoding="utf-8")
                           if (parent_root / rel).exists() else "")
        except (OSError, UnicodeDecodeError):
            continue
        units.extend(group_hunks_by_def(rel, child_text, parent_text))
        if rel.endswith(".py"):
            for name, kind, status, src in extract_changed_defs(parent_text,
                                                                child_text):
                if status == "added":
                    added_defs.append((rel, name, kind, src))

    out = [IMPLEMENTATION_HEADER]
    used = len(IMPLEMENTATION_HEADER) + 1
    omitted: list[str] = []
    for u in units:
        n_add = sum(1 for l in u["added"] if l != "...")
        title = (f"### {u['file']} :: {u['def']} "
                 + (f"({u['kind']}, {u['status']})" if u["kind"]
                    else f"({u['status']})")
                 + f"  +{n_add}/-{u['removed']}")
        if u["status"] == "added" and u["kind"] in ("function", "async function",
                                                    "class"):
            block = title + " — full source below"
        elif u["added"]:
            fence = "```python" if u["file"].endswith(".py") else "```"
            body = "\n".join(("..." if l == "..." else "+" + l)
                             for l in u["added"])
            block = f"{title}\n{fence}\n{body}\n```"
        else:
            block = title + " (removed lines only)"
        if used + len(block) + 1 > char_budget:
            omitted.append(f"{u['file']} :: {u['def']} (+{n_add}/-{u['removed']})")
            stats["hunks_omitted"] += u["hunks"]
            continue
        out.append(block)
        used += len(block) + 1
        stats["hunks_shown"] += u["hunks"]

    shown_header = False
    for rel, name, kind, src in added_defs:
        block = f"### {rel} :: {name} ({kind}, added)\n```python\n{src}\n```"
        extra = 0 if shown_header else len(NEW_DEFS_HEADER) + 1
        if used + extra + len(block) + 1 > char_budget:
            omitted.append(f"{rel} :: {name} (full source {len(src)} chars)")
            stats["defs_omitted"] += 1
            continue
        if not shown_header:
            out.append(NEW_DEFS_HEADER)
            used += len(NEW_DEFS_HEADER) + 1
            shown_header = True
        out.append(block)
        used += len(block) + 1
        stats["defs_shown"] += 1
    if omitted:
        out.append("omitted (budget): " + "; ".join(omitted))
    text = "\n".join(out)
    stats["chars"] = len(text)
    return text, stats


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
