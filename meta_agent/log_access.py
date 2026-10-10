"""Read-only access to a node's evaluation logs for agentic meta-agent roles.

Shared by the BlockSuggester (``agentic_access``) and the AgentEditor
(``agentic_log_access``) so both read the same evidence the same way:

* alias-rooted paths -- ``harness/<rel>`` (the current source, served from an
  in-memory ``sources`` dict), ``logs/<rel>`` (the node's real per-case
  evaluation logs under ``<round_dir>/logs``; escaping that root is refused),
  ``eval_result.json`` and ``cases/<rel>`` (the run's per-test-case reflection
  files, ``<run_dir>/case_reflections`` -- see meta_agent/case_reflections.py);
* ``read_file`` pages by line (``offset``/``limit``, default 200 lines) so a
  large log never floods the context;
* ``grep`` returns a window CENTERED on each match (a single JSON line can hold
  thousands of chars; returning only a line's start hid real matches).

Pure functions; never write anything.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional

READ_PAGE_LINES = 200
GREP_DEFAULT_MATCHES = 12
CASES_DIR = "case_reflections"  # == case_reflections.CASES_DIR (no import: this module stays leaf)


def resolve(
    path: str, *, sources: dict[str, str], round_dir: Path
) -> tuple[Optional[str], Any]:
    """``("harness", text)`` | ``("file", Path)`` | ``(None, error_message)``."""
    path = (path or "").strip().lstrip("/")
    if path == "eval_result.json":
        return "file", round_dir / "eval_result.json"
    if path.startswith("harness/"):
        rel = path[len("harness/"):]
        if rel not in sources:
            return None, (
                f"ERROR: {path!r} not found -- available harness "
                f"files: {', '.join(sorted(sources)) or '(none)'}"
            )
        return "harness", sources[rel]
    if path == "logs" or path.startswith("logs/"):
        rel = path[len("logs/"):] if path.startswith("logs/") else ""
        logs_root = (round_dir / "logs").resolve()
        target = (round_dir / "logs" / rel).resolve()
        if target != logs_root and logs_root not in target.parents:
            return None, f"ERROR: {path!r} escapes the logs/ root."
        return "file", target
    if path == "cases" or path.startswith("cases/"):
        rel = path[len("cases/"):] if path.startswith("cases/") else ""
        cases_root = (round_dir.parent / CASES_DIR).resolve()
        target = (round_dir.parent / CASES_DIR / rel).resolve()
        if target != cases_root and cases_root not in target.parents:
            return None, f"ERROR: {path!r} escapes the cases/ root."
        if not cases_root.is_dir():
            return None, "ERROR: no per-test-case reflection files in this run (cases/ is empty)."
        return "file", target
    return None, (
        f"ERROR: unrecognized path {path!r} -- paths must be exactly "
        "'eval_result.json' or start with 'harness/', 'logs/' or 'cases/'."
    )


def _load(kind: str, payload: Any, raw_path: str) -> tuple[Optional[str], str]:
    if kind == "harness":
        return payload, ""
    fpath = payload
    if fpath.is_dir():
        try:
            entries = sorted(e.name + ("/" if e.is_dir() else "") for e in fpath.iterdir())
        except OSError:
            entries = []
        return None, (f"ERROR: {raw_path!r} is a directory -- read one of: "
                      + (", ".join(entries[:60]) or "(empty)"))
    if not fpath.exists() or not fpath.is_file():
        return None, f"(file not found: {raw_path})"
    try:
        return fpath.read_text(encoding="utf-8", errors="replace"), ""
    except OSError as exc:
        return None, f"ERROR reading {raw_path}: {exc!r}"


def read_file(sources: dict[str, str], round_dir: Path, args: dict[str, Any]) -> str:
    raw_path = args.get("path") or ""
    kind, payload = resolve(raw_path, sources=sources, round_dir=round_dir)
    if kind is None:
        return payload
    text, err = _load(kind, payload, raw_path)
    if text is None:
        return err
    lines = text.splitlines()
    offset = max(0, int(args.get("offset") or 0))
    limit = args.get("limit")
    limit = int(limit) if limit else READ_PAGE_LINES
    chunk_lines = lines[offset:offset + limit]
    chunk = "\n".join(f"L{offset + i}: {line}" for i, line in enumerate(chunk_lines))
    remaining = len(lines) - (offset + limit)
    if remaining > 0:
        chunk += (
            f"\n\n[... {remaining} more lines -- call read_file again "
            f"with offset={offset + limit} ...]"
        )
    return chunk if chunk else "(empty file or offset past end)"


def grep_text(text: str, pattern: str, max_matches: int = GREP_DEFAULT_MATCHES) -> str:
    try:
        rx = re.compile(pattern)
    except re.error as exc:
        return f"ERROR: invalid regex {pattern!r}: {exc!r}"
    matches: list[str] = []
    for i, line in enumerate(text.splitlines()):
        m = rx.search(line)
        if not m:
            continue
        start = max(0, m.start() - 150)
        end = min(len(line), m.end() + 350)
        prefix = "..." if start > 0 else ""
        suffix = "..." if end < len(line) else ""
        matches.append(f"L{i} (char {m.start()}): {prefix}{line[start:end].strip()}{suffix}")
        if len(matches) >= max_matches:
            break
    return "\n".join(matches) if matches else "(no matches)"


def grep(sources: dict[str, str], round_dir: Path, args: dict[str, Any]) -> str:
    raw_path = args.get("path") or ""
    kind, payload = resolve(raw_path, sources=sources, round_dir=round_dir)
    if kind is None:
        return payload
    text, err = _load(kind, payload, raw_path)
    if text is None:
        return err
    try:
        max_matches = int(args.get("max_matches") or GREP_DEFAULT_MATCHES)
    except (TypeError, ValueError):
        max_matches = GREP_DEFAULT_MATCHES
    return grep_text(text, args.get("pattern") or "", max_matches)


def cases_listing(round_dir: Path, cap: int = 40, max_index_chars: int = 3000) -> str:
    """Prompt text for the run's per-test-case reflection files (meta_agent/case_reflections.py):
    the INDEX table INLINE (every case's pass rate across all evaluated nodes, hardest first --
    capped at ``max_index_chars``), the case files of the cases the node at ``round_dir`` was
    evaluated on, and the instruction to read the targeted cases' files before editing. ``""``
    when the run has no per-test-case files.

    Inlined because a pointer alone was read in only 4 of 7 EXPANDs of the live smoke (and cited
    in 2): the cross-node pass rates are cheap and should always be seen."""
    root = Path(round_dir).parent / CASES_DIR
    index = root / "INDEX.md"
    if not index.is_file():
        return ""
    try:
        ids = [c.get("case_id") for c in
               json.loads((Path(round_dir) / "eval_result.json").read_text(encoding="utf-8")).get("per_case") or []]
    except (OSError, ValueError, AttributeError):
        ids = []
    from .reflector import _safe  # lazy: keeps this module importable on its own
    names = [n for n in dict.fromkeys(f"{_safe(i)}.md" for i in ids if i is not None) if (root / n).is_file()]
    more = f" ... (+{len(names) - cap} more)" if len(names) > cap else ""
    try:
        rows = [ln for ln in index.read_text(encoding="utf-8").splitlines() if ln.startswith("|")]
    except OSError:
        rows = []
    table, used = [], 0
    for ln in rows:
        if used + len(ln) + 1 > max_index_chars:
            table.append(f"| ... {len(rows) - len(table)} more rows in cases/INDEX.md |")
            break
        table.append(ln)
        used += len(ln) + 1
    return ("  - cases/INDEX.md -- every test case of this run, hardest first: its pass rate across ALL "
            "evaluated nodes and the task agent's reflections on each evaluation, linked to the node "
            "(which edit it carried). Self-reported and fallible; the pass rates are measured.\n"
            + (("  cases/ files for the parent's evaluated cases: " + ", ".join(names[:cap]) + more + "\n")
               if names else "")
            + ("\n### Per-test-case history across all nodes (cases/INDEX.md)\n" + "\n".join(table) + "\n"
               if table else "")
            + "\nBEFORE YOU DECIDE WHAT TO CHANGE: read `cases/<file>` for each case you are targeting. It shows "
              "how every earlier node fared on that case, which edit each node carried, and what the task agent "
              "said about its own run -- an earlier node may already have tried the same fix and failed. In your "
              "rationale, cite what you used (e.g. `cases/112.md: nodes 1 and 4 added a check for this; still "
              "failing, the agent says it skipped the final check`).\n")


def load_logs_guide(path: Optional[str]) -> str:
    """Text of a project's logs guide (``logs_guide:`` on the editor / block
    suggester config): what lives under ``logs/`` for THIS project, so the
    meta-agent is pointed at the right evidence instead of a fixed description.
    Relative paths resolve against the working directory (like
    ``strategies_path``). ``""`` when unset; a set-but-unreadable path is reported
    once and ignored (a missing guide must never fail an EXPAND)."""
    if not path:
        return ""
    p = Path(path)
    if not p.is_absolute():
        p = Path.cwd() / p
    try:
        return p.read_text(encoding="utf-8").strip()
    except OSError as exc:
        print(f"[log_access] logs_guide {path!r} unreadable ({exc}); using the default description",
              flush=True)
        return ""
