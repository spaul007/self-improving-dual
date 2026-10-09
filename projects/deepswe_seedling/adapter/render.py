"""Render a finished trial's per-role conversations into bounded plain text for the
meta-agent (BlockSuggester agentic read_file/grep over logs/scratch/...).

Why not copy the raw artifacts: ``conv/<role>.<attempt>.json`` is single-line JSON up to
~475 KB and ``trajectory.json`` ~1 MB; the suggester's read_file pages BY LINE, so one read
of a raw file would push ~120K tokens into a 262K context. Rendered output guarantees:
every line <= MAX_LINE chars, each tool output <= MAX_TOOL chars (head + tail), each
reasoning block <= MAX_REASONING chars.

Only the agent's OWN transcript is rendered -- never ``verifier/`` (hidden tests).
"""
from __future__ import annotations

import json
import re
import textwrap
from pathlib import Path
from typing import Any, Iterable

MAX_LINE = 400
MAX_TOOL = 2000
MAX_REASONING = 1500
MAX_CONTENT = 3000
MAX_ARGS = 1200


def _clip(text: str, limit: int) -> str:
    text = text or ""
    if len(text) <= limit:
        return text
    head = limit * 2 // 3
    tail = limit - head
    return f"{text[:head]}\n[... {len(text) - limit} chars elided ...]\n{text[-tail:]}"


def _wrap(text: str) -> list[str]:
    out: list[str] = []
    for line in (text or "").splitlines() or [""]:
        if len(line) <= MAX_LINE:
            out.append(line)
        else:
            out.extend(textwrap.wrap(line, MAX_LINE, break_long_words=True,
                                     replace_whitespace=False, drop_whitespace=False) or [""])
    return out


def _tool_calls(msg: dict) -> Iterable[str]:
    for tc in msg.get("tool_calls") or []:
        fn = (tc.get("function") or {}) if isinstance(tc, dict) else {}
        name = fn.get("name") or "?"
        args = fn.get("arguments") or ""
        try:
            parsed = json.loads(args) if isinstance(args, str) else args
        except ValueError:
            parsed = None
        if isinstance(parsed, dict):
            if name == "Bash":
                desc = parsed.get("description")
                body = f"$ {parsed.get('command', '')}" + (f"    # {desc}" if desc else "")
            elif name in ("Read", "Write", "Edit"):
                keys = {k: v for k, v in parsed.items() if k not in ("content", "old_string", "new_string")}
                sizes = {k: f"{len(str(parsed[k]))} chars" for k in ("content", "old_string", "new_string") if k in parsed}
                body = json.dumps({**keys, **sizes})
            else:
                body = json.dumps(parsed)
        else:
            body = str(args)
        yield f"-> {name}: {_clip(body, MAX_ARGS)}"


def render_conversation(conv: dict) -> str:
    role = conv.get("role", "?")
    attempt = conv.get("attempt", "?")
    lines = [f"# {role.upper()} attempt {attempt}  (stop_reason={conv.get('stop_reason')}, "
             f"messages={conv.get('n_messages')})", ""]
    for i, m in enumerate(conv.get("messages") or []):
        r = m.get("role")
        if r == "system":
            lines += [f"## [{i}] SYSTEM PROMPT ({len(m.get('content') or '')} chars) -- see the role prompt file in harness/", ""]
            continue
        if r == "tool":
            lines += [f"## [{i}] TOOL RESULT", *_wrap(_clip(str(m.get("content") or ""), MAX_TOOL)), ""]
            continue
        head = f"## [{i}] {str(r).upper()}"
        lines.append(head)
        if m.get("reasoning"):
            lines += ["(thinking)", *_wrap(_clip(str(m["reasoning"]), MAX_REASONING))]
        content = m.get("content")
        if content and content != "None":
            lines += _wrap(_clip(str(content), MAX_CONTENT))
        for call in _tool_calls(m):
            lines += _wrap(call)
        lines.append("")
    return "\n".join(lines) + "\n"


def render_trial(trial_dir: Path, dest: Path) -> list[str]:
    """Write one ``<role>.<attempt>.txt`` per conversation snapshot into ``dest``. Returns
    the written file names."""
    dest.mkdir(parents=True, exist_ok=True)
    written = []
    conv_dir = Path(trial_dir) / "agent" / "conv"
    for f in sorted(conv_dir.glob("*.json")) if conv_dir.is_dir() else []:
        if not re.fullmatch(r"[a-z_]+\.\d+\.json", f.name):
            continue
        try:
            conv = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        name = f.name[:-5] + ".txt"
        (dest / name).write_text(render_conversation(conv), encoding="utf-8")
        written.append(name)
    return written


def max_line_len(path: Path) -> int:
    return max((len(l) for l in path.read_text(encoding="utf-8").splitlines()), default=0)


def render_exec_log(trial_dir: Path, dest: Path, max_rows: int = 400) -> bool:
    """A compact, line-bounded table of every tool call (role, step, tool, rc, dur, cmd)."""
    src = Path(trial_dir) / "agent" / "exec_log.jsonl"
    if not src.is_file():
        return False
    rows = []
    for line in src.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
        except ValueError:
            continue
        what = r.get("command") or r.get("path") or ""
        rows.append(f"{r.get('role')}.{r.get('attempt')} s{r.get('step')} {r.get('tool')} "
                    f"rc={r.get('rc')} {r.get('dur_s')}s refused={r.get('refused')} :: "
                    f"{str(what).replace(chr(10), ' ')[:300]}")
    if len(rows) > max_rows:
        rows = rows[: max_rows // 2] + [f"[... {len(rows) - max_rows} rows elided ...]"] + rows[-max_rows // 2:]
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "exec_log.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")
    return True


def run_report(o: dict, dispatches: list[dict]) -> str:
    """The case's ``raw_result`` text: outcome line FIRST, VERIFY's last report LAST (the
    failure report keeps ~500 chars from each end)."""
    from .trial import outcome_line
    lines = [outcome_line(o), ""]
    for r in o.get("role_stats") or []:
        lines.append(
            f"- {r.get('role')}.{r.get('attempt')}: steps={r.get('steps')} wall={r.get('wall_sec')}s/"
            f"{r.get('planned_wall_sec')}s stop={r.get('stop_reason')} edits={r.get('edits')} "
            f"tests_run={r.get('tests_run')} verdict={r.get('verdict')} "
            f"behaviours={r.get('n_behaviours_tested')}/{r.get('n_behaviours')}")
    last_patch = next((d["output"] for d in reversed(dispatches) if d["role"] == "patch"), {})
    last_verify = next((d["output"] for d in reversed(dispatches) if d["role"] == "verify"), {})
    if last_patch:
        lines += ["", "PATCH's last summary: " + _clip(str(last_patch.get("summary") or ""), 600)]
    if last_verify:
        issues = "; ".join(str(i) for i in (last_verify.get("issues") or []))[:500]
        lines += ["", f"VERIFY's last report: verdict={last_verify.get('verdict')} "
                      f"test_command={str(last_verify.get('test_command') or '')[:200]!r} "
                      f"issues={issues!r}"]
    return "\n".join(lines)
