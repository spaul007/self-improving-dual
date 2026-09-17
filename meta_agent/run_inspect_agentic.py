"""Read-only loaders for the agentic editor's transcripts and the
``edit_memory/`` directory. Pure Python, no Streamlit -- sibling of
``run_inspect.py``.

The transcript format (``agentic/transcript.jsonl``, one JSON object per
line with ``t`` + ``kind``) is shared by the editor session under each
``round_NNN/agentic/`` and by the curator sessions under
``edit_memory/window_NNN/agentic/`` and ``edit_memory/instruction_update_NNN/
agentic/`` (see ``meta_agent/agentic/session.py::Transcript``), so one parser
serves all three.

Nothing here reads ``verbose/*_messages.json`` (the full Responses-API
message list -- large and redundant with the transcript).
"""
from __future__ import annotations

import difflib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional
from zoneinfo import ZoneInfo

from .agentic.policy import MEMORY_DIR_NAME
from .agentic.session import SESSION_NAME, TRANSCRIPT_NAME
from .edit_memory.layer import INSTRUCTION_FILE, MEMORY_FILE, STATE_FILE

# Times in the transcripts/state are UTC epochs; the dashboard shows them in
# the user's local zone.
DISPLAY_TZ = ZoneInfo("America/Los_Angeles")

_MEMORY_VERSION_RE = re.compile(r"edit_memory_v(\d+)\.md$")
_INSTRUCTION_VERSION_RE = re.compile(r"instruction_v(\d+)\.md$")
_WINDOW_DIR_RE = re.compile(r"window_(\d+)$")
_UPDATE_DIR_RE = re.compile(r"instruction_update_(\d+)$")


def fmt_time(t: Optional[float]) -> str:
    if t is None:
        return ""
    try:
        return datetime.fromtimestamp(float(t), tz=DISPLAY_TZ).strftime("%Y-%m-%d %H:%M:%S %Z")
    except (OverflowError, OSError, ValueError):
        return str(t)


def _read_json(path: Path) -> Optional[Any]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _read_text(path: Path) -> Optional[str]:
    try:
        return path.read_text(encoding="utf-8") if path.exists() else None
    except OSError:
        return None


# --------------------------------------------------------------------------- #
# Transcript
# --------------------------------------------------------------------------- #


@dataclass
class ToolCallRec:
    i: int
    call_id: str
    name: str
    input: dict[str, Any]
    result: str
    result_chars: int
    elapsed_s: float
    t: float


@dataclass
class LLMStep:
    """One LLM call and everything that happened because of it: the
    assistant's text, its tool calls (in execution order), any error, and the
    i-less events (validation, nudge, budget reminder, wrap-up) that landed
    before the next call."""
    i: int
    t: float
    n_messages: int = 0
    content: str = ""
    stop_reason: Optional[str] = None
    elapsed_s: Optional[float] = None
    usage: dict[str, int] = field(default_factory=dict)
    tool_calls: list[ToolCallRec] = field(default_factory=list)
    error: Optional[str] = None
    notes: list[dict[str, Any]] = field(default_factory=list)
    cum_input: int = 0
    cum_output: int = 0
    cum_reasoning: int = 0


@dataclass
class Transcript:
    steps: list[LLMStep]
    end: Optional[dict[str, Any]]
    validations: list[dict[str, Any]]
    n_events: int
    t_start: Optional[float]
    t_end: Optional[float]
    # The last line failed to parse -- the session is still writing it.
    truncated_tail: bool = False


def parse_transcript_lines(lines: Iterable[str]) -> Transcript:
    """Group raw transcript events by LLM-call index ``i``. Events carrying
    an ``i`` attach to that step (creating it if ``llm_call`` was somehow
    missed); events without one attach to the most recent step's ``notes``.
    ``end`` is kept separately. Cumulative token counters accumulate in step
    order."""
    steps: dict[int, LLMStep] = {}
    order: list[int] = []
    end: Optional[dict[str, Any]] = None
    validations: list[dict[str, Any]] = []
    n_events = 0
    t_start: Optional[float] = None
    t_end: Optional[float] = None
    truncated_tail = False
    last_i: Optional[int] = None

    raw_lines = [l for l in lines if l.strip()]
    for idx, line in enumerate(raw_lines):
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            if idx == len(raw_lines) - 1:
                truncated_tail = True
                break
            continue
        if not isinstance(ev, dict):
            continue
        n_events += 1
        t = ev.get("t")
        if isinstance(t, (int, float)):
            t_start = t if t_start is None else min(t_start, t)
            t_end = t if t_end is None else max(t_end, t)
        kind = ev.get("kind")
        i = ev.get("i")

        def _step(i_: int, t_: Any) -> LLMStep:
            if i_ not in steps:
                steps[i_] = LLMStep(i=i_, t=float(t_) if isinstance(t_, (int, float)) else 0.0)
                order.append(i_)
            return steps[i_]

        if kind == "end":
            end = ev
        elif kind == "validation":
            validations.append(ev)
            if last_i is not None:
                steps[last_i].notes.append(ev)
        elif isinstance(i, int):
            s = _step(i, t)
            last_i = i
            if kind == "llm_call":
                s.t = float(t) if isinstance(t, (int, float)) else s.t
                s.n_messages = int(ev.get("n_messages") or 0)
            elif kind == "llm_response":
                s.content = str(ev.get("content") or "")
                s.stop_reason = ev.get("stop_reason")
                s.elapsed_s = ev.get("elapsed_s")
                s.usage = {k: int(v) for k, v in (ev.get("usage") or {}).items() if isinstance(v, (int, float))}
            elif kind == "tool_call":
                s.tool_calls.append(
                    ToolCallRec(
                        i=i,
                        call_id=str(ev.get("call_id") or ""),
                        name=str(ev.get("name") or "?"),
                        input=dict(ev.get("input") or {}),
                        result=str(ev.get("result") or ""),
                        result_chars=int(ev.get("result_chars") or 0),
                        elapsed_s=float(ev.get("elapsed_s") or 0.0),
                        t=float(t) if isinstance(t, (int, float)) else 0.0,
                    )
                )
            elif kind == "llm_error":
                s.error = str(ev.get("error") or "")
            else:  # nudge etc. with an index
                s.notes.append(ev)
        else:
            # budget_reminder, wrap_up, nudge-without-i ...
            if last_i is not None:
                steps[last_i].notes.append(ev)

    cum_in = cum_out = cum_reason = 0
    ordered = [steps[i] for i in order]
    for s in ordered:
        cum_in += s.usage.get("input_tokens", 0)
        cum_out += s.usage.get("output_tokens", 0)
        cum_reason += s.usage.get("reasoning_tokens", 0)
        s.cum_input, s.cum_output, s.cum_reasoning = cum_in, cum_out, cum_reason

    return Transcript(
        steps=ordered,
        end=end,
        validations=validations,
        n_events=n_events,
        t_start=t_start,
        t_end=t_end,
        truncated_tail=truncated_tail,
    )


def load_transcript(path: Path) -> Optional[Transcript]:
    text = _read_text(path)
    if text is None:
        return None
    return parse_transcript_lines(text.splitlines())


def load_session(path: Path) -> Optional[dict[str, Any]]:
    d = _read_json(path)
    return d if isinstance(d, dict) else None


def tool_call_counts(tr: Transcript) -> dict[str, int]:
    counts: dict[str, int] = {}
    for s in tr.steps:
        for tc in s.tool_calls:
            counts[tc.name] = counts.get(tc.name, 0) + 1
    return counts


def token_curve(tr: Transcript) -> list[dict[str, int]]:
    return [
        {
            "i": s.i,
            "input": s.usage.get("input_tokens", 0),
            "output": s.usage.get("output_tokens", 0),
            "reasoning": s.usage.get("reasoning_tokens", 0),
            "cum_input": s.cum_input,
            "cum_output": s.cum_output,
            "cum_reasoning": s.cum_reasoning,
        }
        for s in tr.steps
    ]


def _short_path(p: Any) -> str:
    s = str(p or "")
    # Paths in editor calls are usually "$NODE_DIR/task_agent/workflow.py";
    # keep the part after task_agent/ when present.
    if "task_agent/" in s:
        return s.split("task_agent/", 1)[1]
    return s.rsplit("/", 1)[-1] if "/" in s else s


def editor_call_summary(inp: dict[str, Any]) -> str:
    """One-line label for an ``editor`` tool call, e.g. ``str_replace
    workflow.py`` or ``view tool_wrapper.py[10:40]``."""
    cmd = str(inp.get("command") or "?")
    path = _short_path(inp.get("path"))
    if cmd == "view":
        rng = inp.get("view_range") or inp.get("range")
        if isinstance(rng, (list, tuple)) and len(rng) == 2:
            return f"view {path}[{rng[0]}:{rng[1]}]"
        return f"view {path}"
    if cmd == "insert":
        return f"insert {path} after line {inp.get('insert_line', inp.get('line', '?'))}"
    if cmd == "replace_lines":
        return f"replace_lines {path} {inp.get('start_line', '?')}-{inp.get('end_line', '?')}"
    return f"{cmd} {path}"


_CREATE_CAP_LINES = 200


def editor_call_as_diff(inp: dict[str, Any]) -> Optional[str]:
    """Render an editor mutation as unified-diff-ish text for ``st.code(...,
    "diff")``. ``view`` (and unknown commands) return ``None``."""
    cmd = inp.get("command")
    path = _short_path(inp.get("path"))
    if cmd == "str_replace":
        old = str(inp.get("old_str") or "").splitlines()
        new = str(inp.get("new_str") or "").splitlines()
        lines = list(difflib.unified_diff(old, new, fromfile=f"a/{path}", tofile=f"b/{path}", lineterm="", n=2))
        return "\n".join(lines) if lines else f"--- a/{path}\n+++ b/{path}\n(no textual change)"
    if cmd == "insert":
        new = str(inp.get("new_str") or "").splitlines()
        where = inp.get("insert_line", inp.get("line", "?"))
        return "\n".join([f"+++ b/{path}", f"@@ insert after line {where} @@"] + [f"+{l}" for l in new])
    if cmd == "replace_lines":
        new = str(inp.get("new_str") or "").splitlines()
        a, b = inp.get("start_line", "?"), inp.get("end_line", "?")
        return "\n".join([f"+++ b/{path}", f"@@ replace lines {a}-{b} @@"] + [f"+{l}" for l in new])
    if cmd == "create":
        body = str(inp.get("file_text") or inp.get("new_str") or "").splitlines()
        shown = body[:_CREATE_CAP_LINES]
        tail = [f"... ({len(body) - _CREATE_CAP_LINES} more lines)"] if len(body) > _CREATE_CAP_LINES else []
        return "\n".join([f"+++ b/{path}", "@@ create @@"] + [f"+{l}" for l in shown] + tail)
    return None


def load_verbose_prompts(round_dir: Path) -> dict[str, str]:
    """The editor's system + instruction prompts from ``verbose/`` when the
    run was launched with ``verbose: true``. Never the messages JSON."""
    out: dict[str, str] = {}
    for key, name in (("system", "editor_agentic_system.txt"), ("instruction", "editor_agentic_instruction.txt")):
        txt = _read_text(round_dir / "verbose" / name)
        if txt:
            out[key] = txt
    return out


# --------------------------------------------------------------------------- #
# edit_memory/
# --------------------------------------------------------------------------- #


@dataclass
class WindowInfo:
    index: int
    dir: Path
    window: dict[str, Any]
    curation_md: Optional[str]
    memory_call: Optional[dict[str, Any]]
    has_agentic: bool


@dataclass
class InstructionUpdateInfo:
    index: int
    dir: Path
    nodes: list[dict[str, Any]]
    q_md: Optional[str]
    update_call: Optional[dict[str, Any]]
    has_agentic: bool


@dataclass
class EditMemoryInfo:
    dir: Path
    state: dict[str, Any]
    memory_versions: list[tuple[int, Path]]
    instruction_versions: list[tuple[int, Path]]
    windows: list[WindowInfo]
    updates: list[InstructionUpdateInfo]

    @property
    def current_memory_path(self) -> Path:
        return self.dir / MEMORY_FILE

    @property
    def current_instruction_path(self) -> Path:
        return self.dir / INSTRUCTION_FILE


def edit_memory_dir(experiment_dir: Path) -> Optional[Path]:
    d = experiment_dir / MEMORY_DIR_NAME
    return d if d.is_dir() else None


def edit_memory_signature(em_dir: Path) -> tuple:
    """Change-detection key: state.json (rewritten on every event) plus the
    directory's own mtime and child count (new window/update dirs)."""
    sig: list[Any] = []
    for p in (em_dir / STATE_FILE, em_dir):
        try:
            st = p.stat()
            sig.append((st.st_mtime, st.st_size))
        except OSError:
            sig.append((None, None))
    try:
        sig.append(sum(1 for _ in em_dir.iterdir()))
    except OSError:
        sig.append(0)
    return tuple(sig)


def _versioned(em_dir: Path, rx: re.Pattern[str]) -> list[tuple[int, Path]]:
    out: list[tuple[int, Path]] = []
    for p in em_dir.iterdir():
        m = rx.match(p.name)
        if m and p.is_file():
            out.append((int(m.group(1)), p))
    out.sort()
    return out


def load_edit_memory(experiment_dir: Path) -> Optional[EditMemoryInfo]:
    em_dir = edit_memory_dir(experiment_dir)
    if em_dir is None:
        return None
    state = _read_json(em_dir / STATE_FILE)
    if not isinstance(state, dict):
        state = {}

    windows: list[WindowInfo] = []
    updates: list[InstructionUpdateInfo] = []
    for p in em_dir.iterdir():
        if not p.is_dir():
            continue
        m = _WINDOW_DIR_RE.match(p.name)
        if m:
            w = _read_json(p / "window.json")
            windows.append(
                WindowInfo(
                    index=int(m.group(1)),
                    dir=p,
                    window=w if isinstance(w, dict) else {},
                    curation_md=_read_text(p / "curation.md"),
                    memory_call=_read_json(p / "memory_call.json"),
                    has_agentic=(p / "agentic" / TRANSCRIPT_NAME).is_file(),
                )
            )
            continue
        m = _UPDATE_DIR_RE.match(p.name)
        if m:
            nodes = _read_json(p / "nodes.json")
            updates.append(
                InstructionUpdateInfo(
                    index=int(m.group(1)),
                    dir=p,
                    nodes=list(nodes) if isinstance(nodes, list) else [],
                    q_md=_read_text(p / "q.md"),
                    update_call=_read_json(p / "update_call.json"),
                    has_agentic=(p / "agentic" / TRANSCRIPT_NAME).is_file(),
                )
            )
    windows.sort(key=lambda w: w.index)
    updates.sort(key=lambda u: u.index)

    return EditMemoryInfo(
        dir=em_dir,
        state=state,
        memory_versions=_versioned(em_dir, _MEMORY_VERSION_RE),
        instruction_versions=_versioned(em_dir, _INSTRUCTION_VERSION_RE),
        windows=windows,
        updates=updates,
    )


def session_path(agentic_dir: Path) -> Path:
    return agentic_dir / SESSION_NAME


def transcript_path(agentic_dir: Path) -> Path:
    return agentic_dir / TRANSCRIPT_NAME


def events_rows(state: dict[str, Any]) -> list[dict[str, Any]]:
    """``state.events`` flattened for a table, newest last."""
    rows: list[dict[str, Any]] = []
    for ev in state.get("events") or []:
        rows.append(
            {
                "time": fmt_time(ev.get("t")),
                "event": ev.get("event"),
                "node_id": ev.get("node_id"),
                "window": ev.get("window"),
                "memory_version": ev.get("memory_version"),
                "update": ev.get("update"),
                "end_reason": ev.get("end_reason"),
                "error": (str(ev["error"])[:200] if ev.get("error") else None),
            }
        )
    return rows


def node_arm_rows(state: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for nid, pair in (state.get("node_arms") or {}).items():
        arm, ver = (list(pair) + [None, None])[:2] if isinstance(pair, (list, tuple)) else (pair, None)
        try:
            nid_int: Any = int(nid)
        except (TypeError, ValueError):
            nid_int = nid
        rows.append({"node_id": nid_int, "arm": arm, "memory_version": ver})
    rows.sort(key=lambda r: (r["node_id"] if isinstance(r["node_id"], int) else 1 << 30))
    return rows


def diff_markdown(old: str, new: str, *, old_label: str, new_label: str) -> str:
    lines = difflib.unified_diff(
        old.splitlines(), new.splitlines(), fromfile=old_label, tofile=new_label, lineterm="", n=2
    )
    return "\n".join(lines)


def memory_call_rows(call: Optional[dict[str, Any]]) -> list[dict[str, Any]]:
    """``memory_call.json`` / ``update_call.json`` attempts flattened."""
    if not call:
        return []
    rows: list[dict[str, Any]] = []
    for a in call.get("attempts") or []:
        rows.append(
            {
                "attempt": a.get("attempt"),
                "elapsed_s": a.get("elapsed_s"),
                "input_tokens": a.get("input_tokens"),
                "output_tokens": a.get("output_tokens"),
                "chars": a.get("chars"),
                "errors": "; ".join(str(e) for e in (a.get("errors") or [])) or None,
            }
        )
    return rows
