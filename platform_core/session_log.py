"""Opt-in per-case LLM session log (for meta_agent/reflector.py).

When ``META_AGENT_SESSION_LOG=1`` (set by the meta-agent's reflector when it is
enabled), every LLM call made inside a case records the conversation it sent plus
the model's reply to::

    $META_AGENT_SCRATCH_DIR/sessions/<case_id>/<key>.json

``key`` hashes the conversation's opening (its first two messages -- normally the
system prompt and the first user turn), so successive calls of one tool loop
overwrite the same file and the last write is the whole session. Framework code
(this module, ``platform_core.llm_wrapper.call_llm``) does the writing, so an
evolved agent cannot switch it off. Off (the default) = nothing is written and no
code path changes. Never raises.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

ENV = "META_AGENT_SESSION_LOG"
SESSIONS_DIR = "sessions"


def enabled() -> bool:
    return os.environ.get(ENV) == "1"


def _plain(o: Any) -> Any:
    dump = getattr(o, "model_dump", None)
    if callable(dump):
        try:
            return dump(exclude_none=True)
        except Exception:  # noqa: BLE001
            pass
    return str(o)


def _first_text(m: Any) -> str:
    m = _plain(m) if not isinstance(m, dict) else m
    c = m.get("content") if isinstance(m, dict) else None
    if isinstance(c, list):
        c = " ".join(str(p.get("text", "")) if isinstance(p, dict) else str(p) for p in c)
    return str(c or "")


def session_key(messages: list) -> str:
    head = "\x1e".join(_first_text(m) for m in (messages or [])[:2])
    return hashlib.sha1(head.encode("utf-8", "replace")).hexdigest()[:12]


def safe_case_dir(case_id: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9._-]+", "_", str(case_id))[:80]
    return clean if clean == str(case_id) else f"{clean}-{hashlib.sha1(str(case_id).encode()).hexdigest()[:6]}"


def log_session(
    messages: list,
    reply: list,
    *,
    fmt: str,
    case_id: Optional[str] = None,
    meta: Optional[dict] = None,
) -> Optional[Path]:
    """Record ``messages`` + ``reply`` (the assistant output items/messages) for
    the current case. ``fmt`` is ``"chat"`` (chat.completions messages) or
    ``"responses"`` (Responses-API input items). Returns the path, or None when
    disabled / outside a case / on any error."""
    if not enabled():
        return None
    try:
        scratch = os.environ.get("META_AGENT_SCRATCH_DIR")
        if case_id is None:
            from . import trace

            case_id = trace._case_id_ctx.get()
        if not scratch or not case_id:
            return None
        d = Path(scratch) / SESSIONS_DIR / safe_case_dir(case_id)
        d.mkdir(parents=True, exist_ok=True)
        dest = d / f"{session_key(messages)}.json"
        rec = {
            "case_id": case_id,
            "format": fmt,
            "updated": time.time(),
            "meta": meta or {},
            "messages": [m if isinstance(m, dict) else _plain(m) for m in list(messages) + list(reply)],
        }
        fd, tmp = tempfile.mkstemp(dir=d, prefix=f".{dest.name}.", suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(rec, f, default=_plain, ensure_ascii=False)
        os.replace(tmp, dest)
        return dest
    except Exception:  # noqa: BLE001 -- logging must never break a case
        return None


def load_sessions(scratch_dir: Path, case_id: str) -> list[dict]:
    """Every logged session of ``case_id`` under ``scratch_dir``, oldest first."""
    d = Path(scratch_dir) / SESSIONS_DIR / safe_case_dir(case_id)
    out = []
    for f in sorted(d.glob("*.json")) if d.is_dir() else []:
        try:
            out.append(json.loads(f.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            continue
    out.sort(key=lambda r: r.get("updated") or 0)
    return out
