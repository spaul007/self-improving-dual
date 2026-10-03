"""Helpers for a project scorer's optional reflection hooks (see meta_agent/reflector.py).

A scorer opts in by defining::

    needs_session_log = True                     # sessions come from platform_core.session_log
    def reflection_sessions(self, case, round_dir) -> {role: {"messages", "format", "preamble"}}
    def grading_outcome(self, case, detail) -> {"text": str, "redact": [str]}

``sessions_from_log`` covers the first for every project whose LLM calls go
through ``platform_core.llm_wrapper.call_llm`` or another ``session_log`` writer;
a project with its own transcripts (e.g. deepswe_seedling's pier ``agent/conv``)
implements it directly.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from .models import CaseResult


def sessions_from_log(
    round_dir: Path,
    case: CaseResult,
    role_of: Callable[[dict], Optional[str]],
    *,
    preamble: str = "You are the {role} agent in a multi-agent system; this is your own session.",
) -> dict[str, dict]:
    """``{role: session}`` from ``<round_dir>/logs/scratch/sessions/<case>/``.
    ``role_of(record)`` names the agent from the logged record (typically from its
    system prompt); ``None`` falls back to ``agent<i>``. A role seen more than once
    gets ``.2``, ``.3`` ... in call order."""
    from platform_core import session_log

    out: dict[str, dict] = {}
    recs = session_log.load_sessions(Path(round_dir) / "logs" / "scratch", case.case_id)
    for i, rec in enumerate(recs):
        role = role_of(rec) or f"agent{i + 1}"
        name, k = role, 2
        while name in out:
            name, k = f"{role}.{k}", k + 1
        meta = rec.get("meta") or {}
        out[name] = {
            "messages": rec.get("messages") or [],
            "format": rec.get("format", "chat"),
            "preamble": preamble.format(role=role),
            "model": meta.get("model"),
            "base_url": meta.get("base_url"),
        }
    return out


def system_text(rec: dict) -> str:
    """The first system/developer message's text of a logged session."""
    for m in rec.get("messages") or []:
        if isinstance(m, dict) and m.get("role") in ("system", "developer"):
            c = m.get("content")
            if isinstance(c, list):
                c = " ".join(str(p.get("text", "")) if isinstance(p, dict) else str(p) for p in c)
            return str(c or "")
    return ""


def role_by_keyword(names: "list[str] | dict[str, str]") -> Callable[[dict], Optional[str]]:
    """``role_of`` that picks whichever keyword appears EARLIEST in the session's
    system prompt (case-insensitive). ``names`` is a list of keywords (the keyword
    is the role) or a ``{keyword: role}`` dict."""
    mapping = dict(names) if isinstance(names, dict) else {n: n for n in names}

    def role_of(rec: dict) -> Optional[str]:
        s = system_text(rec).lower()
        hits = [(s.find(k.lower()), r) for k, r in mapping.items() if k.lower() in s]
        return min(hits)[1] if hits else None
    return role_of
