"""The two single-call steps of the edit-memory layer and their validators.

``generate_memory``: (Z, B_j, I_k) -> B_{j+1}. ``update_instruction``:
(Q, I_k) -> I_{k+1} (the addendum only). Each is one ``call_llm`` with a
validator; one retry with the rejection reasons fed back; on a second
failure the caller keeps the previous document.

The validators are the hard constraints in code form: the required
sections exist, the size cap holds, and — for the memory — no line reads
like a score prediction.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Callable, Optional

from .. import verbose_log
from . import prompts as P

# Lines that forecast an evaluation outcome. Word-bounded so "unpredictable"
# or "will reach the 5-call limit" do not trip it; "expected" alone is fine.
PREDICTION_PATTERNS = (
    re.compile(r"\bpredict(s|ed|ion|ions)?\b", re.I),
    re.compile(r"\bexpected (score|gain|improvement|composite|mean)\b", re.I),
    re.compile(r"\bwill (raise|reach|score|improve|increase|boost)\b[^.\n]*"
               r"(\d\.\d+|\d+\s?%|\b(score|composite|mean|pass rate|accuracy)\b)", re.I),
    re.compile(r"[+\-−]\d+(\.\d+)?\s?(points|pts|score|pp)\b", re.I),
    re.compile(r"\bforecast(s|ed)?\b", re.I),
)


def score_prediction_lines(text: str) -> list[str]:
    """Lines of ``text`` that look like a score prediction (empty = clean)."""
    hits: list[str] = []
    for line in text.splitlines():
        if any(p.search(line) for p in PREDICTION_PATTERNS):
            hits.append(line.strip())
    return hits


def validate_memory(text: str, *, max_chars: int) -> list[str]:
    errors: list[str] = []
    if not text.strip():
        return ["the document is empty"]
    for heading in P.MEMORY_SECTIONS:
        if heading not in text:
            errors.append(f"missing section heading {heading!r}")
    if len(text) > max_chars:
        errors.append(f"document is {len(text)} chars; the limit is {max_chars}")
    for line in score_prediction_lines(text)[:5]:
        errors.append(f"reads like a score prediction (not allowed): {line[:160]!r}")
    return errors


def validate_addendum(text: str, *, max_chars: int) -> list[str]:
    errors: list[str] = []
    if P.CORE_SENTINEL in text or "END OF FIXED CORE" in text:
        errors.append("the addendum must not repeat the fixed core / its sentinel")
    if len(text) > max_chars:
        errors.append(f"addendum is {len(text)} chars; the limit is {max_chars}")
    for line in score_prediction_lines(text)[:5]:
        errors.append(f"asks for a score prediction (not allowed): {line[:160]!r}")
    return errors


def validate_curation(text: str, *, node_ids: list[int]) -> list[str]:
    """The memory curator's document: one section per live node with every
    subsection, plus the cross-window and gradient sections."""
    errors: list[str] = []
    if not text.strip():
        return ["the document is empty"]
    for nid in node_ids:
        head = P.NODE_SECTION_HEADING.format(node_id=nid)
        if head not in text:
            errors.append(f"missing section {head!r}")
            continue
        body = text.split(head, 1)[1]
        # up to the next node section / cross heading
        cut = len(body)
        for other in [P.NODE_SECTION_HEADING.format(node_id=o) for o in node_ids if o != nid] + \
                     [P.CURATION_CROSS_HEADING, P.CURATION_GRADIENT_HEADING]:
            i = body.find(other)
            if i != -1:
                cut = min(cut, i)
        body = body[:cut]
        for sub in P.NODE_SUBSECTIONS:
            if not re.search(rf"^#{{2,4}}\s+{re.escape(sub)}\b", body, re.M | re.I):
                errors.append(f"section {head!r} lacks subsection {sub!r}")
    for heading in (P.CURATION_CROSS_HEADING, P.CURATION_GRADIENT_HEADING):
        if heading not in text:
            errors.append(f"missing section heading {heading!r}")
    return errors


def validate_q(text: str) -> list[str]:
    errors: list[str] = []
    if not text.strip():
        return ["the document is empty"]
    for heading in P.Q_SECTIONS:
        if heading not in text:
            errors.append(f"missing section heading {heading!r}")
    for line in score_prediction_lines(text)[:5]:
        errors.append(f"asks for a score prediction (not allowed): {line[:160]!r}")
    return errors


# ---------------------------------------------------------------------- #
# The two calls
# ---------------------------------------------------------------------- #

class LLMSpec:
    """Model kwargs the layer threads into ``call_llm`` for its own calls."""

    def __init__(self, *, model: Optional[str], reasoning_effort: Optional[str],
                 base_url: Optional[str], api_key_env: Optional[str],
                 llm_timeout_s: Optional[float]) -> None:
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.base_url = base_url
        self.api_key_env = api_key_env
        self.llm_timeout_s = llm_timeout_s

    def kwargs_plain(self) -> dict[str, Any]:
        """The raw fields (for a SessionConfig)."""
        return {"model": self.model, "reasoning_effort": self.reasoning_effort,
                "base_url": self.base_url, "api_key_env": self.api_key_env,
                "llm_timeout_s": self.llm_timeout_s}

    def kwargs(self) -> dict[str, Any]:
        """``call_llm`` kwargs for a single (non-session) call."""
        kw: dict[str, Any] = {}
        if self.model:
            kw["model"] = self.model
        if self.reasoning_effort:
            kw["reasoning_effort"] = self.reasoning_effort
        else:
            kw["temperature"] = 0.2
        if self.base_url:
            kw["base_url"] = self.base_url
        if self.api_key_env:
            kw["api_key_env"] = self.api_key_env
        if self.llm_timeout_s:
            kw["timeout_s"] = self.llm_timeout_s
        return kw


def _one_call(llm: Callable[..., Any], spec: LLMSpec, messages: list[dict[str, str]]) -> tuple[str, dict[str, Any]]:
    t0 = time.time()
    response = llm(messages=messages, **spec.kwargs())
    text = getattr(response, "content", None) or ""
    raw = getattr(response, "raw", None)
    usage = getattr(raw, "usage", None) if raw is not None else None
    meta = {
        "elapsed_s": round(time.time() - t0, 3),
        "input_tokens": getattr(usage, "input_tokens", None),
        "output_tokens": getattr(usage, "output_tokens", None),
    }
    return text.strip(), meta


def _record(path: Path, *, kind: str, messages: list, attempts: list[dict[str, Any]],
            accepted: bool, verbose_dir: Optional[Path]) -> None:
    path.write_text(json.dumps({
        "kind": kind, "accepted": accepted, "attempts": attempts,
        "n_messages": len(messages),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    if verbose_dir is not None and verbose_log.is_enabled():
        verbose_log.write_json(verbose_dir, f"{kind}_messages.json", messages)


def generate_memory(
    llm: Callable[..., Any],
    spec: LLMSpec,
    *,
    previous_memory: str,
    curation: str,
    addendum: str,
    window_meta: dict[str, Any],
    max_chars: int,
    record_path: Path,
    verbose_dir: Optional[Path] = None,
) -> Optional[str]:
    """B_{j+1}, or ``None`` when both attempts failed validation."""
    attempts: list[dict[str, Any]] = []
    rejection: Optional[list[str]] = None
    messages: list = []
    for attempt in (1, 2):
        messages = P.render_memory_generation_messages(
            previous_memory=previous_memory, curation=curation, addendum=addendum,
            window_meta=window_meta, max_chars=max_chars, rejection=rejection,
        )
        try:
            text, meta = _one_call(llm, spec, messages)
        except Exception as exc:  # noqa: BLE001 - call_llm already retried
            attempts.append({"attempt": attempt, "error": f"{type(exc).__name__}: {exc}"})
            _record(record_path, kind="memory_call", messages=messages, attempts=attempts,
                    accepted=False, verbose_dir=verbose_dir)
            return None
        errors = validate_memory(text, max_chars=max_chars)
        attempts.append({"attempt": attempt, **meta, "chars": len(text), "errors": errors})
        if not errors:
            _record(record_path, kind="memory_call", messages=messages + [{"role": "assistant", "content": text}],
                    attempts=attempts, accepted=True, verbose_dir=verbose_dir)
            return text + ("\n" if not text.endswith("\n") else "")
        rejection = errors
    _record(record_path, kind="memory_call", messages=messages, attempts=attempts,
            accepted=False, verbose_dir=verbose_dir)
    return None


def update_instruction(
    llm: Callable[..., Any],
    spec: LLMSpec,
    *,
    addendum: str,
    q: str,
    previous_q: list[str],
    max_chars: int,
    record_path: Path,
    verbose_dir: Optional[Path] = None,
) -> Optional[str]:
    """I_{k+1} (the addendum), or ``None`` when both attempts failed."""
    attempts: list[dict[str, Any]] = []
    rejection: Optional[list[str]] = None
    messages: list = []
    for attempt in (1, 2):
        messages = P.render_instruction_update_messages(
            addendum=addendum, q=q, previous_q=previous_q, max_chars=max_chars,
            rejection=rejection,
        )
        try:
            text, meta = _one_call(llm, spec, messages)
        except Exception as exc:  # noqa: BLE001
            attempts.append({"attempt": attempt, "error": f"{type(exc).__name__}: {exc}"})
            _record(record_path, kind="update_call", messages=messages, attempts=attempts,
                    accepted=False, verbose_dir=verbose_dir)
            return None
        errors = validate_addendum(text, max_chars=max_chars)
        attempts.append({"attempt": attempt, **meta, "chars": len(text), "errors": errors})
        if not errors:
            _record(record_path, kind="update_call", messages=messages + [{"role": "assistant", "content": text}],
                    attempts=attempts, accepted=True, verbose_dir=verbose_dir)
            return text + ("\n" if not text.endswith("\n") else "")
        rejection = errors
    _record(record_path, kind="update_call", messages=messages, attempts=attempts,
            accepted=False, verbose_dir=verbose_dir)
    return None
