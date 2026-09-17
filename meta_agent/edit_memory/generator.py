"""The two single-call steps of the edit-memory layer and their validators.

``generate_memory``: (Z, B_j, I_k) -> B_{j+1}. ``update_instruction``:
(Q, I_k) -> I_{k+1} (the addendum only). Each is one ``call_llm`` with a
validator; a failing draft is regenerated once with the check's findings
quoted back; the FINAL draft is used either way (2026-09-17 policy: never
throw away generated content — the remaining findings are recorded in the
call record and the layer's state instead). Only an LLM failure yields no
document, in which case the caller keeps the previous version.

The validators report the required sections, the size cap and — for the
memory — lines that read like a score prediction.
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


PLACEHOLDER = "(not provided by the curator)"


def salvage_curation(text: str, *, node_ids: list[int]) -> tuple[str, list[str]]:
    """Make a curation structurally complete without an LLM call: append a
    ``## Node <id>`` section for every missing node, insert every missing
    subsection heading (at the end of that node's section), and append the
    cross-window / gradient sections — each with a placeholder line, so the
    generator sees explicitly what the curator did not provide. Returns the
    repaired text and the list of insertions made."""
    if not text.strip():
        return text, []
    lines = text.rstrip("\n").split("\n")
    inserted: list[str] = []

    def section_span(head: str) -> tuple[int, int]:
        start = next(i for i, l in enumerate(lines) if l.strip().startswith(head))
        end = len(lines)
        for j in range(start + 1, len(lines)):
            if lines[j].startswith("## "):
                end = j
                break
        return start, end

    for nid in node_ids:
        head = P.NODE_SECTION_HEADING.format(node_id=nid)
        if not any(l.strip().startswith(head) for l in lines):
            lines += ["", head] + [f"### {sub}\n{PLACEHOLDER}" for sub in P.NODE_SUBSECTIONS]
            inserted.append(head)
            continue
        start, end = section_span(head)
        body = "\n".join(lines[start:end])
        missing = [sub for sub in P.NODE_SUBSECTIONS
                   if not re.search(rf"^#{{2,4}}\s+{re.escape(sub)}\b", body, re.M | re.I)]
        if missing:
            add = []
            for sub in missing:
                add += ["", f"### {sub}", PLACEHOLDER]
                inserted.append(f"{head} / {sub}")
            lines[end:end] = add
    for heading in (P.CURATION_CROSS_HEADING, P.CURATION_GRADIENT_HEADING):
        if not any(l.strip().startswith(heading) for l in lines):
            lines += ["", heading, PLACEHOLDER]
            inserted.append(heading)
    return "\n".join(lines) + "\n", inserted


def salvage_q(text: str) -> tuple[str, list[str]]:
    """Same for the instruction audit: append any missing Q section."""
    if not text.strip():
        return text, []
    lines = text.rstrip("\n").split("\n")
    inserted: list[str] = []
    for heading in P.Q_SECTIONS:
        if not any(l.strip().startswith(heading) for l in lines):
            lines += ["", heading, PLACEHOLDER]
            inserted.append(heading)
    return "\n".join(lines) + "\n", inserted


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
                 llm_timeout_s: Optional[float],
                 extra_body: Optional[dict[str, Any]] = None) -> None:
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.base_url = base_url
        self.api_key_env = api_key_env
        self.llm_timeout_s = llm_timeout_s
        self.extra_body = dict(extra_body) if extra_body else None

    def kwargs_plain(self) -> dict[str, Any]:
        """The raw fields (for a SessionConfig)."""
        return {"model": self.model, "reasoning_effort": self.reasoning_effort,
                "base_url": self.base_url, "api_key_env": self.api_key_env,
                "llm_timeout_s": self.llm_timeout_s, "extra_body": self.extra_body}

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
        if self.extra_body:
            kw["extra_body"] = self.extra_body
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
) -> tuple[Optional[str], list[str]]:
    """``(B_{j+1}, remaining_errors)``. The text is ``None`` only when the
    LLM call itself failed; a draft that still fails the check after the one
    retry is returned with its findings."""
    return _generate(
        llm, spec, kind="memory_call",
        render=lambda rejection: P.render_memory_generation_messages(
            previous_memory=previous_memory, curation=curation, addendum=addendum,
            window_meta=window_meta, max_chars=max_chars, rejection=rejection),
        validate=lambda text: validate_memory(text, max_chars=max_chars),
        record_path=record_path, verbose_dir=verbose_dir,
    )


def _generate(
    llm: Callable[..., Any],
    spec: LLMSpec,
    *,
    kind: str,
    render: Callable[[Optional[list[str]]], list[dict[str, str]]],
    validate: Callable[[str], list[str]],
    record_path: Path,
    verbose_dir: Optional[Path],
) -> tuple[Optional[str], list[str]]:
    attempts: list[dict[str, Any]] = []
    rejection: Optional[list[str]] = None
    messages: list = []
    text, errors = "", ["no draft produced"]
    for attempt in (1, 2):
        messages = render(rejection)
        try:
            text, meta = _one_call(llm, spec, messages)
        except Exception as exc:  # noqa: BLE001 - call_llm already retried
            attempts.append({"attempt": attempt, "error": f"{type(exc).__name__}: {exc}"})
            _record(record_path, kind=kind, messages=messages, attempts=attempts,
                    accepted=False, verbose_dir=verbose_dir)
            return None, [f"LLM call failed: {type(exc).__name__}: {exc}"]
        errors = validate(text) if text.strip() else ["the document is empty"]
        attempts.append({"attempt": attempt, **meta, "chars": len(text), "errors": errors})
        if not errors:
            break
        rejection = errors
    if not text.strip():
        _record(record_path, kind=kind, messages=messages, attempts=attempts,
                accepted=False, verbose_dir=verbose_dir)
        return None, errors
    # Accepted either way; ``accepted`` records whether the check passed.
    _record(record_path, kind=kind, messages=messages + [{"role": "assistant", "content": text}],
            attempts=attempts, accepted=not errors, verbose_dir=verbose_dir)
    return text + ("\n" if not text.endswith("\n") else ""), errors


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
) -> tuple[Optional[str], list[str]]:
    """``(I_{k+1}, remaining_errors)`` — same policy as ``generate_memory``."""
    return _generate(
        llm, spec, kind="update_call",
        render=lambda rejection: P.render_instruction_update_messages(
            addendum=addendum, q=q, previous_q=previous_q, max_chars=max_chars,
            rejection=rejection),
        validate=lambda text: validate_addendum(text, max_chars=max_chars),
        record_path=record_path, verbose_dir=verbose_dir,
    )
