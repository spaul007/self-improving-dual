"""Task-agent self-reflection as evolution evidence (opt-in, off by default).

After a node's evaluation batch, the task agent's own saved sessions are replayed with
ONE extra user turn (no tools) and the answers are given to the meta-agent:

* ``post_grading`` (failed cases): the role is told the task was not solved and -- per
  ``grading_detail`` -- what the grader reported, then asked WHERE / WHY it went wrong,
  WHAT WOULD HAVE CAUGHT IT and the GENERAL LESSON.
* ``unsure`` (any case, blind -- nothing about grading): which parts of its submission it
  is unsure of, each with a 0-100 confidence.
* ``essential`` (resolved cases): what was essential, and the rules the next version of
  the harness must KEEP.

The LLM is stateless, so replaying the saved message list plus one turn IS the continued
session. Pilots on a coding agent found the post-grading answers grounded, mostly correct
and stable across samples; the blind "unsure" list often named the real defect; agents'
own "was it my fault" verdicts over-blame themselves (hence: only lessons are passed on by
default, never verdicts).

Projects opt in by giving their scorer two optional methods (same pattern as
``aggregate``):

* ``reflection_sessions(case, round_dir) -> {role: {"messages": [...], "format":
  "chat"|"responses", "preamble": str, "model"?: str, "base_url"?: str}}`` -- the role's
  saved conversation as it was sent to the model (``model``/``base_url``: the endpoint it ran
  on, used when the reflector config does not name one);
* ``grading_outcome(case, detail) -> {"text": str, "redact": [str, ...]}`` -- what the
  grader reported (``detail`` is ``"full"`` or ``"numeric"``) and the terms (hidden test
  names, file paths...) that must never reach the meta-agent under ``exposure:
  lessons_only``.

A project without them is reported once as unsupported and nothing else happens. A
scorer whose sessions come from ``platform_core.session_log`` also sets
``needs_session_log = True``; the reflector then exports ``META_AGENT_SESSION_LOG=1``
(inherited by every case subprocess). See meta_agent/reflection_hooks.py.

Storage: ``<round_dir>/reflections/<case>.<role>.<mode>.json`` -- deliberately OUTSIDE
``logs/`` so agentic meta-agent roles (which may read ``logs/``) never see raw answers.
What the meta-agent sees is :meth:`Reflector.render_for_steering` under ``exposure``.
"""
from __future__ import annotations

import concurrent.futures as cf
import hashlib
import json
import os
import random
import tempfile
import re
import time
from pathlib import Path
from typing import Any, Callable, Optional

from .models import CaseResult, EvaluationResult
from .registry import register

MODES = ("post_grading", "unsure", "essential")
EXPOSURES = ("lessons_only", "full", "off")
DETAILS = ("full", "numeric", "none")
CONSUMERS = ("failure_summarizer", "block_suggester", "editor")
PHASES = ("root", "expand", "finalize")
REFLECTIONS_DIR = "reflections"

_POST_GRADING = """=== NEW TURN: post-grading review. No tools are available in this turn; do not call any tool. ===

Your team's final submission for this task has now been graded: NOT solved.
{grading}
{preamble}

Using only what you can see in this session, answer under these four headings:

1. WHERE: the specific step(s) or decision(s) in this session where your work went wrong or where you missed
   the problem. Quote a few words of your own message so it can be located.
2. WHY: why you made that decision at the time -- what you believed, and which instruction, evidence, handoff
   or time pressure led you there.
3. WHAT WOULD HAVE CAUGHT IT: a concrete change to your instructions, your tools, the handoff between roles, or
   your time budget. Say which of those, and what exactly.
4. GENERAL LESSON: one or two rules that would help on OTHER tasks. Do not name this task's specific files,
   functions, tests or APIs. If you cannot give a general lesson, say so."""

_UNSURE = """=== NEW TURN: self-assessment before grading. No tools are available; do not call any tool. ===
The final submission has NOT been graded yet, and you will not be told the result. {preamble}
1. UNSURE PARTS: list the specific behaviours, requirements, edge cases or interface details you are unsure your
   team's final submission handles correctly. For each: why you are unsure, and a confidence 0-100 that it is
   handled correctly. List at most 8, most doubtful first, one per line as `- [<confidence>] <item>`.
2. OVERALL: on its own line write `OVERALL_CONFIDENCE: <0-100>`.
3. FIRST CHECK: if you had 30 more minutes, the single thing you would check first."""

_ESSENTIAL = """=== NEW TURN: the final submission was graded and PASSED. No tools are available; do not call any tool. ===
{preamble}
1. ESSENTIAL: which of your steps or decisions in this session were essential to getting it right? Quote them briefly.
2. CLOSE CALLS: where were you closest to getting it wrong, and what saved you?
3. KEEP: if someone rewrites your instructions or the harness, what behaviour must they NOT lose? One or two
   rules, without this task's specific names."""

_TEMPLATES = {"post_grading": _POST_GRADING, "unsure": _UNSURE, "essential": _ESSENTIAL}


def clean_messages(msgs: list[dict]) -> list[dict]:
    """The saved chat list as sent to an OpenAI-compatible server: keep role/content/
    tool_calls/tool_call_id/name/reasoning; an assistant turn always has ``content``."""
    keep = ("role", "content", "tool_calls", "tool_call_id", "name", "reasoning")
    out = []
    for m in msgs or []:
        if not isinstance(m, dict):
            continue
        mm = {k: m[k] for k in keep if k in m and m[k] is not None}
        if mm.get("role") == "assistant" and "content" not in mm:
            mm["content"] = ""
        out.append(mm)
    return out


_HEADINGS = ("WHERE|WHY|WHAT WOULD HAVE CAUGHT IT|GENERAL LESSON|UNSURE PARTS|OVERALL|FIRST CHECK|"
             "ESSENTIAL|CLOSE CALLS|KEEP")


def _section(text: str, heading: str) -> str:
    """Body of a numbered/bold heading that STARTS a line (so a word inside prose or a
    JSON key never matches), up to the next numbered heading."""
    m = re.search(rf"(?mi)^[\s>#*]*(?:\d+[.)]\s*)?\**\s*{heading}\b[^\n:]*:?", text or "")
    if not m:
        return ""
    rest = text[m.end():]
    # End at the next TEMPLATE heading only -- a numbered list inside the section
    # ("**4. GENERAL LESSON**\n\n1. When ...") must not cut it short.
    n = re.search(rf"(?mi)^[\s>#*]*\d+[.)]\s*\**\s*(?:{_HEADINGS})\b", rest)
    return (rest[: n.start()] if n else rest).strip(" :*\n")


def _json_obj(text: str) -> Optional[dict]:
    """The first JSON object in an answer (agents prompted to reply in JSON keep doing so)."""
    text = re.sub(r"(?s)<think>.*?</think>", "", text or "")
    start = text.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                esc = (ch == "\\") and not esc
                if ch == '"' and not esc:
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start:i + 1])
                    except ValueError:
                        break
                    return obj if isinstance(obj, dict) else None
        start = text.find("{", start + 1)
    return None


def _walk(obj: Any):
    """(key, value) pairs of a nested dict, depth first."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield str(k).lower(), v
            yield from _walk(v)


def _flat(v: Any) -> str:
    if isinstance(v, list):
        return " ".join(_flat(x) for x in v)
    if isinstance(v, dict):
        return " ".join(f"{k}: {_flat(x)}" for k, x in v.items())
    return str(v if v is not None else "").strip()


def _from_json(obj: dict, *needles: str) -> str:
    for k, v in _walk(obj):
        if any(n in k for n in needles):
            return _flat(v)
    return ""


def parse_reflection(mode: str, text: str) -> dict[str, Any]:
    """Pull the parts the meta-agent may see out of an answer: numbered free text
    (the template's own format) or a JSON object (agents whose system prompt demands
    JSON answer the reflection in JSON too)."""
    text = text or ""
    obj = _json_obj(text) if not re.search(r"(?m)^\s*\d+[.)]\s", text) else None
    if mode == "post_grading":
        if obj:
            return {"lesson": _from_json(obj, "lesson"), "catch": _from_json(obj, "caught", "catch")}
        return {"lesson": _section(text, "GENERAL LESSON"),
                "catch": _section(text, "WHAT WOULD HAVE CAUGHT IT")}
    if mode == "unsure":
        items: list[dict] = []
        overall = None
        if obj:
            for k, v in _walk(obj):
                if "overall" in k and isinstance(v, (int, float)):
                    overall = int(v)
                if "unsure" in k and isinstance(v, list):
                    for it in v:
                        if isinstance(it, dict):
                            c = next((x for kk, x in it.items() if "conf" in str(kk).lower()), None)
                            body = _flat({kk: x for kk, x in it.items() if "conf" not in str(kk).lower()})
                            if isinstance(c, (int, float)):
                                items.append({"confidence": int(c), "item": body})
        else:
            for line in text.splitlines():
                m = re.match(r"\s*[-*]?\s*\[\s*(\d{1,3})\s*\]\s*[-:)]?\s*(.+)", line)
                if m and 0 <= int(m.group(1)) <= 100:
                    items.append({"confidence": int(m.group(1)), "item": m.group(2).strip()})
            c = re.search(r"OVERALL_CONFIDENCE:\s*\**\s*(\d{1,3})", text)
            overall = int(c.group(1)) if c else None
        return {"items": items[:8], "overall_confidence": overall}
    if mode == "essential":
        return {"keep": _from_json(obj, "keep") if obj else _section(text, "KEEP")}
    return {}


_REASK = ("That reply continued the task. The task is over and no tools are available. Do not plan or "
          "make further changes: answer the numbered questions above, under their headings, now.")


def _answered(parsed: dict) -> bool:
    """Whether a parsed answer contains anything the meta-agent can use."""
    return bool(parsed.get("lesson") or parsed.get("catch") or parsed.get("keep") or parsed.get("items")
                or parsed.get("overall_confidence") is not None)


def redact(text: str, terms: list[str]) -> str:
    """Replace every redact term (>= 4 chars, longest first) with ``[redacted]``."""
    for t in sorted({t for t in terms if t and len(t) >= 4}, key=len, reverse=True):
        text = re.sub(re.escape(t), "[redacted]", text, flags=re.I)
    return text


@register("reflector", "default")
class Reflector:
    """See the module docstring. Constructed by ``build_components`` from the
    ``reflector:`` config section; ``llm_caller`` / ``scorer`` / ``task_agent`` are
    injected. The model defaults to the TASK agent's own (the reflection should come
    from the same model that did the work)."""

    def __init__(
        self,
        *,
        llm_caller: Optional[Callable[..., Any]] = None,
        scorer: Any = None,
        task_agent: Any = None,
        modes: list[str] = list(MODES),
        on_failed: bool = True,
        on_resolved: bool = True,
        max_cases_per_batch: int = 8,
        roles: Optional[list[str]] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        temperature: float = 1.0,
        top_p: Optional[float] = 0.95,
        top_k: Optional[int] = None,
        max_output_tokens: int = 32768,
        ctx_tokens: int = 262144,
        chars_per_token: float = 3.2,
        concurrency: int = 3,
        grading_detail: str = "full",
        exposure: str = "lessons_only",
        consumers: list[str] = list(CONSUMERS),
        phases: list[str] = ["expand"],
        max_steering_chars: int = 6000,
        seed: int = 0,
        chat_caller: Optional[Callable[..., dict]] = None,
    ) -> None:
        for name, vals, allowed in (("modes", modes, MODES), ("consumers", consumers, CONSUMERS),
                                    ("phases", phases, PHASES)):
            bad = sorted(set(vals) - set(allowed))
            if bad:
                raise ValueError(f"reflector.{name}: unknown {bad}; allowed {list(allowed)}")
        if exposure not in EXPOSURES:
            raise ValueError(f"reflector.exposure must be one of {EXPOSURES}, got {exposure!r}")
        if grading_detail not in DETAILS:
            raise ValueError(f"reflector.grading_detail must be one of {DETAILS}, got {grading_detail!r}")
        self.llm_caller, self.scorer = llm_caller, scorer
        self.modes, self.on_failed, self.on_resolved = list(modes), on_failed, on_resolved
        self.max_cases_per_batch, self.roles = max_cases_per_batch, roles
        ta = task_agent
        self.model = model or getattr(ta, "model", None)
        self.base_url = base_url or getattr(ta, "base_url", None)
        self.reasoning_effort = reasoning_effort or getattr(ta, "reasoning_effort", None)
        self.api_key = api_key
        self.temperature, self.top_p, self.top_k = temperature, top_p, top_k
        self.max_output_tokens, self.ctx_tokens = max_output_tokens, ctx_tokens
        self.chars_per_token, self.concurrency = chars_per_token, max(1, concurrency)
        self.grading_detail, self.exposure = grading_detail, exposure
        self.consumers, self.phases = list(consumers), list(phases)
        self.max_steering_chars = max_steering_chars
        self._rng = random.Random(seed)
        self._chat_caller = chat_caller or self._openai_chat
        self._warned_unsupported = False
        if self.supported and getattr(scorer, "needs_session_log", False):
            os.environ["META_AGENT_SESSION_LOG"] = "1"

    # ------------------------------------------------------------------ hooks
    @property
    def supported(self) -> bool:
        return all(callable(getattr(self.scorer, m, None))
                   for m in ("reflection_sessions", "grading_outcome"))

    def wants(self, phase: str, consumer: Optional[str] = None) -> bool:
        return phase in self.phases and (consumer is None or consumer in self.consumers)

    # ------------------------------------------------------------------ main entry
    def reflect(self, round_dir: Path, batch: EvaluationResult, *, phase: str = "expand") -> dict:
        """Reflect on one evaluation batch of the node at ``round_dir``. Never raises;
        returns counters (logged by the manager as ``reflections=k/n``)."""
        stats = {"cases": 0, "calls": 0, "ok": 0, "off_task": 0, "skipped": 0, "errors": 0}
        if phase not in self.phases:
            return stats
        if not self.supported:
            if not self._warned_unsupported:
                print("[reflector] this project's scorer has no reflection_sessions/grading_outcome "
                      "-- reflection disabled", flush=True)
                self._warned_unsupported = True
            return stats
        jobs = []
        for case in self._select(batch.per_case or []):
            stats["cases"] += 1
            try:
                sessions = self.scorer.reflection_sessions(case, Path(round_dir)) or {}
            except Exception as exc:  # noqa: BLE001
                print(f"[reflector] {case.case_id}: reflection_sessions failed: {exc!r}", flush=True)
                stats["errors"] += 1
                continue
            for role, sess in sessions.items():
                if self.roles and role not in self.roles:
                    continue
                for mode in self._modes_for(case):
                    jobs.append((case, role, sess, mode))
        out = Path(round_dir) / REFLECTIONS_DIR
        out.mkdir(parents=True, exist_ok=True)
        with cf.ThreadPoolExecutor(max_workers=self.concurrency) as ex:
            for res in ex.map(lambda j: self._one(out, *j), jobs):
                stats["calls"] += 1
                stats[res] += 1
        return stats

    def _select(self, cases: list[CaseResult]) -> list[CaseResult]:
        """Failed (nearest misses first) then resolved, up to ``max_cases_per_batch``.
        Infra-excluded cases are never reflected on."""
        usable = [c for c in cases if not (c.details or {}).get("excluded") and not c.error]
        failed = sorted([c for c in usable if not c.passed], key=lambda c: -float(c.score or 0.0))
        passed = [c for c in usable if c.passed]
        self._rng.shuffle(passed)
        picked = (failed if self.on_failed else []) + (passed if self.on_resolved else [])
        return picked[: self.max_cases_per_batch]

    def _modes_for(self, case: CaseResult) -> list[str]:
        out = []
        if "unsure" in self.modes:
            out.append("unsure")
        if not case.passed and "post_grading" in self.modes and self.grading_detail != "none":
            out.append("post_grading")
        if case.passed and "essential" in self.modes:
            out.append("essential")
        return out

    def _one(self, out: Path, case: CaseResult, role: str, sess: dict, mode: str) -> str:
        dest = out / f"{_safe(case.case_id)}.{_safe(role)}.{mode}.json"
        rec: dict[str, Any] = {"case_id": case.case_id, "role": role, "mode": mode,
                               "passed": bool(case.passed), "score": case.score,
                               "grading_detail": self.grading_detail}
        try:
            fmt = sess.get("format", "chat")
            msgs = clean_messages(sess.get("messages") or []) if fmt == "chat" else list(sess.get("messages") or [])
            preamble = sess.get("preamble") or f"You are the {role} role; this is your own session."
            # Redact terms for EVERY mode: a solved case's answer can quote the ground
            # truth itself (e.g. the exact product ids). The grader's text goes into the
            # question only for post_grading.
            detail = self.grading_detail if self.grading_detail != "none" else "numeric"
            g = self.scorer.grading_outcome(case, detail) or {}
            terms = list(g.get("redact") or [])
            grading = (g.get("text") or "") if mode == "post_grading" else ""
            turn = _TEMPLATES[mode].format(preamble=preamble, grading=grading)
            msgs = msgs + [{"role": "user", "content": turn}]
            est = int(len(json.dumps(msgs, default=str)) / self.chars_per_token)
            rec.update(redact_terms=terms, est_tokens=est, turn=turn)
            if est + self.max_output_tokens > self.ctx_tokens:
                rec["skipped"] = f"too long ({est} est tokens)"
                status = "skipped"
            else:
                resp = self._call(msgs, fmt, sess.get("model"), sess.get("base_url"))
                parsed = parse_reflection(mode, resp.get("content") or "")
                if (resp.get("content") or "").strip() and not _answered(parsed):
                    # The model CONTINUED its task ("Let me fix it ...") instead of answering
                    # (EXP-049d, 1 of 16). Re-ask once, keeping its off-task reply in context.
                    rec["off_task_first_reply"] = resp
                    retry = msgs + [{"role": "assistant", "content": resp.get("content") or ""},
                                    {"role": "user", "content": _REASK}]
                    resp = self._call(retry, fmt, sess.get("model"), sess.get("base_url"))
                    parsed = parse_reflection(mode, resp.get("content") or "")
                rec["response"], rec["parsed"] = resp, parsed
                if not (resp.get("content") or "").strip():
                    status = "errors"
                else:
                    status = "ok" if _answered(parsed) else "off_task"
        except Exception as exc:  # noqa: BLE001 -- reflection must never break a batch
            rec["error"] = repr(exc)[:500]
            status = "errors"
        _atomic_write(dest, json.dumps(rec, indent=1, default=str))
        return status

    def _call(self, msgs: list[dict], fmt: str, model: Optional[str] = None,
              base_url: Optional[str] = None) -> dict:
        """One reflection call; retried once with a doubled token budget when the
        thinking ate the whole budget (``finish_reason == length``)."""
        model, base_url = self.model or model, self.base_url or base_url
        budget = self.max_output_tokens
        for _ in range(2):
            t0 = time.time()
            if fmt == "responses":
                if self.llm_caller is None:
                    raise RuntimeError("responses-format session but no llm_caller injected")
                r = self.llm_caller(msgs, model=model, base_url=base_url,
                                    reasoning_effort=self.reasoning_effort,
                                    temperature=self.temperature, max_output_tokens=budget)
                resp = {"content": getattr(r, "content", None),
                        "finish_reason": getattr(r, "stop_reason", None)}
            else:
                resp = self._chat_caller(msgs, budget, model=model, base_url=base_url)
            resp["wall_s"] = round(time.time() - t0, 1)
            if resp.get("finish_reason") != "length" or (resp.get("content") or "").strip():
                return resp
            budget = min(budget * 2, max(self.ctx_tokens // 4, budget))
        return resp

    def _openai_chat(self, msgs: list[dict], max_tokens: int, *, model: Optional[str] = None,
                     base_url: Optional[str] = None) -> dict:
        from openai import OpenAI

        client = OpenAI(base_url=base_url or os.environ.get("LLM_BASE_URL"),
                        api_key=self.api_key or os.environ.get("OPENAI_API_KEY") or "EMPTY",
                        timeout=float(os.environ.get("LLM_REQUEST_TIMEOUT_S", "3600")))
        extra: dict[str, Any] = {}
        if self.top_k is not None:
            extra["top_k"] = self.top_k
        if self.reasoning_effort:
            extra["chat_template_kwargs"] = {"reasoning_effort": self.reasoning_effort}
        r = client.chat.completions.create(
            model=model or os.environ.get("LLM_MODEL"), messages=msgs, max_tokens=max_tokens,
            temperature=self.temperature, **({"top_p": self.top_p} if self.top_p is not None else {}),
            **({"extra_body": extra} if extra else {}),
        )
        ch = r.choices[0]
        msg = ch.message
        return {"content": msg.content,
                "reasoning": getattr(msg, "reasoning", None) or getattr(msg, "reasoning_content", None),
                "finish_reason": ch.finish_reason,
                "usage": r.usage.model_dump() if getattr(r, "usage", None) else None}

    # ------------------------------------------------------------------ meta-agent view
    def render_for_steering(self, round_dir: Path) -> str:
        """What the meta-agent may see of this node's reflections (cumulative over
        every batch), under ``exposure``. ``""`` when off or none exist."""
        return render_reflections(Path(round_dir), self.exposure, self.max_steering_chars)


def render_reflections(round_dir: Path, exposure: str = "lessons_only", max_chars: int = 6000) -> str:
    """Module-level so analysis tools can render a run without a configured reflector."""
    if exposure == "off":
        return ""
    files = sorted((round_dir / REFLECTIONS_DIR).glob("*.json"))
    lessons, keeps, unsure, full = [], [], [], []
    for f in files:
        try:
            rec = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        p = rec.get("parsed") or {}
        terms = rec.get("redact_terms") or []
        tag = f"{rec.get('case_id')}/{rec.get('role')}"
        if exposure == "full":
            body = ((rec.get("response") or {}).get("content") or "").strip()
            if body:
                full.append(f"### {tag} ({rec.get('mode')})\n{redact(body, terms)}")
            continue
        if rec.get("mode") == "post_grading" and p.get("lesson"):
            lessons.append(f"- ({tag}) {redact(' '.join(p['lesson'].split()), terms)}")
            if p.get("catch"):
                lessons.append(f"  would have caught it: {redact(' '.join(p['catch'].split()), terms)}")
        elif rec.get("mode") == "essential" and p.get("keep"):
            keeps.append(f"- ({tag}) {redact(' '.join(p['keep'].split()), terms)}")
        elif rec.get("mode") == "unsure":
            for it in (p.get("items") or [])[:3]:
                if it.get("confidence", 100) < 70:
                    unsure.append(f"- ({tag}, {it['confidence']}) {redact(it['item'], terms)}")
    if exposure == "full":
        text = "\n\n".join(full)
    else:
        parts = []
        if lessons:
            parts.append("Lessons the task agent drew from its FAILED runs (its own account -- fallible, "
                         "and biased toward self-blame; weigh against the evidence):\n" + "\n".join(lessons))
        if keeps:
            parts.append("What the task agent says was ESSENTIAL in its SOLVED runs (do not lose these):\n"
                         + "\n".join(keeps))
        if unsure:
            parts.append("Parts the task agent itself was UNSURE of before grading (confidence 0-100):\n"
                         + "\n".join(unsure))
        text = "\n\n".join(parts)
    if len(text) > max_chars:
        text = text[: max_chars - 40].rstrip() + "\n[... truncated]"
    return text


def _safe(s: Any) -> str:
    s = str(s)
    clean = re.sub(r"[^A-Za-z0-9._-]+", "_", s)[:80]
    return clean if clean == s else f"{clean}-{hashlib.sha1(s.encode()).hexdigest()[:6]}"


def _atomic_write(path: Path, text: str) -> None:
    """Write via a temp file in the same directory + ``os.replace`` (a reflection is
    either complete or absent, even if the run is killed mid-write)."""
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
