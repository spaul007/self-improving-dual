"""Task-agent self-reflection as evolution evidence (opt-in, off by default).

After a node's evaluation batch, every evaluated test case gets one reflection PER ROLE: the
role's own saved session is replayed (the LLM is stateless, so the saved message list plus new
turns IS the continued session) and asked two turns, with no tools:

* **turn 1 -- blind** (nothing about grading yet): which parts of its team's submission it is
  unsure of, each with a 0-100 confidence, an ``OVERALL_CONFIDENCE`` and the FIRST CHECK it
  would make with more time;
* **turn 2 -- graded** (the outcome is revealed in the same conversation):
  - a FAILED case is told it was not solved and -- per ``grading_detail`` -- what the grader
    reported, then asked WHERE / WHY it went wrong, WHAT WOULD HAVE CAUGHT IT and the GENERAL
    LESSON;
  - a SOLVED case is asked what was ESSENTIAL, its CLOSE CALLS and what a rewrite must KEEP;
  - plus the node's **probe questions**: 1-3 questions the meta-agent wrote when it created this
    node, to test its own hypothesis about the edit (``EvolutionStrategy.probe_questions``).

Pilots on a coding agent found post-grading answers grounded, mostly correct and stable across
samples; the blind list often named the real defect; agents' own "was it my fault" verdicts
over-blame themselves (hence: under ``exposure: lessons_only`` only lessons and the parsed
fields are passed on, never verdicts).

Projects opt in by giving their scorer two optional methods (same pattern as ``aggregate``):

* ``reflection_sessions(case, round_dir) -> {role: {"messages": [...], "format":
  "chat"|"responses", "preamble": str, "model"?: str, "base_url"?: str}}``;
* ``grading_outcome(case, detail) -> {"text": str, "redact": [str, ...]}`` -- what the grader
  reported (``detail`` is ``"full"`` or ``"numeric"``) and the terms (hidden test names, file
  paths...) that must never reach the meta-agent.

A project without them is reported once as unsupported and nothing else happens. A scorer whose
sessions come from ``platform_core.session_log`` also sets ``needs_session_log = True``; the
reflector then exports ``META_AGENT_SESSION_LOG=1``. See meta_agent/reflection_hooks.py.

Storage: one record per (node, case, role, evaluation) at
``<round_dir>/reflections/<case>.<role>.e<k>.json`` (``k`` = the k-th evaluation of that case on
that node, so a re-evaluation never overwrites an earlier one) -- deliberately OUTSIDE ``logs/``,
so agentic meta-agent roles never see raw answers. What the meta-agent sees is redacted and
rendered: :meth:`Reflector.render_for_steering` (the parent node's own reflections) and the
per-test-case files built by :mod:`meta_agent.case_reflections` (every node's reflections on one
case, plus that case's pass rate across nodes).
"""
from __future__ import annotations

import concurrent.futures as cf
import hashlib
import json
import os
import random
import re
import tempfile
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
RECORD_VERSION = 2
MAX_PROBES = 3
MAX_PROBE_CHARS = 300

_BLIND = """=== NEW TURN: self-assessment before grading. No tools are available; do not call any tool. ===
The final submission has NOT been graded yet. {preamble}
1. UNSURE PARTS: list the specific behaviours, requirements, edge cases or interface details you are unsure your
   team's final submission handles correctly. For each: why you are unsure, and a confidence 0-100 that it is
   handled correctly. List at most 8, most doubtful first, one per line as `- [<confidence>] <item>`.
2. OVERALL: on its own line write `OVERALL_CONFIDENCE: <0-100>`.
3. FIRST CHECK: if you had 30 more minutes, the single thing you would check first."""

_GRADED_FAILED = """=== NEW TURN: post-grading review. No tools are available in this turn; do not call any tool. ===

Your team's final submission for this task has now been graded: NOT solved.
{grading}

Using only what you can see in this session (including your self-assessment above), answer under these headings:

1. WHERE: the specific step(s) or decision(s) in this session where your work went wrong or where you missed
   the problem. Quote a few words of your own message so it can be located.
2. WHY: why you made that decision at the time -- what you believed, and which instruction, evidence, handoff
   or time pressure led you there.
3. WHAT WOULD HAVE CAUGHT IT: a concrete change to your instructions, your tools, the handoff between roles, or
   your time budget. Say which of those, and what exactly.
4. GENERAL LESSON: one or two rules that would help on OTHER tasks. Do not name this task's specific files,
   functions, tests or APIs. If you cannot give a general lesson, say so.{probes}"""

_GRADED_PASSED = """=== NEW TURN: the final submission was graded and PASSED. No tools are available; do not call any tool. ===

Using only what you can see in this session (including your self-assessment above), answer under these headings:

1. ESSENTIAL: which of your steps or decisions in this session were essential to getting it right? Quote them briefly.
2. CLOSE CALLS: where were you closest to getting it wrong, and what saved you?
3. KEEP: if someone rewrites your instructions or the harness, what behaviour must they NOT lose? One or two
   rules, without this task's specific names.{probes}"""

_GRADED_PROBES_ONLY = """=== NEW TURN: the final submission was graded: {verdict}. No tools are available; do not call any tool. ===

Using only what you can see in this session, answer under these headings:{probes}"""

_PROBES_INTRO = ("\n\nThe people who last changed your instructions or tools asked these questions about this change. "
                 "Answer each under its own heading, from what you actually did in this session (say so if it "
                 "does not apply):\n")

# Legacy single-turn templates (records written before the two-turn design; parse_reflection
# still reads them so older runs render).
_POST_GRADING, _UNSURE = _GRADED_FAILED, _BLIND
_TEMPLATES = {"unsure": _BLIND, "post_grading": _GRADED_FAILED, "essential": _GRADED_PASSED}


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


def clean_probes(qs: Any) -> list[str]:
    """At most ``MAX_PROBES`` distinct, non-empty, single-line questions of at most
    ``MAX_PROBE_CHARS`` characters (anything else the meta-agent emitted is dropped)."""
    out: list[str] = []
    if isinstance(qs, str):
        qs = [qs]
    for q in qs if isinstance(qs, list) else []:
        q = " ".join(str(q or "").split())[:MAX_PROBE_CHARS]
        if q and q.lower() not in {x.lower() for x in out}:
            out.append(q)
        if len(out) >= MAX_PROBES:
            break
    return out


def _probe_block(probes: list[str], first_number: int) -> str:
    if not probes:
        return ""
    lines = [f"{first_number + i}. PROBE {i + 1}: {q}" for i, q in enumerate(probes)]
    return _PROBES_INTRO + "\n".join(lines)


_HEADINGS = ("WHERE|WHY|WHAT WOULD HAVE CAUGHT IT|GENERAL LESSON|UNSURE PARTS|OVERALL|FIRST CHECK|"
             r"ESSENTIAL|CLOSE CALLS|KEEP|PROBE \d+")


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


def parse_reflection(mode: str, text: str, n_probes: int = 0) -> dict[str, Any]:
    """Pull the parts the meta-agent may see out of an answer: numbered free text (the
    template's own format) or a JSON object (agents whose system prompt demands JSON answer
    the reflection in JSON too). ``mode``: ``unsure`` (turn 1), ``post_grading`` (turn 2,
    failed) or ``essential`` (turn 2, solved); ``n_probes`` probe answers are parsed too."""
    text = text or ""
    obj = _json_obj(text) if not re.search(r"(?m)^\s*\d+[.)]\s", text) else None
    out: dict[str, Any] = {}
    if mode == "post_grading":
        if obj:
            out = {"lesson": _from_json(obj, "lesson"), "catch": _from_json(obj, "caught", "catch"),
                   "where": _from_json(obj, "where"), "why": _from_json(obj, "why")}
        else:
            out = {"lesson": _section(text, "GENERAL LESSON"),
                   "catch": _section(text, "WHAT WOULD HAVE CAUGHT IT"),
                   "where": _section(text, "WHERE"), "why": _section(text, "WHY")}
    elif mode == "unsure":
        items: list[dict] = []
        overall = None
        first = ""
        if obj:
            for k, v in _walk(obj):
                if "overall" in k and isinstance(v, (int, float)):
                    overall = int(v)
                if "first" in k and isinstance(v, str):
                    first = v.strip()
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
            first = _section(text, "FIRST CHECK")
        out = {"items": items[:8], "overall_confidence": overall, "first_check": first}
    elif mode == "essential":
        if obj:
            out = {"keep": _from_json(obj, "keep"), "essential": _from_json(obj, "essential"),
                   "close_calls": _from_json(obj, "close")}
        else:
            out = {"keep": _section(text, "KEEP"), "essential": _section(text, "ESSENTIAL"),
                   "close_calls": _section(text, "CLOSE CALLS")}
    if n_probes:
        answers = []
        for i in range(n_probes):
            a = _from_json(obj, f"probe {i + 1}", f"probe_{i + 1}", f"probe{i + 1}") if obj \
                else _section(text, rf"PROBE {i + 1}")
            answers.append(a)
        out["probe_answers"] = answers
    return out


_REASK = ("That reply continued the task. The task is over and no tools are available. Do not plan or "
          "make further changes: answer the numbered questions above, under their headings, now.")


def _answered(parsed: dict) -> bool:
    """Whether a parsed answer contains anything the meta-agent can use."""
    return bool(parsed.get("lesson") or parsed.get("catch") or parsed.get("keep") or parsed.get("items")
                or parsed.get("essential") or parsed.get("overall_confidence") is not None
                or any(parsed.get("probe_answers") or []))


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
        max_cases_per_batch: Optional[int] = None,
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
        probe_questions: bool = True,
        case_files: bool = True,
        max_case_file_chars: int = 60000,
        max_case_field_chars: Optional[int] = 600,
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
        if max_cases_per_batch is not None and int(max_cases_per_batch) < 1:
            raise ValueError("reflector.max_cases_per_batch must be >= 1 or null (all cases)")
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
        self.probe_questions, self.case_files = bool(probe_questions), bool(case_files)
        self.max_case_file_chars = max_case_file_chars
        self.max_case_field_chars = max_case_field_chars
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
    def reflect(self, round_dir: Path, batch: EvaluationResult, *, phase: str = "expand",
                node_id: Optional[int] = None, parent_id: Optional[int] = None,
                probes: Optional[list[str]] = None) -> dict:
        """Reflect on one evaluation batch of the node at ``round_dir``: one two-turn
        session per (case, role). Never raises; returns counters (logged by the manager as
        ``reflections=ok/calls``; ``calls`` counts sessions, ``llm_calls`` model calls)."""
        stats = {"cases": 0, "calls": 0, "ok": 0, "off_task": 0, "skipped": 0, "errors": 0,
                 "llm_calls": 0}
        if phase not in self.phases:
            return stats
        if not self.supported:
            if not self._warned_unsupported:
                print("[reflector] this project's scorer has no reflection_sessions/grading_outcome "
                      "-- reflection disabled", flush=True)
                self._warned_unsupported = True
            return stats
        probes = clean_probes(probes) if self.probe_questions else []
        out = Path(round_dir) / REFLECTIONS_DIR
        out.mkdir(parents=True, exist_ok=True)
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
                # The evaluation index is fixed HERE (single-threaded), so two sessions can
                # never pick the same file.
                k = len(list(out.glob(f"{_safe(case.case_id)}.{_safe(role)}.e*.json"))) + 1
                dest = out / f"{_safe(case.case_id)}.{_safe(role)}.e{k}.json"
                meta = {"node_id": node_id, "parent_id": parent_id, "phase": phase, "eval_index": k}
                jobs.append((dest, case, role, sess, probes, meta))
        with cf.ThreadPoolExecutor(max_workers=self.concurrency) as ex:
            for status, n_calls in ex.map(lambda j: self._session(*j), jobs):
                stats["calls"] += 1
                stats["llm_calls"] += n_calls
                stats[status] += 1
        return stats

    def _select(self, cases: list[CaseResult]) -> list[CaseResult]:
        """Every usable case of the batch, failed (nearest misses first) then solved;
        capped only when ``max_cases_per_batch`` is set. Infra-excluded or errored cases
        are never reflected on (there is no meaningful session)."""
        usable = [c for c in cases if not (c.details or {}).get("excluded") and not c.error]
        failed = sorted([c for c in usable if not c.passed], key=lambda c: -float(c.score or 0.0))
        passed = [c for c in usable if c.passed]
        self._rng.shuffle(passed)
        picked = (failed if self.on_failed else []) + (passed if self.on_resolved else [])
        return picked if self.max_cases_per_batch is None else picked[: int(self.max_cases_per_batch)]

    def _graded_turn(self, case: CaseResult, grading: str, probes: list[str]) -> tuple[Optional[str], str]:
        """(turn text, parse mode) for turn 2, or (None, "") when nothing is asked."""
        if case.passed and "essential" in self.modes:
            return _GRADED_PASSED.format(probes=_probe_block(probes, 4)), "essential"
        if not case.passed and "post_grading" in self.modes:
            return _GRADED_FAILED.format(grading=grading, probes=_probe_block(probes, 5)), "post_grading"
        if probes:
            verdict = "PASSED" if case.passed else "NOT solved"
            return _GRADED_PROBES_ONLY.format(verdict=verdict, probes=_probe_block(probes, 1)), "probes"
        return None, ""

    def _ask(self, msgs: list[dict], turn: str, mode: str, n_probes: int, sess: dict, fmt: str
             ) -> tuple[dict, dict, int, list[dict]]:
        """One question turn with the off-task re-ask. Returns (response, parsed, model
        calls, the messages including this turn and its answer)."""
        msgs = msgs + [{"role": "user", "content": turn}]
        resp = self._call(msgs, fmt, sess.get("model"), sess.get("base_url"))
        calls = 1
        parse_mode = "post_grading" if mode == "probes" else mode
        parsed = parse_reflection(parse_mode, resp.get("content") or "", n_probes)
        if (resp.get("content") or "").strip() and not _answered(parsed):
            # The model CONTINUED its task ("Let me fix it ...") instead of answering
            # (EXP-049d, 1 of 16). Re-ask once, keeping its off-task reply in context.
            first = resp.get("content") or ""
            msgs = msgs + [{"role": "assistant", "content": first},
                           {"role": "user", "content": _REASK}]
            resp = self._call(msgs, fmt, sess.get("model"), sess.get("base_url"))
            resp["off_task_first_reply"] = first
            calls += 1
            parsed = parse_reflection(parse_mode, resp.get("content") or "", n_probes)
        msgs = msgs + [{"role": "assistant", "content": resp.get("content") or ""}]
        return resp, parsed, calls, msgs

    def _session(self, dest: Path, case: CaseResult, role: str, sess: dict, probes: list[str],
                 meta: dict) -> tuple[str, int]:
        rec: dict[str, Any] = {"version": RECORD_VERSION, "case_id": case.case_id, "role": role,
                               "passed": bool(case.passed), "score": case.score,
                               "grading_detail": self.grading_detail, "probe_questions": probes,
                               "ts": time.time(), **meta}
        calls = 0
        status = "errors"
        try:
            fmt = sess.get("format", "chat")
            msgs = clean_messages(sess.get("messages") or []) if fmt == "chat" else list(sess.get("messages") or [])
            preamble = sess.get("preamble") or f"You are the {role} role; this is your own session."
            # Redact terms for EVERY case: a solved case's answer can quote the ground
            # truth itself (e.g. the exact product ids). The grader's text is shown only
            # for a failed case.
            detail = self.grading_detail if self.grading_detail != "none" else "numeric"
            g = self.scorer.grading_outcome(case, detail) or {}
            rec["redact_terms"] = list(g.get("redact") or [])
            # grading_detail "none": the verdict alone (NOT solved), no grader text.
            grading = (g.get("text") or "") if not case.passed and self.grading_detail != "none" else ""
            turns: list[dict] = []
            parsed: dict[str, Any] = {}
            est = int(len(json.dumps(msgs, default=str)) / self.chars_per_token)
            rec["est_tokens"] = est
            if est + 2 * self.max_output_tokens > self.ctx_tokens:
                rec["skipped"] = f"too long ({est} est tokens)"
                status = "skipped"
            else:
                if "unsure" in self.modes:
                    t1 = _BLIND.format(preamble=preamble)
                    resp, p1, n, msgs = self._ask(msgs, t1, "unsure", 0, sess, fmt)
                    calls += n
                    turns.append({"name": "blind", "question": t1, "response": resp})
                    parsed.update({"unsure_items": p1.get("items") or [],
                                   "overall_confidence": p1.get("overall_confidence"),
                                   "first_check": p1.get("first_check") or ""})
                t2, mode2 = self._graded_turn(case, grading, probes)
                if t2 is not None:
                    if not turns:
                        t2 = f"{preamble}\n\n{t2}"
                    est2 = int(len(json.dumps(msgs, default=str)) / self.chars_per_token)
                    if est2 + self.max_output_tokens > self.ctx_tokens:
                        rec["graded_skipped"] = f"too long ({est2} est tokens)"
                    else:
                        resp, p2, n, msgs = self._ask(msgs, t2, mode2, len(probes), sess, fmt)
                        calls += n
                        turns.append({"name": "graded", "question": t2, "response": resp})
                        for k in ("lesson", "catch", "where", "why", "keep", "essential", "close_calls"):
                            if p2.get(k):
                                parsed[k] = p2[k]
                        if probes:
                            ans = p2.get("probe_answers") or [""] * len(probes)
                            parsed["probes"] = [{"q": q, "a": a} for q, a in zip(probes, ans)]
                rec["turns"], rec["parsed"] = turns, parsed
                contents = [(t["response"].get("content") or "").strip() for t in turns]
                if not turns:
                    status = "skipped"
                elif not any(contents):
                    status = "errors"
                else:
                    useful = (parsed.get("unsure_items") or parsed.get("overall_confidence") is not None
                              or any(parsed.get(k) for k in ("lesson", "catch", "keep", "essential"))
                              or any(p.get("a") for p in parsed.get("probes") or []))
                    status = "ok" if useful else "off_task"
        except Exception as exc:  # noqa: BLE001 -- reflection must never break a batch
            rec["error"] = repr(exc)[:500]
            status = "errors"
        rec["status"] = status
        _atomic_write(dest, json.dumps(rec, indent=1, default=str))
        return status, calls

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


def load_records(round_dir: Path) -> list[dict]:
    """Every reflection record of one node, sorted by file name (legacy single-turn records
    are converted to the two-turn shape: ``mode`` folded into ``parsed``)."""
    out = []
    for f in sorted((Path(round_dir) / REFLECTIONS_DIR).glob("*.json")):
        try:
            rec = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(rec, dict):
            continue
        rec.setdefault("_file", f.name)
        if rec.get("version") != RECORD_VERSION:
            rec = _upgrade_legacy(rec)
        out.append(rec)
    return out


def _upgrade_legacy(rec: dict) -> dict:
    p = rec.get("parsed") or {}
    mode = rec.get("mode")
    parsed: dict[str, Any] = {}
    if mode == "unsure":
        parsed = {"unsure_items": p.get("items") or [], "overall_confidence": p.get("overall_confidence")}
    elif mode == "post_grading":
        parsed = {k: p[k] for k in ("lesson", "catch") if p.get(k)}
    elif mode == "essential":
        parsed = {"keep": p["keep"]} if p.get("keep") else {}
    body = ((rec.get("response") or {}).get("content") or "")
    turns = [{"name": "blind" if mode == "unsure" else "graded", "question": rec.get("turn", ""),
              "response": rec.get("response") or {}}] if body else []
    return {**rec, "parsed": parsed, "turns": turns, "legacy_mode": mode}


def _oneline(s: Any) -> str:
    return " ".join(str(s or "").split())


def _field(v: Any, terms: list[str], cap: Optional[int]) -> str:
    """One answer field on one line: redacted FIRST (a cut must never leave half a
    redact term behind), then clipped to ``cap`` characters."""
    text = redact(_oneline(v), terms)
    return text if not cap or len(text) <= cap else text[: max(1, cap - 1)].rstrip() + "\u2026"


def render_record(rec: dict, exposure: str, extra_terms: Optional[list[str]] = None,
                  max_field_chars: Optional[int] = None) -> list[str]:
    """Lines describing ONE reflection record for the meta-agent, redacted. ``lessons_only``
    shows only the parsed fields (each clipped to ``max_field_chars`` when set); ``full`` the
    whole answers."""
    terms = list(rec.get("redact_terms") or []) + list(extra_terms or [])
    if exposure == "full":
        lines = []
        for t in rec.get("turns") or []:
            body = ((t.get("response") or {}).get("content") or "").strip()
            if body:
                lines.append(f"[{t.get('name')}] {redact(body, terms)}")
        return lines
    cap = max_field_chars
    p = rec.get("parsed") or {}
    lines = []
    if p.get("overall_confidence") is not None:
        lines.append(f"blind confidence: {p['overall_confidence']}")
    for it in (p.get("unsure_items") or [])[:4]:
        if it.get("confidence", 100) < 70:
            lines.append(f"unsure ({it['confidence']}): {_field(it.get('item'), terms, cap)}")
    if p.get("first_check"):
        lines.append(f"first check: {_field(p['first_check'], terms, cap)}")
    for key, label in (("lesson", "lesson"), ("catch", "would have caught it"), ("essential", "essential"),
                       ("close_calls", "close calls"), ("keep", "keep")):
        if p.get(key):
            lines.append(f"{label}: {_field(p[key], terms, cap)}")
    for i, pq in enumerate(p.get("probes") or [], 1):
        if pq.get("a"):
            lines.append(f"probe {i} -- {_field(pq.get('q'), terms, None)} => {_field(pq['a'], terms, cap)}")
    return lines


def render_reflections(round_dir: Path, exposure: str = "lessons_only", max_chars: int = 6000) -> str:
    """One node's reflections for the meta-agent (module-level so analysis tools can render
    a run without a configured reflector)."""
    if exposure == "off":
        return ""
    recs = load_records(Path(round_dir))
    if exposure == "full":
        blocks = []
        for rec in recs:
            body = "\n".join(render_record(rec, "full"))
            if body:
                blocks.append(f"### {rec.get('case_id')}/{rec.get('role')} "
                              f"({'PASSED' if rec.get('passed') else 'FAILED'})\n{body}")
        text = "\n\n".join(blocks)
    else:
        lessons, keeps, unsure, probes = [], [], [], []
        for rec in recs:
            p = rec.get("parsed") or {}
            terms = rec.get("redact_terms") or []
            tag = f"{rec.get('case_id')}/{rec.get('role')}"
            if p.get("lesson"):
                lessons.append(f"- ({tag}) {redact(_oneline(p['lesson']), terms)}")
                if p.get("catch"):
                    lessons.append(f"  would have caught it: {redact(_oneline(p['catch']), terms)}")
            if p.get("keep"):
                keeps.append(f"- ({tag}) {redact(_oneline(p['keep']), terms)}")
            for it in (p.get("unsure_items") or [])[:3]:
                if it.get("confidence", 100) < 70:
                    unsure.append(f"- ({tag}, {it['confidence']}) {redact(_oneline(it.get('item')), terms)}")
            for pq in p.get("probes") or []:
                if pq.get("a"):
                    probes.append(f"- ({tag}) {redact(_oneline(pq.get('q')), terms)} => "
                                  f"{redact(_oneline(pq['a']), terms)}")
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
        if probes:
            parts.append("Answers to the probe questions written for this node's edit:\n" + "\n".join(probes))
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
