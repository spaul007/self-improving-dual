"""Per-round failure summarizer.

Companion to ``behavior_summarizer.py``, same philosophy: deterministic
selection in code (cheap, exact, auditable — which cases are failing, sorted
worst-first), one LLM call to narrate them into prose the editor can actually
use. This exists because the editor's OTHER source of case-level evidence
(``failure_report.py``'s ``hard_cases``, rendered via ``render_failure_report``
into ``_format_feedback``) is a small, char-capped sample meant for direct
display — even after fixing its head-truncation (see ``_truncate_middle`` in
``failure_report.py``), it is still just 1-3 raw examples, never a synthesis
across all failing cases.

Key difference from ``BehaviorSummarizer``: this runs on EVERY evaluation
batch, including the root/seed's (no parent or diff required) — there's
nothing to diff, but there's still something to fail, and the very first
round is exactly when a clear failure catalog is most useful (nothing has
been tried yet).

Design:
    - Reads ``eval_result.per_case`` directly (not through
      ``failure_report.build_failure_report``'s category/example-selection
      machinery, which is oriented at picking a SMALL rendered sample for
      direct display) — takes every failing case, sorted worst-score-first,
      capped at ``_MAX_CASES`` for prompt-size sanity on large eval sets.
    - Truncates each case's query/raw-output with ``_truncate_middle`` (head
      AND tail kept, middle elided) at generous caps (much larger than
      ``failure_report.py``'s display caps) — this call's whole purpose is to
      give the LLM enough real signal to synthesize correctly, so it should
      see much more than what ultimately gets shown to a human/the editor
      directly.
    - One LLM call, explicitly forbidden from proposing fixes (that's the
      editor's job) and instructed to cite case_ids and ground every claim in
      what's shown — same anti-hallucination framing already proven to work
      for ``BehaviorSummarizer``.
    - Persists ``failure_summary_aggregate.json`` (pre-LLM structured input)
      and ``failure_summary_prompt.txt`` (the literal prompt) for forensics,
      alongside the final ``failure_summary.md``, mirroring
      ``behavior_summarizer.py``'s own artifact conventions exactly.
    - Graceful when missing/failing: no failing cases, an LLM error, or an
      empty response all return ``None`` rather than raising — a round's
      evaluation must never be lost to this being unavailable.
"""
from __future__ import annotations

import json
import os
import traceback
from pathlib import Path
from typing import Any, Callable, Optional

from .failure_report import _as_text, _truncate_middle
from .models import CaseResult, EvaluationResult
from .registry import register

_QUERY_CAP = 800
_RAW_OUTPUT_CAP = 4000
_MAX_CASES = 12  # bounds prompt size for large eval sets; failing cases beyond
                 # this (sorted worst-first) are simply not shown to this call.


@register("failure_summarizer", "default")
class FailureSummarizer:
    """LLM-synthesized cross-case failure summary.

    Args:
        llm_caller: same callable injected into ``AgentEditor``/
            ``BehaviorSummarizer`` (the project's
            ``platform_core.llm_wrapper.call_llm``).
        model / reasoning_effort / base_url: same meaning as
            ``BehaviorSummarizer``'s.
        domain_label: same fallback chain as ``BehaviorSummarizer``
            (``META_AGENT_PROJECT`` env var, else ``"agent"``).
    """

    def __init__(
        self,
        llm_caller: Callable[..., object],
        *,
        model: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        base_url: Optional[str] = None,
        domain_label: Optional[str] = None,
        # Caps runaway generation. 16384 matches the task_agent default.
        # None disables the cap.
        max_output_tokens: Optional[int] = 16384,
        # How many of the hardest (worst-scoring) shown cases also get
        # their literal trace.jsonl tool_call -> tool_result narrative
        # appended to their digest -- e.g. an argument value copied
        # verbatim from an earlier tool's own free-text output that then
        # 404'd, followed by a silent substitution, is a concrete,
        # correctly-diagnosable story a final score/error string alone
        # can't tell (confirmed live this session: a hard-constraint
        # failure whose error just named a missing attraction turned out,
        # via the trace, to be a tool-output-format bug plus a bad-
        # recovery-strategy bug -- two fixable things a generic "verify
        # names better" diagnosis would have missed). 0 (default)
        # disables this entirely -- byte-identical to today.
        trace_digest_case_count: int = 0,
        # Bounds prompt size per digested case.
        trace_digest_max_calls: int = 15,
        # No point exceeding trace.jsonl's own result_preview truncation.
        trace_digest_preview_chars: int = 200,
    ) -> None:
        self.llm = llm_caller
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.base_url = base_url
        self.domain_label = (
            domain_label or os.environ.get("META_AGENT_PROJECT") or "agent"
        )
        self.max_output_tokens = max_output_tokens
        self.trace_digest_case_count = trace_digest_case_count
        self.trace_digest_max_calls = trace_digest_max_calls
        self.trace_digest_preview_chars = trace_digest_preview_chars

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def summarize(
        self, *, eval_result: EvaluationResult, round_dir: Path, node_id: int
    ) -> Optional[Path]:
        """Produce (or refresh) ``failure_summary.md`` for ``round_dir``.

        Unlike ``BehaviorSummarizer.summarize``, this takes no
        ``parent_round_dir`` — there's no diff involved, so it works for the
        root/seed round too. Overwrites the file each call (the cumulative
        ``eval_result`` passed in already reflects every case the node has
        seen so far), so it stays current across batches without needing an
        UPDATE-mode prompt variant.
        """
        cases = list(eval_result.per_case or [])
        failing = [c for c in cases if (not c.passed) or float(c.score) < 1.0]
        if not failing:
            return None

        failing.sort(key=lambda c: (float(c.score), str(c.case_id)))
        shown = failing[:_MAX_CASES]

        case_events_by_id: dict[str, list[dict[str, Any]]] = {}
        if self.trace_digest_case_count > 0:
            try:
                events = self._read_trace(round_dir / "logs" / "trace.jsonl")
                case_events_by_id = self._group_tool_events_by_case(events)
            except Exception:
                case_events_by_id = {}

        aggregate = self._aggregate(cases, shown, node_id, case_events_by_id)

        try:
            (round_dir / "failure_summary_aggregate.json").write_text(
                json.dumps(aggregate, indent=2, default=str), encoding="utf-8"
            )
        except Exception:
            pass

        prompt_user, prompt_system = self._build_prompt(aggregate)
        try:
            (round_dir / "failure_summary_prompt.txt").write_text(
                f"### SYSTEM\n{prompt_system}\n\n### USER\n{prompt_user}",
                encoding="utf-8",
            )
        except Exception:
            pass

        try:
            response = self._call_llm(prompt_system, prompt_user)
        except Exception:
            print(
                f"[failure_summarizer] LLM call failed for node {node_id}:\n"
                + traceback.format_exc(limit=3),
                flush=True,
            )
            return None

        text = (response or "").strip()
        if not text:
            print(
                f"[failure_summarizer] empty summary returned for node "
                f"{node_id} — skipped",
                flush=True,
            )
            return None

        path = round_dir / "failure_summary.md"
        path.write_text(text, encoding="utf-8")
        print(
            f"[failure_summarizer] node {node_id}: wrote failure_summary.md "
            f"({len(text)} chars, {len(shown)}/{len(failing)} failing case(s) shown)",
            flush=True,
        )
        return path

    # ------------------------------------------------------------------ #
    # Aggregation + prompt
    # ------------------------------------------------------------------ #

    def _aggregate(
        self,
        all_cases: list[CaseResult],
        shown: list[CaseResult],
        node_id: int,
        case_events_by_id: Optional[dict[str, list[dict[str, Any]]]] = None,
    ) -> dict[str, Any]:
        case_events_by_id = case_events_by_id or {}
        case_records = []
        for i, c in enumerate(shown):
            det = c.details or {}
            # c.error (top-level CaseResult.error) is reserved for
            # harness-level crashes and is None for a scorer-judged
            # failure -- the actual descriptive message (e.g. "plan
            # conversion failed: agent produced no plan (agent_metadata:
            # ...)") lives in details["error"] instead. Previously this
            # read c.error only, so every scorer-judged failure (the
            # common case) showed up here as error=None -- the LLM had
            # nothing but an empty raw_output to reason from, unable to
            # distinguish "nothing was ever generated" from "a complete
            # plan was generated and then discarded by the workflow's own
            # validation gate" (confirmed live: both looked identical).
            record = {
                "case_id": str(c.case_id),
                "score": round(float(c.score), 4),
                "error": det.get("error") or c.error,
                "query": _truncate_middle(_as_text(det.get("query")), _QUERY_CAP),
                "raw_output": _truncate_middle(
                    _as_text(det.get("raw_result")), _RAW_OUTPUT_CAP
                ),
            }
            if i < self.trace_digest_case_count:
                case_events = case_events_by_id.get(str(c.case_id))
                if case_events:
                    record["tool_trace"] = self._render_case_tool_trace(
                        case_events,
                        self.trace_digest_max_calls,
                        self.trace_digest_preview_chars,
                    )
            case_records.append(record)
        n_failing_total = sum(
            1 for c in all_cases if (not c.passed) or float(c.score) < 1.0
        )
        return {
            "node_id": node_id,
            "n_total": len(all_cases),
            "n_failing": n_failing_total,
            "n_shown": len(case_records),
            "mean_score": round(
                sum(float(c.score) for c in all_cases) / len(all_cases), 4
            )
            if all_cases
            else 0.0,
            "cases": case_records,
        }

    # ------------------------------------------------------------------ #
    # Trace digest (opt-in via trace_digest_case_count)
    # ------------------------------------------------------------------ #

    def _read_trace(self, path: Path) -> list[dict[str, Any]]:
        """Parse a trace.jsonl file into a list of event dicts. Tolerates
        malformed lines (skipped, not fatal) and a missing file (returns
        ``[]``) -- same convention as the independent copies of this
        helper in ``behavior_summarizer.py``/``feedback_gatherer.py``."""
        if not path.exists():
            return []
        events: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict):
                events.append(event)
        return events

    def _group_tool_events_by_case(
        self, events: list[dict[str, Any]]
    ) -> dict[str, list[dict[str, Any]]]:
        """Bucket ``tool_call``/``tool_result`` events by their payload's
        ``case_id``, preserving original (chronological) order within each
        bucket. Events of other kinds, or without a usable payload/case_id,
        are dropped -- this is only ever used to render a tool-call
        narrative, not a full trace."""
        grouped: dict[str, list[dict[str, Any]]] = {}
        for event in events:
            if event.get("kind") not in ("tool_call", "tool_result"):
                continue
            payload = event.get("payload")
            if not isinstance(payload, dict):
                continue
            case_id = payload.get("case_id")
            if not isinstance(case_id, str):
                continue
            grouped.setdefault(case_id, []).append(event)
        return grouped

    def _render_case_tool_trace(
        self,
        events_for_case: list[dict[str, Any]],
        max_calls: int,
        preview_chars: int,
    ) -> str:
        """Render one case's ordered tool_call -> tool_result narrative,
        e.g.:
            1. query_attraction_details(attraction_name='Nanbin Road...')
               -> Detailed information not found for attraction Nanbin...
        Pairs a tool_call with its tool_result by their shared ``id``; an
        unmatched call renders ``-> (no result)``. Capped at ``max_calls``
        calls and ``preview_chars`` chars of each result -- this exists to
        bound prompt size, not to be a complete trace."""
        results_by_id: dict[str, dict[str, Any]] = {}
        for event in events_for_case:
            if event.get("kind") == "tool_result":
                payload = event.get("payload") or {}
                call_id = payload.get("id")
                if isinstance(call_id, str):
                    results_by_id[call_id] = payload

        calls = [
            event.get("payload") or {}
            for event in events_for_case
            if event.get("kind") == "tool_call"
        ]
        lines: list[str] = []
        for i, call in enumerate(calls[:max_calls], 1):
            name = call.get("name", "(unknown tool)")
            arguments = call.get("arguments", {})
            lines.append(f"{i}. {name}({arguments})")
            call_id = call.get("id")
            result = results_by_id.get(call_id) if isinstance(call_id, str) else None
            if result is None:
                lines.append("   -> (no result)")
                continue
            preview = result.get("result_preview")
            preview = preview if isinstance(preview, str) else ""
            lines.append(f"   -> {preview[:preview_chars]}")
        return "\n".join(lines)

    def _build_prompt(self, aggregate: dict[str, Any]) -> tuple[str, str]:
        system = (
            f"You are summarizing evaluation failures for a self-evolving "
            f"{self.domain_label} agent, for the NEXT self-improvement step "
            "to read. You are given the failing cases from this round's "
            "evaluation, worst-scoring first: each case's query, its (near-)"
            "full raw agent output, its score, and any runtime error.\n\n"
            "When `error` is present for an empty-output case, it often "
            "distinguishes WHY nothing was produced (e.g. an "
            "`agent_metadata` annotation showing which internal stage "
            "failed, or that a complete result WAS generated internally "
            "but was rejected by the workflow's own validation logic "
            "before being returned) -- use this to correctly separate "
            "'the agent never managed to produce anything' from 'the "
            "agent produced something reasonable but the workflow itself "
            "discarded it,' since those call for very different fixes. "
            "Don't guess at a cause the error text doesn't support.\n\n"
            "Produce a concise markdown summary with exactly two sections:\n"
            "  ## Main failure patterns — group the failing cases into 1-4 "
            "recurring themes you can actually support from the text shown "
            "(e.g. a specific class of reasoning error, a formatting/"
            "instruction-following slip, a scoring/normalization mismatch). "
            "Cite case_ids for each theme. If the cases don't share a clear "
            "pattern, say so rather than inventing one.\n"
            "  ## Hardest cases — call out 1-2 of the most illustrative "
            "failing cases by case_id, with a short, SPECIFIC, VERIFIED "
            "description of what went wrong. Quote or closely paraphrase the "
            "actual text you were shown for that exact case rather than "
            "guessing or generalizing from a different case.\n\n"
            "This is an OBSERVATION only — do NOT propose or suggest fixes; "
            "leave that to the editor. Stay under 250 words. Never invent a "
            "case_id, a quote, or a detail that isn't present in what's shown "
            "above — if you're not sure, say the evidence is limited rather "
            "than guessing.\n\n"
            "IMPORTANT: `raw_output` is a JSON-serialized DEBUG DUMP of "
            "whatever object the agent's workflow internally returned (a "
            "project-specific shape, e.g. separate `prediction`/`raw text` "
            "fields) — it is a display artifact for YOUR benefit, not "
            "something the agent output as JSON to be graded. Do NOT treat "
            "the JSON structure/nesting itself as a formatting bug or "
            "extraction problem. Only call out an actual formatting issue if "
            "the real text CONTENT shows one (e.g. a required tag is truly "
            "missing from the text, or the text is garbled/cut off) — judge "
            "the prose content, not the debug wrapper around it.\n\n"
            "Some of the hardest cases also show a `tool call trace` — the "
            "literal, ordered sequence of tool calls the agent made and "
            "what each tool actually returned. Use it to tell apart 'the "
            "agent's own reasoning went wrong' from 'the agent behaved "
            "reasonably given what a tool actually told it': e.g. an "
            "argument value copied verbatim from an earlier tool's own "
            "output that then failed/came back empty, followed by the "
            "agent silently substituting something else, is a different, "
            "more specific failure than a generic reasoning slip — say so "
            "explicitly when the trace shows it, don't default to a vaguer "
            "'didn't verify carefully' framing when the trace shows it did."
        )
        lines = [
            f"## Round summary\nnode {aggregate['node_id']}  "
            f"{aggregate['n_failing']}/{aggregate['n_total']} failing "
            f"(showing the {aggregate['n_shown']} worst-scoring)  "
            f"mean_score={aggregate['mean_score']}\n"
        ]
        for c in aggregate["cases"]:
            lines.append(f"\n### case {c['case_id']}  (score={c['score']})")
            if c["error"]:
                lines.append(f"error: {c['error']}")
            lines.append(f"query: {c['query']}")
            lines.append(f"raw_output: {c['raw_output']}")
            if c.get("tool_trace"):
                lines.append(f"tool call trace:\n{c['tool_trace']}")
        return "\n".join(lines), system

    def _call_llm(self, system: str, user: str) -> str:
        kwargs: dict[str, Any] = {
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if self.model:
            kwargs["model"] = self.model
        if self.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort
        else:
            kwargs["temperature"] = 0.2
        if self.base_url:
            kwargs["base_url"] = self.base_url
        if self.max_output_tokens is not None:
            kwargs["max_output_tokens"] = self.max_output_tokens
        response = self.llm(**kwargs)
        return getattr(response, "content", None) or ""


def render_failure_summary_for_steering(
    round_dir: Path, *, cap_chars: Optional[int] = None
) -> Optional[str]:
    """Read a previously-written ``failure_summary.md`` for use in steering.
    Mirrors ``behavior_summarizer.render_memory_for_steering`` exactly."""
    path = round_dir / "failure_summary.md"
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8")
    if cap_chars is not None and len(text) > cap_chars:
        return text[:cap_chars].rstrip() + "\n<... truncated ...>"
    return text
