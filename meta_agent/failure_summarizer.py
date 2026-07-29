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
    ) -> None:
        self.llm = llm_caller
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.base_url = base_url
        self.domain_label = (
            domain_label or os.environ.get("META_AGENT_PROJECT") or "agent"
        )

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

        aggregate = self._aggregate(cases, shown, node_id)

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
        self, all_cases: list[CaseResult], shown: list[CaseResult], node_id: int
    ) -> dict[str, Any]:
        case_records = []
        for c in shown:
            det = c.details or {}
            case_records.append(
                {
                    "case_id": str(c.case_id),
                    "score": round(float(c.score), 4),
                    "error": c.error,
                    "query": _truncate_middle(_as_text(det.get("query")), _QUERY_CAP),
                    "raw_output": _truncate_middle(
                        _as_text(det.get("raw_result")), _RAW_OUTPUT_CAP
                    ),
                }
            )
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

    def _build_prompt(self, aggregate: dict[str, Any]) -> tuple[str, str]:
        system = (
            f"You are summarizing evaluation failures for a self-evolving "
            f"{self.domain_label} agent, for the NEXT self-improvement step "
            "to read. You are given the failing cases from this round's "
            "evaluation, worst-scoring first: each case's query, its (near-)"
            "full raw agent output, its score, and any runtime error.\n\n"
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
            "the prose content, not the debug wrapper around it."
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
