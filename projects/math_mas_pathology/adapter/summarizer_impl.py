"""math_mas_pathology's named behavior-summarizer extension point.

Like math_mas's own summarizer, `_extract_failure_hint` builds a per-case
diagnostic string from `MathMASPathologyScorer.score()`'s `details` dict
(none of its keys match `BehaviorSummarizer`'s generic conventions, so the
base class's hint would otherwise come out empty for every case).

Beyond math_mas's predictor/reflector correctness fingerprint, this adds a
three-stage correctness chain (predictor -> verifier -> reflector) plus an
explicit statement of which pathologies were active and, when stale-context
injection was on, which context object actually reached the reflector vs.
what was available -- so a human or an HGM editor can attribute a failure to
a specific hand-off rather than just "the MAS got it wrong."
"""
from __future__ import annotations

import re
from typing import Any, Optional

from meta_agent.behavior_summarizer import BehaviorSummarizer
from meta_agent.models import CaseResult
from meta_agent.registry import register

_HINT_CHAR_CAP = 2200
_EXCERPT_CAP = 350

_ANSWER_TAG_RE = re.compile(r"<answer>.*?</answer>", re.IGNORECASE | re.DOTALL)


def _tail_excerpt(text: Optional[str], n: int = _EXCERPT_CAP) -> str:
    """Last `n` chars of an agent's raw reasoning, with the trailing
    `<answer>...</answer>` tag stripped first -- the answer itself is
    already surfaced separately."""
    if not text:
        return ""
    stripped = _ANSWER_TAG_RE.sub("", text).strip()
    if len(stripped) <= n:
        return stripped
    return "…" + stripped[-n:]


@register("summarizer", "math_mas_pathology_default")
class MathMASPathologyBehaviorSummarizer(BehaviorSummarizer):
    @staticmethod
    def _extract_failure_hint(case: CaseResult) -> str:
        parts: list[str] = []

        if case.error:
            parts.append(f"runtime_error: {case.error}")

        details: dict[str, Any] = case.details or {}

        if details.get("error"):
            parts.append(f"agent_error: {details['error']}")

        predictor_correct = details.get("predictor_correct")
        verifier_correct = details.get("verifier_correct")
        final_correct = details.get("final_correct")
        if predictor_correct is not None and final_correct is not None:
            parts.append(
                f"predictor_correct={predictor_correct} "
                f"verifier_correct={verifier_correct} final_correct={final_correct}"
            )

        pathology_flags = details.get("pathology_flags") or {}
        if pathology_flags:
            active = ", ".join(k for k, v in pathology_flags.items() if v and k != "verify_rounds")
            parts.append(f"pathologies_active=[{active or 'none'}]")

        if pathology_flags.get("stale_context") and details.get("context_divergent"):
            # The verifier's real conclusion differed from the stale
            # first-draft context the reflector actually received -- the
            # single most attributable failure signal this project can
            # surface (see README.md pathology 2).
            parts.append(
                f"reflector_received_stale_context=True "
                f"(context_used_excerpt={_tail_excerpt(details.get('context_used_by_reflector'), 200)!r})"
            )

        if details.get("deafness_active"):
            parts.append(
                f"selective_deafness_dropped="
                f"{details.get('deafness_sentences_dropped')}_sentences/"
                f"{details.get('deafness_chars_dropped')}_chars"
            )

        if details.get("verifier_turn_drift"):
            parts.append(
                f"verifier_turn_drift=True (rounds_run={details.get('verifier_rounds_run')}, "
                f"unique_answers={details.get('verifier_unique_answer_count')}) -- the "
                "repeated verifier calls disagreed with themselves, yet only the last "
                "was ever usable downstream"
            )

        prediction = details.get("prediction")
        gold = details.get("gold_answer")
        if prediction is not None or gold is not None:
            parts.append(f"prediction={prediction!r} gold={gold!r}")

        if not parts:
            return ""

        text = " | ".join(parts).replace("\n", " ")
        if len(text) > _HINT_CHAR_CAP:
            text = text[: _HINT_CHAR_CAP - 1].rstrip() + "…"
        return text

    def _build_prompt(
        self, aggregate: dict[str, Any], prior_memory: Optional[str] = None
    ) -> tuple[str, str]:
        user, system = super()._build_prompt(aggregate, prior_memory=prior_memory)
        system += (
            "\n\nADDITIONALLY, math_mas_pathology has three sub-agents: PREDICTOR "
            "(solves from scratch), VERIFIER (re-derives the predictor's answer, "
            "called multiple identical times per case -- only its last turn is ever "
            "used), and REFLECTOR (reviews whatever context it's handed and may "
            "overturn the answer). This project deliberately contains three "
            "communication pathologies at specific hand-off points: (1) "
            "repetition-then-ignore -- the verifier's earlier turns are computed but "
            "discarded; (2) stale context injection -- the reflector may receive the "
            "predictor's ORIGINAL first draft instead of the verifier's real "
            "conclusion; (3) selective deafness -- the reflector may only read the "
            "last sentence of whatever context it's given. Per-case hints above state "
            "which pathologies were active and, when relevant, what was actually lost "
            "(stale context excerpt, dropped sentence/char counts, verifier turn "
            "drift). Using ONLY those hints and the per-case table, add one more "
            "section:\n"
            "  ## Pathology impact\n"
            "  - For each pathology that was active this round, state whether the "
            "per-case hints actually show it changing an outcome (a case that would "
            "have been correct if the verifier's real conclusion, or the full "
            "context, had reached the reflector) -- cite case_ids.\n"
            "  - If a pathology's toggle was off this round, say so explicitly rather "
            "than guessing at its effect.\n"
            "  Do not speculate beyond what the excerpts/hints actually show; if this "
            "batch is too small/limited to support a claim, say so instead of "
            "inventing a pattern."
        )
        return user, system
