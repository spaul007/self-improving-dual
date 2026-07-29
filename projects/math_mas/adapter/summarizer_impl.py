"""math_mas's named behavior-summarizer extension point.

`meta_agent.behavior_summarizer.BehaviorSummarizer._extract_failure_hint` only
recognizes generic scorer conventions (`failed_checks`, a nested
`{name: {"passed": bool}}` map, `missing_*`/`extra_*` collections, a bare
`error` string) -- none of `MathMASScorer.score()`'s actual keys
(`prediction`/`gold_answer`/`predictor_answer`/`predictor_correct`/
`final_correct`) match those conventions, so the base class's hint comes out
empty for every math_mas case.

Two overrides here, both needed to make the memo diagnostically useful rather
than just a pass/fail tally:

1. `_extract_failure_hint` -- besides the compact predictor/final-correctness
   fingerprint, pulls a short, bounded excerpt of whichever sub-agent's raw
   reasoning is actually relevant to that case's outcome. The full reasoning
   trace is always available in `case.details["agent_metadata"]["trajectory"]`
   (injected automatically by `meta_agent/evaluator.py`'s generic
   `agent_artifact` merge -- not something this project's scorer has to
   produce), but inlining it for every case would blow up the prompt, so an
   excerpt is only attached to the "interesting" outcome classes (a
   correctness flip, or both agents wrong) -- exactly where a reasoning-level
   signal is worth the tokens; clean unchanged-correct cases get only the
   compact line.
2. `_build_prompt` -- appends an extra system-prompt section asking the
   summarizer LLM to explicitly name recurring PREDICTOR/REFLECTOR reasoning
   issues (not just "did the edit help"), grounded only in the excerpts/hints
   actually shown -- this is what makes `behavior_memory.md` describe *why*
   the MAS is failing, not just *whether* a given round's edit helped.
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
    `<answer>...</answer>` tag stripped first -- the answer itself is already
    surfaced separately (`prediction`/`predictor_answer`), so the excerpt's
    tokens are better spent on the concluding reasoning/critique paragraph
    that usually precedes it, where an agent states whether/why it found an
    error."""
    if not text:
        return ""
    stripped = _ANSWER_TAG_RE.sub("", text).strip()
    if len(stripped) <= n:
        return stripped
    return "…" + stripped[-n:]


def _trajectory_stage(details: dict[str, Any], agent_name: str) -> str:
    agent_meta = details.get("agent_metadata") or {}
    for stage in agent_meta.get("trajectory") or []:
        if stage.get("agent") == agent_name:
            return stage.get("raw", "")
    return ""


@register("summarizer", "math_mas_default")
class MathMASBehaviorSummarizer(BehaviorSummarizer):
    @staticmethod
    def _extract_failure_hint(case: CaseResult) -> str:
        parts: list[str] = []

        if case.error:
            parts.append(f"runtime_error: {case.error}")

        details: dict[str, Any] = case.details or {}

        if details.get("error"):
            parts.append(f"agent_error: {details['error']}")

        predictor_correct = details.get("predictor_correct")
        final_correct = details.get("final_correct")
        if predictor_correct is not None and final_correct is not None:
            if predictor_correct and not final_correct:
                parts.append("reflector_broke_correct_answer=True")
                excerpt = _tail_excerpt(_trajectory_stage(details, "reflector"))
                if excerpt:
                    parts.append(f"reflector_critique_excerpt={excerpt!r}")
            elif not predictor_correct and final_correct:
                parts.append("reflector_fixed_wrong_answer=True")
                excerpt = _tail_excerpt(_trajectory_stage(details, "reflector"))
                if excerpt:
                    parts.append(f"reflector_critique_excerpt={excerpt!r}")
            elif not predictor_correct and not final_correct:
                # Both wrong: the predictor's own reasoning shows where it
                # went wrong; the reflector's excerpt shows whether it
                # noticed anything was off (a rubber-stamping reflector is
                # itself a distinct, worth-naming failure mode).
                pred_excerpt = _tail_excerpt(_trajectory_stage(details, "predictor"))
                if pred_excerpt:
                    parts.append(f"predictor_reasoning_excerpt={pred_excerpt!r}")
                refl_excerpt = _tail_excerpt(_trajectory_stage(details, "reflector"))
                if refl_excerpt:
                    parts.append(f"reflector_critique_excerpt={refl_excerpt!r}")
            parts.append(f"predictor_correct={predictor_correct} final_correct={final_correct}")

        prediction = details.get("prediction")
        gold = details.get("gold_answer")
        if prediction is not None or gold is not None:
            parts.append(f"prediction={prediction!r} gold={gold!r}")

        if not parts:
            return ""

        # Flatten to one line -- the per-case prompt renders one line per
        # case (see BehaviorSummarizer._build_prompt), so embedded newlines
        # would break that layout.
        text = " | ".join(parts).replace("\n", " ")
        if len(text) > _HINT_CHAR_CAP:
            text = text[: _HINT_CHAR_CAP - 1].rstrip() + "…"
        return text

    def _build_prompt(
        self, aggregate: dict[str, Any], prior_memory: Optional[str] = None
    ) -> tuple[str, str]:
        user, system = super()._build_prompt(aggregate, prior_memory=prior_memory)
        system += (
            "\n\nADDITIONALLY, math_mas has exactly two sub-agents: PREDICTOR "
            "(solves the problem from scratch) and REFLECTOR (reviews the "
            "predictor's work and may overturn its answer). Per-case hints above "
            "may include a short excerpt of whichever agent's raw reasoning is "
            "most relevant to that case's outcome (the reflector's critique when "
            "predictor/final correctness differ, or both agents' excerpts when "
            "both were wrong). Using ONLY those excerpts and the per-case table, "
            "add one more section:\n"
            "  ## Sub-agent reasoning issues\n"
            "  - Predictor: describe any recurring reasoning-error pattern you "
            "can actually support from the excerpts shown (e.g. a specific "
            "arithmetic slip, a misapplied formula, a misread constraint) -- "
            "cite case_ids. If the shown excerpts don't support a pattern, say "
            "so explicitly rather than guessing.\n"
            "  - Reflector: state whether it justifies its corrections/overturns "
            "with a concrete, specific error, or whether it appears to overturn "
            "already-correct answers without pointing to any real mistake (a "
            "known failure mode for this MAS) -- cite case_ids either way.\n"
            "  Do not speculate beyond what the excerpts/hints actually show; if "
            "this batch is too small/limited to support a claim, say so instead "
            "of inventing a pattern."
        )
        return user, system
