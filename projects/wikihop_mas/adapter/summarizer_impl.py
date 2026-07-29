"""wikihop_mas's named behavior-summarizer extension point.

`meta_agent.behavior_summarizer.BehaviorSummarizer._extract_failure_hint`
only recognizes generic scorer conventions (`failed_checks`, a nested
`{name: {"passed": bool}}` map, `missing_*`/`extra_*` collections, a bare
`error` string) -- none of `WikihopMASScorer.score()`'s actual keys
(`prediction`/`gold_answer`/`answer_em`/`sp_em`/`predicted_type`/
`gold_type`/`type_correct`/`concluder_rounds`/`error`) match those
conventions, so the base class's hint comes out empty for every
wikihop_mas case without this override.

Two overrides, mirroring math_mas's summarizer_impl.py:

1. `_extract_failure_hint` -- a compact per-case fingerprint (answer
   correctness, supporting-fact correctness, type classification, whether a
   grounding retry fired), plus a short excerpt of the Concluder's own
   `reasoning` field (`case.details["agent_metadata"]["trajectory"]
   ["concluder_calls"][-1]["reasoning"]`, always available -- injected
   automatically by `meta_agent/evaluator.py`'s generic `agent_metadata`
   merge, nothing wikihop_mas-specific had to be added to capture it) for
   "interesting" outcomes only (wrong answer, wrong type classification, or
   a retry fired) -- exactly where the Concluder's own stated reasoning is
   worth the tokens; clean unchanged-correct cases get only the compact
   line.
2. `_build_prompt` -- appends a section describing the real
   Decomposer -> (independent hops | chained hops) -> Concluder
   (+ bounded grounding retry) topology and asks the summarizer LLM to name
   recurring per-agent issues, grounded only in the hints/excerpts shown.
"""
from __future__ import annotations

from typing import Any, Optional

from meta_agent.behavior_summarizer import BehaviorSummarizer
from meta_agent.models import CaseResult
from meta_agent.registry import register

_HINT_CHAR_CAP = 2200
_EXCERPT_CAP = 350


def _tail_excerpt(text: Optional[str], n: int = _EXCERPT_CAP) -> str:
    if not text:
        return ""
    stripped = text.strip()
    if len(stripped) <= n:
        return stripped
    return "…" + stripped[-n:]


def _concluder_reasoning(details: dict[str, Any]) -> str:
    agent_meta = details.get("agent_metadata") or {}
    trajectory = agent_meta.get("trajectory") or {}
    calls = trajectory.get("concluder_calls") or []
    if not calls:
        return ""
    return str(calls[-1].get("reasoning", ""))


@register("summarizer", "wikihop_mas_default")
class WikihopMASBehaviorSummarizer(BehaviorSummarizer):
    @staticmethod
    def _extract_failure_hint(case: CaseResult) -> str:
        parts: list[str] = []

        if case.error:
            parts.append(f"runtime_error: {case.error}")

        details: dict[str, Any] = case.details or {}

        if details.get("error"):
            parts.append(f"agent_error: {details['error']}")

        answer_em = details.get("answer_em")
        sp_em = details.get("sp_em")
        type_correct = details.get("type_correct")
        concluder_rounds = details.get("concluder_rounds")
        retried = concluder_rounds == 2

        if answer_em is not None:
            parts.append(
                f"answer_em={answer_em} sp_em={sp_em} type_correct={type_correct} "
                f"concluder_rounds={concluder_rounds}"
            )

        interesting = (answer_em is False) or (type_correct is False) or retried
        if interesting:
            excerpt = _tail_excerpt(_concluder_reasoning(details))
            if excerpt:
                parts.append(f"concluder_reasoning_excerpt={excerpt!r}")

        prediction = details.get("prediction")
        gold = details.get("gold_answer")
        if prediction is not None or gold is not None:
            parts.append(f"prediction={prediction!r} gold={gold!r}")

        if type_correct is False:
            parts.append(
                f"predicted_type={details.get('predicted_type')!r} "
                f"gold_type={details.get('gold_type')!r}"
            )

        if not parts:
            return ""

        # Flatten to one line -- BehaviorSummarizer renders one line per
        # case in the per-case table, so embedded newlines would break it.
        text = " | ".join(parts).replace("\n", " ")
        if len(text) > _HINT_CHAR_CAP:
            text = text[: _HINT_CHAR_CAP - 1].rstrip() + "…"
        return text

    def _build_prompt(
        self, aggregate: dict[str, Any], prior_memory: Optional[str] = None
    ) -> tuple[str, str]:
        user, system = super()._build_prompt(aggregate, prior_memory=prior_memory)
        system += (
            "\n\nADDITIONALLY, wikihop_mas has a non-sequential, controller-driven "
            "4-agent topology for 2WikiMultihopQA: DECOMPOSER (classifies the "
            "question's reasoning type and emits a hop-plan: independent hops for "
            "comparison/bridge_comparison questions, or chained hops -- hop2's "
            "question gets hop1's answer substituted in -- for "
            "inference/compositional questions) -> per hop, RETRIEVER (BM25 tool "
            "search over the question's own ~10 context paragraphs, closed-book) "
            "-> EXTRACTOR (pulls an answer + quote + source from the retrieved "
            "text) -> CONCLUDER (aggregates all hops, judges whether each hop's "
            "quote is well-grounded, emits the final answer; if any hop is judged "
            "ungrounded, that ONE hop is rerun and the Concluder is called a "
            "second, final time). Per-case hints above may include a short "
            "excerpt of the Concluder's own stated `reasoning` for cases where "
            "the answer was wrong, the Decomposer misclassified the question "
            "type, or a grounding retry fired. Using ONLY those excerpts and the "
            "per-case table, add one more section:\n"
            "  ## Sub-agent reasoning issues\n"
            "  - Decomposer: any recurring question-type misclassification "
            "pattern (cite case_ids) -- misclassifying independent vs. "
            "dependent changes the whole hop plan, not just one field.\n"
            "  - Retriever/Extractor: any recurring pattern where the wrong "
            "quote/answer was pulled from the context, if the excerpts show it.\n"
            "  - Concluder: whether its final answer/grounding judgment is "
            "actually justified by the hop evidence it was given, or whether it "
            "appears to guess/hallucinate a final_answer not supported by any "
            "hop -- cite case_ids either way.\n"
            "  - Grounding retry: for cases where concluder_rounds=2, whether the "
            "retry tended to fix or break the answer (a case-by-case read, not "
            "just the aggregate fixed/broken counts already shown above).\n"
            "  Do not speculate beyond what the excerpts/hints actually show; if "
            "this batch is too small/limited to support a claim, say so instead "
            "of inventing a pattern."
        )
        return user, system
