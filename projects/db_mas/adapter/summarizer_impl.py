"""db_mas's named behavior-summarizer extension point.

`meta_agent.behavior_summarizer.BehaviorSummarizer._extract_failure_hint`
(the per-case "why did this fail" fingerprint fed into the summarizer's
prompt) is deliberately domain-agnostic: it only recognizes generic scorer
conventions (`failed_checks`, a nested `{name: {"passed": bool}}` map,
`missing_*`/`extra_*` collections, a bare `error` string) and truncates
whatever it finds to 180 chars. None of `DBMASScorer.score()`'s actual keys
(`precision`/`recall`/`f1`/`false_positives`/`false_negatives`/
`predicted_root_causes`/`gold_root_causes`) match any of those conventions,
so the base class's hint comes out empty for every db_mas case.

This override replaces just `_extract_failure_hint` with one that reads
db_mas's actual shapes -- including `agent_metadata` (db-mas's own
TaskResult.error / validation_error / forced_fallback, carried through by
the evaluator's `agent_artifact` injection, see `meta_agent/evaluator.py`) --
and keeps a much larger cap (2000 chars vs. 180) so it reads as close to a
full error log as the per-case line format allows. Everything else --
`_aggregate`, prompt sections (What was added / How it behaved / What
helped / What didn't help), the diff/mutable_log/tool_calls machinery, the
cumulative UPDATE-mode behavior -- is unchanged from `BehaviorSummarizer`.
"""
from __future__ import annotations

from typing import Any

from meta_agent.behavior_summarizer import BehaviorSummarizer
from meta_agent.models import CaseResult
from meta_agent.registry import register

_HINT_CHAR_CAP = 2000


@register("summarizer", "db_mas_default")
class DBMASBehaviorSummarizer(BehaviorSummarizer):
    @staticmethod
    def _extract_failure_hint(case: CaseResult) -> str:
        parts: list[str] = []

        if case.error:
            parts.append(f"runtime_error: {case.error}")

        details: dict[str, Any] = case.details or {}
        agent_meta: dict[str, Any] = details.get("agent_metadata") or {}

        if agent_meta.get("error"):
            parts.append(f"agent_error: {agent_meta['error']}")
        if agent_meta.get("validation_error"):
            parts.append(f"validation_error: {agent_meta['validation_error']}")
        if agent_meta.get("forced_fallback"):
            parts.append("forced_fallback=True")

        predicted = details.get("predicted_root_causes")
        gold = details.get("gold_root_causes")
        if predicted is not None or gold is not None:
            parts.append(f"predicted={predicted} gold={gold}")

        false_positives = details.get("false_positives")
        if false_positives:
            parts.append(f"false_positives={false_positives}")
        false_negatives = details.get("false_negatives")
        if false_negatives:
            parts.append(f"false_negatives={false_negatives}")

        precision, recall = details.get("precision"), details.get("recall")
        if precision is not None and recall is not None:
            parts.append(f"precision={precision:.2f} recall={recall:.2f}")

        if not parts:
            return ""

        # Flatten to one line -- the per-case prompt renders one line per
        # case (see BehaviorSummarizer._build_prompt), so embedded newlines
        # would break that layout.
        text = " | ".join(parts).replace("\n", " ")
        if len(text) > _HINT_CHAR_CAP:
            text = text[: _HINT_CHAR_CAP - 1].rstrip() + "…"
        return text
