"""db_mas_snapshot's named behavior-summarizer extension point.

`meta_agent.behavior_summarizer.BehaviorSummarizer._extract_failure_hint`
only recognizes generic scorer conventions (`failed_checks`, a nested
`{name: {"passed": bool}}` map, `missing_*`/`extra_*` collections, a bare
`error` string) -- none of `DBMasSnapshotScorer.score()`'s actual keys
(`recall`/`predicted`/`extraction`/`n_named`/`n_requested`/`snapshot_found`)
match those conventions, so the base class's hint comes out empty for every
case here too.

Two overrides:

1. `_extract_failure_hint` -- surfaces `snapshot_found=False` FIRST and most
   prominently, ahead of everything else: a case whose `unique_id` never
   matched a `db_cache/<id>.json` recorded snapshot is a WIRING bug (no
   evidence was ever available to any `query_db` call), not a genuine
   agent-behavior failure -- flagging it distinctly so the editor doesn't
   waste an edit "fixing" investigator prompts for a case with no evidence
   to find. Otherwise surfaces recall/precision/predicted/n_named/
   n_requested/extraction, plus a bounded excerpt of the lead DBA's raw
   diagnosis text (the trajectory's LAST entry, per `mas_workflow.run_task`:
   `trajectory = [out.to_dict() for out in inv_outs] + [lead_out.to_dict()]`)
   only when the verdict FORMAT itself went wrong (`extraction !=
   "deterministic"`, i.e. no "FINAL: <LABEL>" line was found/parsed, or
   `n_named > n_requested`, i.e. the model ignored the requested label
   count) -- exactly where a reasoning-level signal is worth the tokens;
   clean, cleanly-extracted verdicts get only the compact fingerprint line.
2. `_build_prompt` -- appends a short project-specific addendum asking the
   summarizer to name recurring verdict-FORMAT issues (missed "FINAL:"
   line, wrong label count) separately from recurring LABEL-CHOICE issues
   (which candidate gets over/under-selected), grounded only in the
   per-case data actually shown.
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


def _lead_dba_raw(details: dict[str, Any]) -> str:
    trajectory = details.get("trajectory") or []
    if not trajectory:
        return ""
    return trajectory[-1].get("raw", "") if isinstance(trajectory[-1], dict) else ""


@register("summarizer", "db_mas_snapshot_default")
class DBMasSnapshotBehaviorSummarizer(BehaviorSummarizer):
    @staticmethod
    def _extract_failure_hint(case: CaseResult) -> str:
        parts: list[str] = []

        if case.error:
            parts.append(f"runtime_error: {case.error}")

        details: dict[str, Any] = case.details or {}

        if details.get("error"):
            parts.append(f"agent_error: {details['error']}")

        # WIRING bug, not agent-behavior -- surfaced first, distinctly.
        if details.get("snapshot_found") is False:
            parts.append(
                "snapshot_found=False (WIRING BUG, not an agent-quality issue: "
                "this case's unique_id never matched any recorded db_cache "
                "snapshot, so no evidence was ever available to query_db -- "
                "do not attribute this case's low score to investigator/lead "
                "prompt quality)"
            )

        recall = details.get("recall")
        precision = details.get("precision")
        if recall is not None:
            parts.append(
                f"recall={recall} precision={precision} "
                f"predicted={details.get('predicted')!r} "
                f"gold={details.get('root_causes')!r} "
                f"n_named={details.get('n_named')} n_requested={details.get('n_requested')} "
                f"extraction={details.get('extraction')!r}"
            )

        # Verdict-format failure (no parseable "FINAL:" line, or the model
        # ignored the requested label count) -- attach the lead DBA's raw
        # text only here, where a reasoning-level signal is worth the tokens.
        extraction = details.get("extraction")
        over_named = (details.get("n_named") or 0) > (details.get("n_requested") or 0)
        if extraction not in ("deterministic",) or over_named:
            excerpt = _tail_excerpt(_lead_dba_raw(details))
            if excerpt:
                parts.append(f"lead_dba_excerpt={excerpt!r}")

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
            "\n\nADDITIONALLY, this MAS has 5 fixed specialist investigators "
            "(one per candidate root cause) feeding into a single terminal "
            "lead DBA agent, who must end its diagnosis with a line like "
            "\"FINAL: <LABEL>[, <LABEL>]\" -- scoring parses this line "
            "deterministically. Per-case hints above distinguish two DIFFERENT "
            "failure classes; treat them separately:\n"
            "  ## Verdict format issues\n"
            "  - Cases where extraction is not 'deterministic' (no parseable "
            "FINAL: line found) or the model named more/fewer labels than "
            "requested (n_named vs n_requested) -- this is a FORMAT compliance "
            "problem in the lead DBA's own instructions, not a diagnostic one. "
            "Cite case_ids and quote the lead_dba_excerpt where shown.\n"
            "  ## Label-choice issues\n"
            "  - Among cases with clean, correctly-formatted verdicts, name any "
            "recurring pattern in WHICH candidate gets over- or under-selected "
            "(e.g. a specific investigator's evidence being over-weighted or "
            "ignored) -- cite case_ids.\n"
            "  Also flag explicitly if any case in this batch has "
            "snapshot_found=False -- that is a wiring bug unrelated to prompt "
            "quality, not something the next edit should try to fix.\n"
            "  Do not speculate beyond what the hints/excerpts actually show; "
            "if this batch is too small/limited to support a claim, say so "
            "instead of inventing a pattern."
        )
        return user, system
