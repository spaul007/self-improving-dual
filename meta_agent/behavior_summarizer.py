"""Per-round behavior summarizer.

After each round's evaluation, this module distills the round's signal —
(diff vs parent) + (filtered trace events) + (per-case scorer details) —
into a narrative ``behavior_memory.md`` that the next editor call sees as
part of its steering context. Forensics also persisted as
``behavior_aggregate.json`` (the structured pre-aggregation fed to the LLM)
and ``behavior_summarizer_prompt.txt`` (the literal prompt sent).

Design:
    - Deterministic pre-aggregation in code (cheap, exact, auditable):
      diff mutable files, bucket ``mutable_log`` events by label/verdict,
      cross-tab with case pass/fail outcomes.
    - One LLM call summarizes that structured table into prose. This avoids
      the failure mode where the LLM hallucinates from raw event noise —
      it summarizes a clean view you can also inspect by hand.
    - Graceful when missing: if anything goes wrong (missing trace file,
      LLM failure), the summarizer logs and returns ``None`` rather than
      crashing the round. Descendants then just don't see this ancestor's
      memory block.

Hook points:
    - ``HGMManager._evaluate`` calls this after ``_refresh_node_feedback``
      so the node's feedback is final.
    - ``HGMDualManager._expand`` calls this after ``_promote_winner`` +
      ``gatherer.compile`` so the winner's feedback is final. For dual the
      ``eval_dir`` is the winner's variant subdir (where the winning
      agent's traces live), not the canonical round dir.
"""
from __future__ import annotations

import json
import os
import re
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from . import verbose_log
from .edit_diff import DIFF_CHAR_CAP, changed_mutable_files, diff_mutable_files
from .models import EvaluationResult, CaseResult
from .registry import register


# Per-section caps that bound the prompt size sent to the summarizer LLM.
_DIFF_CHAR_CAP = DIFF_CHAR_CAP
_TRACE_LABEL_CAP = 12          # at most N distinct labels surfaced
_EVENTS_PER_LABEL_SAMPLE = 3   # sample events per label included verbatim
_PER_CASE_LINE_CAP = 25        # at most N per-case outcome lines
_TOOL_CALL_CAP = 20            # at most N distinct tools in the usage table

# A tool_result.result_preview that looks like an error (mirrors the gatherer).
_TOOL_ERROR_RE = re.compile(r'^Error\b|"error"\s*:', re.IGNORECASE)


@register("summarizer", "default")
class BehaviorSummarizer:
    """LLM-summarized per-round behavior memory.

    Args:
        llm_caller: same callable injected into ``AgentEditor`` (the
            project's ``platform_core.llm_wrapper.call_llm``). Injected
            by ``meta_agent.config.build_components``.
        model: LLM identifier (e.g. ``"gpt-5.4-mini"``). Defaults to None
            so the wrapper's default is used.
        reasoning_effort: ``"low"`` | ``"medium"`` | ``"high"`` — passed
            through to the LLM call. ``None`` falls back to ``temperature``.
        base_url: optional alternate OpenAI-compatible endpoint.
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
        # Domain noun for the summarizer prompt — keeps the framework
        # project-agnostic. Precedence: explicit config value, then the active
        # project name (exported as ``META_AGENT_PROJECT`` by
        # ``meta_agent.runtime_env``), then a generic fallback.
        self.domain_label = (
            domain_label or os.environ.get("META_AGENT_PROJECT") or "agent"
        )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def summarize(
        self,
        *,
        round_dir: Path,
        parent_round_dir: Optional[Path],
        eval_dir: Path,
        eval_result: EvaluationResult,
        node_id: int,
        parent_id: Optional[int],
    ) -> Optional[Path]:
        """Produce a behavior memory for ``round_dir``.

        ``round_dir``: canonical round folder where outputs go.
        ``parent_round_dir``: parent's round folder, for the diff. ``None``
            for the seed round (no parent → no diff → no memory).
        ``eval_dir``: where ``logs/trace.jsonl`` lives (== ``round_dir`` for
            vanilla HGM; the winner's variant subdir for hgm_dual).
        ``eval_result``: per-case outcomes for the round.

        Returns the path to the written ``behavior_memory.md`` on success,
        or ``None`` if skipped or on error.
        """
        if parent_round_dir is None:
            # Root: nothing was edited, nothing to summarize.
            return None
        try:
            aggregate = self._aggregate(
                round_dir=round_dir,
                parent_round_dir=parent_round_dir,
                eval_dir=eval_dir,
                eval_result=eval_result,
                node_id=node_id,
                parent_id=parent_id,
            )
        except Exception as exc:
            print(
                f"[summarizer] aggregate failed for node {node_id}: {exc!r}",
                flush=True,
            )
            return None

        # Persist the structured aggregate verbatim for forensics, even if
        # the LLM call fails afterwards.
        try:
            (round_dir / "behavior_aggregate.json").write_text(
                json.dumps(aggregate, indent=2, default=str), encoding="utf-8"
            )
        except Exception as exc:
            print(
                f"[summarizer] failed to write behavior_aggregate.json for "
                f"node {node_id}: {exc!r}",
                flush=True,
            )

        # If a memo already exists for this round (a prior evaluation batch),
        # update it cumulatively rather than overwriting from scratch — the
        # memo carries the running synthesis across batches, so trace.jsonl can
        # stay per-batch.
        prior_memory: Optional[str] = None
        existing = round_dir / "behavior_memory.md"
        if existing.exists():
            try:
                prior_memory = existing.read_text(encoding="utf-8").strip() or None
            except OSError:
                prior_memory = None

        try:
            prompt_user, prompt_system = self._build_prompt(
                aggregate, prior_memory=prior_memory
            )
        except Exception as exc:
            print(
                f"[summarizer] prompt-build failed for node {node_id}: {exc!r}",
                flush=True,
            )
            return None

        # Persist the literal prompt for forensics. Written regardless of
        # whether the LLM call ultimately succeeds.
        try:
            (round_dir / "behavior_summarizer_prompt.txt").write_text(
                f"### SYSTEM\n{prompt_system}\n\n### USER\n{prompt_user}",
                encoding="utf-8",
            )
        except Exception:
            pass

        try:
            response = self._call_llm(prompt_system, prompt_user)
        except Exception:
            print(
                f"[summarizer] LLM call failed for node {node_id}:\n"
                + traceback.format_exc(limit=3),
                flush=True,
            )
            return None

        summary_text = (response or "").strip()
        if not summary_text:
            print(
                f"[summarizer] empty summary returned for node {node_id} — skipped",
                flush=True,
            )
            return None

        memory_path = round_dir / "behavior_memory.md"
        memory_path.write_text(summary_text, encoding="utf-8")
        if verbose_log.is_enabled():
            verbose_log.write_text(
                round_dir,
                "behavior_summarizer_response.txt",
                summary_text,
            )
        print(
            f"[summarizer] node {node_id}: wrote behavior_memory.md "
            f"({len(summary_text)} chars)",
            flush=True,
        )
        return memory_path

    # ------------------------------------------------------------------ #
    # Aggregation — deterministic structured table the LLM summarizes.
    # ------------------------------------------------------------------ #

    def _aggregate(
        self,
        *,
        round_dir: Path,
        parent_round_dir: Path,
        eval_dir: Path,
        eval_result: EvaluationResult,
        node_id: int,
        parent_id: Optional[int],
    ) -> dict[str, Any]:
        diff_text = self._diff_mutable_files(parent_round_dir, round_dir)

        events = self._read_trace(eval_dir / "logs" / "trace.jsonl")
        # Per-case event grouping uses ``case_id`` when the platform tags
        # events with one; fall back to ``"global"`` for untagged events.
        events_by_case = self._group_events_by_case(events)

        mutable_log_aggregate = self._aggregate_mutable_log(
            events, eval_result.per_case, events_by_case
        )

        tool_calls_aggregate = self._aggregate_tool_calls(
            events, eval_result.per_case, events_by_case
        )

        per_case = self._per_case_summary(eval_result.per_case, events_by_case)

        # Identify which mutable files actually changed vs parent — useful
        # for the LLM's "what was added/modified" section.
        changed_files = self._changed_mutable_files(parent_round_dir, round_dir)

        return {
            "node_id": node_id,
            "parent_id": parent_id,
            "score": round(eval_result.score, 4),
            "passed": eval_result.passed,
            "failed": eval_result.failed,
            "n_cases": len(eval_result.per_case),
            "changed_files": changed_files,
            "diff": diff_text,
            "mutable_log": mutable_log_aggregate,
            "tool_calls": tool_calls_aggregate,
            "per_case": per_case,
        }

    def _changed_mutable_files(
        self, parent_round_dir: Path, round_dir: Path
    ) -> list[str]:
        """Thin wrapper — see :func:`meta_agent.edit_diff.changed_mutable_files`."""
        return changed_mutable_files(parent_round_dir, round_dir)

    def _diff_mutable_files(
        self, parent_round_dir: Path, round_dir: Path
    ) -> str:
        """Thin wrapper — see :func:`meta_agent.edit_diff.diff_mutable_files`."""
        return diff_mutable_files(parent_round_dir, round_dir, char_cap=_DIFF_CHAR_CAP)

    def _read_trace(self, path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        out: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return out

    def _group_events_by_case(
        self, events: list[dict[str, Any]]
    ) -> dict[str, list[dict[str, Any]]]:
        """Bucket events by ``case_id`` if the event payload carries one,
        else by ``"global"``. The runner stamps per-case events with
        ``case_id``; bare trace.emit() calls from mutable code don't, but
        the per-case subprocess scope means they're effectively per-case
        already (one trace file per case-running subprocess writes are
        interleaved — case_id is the only reliable disambiguator)."""
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for ev in events:
            payload = ev.get("payload") or {}
            case_id = payload.get("case_id") or "global"
            groups[str(case_id)].append(ev)
        return dict(groups)

    def _aggregate_mutable_log(
        self,
        events: list[dict[str, Any]],
        per_case: list[CaseResult],
        events_by_case: dict[str, list[dict[str, Any]]],
    ) -> dict[str, Any]:
        """Cross-tab ``mutable_log`` events by label/verdict against case
        pass/fail. The result has, per label:

            - ``total_fires``: total count of events with this label
            - ``by_verdict``: {verdict: count}
            - ``by_name``: {name: count} (when payload has ``name``)
            - ``cases_pass_when_fired``: # of passed cases where this label fired
            - ``cases_fail_when_fired``: # of failed cases where this label fired
            - ``sample_events``: up to ``_EVENTS_PER_LABEL_SAMPLE`` example payloads
        """
        pass_ids = {str(c.case_id) for c in per_case if c.passed}
        fail_ids = {str(c.case_id) for c in per_case if not c.passed}

        per_label: dict[str, dict[str, Any]] = defaultdict(
            lambda: {
                "total_fires": 0,
                "by_verdict": defaultdict(int),
                "by_name": defaultdict(int),
                "cases_pass_when_fired": set(),
                "cases_fail_when_fired": set(),
                "sample_events": [],
            }
        )

        for ev in events:
            if ev.get("kind") != "mutable_log":
                continue
            payload = ev.get("payload") or {}
            label = str(payload.get("label") or "(unlabeled)")
            entry = per_label[label]
            entry["total_fires"] += 1
            verdict = payload.get("verdict")
            if verdict is not None:
                entry["by_verdict"][str(verdict)] += 1
            name = payload.get("name")
            if name is not None:
                entry["by_name"][str(name)] += 1
            if len(entry["sample_events"]) < _EVENTS_PER_LABEL_SAMPLE:
                # Drop noisy keys from the sample so the prompt stays compact.
                sample = {
                    k: v for k, v in payload.items() if k != "label"
                }
                entry["sample_events"].append(sample)

        # Per-case attribution: walk the per-case event list, gather labels
        # that fired in that case, then bucket the case_id into the right set.
        for case_id, ev_list in events_by_case.items():
            labels_fired = {
                str((ev.get("payload") or {}).get("label") or "(unlabeled)")
                for ev in ev_list
                if ev.get("kind") == "mutable_log"
            }
            for label in labels_fired:
                if case_id in pass_ids:
                    per_label[label]["cases_pass_when_fired"].add(case_id)
                elif case_id in fail_ids:
                    per_label[label]["cases_fail_when_fired"].add(case_id)

        # Materialize sets to counts (json-friendly), cap by total_fires desc.
        materialized: dict[str, Any] = {}
        items = sorted(
            per_label.items(), key=lambda kv: -kv[1]["total_fires"]
        )[:_TRACE_LABEL_CAP]
        for label, entry in items:
            materialized[label] = {
                "total_fires": entry["total_fires"],
                "by_verdict": dict(entry["by_verdict"]),
                "by_name": dict(entry["by_name"]),
                "cases_pass_when_fired": len(entry["cases_pass_when_fired"]),
                "cases_fail_when_fired": len(entry["cases_fail_when_fired"]),
                "sample_events": entry["sample_events"],
            }
        return materialized

    def _aggregate_tool_calls(
        self,
        events: list[dict[str, Any]],
        per_case: list[CaseResult],
        events_by_case: dict[str, list[dict[str, Any]]],
    ) -> dict[str, Any]:
        """Cross-tab ``tool_call`` / ``tool_result`` events by tool name against
        case pass/fail. Captures BOTH immutable and mutable tools (mutable ones
        flagged via the ``mutable`` payload key set by
        ``platform_core.tools.call_mutable_tool``), so the summary can report
        whether editor-added tools were actually used and how they correlated
        with outcomes. Per tool:

            - ``calls``: total tool_call events
            - ``errors``: tool_result events whose preview looks like an error
            - ``mutable``: True if any call carried the mutable flag
            - ``cases_pass_when_called`` / ``cases_fail_when_called``
        """
        pass_ids = {str(c.case_id) for c in per_case if c.passed}
        fail_ids = {str(c.case_id) for c in per_case if not c.passed}

        per_tool: dict[str, dict[str, Any]] = defaultdict(
            lambda: {
                "calls": 0,
                "errors": 0,
                "mutable": False,
                "cases_pass_when_called": set(),
                "cases_fail_when_called": set(),
            }
        )
        for ev in events:
            kind = ev.get("kind")
            payload = ev.get("payload") or {}
            if kind == "tool_call":
                name = str(payload.get("name") or "?")
                entry = per_tool[name]
                entry["calls"] += 1
                if payload.get("mutable"):
                    entry["mutable"] = True
            elif kind == "tool_result":
                preview = payload.get("result_preview") or ""
                if isinstance(preview, str) and _TOOL_ERROR_RE.search(preview):
                    per_tool[str(payload.get("name") or "?")]["errors"] += 1

        for case_id, ev_list in events_by_case.items():
            tools_called = {
                str((ev.get("payload") or {}).get("name") or "?")
                for ev in ev_list
                if ev.get("kind") == "tool_call"
            }
            for name in tools_called:
                if case_id in pass_ids:
                    per_tool[name]["cases_pass_when_called"].add(case_id)
                elif case_id in fail_ids:
                    per_tool[name]["cases_fail_when_called"].add(case_id)

        # Mutable tools first (most relevant for "was the new tool used?"), then
        # by call count; cap total distinct tools surfaced.
        ordered = sorted(
            per_tool.items(),
            key=lambda kv: (not kv[1]["mutable"], -kv[1]["calls"], kv[0]),
        )[:_TOOL_CALL_CAP]
        out: dict[str, Any] = {}
        for name, entry in ordered:
            out[name] = {
                "calls": entry["calls"],
                "errors": entry["errors"],
                "mutable": entry["mutable"],
                "cases_pass_when_called": len(entry["cases_pass_when_called"]),
                "cases_fail_when_called": len(entry["cases_fail_when_called"]),
            }
        return out

    def _per_case_summary(
        self,
        per_case: list[CaseResult],
        events_by_case: dict[str, list[dict[str, Any]]],
    ) -> list[dict[str, Any]]:
        """Per-case roll-up that captures: passed/failed, the labels that
        fired, and a short ``failure_hint`` from ``case.details`` so the
        LLM can correlate behavior with what went wrong."""
        rows: list[dict[str, Any]] = []
        for case in per_case[:_PER_CASE_LINE_CAP]:
            ev_list = events_by_case.get(str(case.case_id), [])
            labels_fired = sorted({
                f'{(ev.get("payload") or {}).get("label", "?")}/'
                f'{(ev.get("payload") or {}).get("verdict", "?")}'
                for ev in ev_list
                if ev.get("kind") == "mutable_log"
            })
            failure_hint = self._extract_failure_hint(case)
            rows.append({
                "case_id": str(case.case_id),
                "passed": case.passed,
                "score": round(case.score, 3),
                "labels_fired": labels_fired,
                "failure_hint": failure_hint,
            })
        return rows

    @staticmethod
    def _extract_failure_hint(case: CaseResult) -> str:
        """Pull the most informative single string from a CaseResult's
        ``details`` to give the LLM a fingerprint of why the case failed.

        Domain-agnostic: reads only generic, cross-project scorer conventions —
        it assumes no particular task schema. Order of preference:
        runtime error → a flat ``failed_checks`` list → any nested
        ``{name: {"passed": bool}}`` check map (failing names) → non-empty
        ``missing_*`` / ``extra_*`` collections (a common "what's wrong"
        convention) → a generic scorer ``error`` string."""
        if case.error:
            return f"runtime: {case.error[:180]}"
        details = case.details or {}
        # Flattest signal, if the scorer emits one.
        fc = details.get("failed_checks")
        if isinstance(fc, list) and fc:
            head = ", ".join(str(x) for x in fc[:4])
            return f"failed_checks: {head[:180]}"
        # Any nested check map of the shape {name: {"passed": bool, ...}} —
        # surface the failing names. Schema-agnostic: no fixed key name.
        for key, val in details.items():
            if not isinstance(val, dict) or not val:
                continue
            failed = [
                n for n, v in val.items()
                if isinstance(v, dict) and "passed" in v and not v.get("passed", True)
            ]
            if failed:
                return f"{key}: {', '.join(failed[:4])[:180]}"
        # Non-empty "missing_*" / "extra_*" collections (a common scorer
        # convention for what's wrong, e.g. missing_products / extra_coupons).
        flags = [
            f"{k}={len(v)}"
            for k, v in details.items()
            if (k.startswith("missing") or k.startswith("extra"))
            and isinstance(v, (list, dict)) and v
        ]
        if flags:
            return "; ".join(flags[:5])[:180]
        # Generic scorer error string.
        if isinstance(details.get("error"), str) and details["error"]:
            return f"error: {details['error'][:180]}"
        return ""

    # ------------------------------------------------------------------ #
    # Prompt building + LLM call.
    # ------------------------------------------------------------------ #

    def _build_prompt(
        self, aggregate: dict[str, Any], prior_memory: Optional[str] = None
    ) -> tuple[str, str]:
        """Returns (user_message, system_message).

        When ``prior_memory`` is given (the node was already summarized on an
        earlier evaluation batch), the prompt switches to UPDATE mode: the model
        merges the new batch's observations into the prior cumulative memo
        instead of writing from scratch. This keeps ``behavior_memory.md``
        cumulative across batches without growing ``trace.jsonl``."""
        update_clause = (
            ""
            if not prior_memory
            else (
                "\nUPDATE MODE: a prior cumulative memo for THIS agent already "
                "exists (shown below under '## Prior memo'); the structured data "
                "in this message is from an ADDITIONAL evaluation batch of "
                f"{aggregate.get('n_cases')} case(s). Produce an UPDATED "
                "cumulative memo that folds the new batch into the prior one: "
                "keep still-valid points, revise claims the new evidence "
                "changes, and accumulate case coverage (don't drop earlier "
                "findings just because they're not in this batch). The diff is "
                "unchanged across batches, so keep '## What was added' stable. "
                "Output the full updated memo (same sections), not a delta.\n"
            )
        )
        system = (
            f"You are a behavior summarizer for a self-evolving "
            f"{self.domain_label} agent. After each edit, you receive: (a) the "
            "diff between parent "
            "and child agent code, (b) a structured table of in-code "
            "instrumentation events (`mutable_log`) cross-tabbed against case "
            "outcomes, (c) a per-case roll-up of what passed/failed and "
            "which instrumentation fired, and (d) a per-tool call-usage table "
            "(immutable and mutable tools; mutable = editor-added) cross-tabbed "
            "against case outcomes.\n\n"
            "Produce a concise markdown memo for the NEXT editor. Sections:\n"
            "  ## What was added — one bullet per new/changed tool, helper, or "
            "verifier: name it and give a one-line description of what it does "
            "(read from the diff), so the next editor knows the component's "
            "purpose without seeing the code.\n"
            "  ## How it behaved (observed)\n"
            "  ## What helped\n"
            "  ## What didn't help (or hurt)\n\n"
            "This is an OBSERVED-BEHAVIOR ANALYSIS only — do NOT propose or "
            "suggest what to change next; leave the next edit to the editor. "
            "Stay under 300 words total. Be specific: name the verifiers / "
            "branches / helpers by their `name` field. Cite case_ids when "
            "useful. If `mutable_log` is empty, say so — the editor failed to "
            "instrument and you can only summarize from the diff + outcomes. "
            "If the diff added or changed a tool, use the tool-usage table to "
            "state whether it was actually called and how its calls correlated "
            "with pass/fail.\n"
            "CRITICAL — verdict vs case outcome: `by_verdict` (and the "
            "`label/verdict` tags) are the instrumentation's OWN self-reported "
            "verdict (pass/fail/skip of that check), NOT whether the CASE "
            "passed. Whether a case passed is given ONLY by "
            "`cases_pass_when_fired` / `cases_fail_when_fired` and the per-case "
            "`passed` field. Never say a branch 'fired on pass cases' from "
            "`by_verdict`; use the case-outcome counts. Only call a specific "
            "case_id passed/failed using the per-case table, and do NOT "
            "attribute a skip/verdict to a particular case unless a "
            "`sample_event` names that case_id. Do not invent verdicts, counts, "
            "case_ids, or component names the data doesn't support; state only "
            "what these fields show."
            + update_clause
        )

        diff_block = aggregate.get("diff") or "(no diff — no mutable changes)"
        per_case_lines = []
        for row in aggregate.get("per_case") or []:
            mark = "✓" if row["passed"] else "✗"
            labels = ", ".join(row["labels_fired"]) or "(none fired)"
            hint = row["failure_hint"] or ""
            per_case_lines.append(
                f"  {mark} case {row['case_id']}  score={row['score']:.3f}  "
                f"labels=[{labels}]  hint={hint}"
            )

        mutable_log_lines = []
        for label, entry in (aggregate.get("mutable_log") or {}).items():
            by_v = entry["by_verdict"] or {}
            by_n = entry["by_name"] or {}
            mutable_log_lines.append(
                f"  - label={label}: total_fires={entry['total_fires']}  "
                f"by_verdict={by_v}  by_name={by_n}  "
                f"pass_when_fired={entry['cases_pass_when_fired']}  "
                f"fail_when_fired={entry['cases_fail_when_fired']}"
            )
            for sample in entry["sample_events"][:2]:
                mutable_log_lines.append(f"      sample: {sample}")
        if not mutable_log_lines:
            mutable_log_lines.append(
                "  (no mutable_log events emitted — edit was uninstrumented "
                "or didn't fire on these cases)"
            )

        tool_call_lines = []
        for name, entry in (aggregate.get("tool_calls") or {}).items():
            tag = " [mutable/editor-added]" if entry.get("mutable") else ""
            tool_call_lines.append(
                f"  - {name}{tag}: calls={entry['calls']}  errors={entry['errors']}  "
                f"pass_when_called={entry['cases_pass_when_called']}  "
                f"fail_when_called={entry['cases_fail_when_called']}"
            )
        if not tool_call_lines:
            tool_call_lines.append("  (no tool calls recorded)")

        user = (
            f"## Round summary\n"
            f"node {aggregate['node_id']} (parent={aggregate['parent_id']})  "
            f"score={aggregate['score']:.3f}  "
            f"passed={aggregate['passed']}/{aggregate['n_cases']}\n"
            f"changed files: {', '.join(aggregate.get('changed_files') or []) or '(none)'}\n\n"
            f"## Diff vs parent\n```\n{diff_block}\n```\n\n"
            f"## mutable_log aggregates (cross-tabbed with case outcomes)\n"
            + "\n".join(mutable_log_lines)
            + "\n\n## Tool usage (cross-tabbed with case outcomes)\n"
            + "\n".join(tool_call_lines)
            + "\n\n## Per-case outcomes\n"
            + "\n".join(per_case_lines)
        )
        if prior_memory:
            user += (
                "\n\n## Prior memo (cumulative so far — update this)\n"
                + prior_memory
            )
        return user, system

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


def render_memory_for_steering(
    round_dir: Path, *, cap_chars: Optional[int] = None
) -> Optional[str]:
    """Read a previously-written ``behavior_memory.md`` for use in steering.

    Returns ``None`` when the file doesn't exist (memory wasn't generated
    for that ancestor — e.g. seed round, summarizer disabled, or write
    failed). When ``cap_chars`` is set, the head of the file is returned
    with an elision marker on truncation.
    """
    path = round_dir / "behavior_memory.md"
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8")
    if cap_chars is not None and len(text) > cap_chars:
        return text[:cap_chars].rstrip() + "\n<... truncated ...>"
    return text
