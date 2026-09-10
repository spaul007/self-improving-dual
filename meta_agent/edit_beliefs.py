"""Meta-cognitive belief layer over the deterministic edit history.

The belief document (``edit_memory_beliefs.md``, run root) is a list of
SCOPED, SCORED probabilities written by an LLM — rewritten wholesale at every
update — under a fixed contract (``belief_contract``) and a LEARNED guidance
text (``belief_optimizer``). Everything machine-owned lives in a sidecar the
LLM never writes (``edit_memory_beliefs_state.json``); the only code-written
text inside the document is the ``- track:`` calibration line under each
belief, regenerated on every write and stripped before any parse.

The loop, per node:
  1. EXPAND — the editor reads the document; right after the node's record is
     tagged, ``register()`` looks up the belief that covers its (strategy,
     area) per kind and freezes that p in ``round_NNN/belief_prediction.json``
     (a prediction made BEFORE the outcome, by construction).
  2. EVAL — once the node is measured and the analysis LLM has judged its
     implementation, ``belief_scoring.resolve`` pays Brier loss per kind.
  3. UPDATE — the maintainer sees its calibration record (per-belief track
     lines + the report) and the changed evidence, and rewrites the document;
     the contract validator rejects structural violations (one retry).
  4. OPTIMIZE — every ``optimize_every`` scored predictions, one LLM call
     revises the guidance text from the misses (hill-climb with rollback).

Facts vs beliefs: the registry and per-node records stay deterministic truth;
this layer is interpretation, kept honest by the scores.

Update cadence: the manager triggers ``update()`` after every eval batch and
every expand; an evidence signature makes a no-change trigger cost zero LLM
calls, so cost tracks actual evidence movement.

Import discipline: ``edit_memory`` imports this module at load time, so
imports of ``edit_memory`` / ``edit_memory_render`` here are deferred into
function bodies to avoid a cycle.
"""
from __future__ import annotations

import functools
import json
import os
import re
import shutil
import tempfile
from hashlib import blake2b
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

from . import belief_scoring
from .belief_contract import (
    BELIEF_KINDS,
    EXAMPLE,
    GRAMMAR,
    P_MAX,
    P_MIN,
    SUMMARY_CHAR_CAP,
    ParsedDoc,
    inject_track_lines,
    match_belief,
    parse_anchors,
    parse_citations,
    parse_document,
    render_violations,
    strip_track_lines,
    validate_document,
    verify_citations,
)
from .belief_optimizer import (  # noqa: F401  (re-exported for callers/tests)
    INSTRUCTION_ARCHIVE_DIR,
    INSTRUCTION_NAME,
    SEED_INSTRUCTION,
    InstructionOptimizer,
)
from .belief_scoring import Scored
from .edit_outcome import MIN_SHARED_FOR_VERDICT, NEUTRAL_BAND, run_context

__all__ = [
    "BELIEFS_NAME", "BELIEFS_STATE_NAME", "BELIEFS_ARCHIVE_DIR",
    "BELIEF_PROMPT_DIR", "BELIEF_PREDICTION_NAME", "PREDICTION_NAME",
    "BELIEF_FORMAT", "BeliefStore", "parse_anchors", "parse_citations",
    "INSTRUCTION_NAME", "INSTRUCTION_ARCHIVE_DIR", "SEED_INSTRUCTION",
]

BELIEFS_NAME = "edit_memory_beliefs.md"
BELIEFS_STATE_NAME = "edit_memory_beliefs_state.json"
BELIEFS_ARCHIVE_DIR = "edit_memory_beliefs_archive"
BELIEF_PROMPT_DIR = "edit_memory_beliefs_prompts"
BELIEF_PREDICTION_NAME = "belief_prediction.json"
# The two-stage editor's own prediction sidecar (which belief its proposal
# relied on). Joined for the report only; scoring is registration-based.
PREDICTION_NAME = "edit_prediction.json"
# Salted into the evidence signature: bumping forces one update on resume.
BELIEF_FORMAT = 2

SCORING_JUDGE = """  strategy beliefs predict P(the JUDGE finds the mechanism improved its target | implementation
    sound). The judge is the per-node analysis: it reads the edit's implementation, its runtime
    traces, the per-check results and the per-case rows, and gives each sub-edit an effect
    verdict — `improved` (y=1), `no_effect` or `regressed` (y=0), `unclear` (not scored) — with
    an evidence grade (strong / moderate / weak; weak verdicts are not scored). Scored only when
    the judge also found the implementation sound (an unsound implementation says nothing about
    the strategy). No shared cases with the parent are needed.
  implementation beliefs predict P(implementation sound) — y comes from that same analysis.
The benchmark score Δ is shown in every record as CONTEXT with its standard error: ~16-case
batches put the SE near ±0.1, so a Δ smaller than 2×SE says nothing; the judge's reasons
(which components fired, on which cases, what the checks did) are the evidence to read.
Cite nodes by the judge's verdict — `[node N: improved]` — and add the Δ only when it is
well measured (≥ 16 shared cases, or |unpaired Δ| > 2×SE)."""

SCORING_DELTA = """  strategy beliefs predict P(helped | implementation sound) — y=1 iff Δ vs parent ≥ +{threshold}
    over ≥ {min_shared} shared cases; scored only when the per-node analysis judged the
    implementation sound (an unsound implementation says nothing about the strategy).
  implementation beliefs predict P(implementation sound) — y comes from that analysis verdict.
NOISE: evaluation batches are ~16 random cases; a single-node Δ over few shared cases is often
luck — let the number of measured nodes, not one number, move p."""

CONTRACT_SYSTEM = """You maintain the BELIEF DOCUMENT of a self-improving agent run. The agent editor reads it
before every edit. Every belief is a SCOPED PROBABILITY that is SCORED: when a new edit lands,
the code matches it to your beliefs by its registry tags (strategy, area) and registers the
matched p; when that edit's outcome is judged, each matched belief pays Brier loss (p - y)^2.
{scoring}
A judged edit that no belief of a kind covers is scored at p=0.5 (loss 0.25): silence is not
free — and a belief written at p=0.5 scores exactly like silence. Commit to the probability you
actually expect; the calibration record — the `- track:` line under each belief and the
calibration report in this prompt — is your loss signal, and it corrects you either way.

FORMAT (machine-checked; a violation costs a retry, a second violation discards the update):
{grammar}

Example section:
{example}

Rules: scope ids must come from the registry list in this prompt; p in [{p_min}, {p_max}]; no
sections other than an optional leading `## Summary` (≤ {summary_cap} chars) and `### belief:`
sections; no bullets other than kind/scope/predict/evidence/next; never write `- track:` lines
(code regenerates them); at most one belief per (kind, scope); the whole document must stay
under {doc_char_cap} characters. Every node you rely on carries a citation in one of the exact
forms `[node N: improved]` (the judge's effect verdict: improved / no_effect / regressed /
unclear), `[node N: improved; Δ+0.0310/12]` (verdict plus the paired Δ over 12 shared cases),
`[node N: Δ-0.0117±0.1461]` (the unpaired Δ ± SE) or `[node N: unmeasured]`; citations are
verified against the records.
Submit the complete new document via `submit_belief_update` (document, change_note).

## Guidance (learned — revised by an optimizer from your calibration record)
{instruction}"""

BELIEF_TOOL = {"type": "function", "function": {
    "name": "submit_belief_update",
    "description": "Submit the complete rewritten belief document.",
    "parameters": {"type": "object", "properties": {
        "document": {"type": "string"},
        "change_note": {"type": "string"}},
        "required": ["document"]}}}


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


class BeliefStore:
    """Maintains the belief document + sidecar + guidance. Constructed by
    ``EditMemory`` from the ``beliefs:`` config subdict (every key below is a
    real kwarg — an unknown key is a config error); all entry points are
    best-effort."""

    def __init__(
        self,
        llm_caller: Callable[..., object],
        *,
        model: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        base_url: Optional[str] = None,
        enabled: bool = True,
        doc_char_cap: int = 40000,
        max_delta_records: int = 12,
        evidence_char_budget: int = 60000,
        threshold: float = NEUTRAL_BAND,
        min_shared: int = MIN_SHARED_FOR_VERDICT,
        optimize_enabled: bool = True,
        optimize_every: int = 8,
        optimize_min_scored: int = 8,
        optimize_rollback_margin: float = 0.02,
        instruction_char_cap: int = 2500,
        optimize_model: Optional[str] = None,
        optimize_reasoning_effort: Optional[str] = None,
        label_source: str = "judge",
        min_evidence: str = "moderate",
    ) -> None:
        self.llm = llm_caller
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.base_url = base_url
        self.enabled = bool(enabled)
        self.doc_char_cap = max(1000, int(doc_char_cap))
        self.max_delta_records = max(1, int(max_delta_records))
        self.evidence_char_budget = max(1000, int(evidence_char_budget))
        self.threshold = float(threshold)
        self.min_shared = int(min_shared)
        if label_source not in belief_scoring.LABEL_SOURCES:
            raise ValueError(f"beliefs.label_source must be one of "
                             f"{belief_scoring.LABEL_SOURCES}, got {label_source!r}")
        if min_evidence not in ("strong", "moderate", "weak"):
            raise ValueError("beliefs.min_evidence must be 'strong', 'moderate' "
                             f"or 'weak', got {min_evidence!r}")
        self.label_source = label_source
        self.min_evidence = min_evidence
        self.optimize_enabled = bool(optimize_enabled)
        self._opt = InstructionOptimizer(
            functools.partial(self._call, model=optimize_model,
                              reasoning_effort=optimize_reasoning_effort),
            enabled=optimize_enabled, every=optimize_every,
            min_scored=optimize_min_scored,
            rollback_margin=optimize_rollback_margin,
            char_cap=instruction_char_cap)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def update(self, experiment_dir: Path, tree: Any) -> bool:
        """Score, maybe optimize, then one sig-gated belief rewrite. Returns
        True when the document was rewritten. Never raises; on a rejected
        or failed rewrite the previous document and signature stay on disk,
        so the update retries at the next evidence change."""
        if not self.enabled:
            return False
        experiment_dir = Path(experiment_dir)
        try:
            from .edit_memory_render import _load_records
            records = _load_records(experiment_dir)
            if not records:
                return False
            state = self._load_state(experiment_dir)
            registry = self._load_registry(experiment_dir)
            predictions = self._load_belief_predictions(experiment_dir)
            n_updates = int(state.get("n_updates", 0))
            self._opt.ensure_seed(experiment_dir, state)

            # 1. Score every registered prediction whose node is now
            #    measurable with an implementation verdict.
            scored = [Scored.from_dict(d) for d in (state.get("scored") or [])]
            skipped = list(state.get("skipped") or [])
            already = ({(s.node, s.kind) for s in scored}
                       | {(int(k["node"]), str(k["kind"])) for k in skipped})
            new_scored, pending, new_skipped = belief_scoring.resolve(
                predictions, records, threshold=self.threshold,
                min_shared=self.min_shared, already=already,
                n_updates=n_updates, label_source=self.label_source,
                min_evidence=self.min_evidence)
            state_changed = False
            if new_skipped:
                skipped.extend(new_skipped)
                state["skipped"] = skipped
                state_changed = True
            if new_scored:
                scored.extend(new_scored)
                state["scored"] = [s.as_dict() for s in scored]
                state["scored_since_step"] = (
                    int(state.get("scored_since_step", 0) or 0) + len(new_scored))
                state_changed = True
                self._refresh_track_lines(experiment_dir, records, scored)
                print(f"[edit_beliefs] scored {len(new_scored)} prediction(s); "
                      f"running mean Brier "
                      f"{belief_scoring.mean_brier(scored):.3f}", flush=True)

            # 2. The calibration report (shared by the optimizer step and
            #    the update prompt) and the optimizer step itself.
            doc = self._load_doc(experiment_dir)
            joins = self._join_predictions(experiment_dir, records)
            parsed_now = parse_document(doc) if doc else ParsedDoc()
            soft_now = verify_citations(strip_track_lines(doc), records) if doc else []
            info = state.get("instruction") or {}
            report = belief_scoring.render_calibration_report(
                parsed=parsed_now, scored=scored, pending=pending,
                records=records, predictions=predictions,
                soft_violations=soft_now, joins=joins,
                version_history=info.get("versions") or [],
                current_version=int(info.get("version", 0) or 0),
                min_shared=self.min_shared, n_skipped=len(skipped),
                label_source=self.label_source, threshold=self.threshold)
            if self._opt.maybe_step(experiment_dir, state, scored=scored,
                                    predictions=predictions, records=records,
                                    calibration_report=report,
                                    n_updates=n_updates):
                state_changed = True
            if state_changed:
                self._save_state(experiment_dir, state)

            # 3. Evidence gate — no movement, no LLM call.
            per_node_sigs = self._per_node_sigs(experiment_dir, records)
            sig = self._evidence_signature(experiment_dir, per_node_sigs, joins)
            if sig == state.get("updated_at_signature"):
                return False

            # 4. The rewrite, with one contract-driven retry.
            delta_nodes = self._delta_nodes(state, per_node_sigs)
            system = self._system_prompt(self._opt.current_text(experiment_dir))
            user = self._build_update_prompt(
                experiment_dir, tree, records, registry, doc, report,
                delta_nodes)
            got = self._call(system, user, BELIEF_TOOL, "belief update")
            body = strip_track_lines(str((got or {}).get("document") or ""))
            parsed = validate_document(body, registry=registry,
                                       doc_char_cap=self.doc_char_cap,
                                       records=records)
            retry_text = ""
            if parsed.violations or not body.strip():
                retry_text = (
                    "\n\n## Your previous submission was rejected\n"
                    + (render_violations(parsed.violations)
                       if parsed.violations else
                       "1. [HARD] no document was submitted")
                    + "\n\n## Rejected submission\n"
                    + body[: 2 * self.doc_char_cap]
                    + "\n\nResubmit the complete corrected document.")
                got2 = self._call(system, user + retry_text, BELIEF_TOOL,
                                  "belief update (retry)")
                body2 = strip_track_lines(str((got2 or {}).get("document") or ""))
                parsed2 = validate_document(body2, registry=registry,
                                            doc_char_cap=self.doc_char_cap,
                                            records=records)
                # Take the retry unless it made things worse than a
                # soft-only original.
                if not parsed2.hard and body2.strip():
                    body, parsed, got = body2, parsed2, got2
                elif parsed.hard or not body.strip():
                    body, parsed, got = body2, parsed2, got2
            self._dump_prompt(experiment_dir, n_updates + 1, system, user,
                              retry_text)
            if parsed.hard or not body.strip():
                first = parsed.hard[0].render()[:160] if parsed.hard else "empty"
                print(f"[edit_beliefs] update rejected after retry "
                      f"({len(parsed.hard)} hard violation(s): {first}); kept "
                      "the previous version", flush=True)
                return False

            soft_counts: dict[str, int] = {}
            for v in parsed.soft:
                if v.slug:
                    soft_counts[v.slug] = soft_counts.get(v.slug, 0) + 1
            track = belief_scoring.track_lines(parsed.beliefs, scored, soft_counts)
            final = inject_track_lines(body, track)
            if doc:
                self._archive(experiment_dir, state, n_updates)
            _atomic_write(experiment_dir / BELIEFS_NAME, final)
            state.update({
                "belief_format": BELIEF_FORMAT,
                "n_updates": n_updates + 1,
                "updated_at_signature": sig,
                "per_node_sigs": per_node_sigs,
                "prediction_joins": joins,
                "change_note": str((got or {}).get("change_note") or "")[:300],
                "doc_violations_last": [v.render() for v in parsed.violations],
                "beliefs_index": {b.slug: {"kind": b.kind, "strategy": b.strategy,
                                           "area": b.area, "p": b.p}
                                  for b in parsed.beliefs},
            })
            state.setdefault("scored", [])
            state.setdefault("scored_since_step", 0)
            self._save_state(experiment_dir, state)
            print(f"[edit_beliefs] update {n_updates + 1}: "
                  f"{len(parsed.beliefs)} belief(s), {len(parsed.soft)} soft "
                  f"issue(s), {len(delta_nodes)} node(s) of new evidence"
                  + (" (after retry)" if retry_text else ""), flush=True)
            return True
        except Exception as exc:  # noqa: BLE001
            print(f"[edit_beliefs] update failed: {exc!r}", flush=True)
            return False

    def register(self, experiment_dir: Path, node_id: int,
                 round_dir: Path) -> Optional[dict]:
        """Freeze the beliefs that cover a freshly recorded node — the p the
        editor was shown — into ``round_dir/belief_prediction.json``.
        Idempotent; ``None`` when the node has no record (tagger failed)."""
        if not self.enabled:
            return None
        experiment_dir, round_dir = Path(experiment_dir), Path(round_dir)
        path = round_dir / BELIEF_PREDICTION_NAME
        try:
            if path.exists():
                return json.loads(path.read_text(encoding="utf-8"))
            from .edit_memory_render import _load_records
            records = _load_records(experiment_dir)
            rec = records.get(int(node_id))
            if rec is None:
                return None
            state = self._load_state(experiment_dir)
            doc = self._load_doc(experiment_dir)
            parsed = parse_document(doc) if doc else ParsedDoc()
            tags = [{"edit": t.get("edit"), "strategy": t.get("strategy"),
                     "area": t.get("area") or None,
                     "fit": t.get("fit") or "exact"}
                    for t in (rec.get("tags") or [])]
            # A cap-forced tag is the nearest registry id by shared token,
            # not what the tagger meant: no belief may be matched to it or
            # charged for it. A node whose tags are all forced is treated as
            # uncoverable (retired unscored), never as silence.
            usable = [t for t in tags if t["fit"] != "forced"]
            try:
                parent = int(rec["fm"].get("parent"))
            except (TypeError, ValueError):
                parent = None
            # Could a belief have covered this node? Only if some EARLIER
            # node already used one of its strategies — beliefs can only be
            # scoped to registry ids, and the registry grows with the edits.
            prior = {t.get("strategy") for n, r in records.items()
                     if n != int(node_id) for t in (r.get("tags") or [])
                     if t.get("strategy")}
            coverable = any(t.get("strategy") in prior for t in usable)
            payload: dict[str, Any] = {
                "version": 1, "node": int(node_id), "parent": parent,
                "belief_version": int(state.get("n_updates", 0)),
                "instruction_version": self._opt.current_version(state),
                "tags": tags, "coverable": coverable,
                "strategy": None, "implementation": None,
            }
            for kind in BELIEF_KINDS:
                m = match_belief(parsed.beliefs, kind, usable)
                if m is None:
                    continue
                b, idx = m
                payload[kind] = {
                    "slug": b.slug, "p": b.p,
                    "scope": {"strategy": b.strategy, "area": b.area},
                    "matched_edit": idx, "section": b.section_text[:1500]}
            _atomic_write(path, json.dumps(payload, indent=2) + "\n")
            return payload
        except Exception as exc:  # noqa: BLE001
            print(f"[edit_beliefs] registration failed for node {node_id}: "
                  f"{exc!r}", flush=True)
            return None

    def render_block(self, experiment_dir: Path) -> str:
        """The document exactly as written (track lines included) — never
        cut. ``""`` when absent."""
        return self._load_doc(Path(experiment_dir))

    def calibration_line(self, experiment_dir: Path) -> str:
        state = self._load_state(Path(experiment_dir))
        scored = state.get("scored") or []
        if not scored:
            return ""
        mean = sum(float(s.get("brier", 0.0)) for s in scored) / len(scored)
        return (f"{len(scored)} scored prediction(s) so far, mean Brier "
                f"{mean:.2f} (0.25 = uninformative); guidance "
                f"v{self._opt.current_version(state)}.")

    # ------------------------------------------------------------------ #
    # Prompt assembly
    # ------------------------------------------------------------------ #
    def _system_prompt(self, instruction: str) -> str:
        text = CONTRACT_SYSTEM.replace(
            "{scoring}", SCORING_JUDGE if self.label_source == "judge"
            else SCORING_DELTA)
        for key, val in (("{threshold}", f"{self.threshold:.2f}"),
                         ("{min_shared}", str(self.min_shared)),
                         ("{grammar}", GRAMMAR), ("{example}", EXAMPLE),
                         ("{p_min}", f"{P_MIN:.2f}"), ("{p_max}", f"{P_MAX:.2f}"),
                         ("{summary_cap}", str(SUMMARY_CHAR_CAP)),
                         ("{doc_char_cap}", str(self.doc_char_cap)),
                         ("{instruction}", instruction.strip() or SEED_INSTRUCTION)):
            text = text.replace(key, val)
        return text

    def _build_update_prompt(self, experiment_dir: Path, tree: Any,
                             records: Mapping[int, Any],
                             registry: Mapping[str, Any], doc: str,
                             report: str, delta_nodes: list[int]) -> str:
        from .edit_memory_render import build_ledger, judge_ledger_lines
        parts: list[str] = []

        rc = run_context(tree) or {}
        if rc and self.label_source == "judge":
            parts.append(
                "## Run context\nThe run optimizes what the judge finds: each "
                "node's per-node analysis grades its mechanisms on their own "
                "traces and per-check results, and strategy beliefs are scored "
                "against those verdicts. Score context (noisy, for orientation "
                "only): seed %.4f/%d · best so far %.4f/%d (node %d)."
                % (rc.get("seed_mean", 0.0), rc.get("seed_n", 0),
                   rc.get("best_mean", 0.0), rc.get("best_n", 0),
                   rc.get("best_node", -1)))
        elif rc:
            parts.append(
                "## Run context\nseed %.4f/%d · best so far %.4f/%d (node %d). "
                "The goal is the highest ABSOLUTE score."
                % (rc.get("seed_mean", 0.0), rc.get("seed_n", 0),
                   rc.get("best_mean", 0.0), rc.get("best_n", 0),
                   rc.get("best_node", -1)))

        parts.append("## Registry ids you may use in scope lines\n"
                     + self._render_registry(registry))

        parts.append("## Current belief document (with code-generated track lines)\n"
                     + (doc.rstrip() if doc.strip() else
                        "(none yet — this is the first update; write the first "
                        "version)"))

        parts.append(report)

        ledger = build_ledger(registry, records, threshold=self.threshold,
                              min_shared=self.min_shared)
        if ledger and self.label_source == "judge":
            L = ["## Per-strategy outcomes (the judge's verdicts per sub-edit, the "
                 "checks they targeted, regressions and implementation verdicts; "
                 "the score Δ is trailing context)"]
            L += judge_ledger_lines(ledger)
            parts.append("\n".join(L))
        elif ledger:
            L = ["## Deterministic per-strategy ledger (ground truth)"]
            for r in ledger:
                tally = ", ".join(f"{k} {v}" for k, v in r["tally"].items())
                med = ("Δ median %+.4f" % r["median"]
                       if r["median"] is not None else "no measured Δ")
                L.append(f"- `{r['id']}` — {r['n_nodes']} node(s) "
                         f"({', '.join(str(n) for n in r['nodes'])}) · "
                         f"{med} · {tally} — {r['definition']}")
            parts.append("\n".join(L))

        if delta_nodes:
            # Whole records only, newest first; the budget drops the OLDEST
            # whole records rather than cutting any record mid-text.
            L = ["## New/changed evidence since your last update (full records)"]
            used = 0
            shown = 0
            for n in delta_nodes[:self.max_delta_records]:
                text = records[n].get("text") or records[n].get("body") or ""
                if shown and used + len(text) > self.evidence_char_budget:
                    break
                L.append(f"### node {n}\n{text.rstrip()}")
                used += len(text)
                shown += 1
            if shown < len(delta_nodes):
                L.append(f"(+{len(delta_nodes) - shown} older changed node(s) "
                         "not shown — see the ledger)")
            parts.append("\n\n".join(L))
        return "\n\n".join(parts)

    @staticmethod
    def _render_registry(registry: Mapping[str, Any]) -> str:
        strategies = registry.get("strategies") or {}
        areas = registry.get("areas") or {}
        rows = []
        for sid, e in strategies.items():
            nodes = {r.get("node") for r in (e.get("edits") or [])}
            rows.append((-len(nodes), sid, len(nodes), e.get("definition", "")))
        rows.sort()
        L = ["strategies:"]
        L += [f"- `{sid}` — {n} node(s) — {d}" for _, sid, n, d in rows] or ["- (none yet)"]
        L.append("areas:")
        L += [f"- `{aid}` — {e.get('definition', '')}"
              for aid, e in sorted(areas.items())] or ["- (none yet)"]
        return "\n".join(L)

    # ------------------------------------------------------------------ #
    # Track-line refresh (deterministic; keeps the editor's copy current)
    # ------------------------------------------------------------------ #
    def _refresh_track_lines(self, experiment_dir: Path,
                             records: Mapping[int, Any],
                             scored: list[Scored]) -> None:
        doc = self._load_doc(experiment_dir)
        if not doc.strip():
            return
        parsed = parse_document(doc)
        soft_counts: dict[str, int] = {}
        for v in verify_citations(strip_track_lines(doc), records):
            if v.slug:
                soft_counts[v.slug] = soft_counts.get(v.slug, 0) + 1
        track = belief_scoring.track_lines(parsed.beliefs, scored, soft_counts)
        new = inject_track_lines(doc, track)
        if new != doc:
            _atomic_write(experiment_dir / BELIEFS_NAME, new)

    # ------------------------------------------------------------------ #
    # Evidence signature + delta detection
    # ------------------------------------------------------------------ #
    def _per_node_sigs(self, experiment_dir: Path,
                       records: Mapping[int, Any]) -> dict[str, dict[str, str]]:
        from .edit_memory import _load_state as load_round_state
        out: dict[str, dict[str, str]] = {}
        for n in sorted(records):
            st = load_round_state(experiment_dir / f"round_{n:03d}")
            out[str(n)] = {
                "case_sig": str(st.get("child_case_sig") or ""),
                "analysis_sig": str(st.get("analysis_sig") or ""),
            }
        return out

    def _evidence_signature(self, experiment_dir: Path,
                            per_node_sigs: Mapping[str, Mapping[str, str]],
                            joins: list[dict[str, Any]]) -> str:
        from .edit_memory import REGISTRY_NAME
        try:
            reg = (experiment_dir / REGISTRY_NAME).read_bytes()
        except OSError:
            reg = b""
        rows = [f"{n}:{v['case_sig']}:{v['analysis_sig']}"
                for n, v in sorted(per_node_sigs.items())]
        rows += sorted(
            f"{j['node']}:{j.get('belief_id')}:{j.get('measured_delta')}:"
            f"{j.get('n_shared')}" for j in joins)
        h = blake2b(digest_size=8)
        h.update(f"v{BELIEF_FORMAT}|".encode("utf-8"))
        h.update(blake2b(reg, digest_size=8).hexdigest().encode("utf-8"))
        h.update("|".join(rows).encode("utf-8"))
        return h.hexdigest()

    def _delta_nodes(self, state: Mapping[str, Any],
                     per_node_sigs: Mapping[str, Mapping[str, str]]) -> list[int]:
        old = state.get("per_node_sigs") or {}
        changed = [int(n) for n, v in per_node_sigs.items()
                   if old.get(n) != v]
        return sorted(changed, reverse=True)  # newest first

    # ------------------------------------------------------------------ #
    # Proposal-prediction joins (report only — "cited by N proposals")
    # ------------------------------------------------------------------ #
    def _join_predictions(self, experiment_dir: Path,
                          records: Mapping[int, Any]) -> list[dict[str, Any]]:
        joins: list[dict[str, Any]] = []
        for pred_path in sorted(experiment_dir.glob(f"round_*/{PREDICTION_NAME}")):
            try:
                pred = json.loads(pred_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            m = re.search(r"round_(\d+)", pred_path.parent.name)
            if not m:
                continue
            node = int(m.group(1))
            rec = records.get(node) or {}
            bid = str(pred.get("belief_id") or "")[:120]
            if bid.lower().startswith("belief:"):
                bid = bid[len("belief:"):]
            joins.append({
                "node": node,
                "belief_id": bid,
                "expected_direction": str(pred.get("expected_direction") or ""),
                "expected_delta": pred.get("expected_delta"),
                # v2 sidecar: the judge verdict and checks the planner expected.
                "expected_effect": str(pred.get("expected_effect") or ""),
                "expected_targets": [str(t)[:80] for t in
                                     (pred.get("expected_targets") or [])][:8],
                "why": str(pred.get("why") or "")[:300],
                "measured_delta": rec.get("delta"),
                "n_shared": rec.get("n_shared") or 0,
            })
        return joins

    # ------------------------------------------------------------------ #
    # Files
    # ------------------------------------------------------------------ #
    @staticmethod
    def _load_belief_predictions(experiment_dir: Path) -> dict[int, dict[str, Any]]:
        out: dict[int, dict[str, Any]] = {}
        for p in sorted(Path(experiment_dir).glob(f"round_*/{BELIEF_PREDICTION_NAME}")):
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            m = re.search(r"round_(\d+)", p.parent.name)
            if isinstance(d, dict) and m:
                out[int(d.get("node", m.group(1)))] = d
        return out

    @staticmethod
    def _load_registry(experiment_dir: Path) -> dict[str, Any]:
        from .edit_memory import REGISTRY_NAME
        try:
            got = json.loads((experiment_dir / REGISTRY_NAME)
                             .read_text(encoding="utf-8"))
            return got if isinstance(got, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    def _load_doc(self, experiment_dir: Path) -> str:
        path = experiment_dir / BELIEFS_NAME
        try:
            return path.read_text(encoding="utf-8") if path.exists() else ""
        except (OSError, UnicodeDecodeError):
            return ""

    def _dump_prompt(self, experiment_dir: Path, n: int, system: str,
                     user: str, retry_text: str) -> None:
        try:
            _atomic_write(experiment_dir / BELIEF_PROMPT_DIR / f"update_{n:04d}.txt",
                          "### SYSTEM\n" + system + "\n\n### USER\n" + user
                          + (retry_text + "\n" if retry_text else "\n"))
        except Exception as exc:  # noqa: BLE001
            print(f"[edit_beliefs] prompt dump failed: {exc!r}", flush=True)

    def _archive(self, experiment_dir: Path, state: dict[str, Any],
                 n_updates: int) -> None:
        """Copy the current document (track lines included) aside before
        the new version replaces it."""
        try:
            dest_dir = experiment_dir / BELIEFS_ARCHIVE_DIR
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / f"beliefs_{n_updates:04d}.md"
            src = experiment_dir / BELIEFS_NAME
            if src.exists():
                shutil.copyfile(src, dest)
            versions = list(state.get("versions") or [])
            if dest.name not in versions:
                versions.append(dest.name)
            state["versions"] = versions
        except Exception as exc:  # noqa: BLE001
            print(f"[edit_beliefs] archive failed: {exc!r}", flush=True)

    def _load_state(self, experiment_dir: Path) -> dict[str, Any]:
        path = experiment_dir / BELIEFS_STATE_NAME
        try:
            if path.exists():
                got = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(got, dict):
                    return got
        except (OSError, json.JSONDecodeError):
            pass
        return {}

    def _save_state(self, experiment_dir: Path, state: Mapping[str, Any]) -> None:
        _atomic_write(experiment_dir / BELIEFS_STATE_NAME,
                      json.dumps(dict(state), indent=1, default=str) + "\n")

    # ------------------------------------------------------------------ #
    def _call(self, system: str, user: str, tool: dict, tag: str, *,
              model: Optional[str] = None,
              reasoning_effort: Optional[str] = None) -> Optional[dict]:
        kwargs: dict[str, Any] = {
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "tools": [tool],
        }
        model = model or self.model
        effort = reasoning_effort or self.reasoning_effort
        if model:
            kwargs["model"] = model
        if effort:
            kwargs["reasoning_effort"] = effort
        else:
            kwargs["temperature"] = 0.2
        if self.base_url:
            kwargs["base_url"] = self.base_url
        try:
            resp = self.llm(**kwargs)
        except Exception as exc:  # noqa: BLE001
            print(f"[edit_beliefs] {tag}: llm call failed: {exc!r}", flush=True)
            return None
        name = tool["function"]["name"]
        for tc in (getattr(resp, "tool_calls", None) or []):
            if getattr(tc, "name", None) == name:
                return tc.arguments
        m = re.search(r"```json\s*(\{.*?\})\s*```",
                      getattr(resp, "content", None) or "", re.S)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                pass
        print(f"[edit_beliefs] {tag}: no structured output", flush=True)
        return None
