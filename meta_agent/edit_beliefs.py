"""Meta-cognitive belief layer over the deterministic edit history.

The belief document (``edit_memory_beliefs.md``, run root) is free-form
markdown OWNED BY THE META-AGENT — content and structure alike, rewritten
wholesale at every update. Code fixes only three thin conventions (belief
anchors, an inline citation format, a self-describing structure section) and
keeps all machine bookkeeping in a sidecar the LLM never writes
(``edit_memory_beliefs_state.json``).

Facts vs beliefs: the registry and per-node records stay deterministic truth;
this layer is interpretation. It is kept honest by a deterministic verifier
that fact-checks every inline citation against the records and narrates the
findings in a code-generated appendix section — correction pressure the next
update must confront (the textgrad-style loop), and evidential context for
the agent editor reading the document downstream.

Update cadence: the manager triggers ``update()`` after every eval batch and
every expand; an evidence signature makes a no-change trigger cost zero LLM
calls, so cost tracks actual evidence movement.

Import discipline: ``edit_memory`` imports this module at load time, so
imports of ``edit_memory`` / ``edit_memory_render`` here are deferred into
function bodies to avoid a cycle.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from hashlib import blake2b
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

from .edit_diff import truncate_middle
from .edit_outcome import run_context

BELIEFS_NAME = "edit_memory_beliefs.md"
BELIEFS_STATE_NAME = "edit_memory_beliefs_state.json"
BELIEFS_ARCHIVE_DIR = "edit_memory_beliefs_archive"
BELIEF_FORMAT = 1
PREDICTION_NAME = "edit_prediction.json"
MACHINE_SECTION = "## Machine appendix (code-generated — do not write this section)"

# Anchor convention: every belief section heads with "### belief:<slug>".
# Liberal on separators: the prompt suggests kebab-case but models often write
# snake_case — truncating at the first "_" broke the citation-credit join.
_ANCHOR_RE = re.compile(r"^###\s+belief:([a-z0-9][a-z0-9_-]*)", re.M)
# Citation convention: "[node 17: Δ-0.04/22]" or "[node 17: unmeasured]".
_CITE_RE = re.compile(
    r"\[node\s+(\d+):\s*(?:Δ\s*([+-]?\d+(?:\.\d+)?)\s*/\s*(\d+)|unmeasured)\]")
# How far a quoted delta may sit from the record's before it is a misquote.
_DELTA_TOL = 0.005

BELIEF_SYSTEM = """You maintain the BELIEF DOCUMENT of a self-improving agent run: your run-global
understanding of every edit strategy tried so far — what works, what fails, why, and what to do
next. The agent editor reads this document before every new edit; its quality directly shapes the
next edit. You are called after each evaluation batch with the evidence that changed.

YOUR FIRST JOB IS TO GET THE BELIEFS RIGHT: fold the new evidence in, revise whatever the measured
outcomes contradict, and keep next moves concrete enough to act on. You also own this document's
structure — reorganize it whenever the structure itself is failing you (predictions from one
section keep going wrong, sections nobody ever cites) — but a correct belief in a plain layout
beats an elegant layout around a stale one.

WHAT TO REASON ABOUT for each strategy or notable edit:
- Is it useful, harmful, mixed — or simply UNPROVEN? Unproven is the default and is an invitation
  to test, never a rejection.
- Is the evidence sufficient to conclude anything? Say explicitly what is missing.
- When something failed: is the STRATEGY bad, or only its IMPLEMENTATION? Dead components
  ("0 calls", "never fired"), SUSPECT VERIFIER flags, and gates disagreeing with the scorer all
  point at broken implementation — the idea may still be sound.
- What is the actionable next move: build on it, repair a specific mechanism, combine, or stop?
- Are generated tools/skills useful, and does any need gating before it can help?

NOISE — the facts of this setup: evaluation batches are ~16 randomly sampled cases and per-case
scores vary a lot, so a delta over a small shared set is frequently sampling noise. Treat any
single-node delta as weak evidence; look for consistency across multiple nodes before a confident
verdict; for every pattern, ask whether it could be luck. The judgment is yours — no threshold is
imposed on you.

CONVENTIONS (machine-checked; violations are reported back to you, not fixed for you):
1. Every distinct belief starts a section headed exactly `### belief:<slug> — <title>` with a
   stable kebab-case slug. If you rename a slug, add `renamed from:<old-slug>` on the heading line.
2. Every quantitative claim carries an inline citation in the exact form `[node 17: Δ-0.040/22]`
   (node id, delta over shared cases, number of shared cases) or `[node 17: unmeasured]`. Cited
   numbers are verified against the actual records; misquotes are called out in the machine
   appendix until corrected.
3. Keep a `## Document structure` section explaining how you currently organize this document,
   what you track, and what you deliberately dropped.

The `## Machine appendix` section at the bottom of the current document is code-generated fact
tables — do NOT write or copy it; it is regenerated after your update. Where it corrects one of
your numbers, adopt the correction. Where it shows a prediction went wrong, judge whether that is
a real miss or noise, and revise or explicitly defend the belief.

Submit the complete new document via `submit_belief_update` (field `document`, everything except
the machine appendix) plus a one-line `change_note`."""

BELIEF_TOOL = {"type": "function", "function": {
    "name": "submit_belief_update",
    "description": "Submit the complete rewritten belief document.",
    "parameters": {"type": "object", "properties": {
        "document": {"type": "string"},
        "change_note": {"type": "string"}},
        "required": ["document"]}}}

REFLECT_SYSTEM = """You are reviewing the BELIEF DOCUMENT of a self-improving agent run — not to update its
content, but to judge whether it is REPRESENTED well. You see only the document and the machine
statistics about how it has been used; no new evidence.

Answer, concretely and briefly:
- Which sections never get cited by any edit proposal (dead weight — compress or drop)?
- Where did predictions cluster wrong — what distinction is the representation missing there
  (e.g. one belief conflating two mechanisms that behave differently)?
- What should this document track that it currently doesn't, given the failures on record?
- How should it be reorganized, if at all?

Remember content beats form: recommend restructuring only where the structure is demonstrably
failing. Submit 3-8 short directives via `submit_belief_reflection`; they will be handed to the
next regular update, which does the actual rewrite with evidence in hand."""

REFLECT_TOOL = {"type": "function", "function": {
    "name": "submit_belief_reflection",
    "description": "Submit representation-level directives for the next update.",
    "parameters": {"type": "object", "properties": {
        "directives": {"type": "array", "items": {"type": "string"}}},
        "required": ["directives"]}}}

SEED_SKELETON = """No belief document exists yet — this is the first update. A suggested starting
shape (NOT mandatory; from the next update on the structure is entirely yours):

## Document structure
(explain your organization here)

### belief:<strategy-slug> — <title>
- stance: unproven | useful | harmful | mixed — with the evidence, cited
- evidence: is it sufficient? what is missing?
- attribution: strategy vs implementation, when something failed
- next move: one concrete, actionable suggestion
- open questions
"""


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


def strip_machine_section(document: str) -> str:
    """The body the LLM owns — everything above the machine appendix."""
    idx = document.find(MACHINE_SECTION)
    return (document[:idx] if idx != -1 else document).rstrip("\n")


def parse_anchors(document: str) -> list[str]:
    return _ANCHOR_RE.findall(document)


def parse_citations(document: str) -> list[dict[str, Any]]:
    """Every inline citation with the slug of the belief section it sits in
    (``None`` when it appears above the first anchor)."""
    anchors = [(m.start(), m.group(1)) for m in _ANCHOR_RE.finditer(document)]
    out: list[dict[str, Any]] = []
    for m in _CITE_RE.finditer(document):
        slug = None
        for pos, s in anchors:
            if pos <= m.start():
                slug = s
            else:
                break
        out.append({
            "slug": slug,
            "node": int(m.group(1)),
            "delta": float(m.group(2)) if m.group(2) is not None else None,
            "n_shared": int(m.group(3)) if m.group(3) is not None else None,
            "raw": m.group(0),
        })
    return out


class BeliefStore:
    """Maintains the belief document + sidecar. Constructed by ``EditMemory``
    from the ``beliefs:`` config subdict; all entry points best-effort."""

    def __init__(
        self,
        llm_caller: Callable[..., object],
        *,
        model: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        base_url: Optional[str] = None,
        enabled: bool = True,
        reflect_every: int = 0,
        doc_char_cap: int = 48000,
        max_delta_records: int = 12,
        record_char_cap: int = 4000,
    ) -> None:
        self.llm = llm_caller
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.base_url = base_url
        self.enabled = bool(enabled)
        self.reflect_every = max(0, int(reflect_every))
        self.doc_char_cap = max(1000, int(doc_char_cap))
        self.max_delta_records = max(1, int(max_delta_records))
        self.record_char_cap = max(500, int(record_char_cap))

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def update(self, experiment_dir: Path, tree: Any) -> bool:
        """One sig-gated belief update. Returns True when the document was
        rewritten. Never raises; on any failure the previous document and
        signature stay on disk so the update retries at the next evidence
        change (same retry semantics as the analysis sig)."""
        if not self.enabled:
            return False
        experiment_dir = Path(experiment_dir)
        try:
            from .edit_memory_render import _load_records
            records = _load_records(experiment_dir)
            if not records:
                return False
            state = self._load_state(experiment_dir)
            per_node_sigs = self._per_node_sigs(experiment_dir, records)
            joins = self._join_predictions(experiment_dir, records)
            sig = self._evidence_signature(experiment_dir, per_node_sigs, joins)
            if sig == state.get("updated_at_signature"):
                return False

            doc = self._load_doc(experiment_dir)
            n_updates = int(state.get("n_updates", 0))
            directives = self._maybe_reflect(state, doc, n_updates)

            delta_nodes = self._delta_nodes(state, per_node_sigs)
            user = self._build_update_prompt(
                experiment_dir, tree, records, doc, delta_nodes, joins,
                directives)
            got = self._call(BELIEF_SYSTEM, user, BELIEF_TOOL, "belief update")
            new_doc = strip_machine_section(str((got or {}).get("document") or ""))
            if not new_doc.strip():
                print("[edit_beliefs] update produced no document; kept the "
                      "previous version", flush=True)
                return False
            anchors = parse_anchors(new_doc)
            if not anchors:
                print("[edit_beliefs] update rejected: no `### belief:` "
                      "anchors; kept the previous version", flush=True)
                return False
            if len(new_doc) > 2 * self.doc_char_cap:
                print(f"[edit_beliefs] update rejected: {len(new_doc)} chars "
                      f"exceeds 2x cap {self.doc_char_cap}; kept the previous "
                      "version", flush=True)
                return False

            verification = self._verify(new_doc, records)
            citation_outcomes = self._citation_outcomes(joins)
            appendix = self._render_appendix(
                new_doc, records, verification, joins, citation_outcomes,
                anchors)

            if doc:
                self._archive(experiment_dir, state, doc, n_updates)
            _atomic_write(experiment_dir / BELIEFS_NAME,
                          new_doc + "\n\n" + appendix + "\n")
            state.update({
                "belief_format": BELIEF_FORMAT,
                "n_updates": n_updates + 1,
                "updated_at_signature": sig,
                "per_node_sigs": per_node_sigs,
                "prediction_joins": joins,
                "citation_outcomes": citation_outcomes,
                "verification": verification,
                "change_note": str((got or {}).get("change_note") or "")[:300],
            })
            self._save_state(experiment_dir, state)
            print(f"[edit_beliefs] update {n_updates + 1}: {len(anchors)} "
                  f"belief(s), {len(verification['bad_citations'])} citation "
                  f"issue(s), {len(delta_nodes)} node(s) of new evidence",
                  flush=True)
            return True
        except Exception as exc:  # noqa: BLE001
            print(f"[edit_beliefs] update failed: {exc!r}", flush=True)
            return False

    def render_block(self, experiment_dir: Path) -> str:
        """The full document (machine appendix included — it is signal),
        capped for steering. ``""`` when absent."""
        doc = self._load_doc(Path(experiment_dir), with_appendix=True)
        if not doc:
            return ""
        return truncate_middle(doc, self.doc_char_cap)

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
    # Prediction joins (calibration raw data — no correctness judgment)
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
                "why": str(pred.get("why") or "")[:300],
                "measured_delta": rec.get("delta"),
                "n_shared": rec.get("n_shared") or 0,
            })
        return joins

    @staticmethod
    def _citation_outcomes(joins: list[dict[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for j in joins:
            slug = j.get("belief_id") or "(no belief named)"
            entry = out.setdefault(slug, {"cited_by_nodes": [], "measured": []})
            if j["node"] not in entry["cited_by_nodes"]:
                entry["cited_by_nodes"].append(j["node"])
            if j.get("measured_delta") is not None:
                entry["measured"].append({
                    "node": j["node"], "delta": j["measured_delta"],
                    "n_shared": j["n_shared"]})
        return out

    # ------------------------------------------------------------------ #
    # Verifier — fact-checking only, never an epistemic judgment
    # ------------------------------------------------------------------ #
    def _verify(self, document: str,
                records: Mapping[int, Any]) -> dict[str, Any]:
        bad: list[dict[str, Any]] = []
        outside: list[str] = []
        for c in parse_citations(document):
            if c["slug"] is None:
                outside.append(c["raw"])
            rec = records.get(c["node"])
            if rec is None:
                bad.append({**c, "reason": "no record for this node"})
                continue
            if c["delta"] is None:  # "[node N: unmeasured]"
                if rec.get("delta") is not None:
                    bad.append({**c, "reason":
                                f"claimed unmeasured, but the record shows "
                                f"Δ{rec['delta']:+.4f} over {rec['n_shared']} "
                                f"shared"})
                continue
            r_delta, r_n = rec.get("delta"), rec.get("n_shared") or 0
            if r_delta is None:
                bad.append({**c, "reason": "the record has no measured Δ yet"})
            elif (c["n_shared"] != r_n
                  or (c["delta"] > 0) != (r_delta > 0) and abs(r_delta) > 1e-9
                  or abs(c["delta"] - r_delta) > _DELTA_TOL):
                bad.append({**c, "reason":
                            f"the record shows Δ{r_delta:+.4f} over {r_n} "
                            f"shared"})
        return {"bad_citations": bad, "citations_outside_beliefs": outside}

    # ------------------------------------------------------------------ #
    # Machine appendix — verbose, template-rendered, facts only
    # ------------------------------------------------------------------ #
    def _render_appendix(self, document: str, records: Mapping[int, Any],
                         verification: Mapping[str, Any],
                         joins: list[dict[str, Any]],
                         citation_outcomes: Mapping[str, Any],
                         anchors: list[str]) -> str:
        cites = parse_citations(document)
        by_slug: dict[str, list[dict[str, Any]]] = {}
        for c in cites:
            by_slug.setdefault(c["slug"] or "", []).append(c)
        bad = list(verification.get("bad_citations") or [])
        bad_raw = {b["raw"] for b in bad}

        L = [MACHINE_SECTION,
             "_Regenerated by code after every belief update. Facts checked "
             "against the actual records — when a number here disagrees with "
             "the body text above, this section is what the records say. No "
             "judgments: whether evidence is thin or a prediction truly "
             "missed is the belief maintainer's call._", "",
             "### Citation checks"]
        if not bad:
            L.append(f"- All {len(cites)} inline citations match the records.")
        for b in bad:
            where = f"belief:{b['slug']}" if b.get("slug") else \
                "outside any belief section"
            L.append(f"- {where} quotes `{b['raw']}`, but {b['reason']} — "
                     "the body text misquotes the record.")
        if verification.get("citations_outside_beliefs"):
            L.append("- Citations found above the first belief anchor: "
                     + ", ".join(f"`{r}`" for r in
                                 verification["citations_outside_beliefs"][:5]))

        L += ["", "### Evidence base per belief"]
        for slug in anchors:
            mine = by_slug.get(slug, [])
            nodes = sorted({c["node"] for c in mine})
            if not nodes:
                L.append(f"- belief:{slug} contains no verifiable citations — "
                         "its claims cannot be checked against any record.")
                continue
            shared = sum(records.get(n, {}).get("n_shared") or 0 for n in nodes)
            ok = sum(1 for c in mine if c["raw"] not in bad_raw)
            L.append(
                f"- belief:{slug} cites {len(nodes)} node(s) — "
                f"{', '.join(str(n) for n in nodes)} — covering {shared} "
                f"shared case(s) in total; {ok} of its {len(mine)} citations "
                "check out.")

        L += ["", "### Proposal outcomes (edits justified by each belief)"]
        cited_slugs = set()
        for slug, entry in sorted(citation_outcomes.items()):
            cited_slugs.add(slug)
            parts = []
            measured = {m["node"]: m for m in entry.get("measured", [])}
            for n in entry.get("cited_by_nodes", []):
                m = measured.get(n)
                if m is not None:
                    parts.append(f"node {n} (measured Δ{m['delta']:+.4f} over "
                                 f"{m['n_shared']} shared)")
                else:
                    parts.append(f"node {n} (not yet measured)")
            L.append(f"- belief:{slug} justified the edit(s) at "
                     + "; ".join(parts) + ".")
            if slug not in anchors and slug != "(no belief named)":
                L.append(f"  - note: belief:{slug} is no longer present in the "
                         "document, but predictions still reference it.")
        never = [s for s in anchors if s not in cited_slugs]
        if never:
            L.append("- Never cited by any proposal so far: "
                     + ", ".join(f"belief:{s}" for s in never) + ".")

        n_measured = sum(1 for j in joins if j.get("measured_delta") is not None)
        L += ["", "### Run totals",
              f"- {len(anchors)} belief(s); {len(joins)} prediction(s) "
              f"recorded, {n_measured} with a measured outcome so far."]
        return "\n".join(L)

    # ------------------------------------------------------------------ #
    # Prompt assembly
    # ------------------------------------------------------------------ #
    def _build_update_prompt(self, experiment_dir: Path, tree: Any,
                             records: Mapping[int, Any], doc: str,
                             delta_nodes: list[int],
                             joins: list[dict[str, Any]],
                             directives: list[str]) -> str:
        from .edit_memory import REGISTRY_NAME
        from .edit_memory_render import build_ledger
        parts: list[str] = []

        rc = run_context(tree) or {}
        if rc:
            parts.append(
                "## Run context\nseed %.4f/%d · best so far %.4f/%d (node %d). "
                "The goal is the highest ABSOLUTE score."
                % (rc.get("seed_mean", 0.0), rc.get("seed_n", 0),
                   rc.get("best_mean", 0.0), rc.get("best_n", 0),
                   rc.get("best_node", -1)))

        parts.append("## Current belief document")
        parts.append(doc if doc else SEED_SKELETON)

        try:
            registry = json.loads(
                (experiment_dir / REGISTRY_NAME).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            registry = {}
        ledger = build_ledger(registry, records, threshold=0.02, min_shared=8)
        if ledger:
            parts.append("## Deterministic per-strategy ledger (ground truth)")
            for r in ledger:
                tally = ", ".join(f"{k} {v}" for k, v in r["tally"].items())
                med = ("Δ median %+.4f" % r["median"]
                       if r["median"] is not None else "no measured Δ")
                parts.append(f"- `{r['id']}` — {r['n_nodes']} node(s) "
                             f"({', '.join(str(n) for n in r['nodes'])}) · "
                             f"{med} · {tally} — {r['definition']}")

        if delta_nodes:
            parts.append("## New/changed evidence since your last update "
                         "(full records)")
            for n in delta_nodes[:self.max_delta_records]:
                text = records[n].get("text") or records[n].get("body") or ""
                parts.append(f"### node {n}\n"
                             + truncate_middle(text, self.record_char_cap))
            if len(delta_nodes) > self.max_delta_records:
                parts.append(f"(+{len(delta_nodes) - self.max_delta_records} "
                             "more changed node(s) — see the ledger)")

        pred_lines = []
        for j in joins:
            who = f"belief:{j['belief_id']}" if j["belief_id"] else \
                "a proposal that named no belief"
            if j.get("measured_delta") is not None:
                pred_lines.append(
                    f"- {who} predicted {j['expected_direction'] or '?'} for "
                    f"the edit at node {j['node']}; measured Δ"
                    f"{j['measured_delta']:+.4f} over {j['n_shared']} shared — "
                    "judge whether this is a real miss or hit vs noise, and "
                    "revise or defend the belief.")
            else:
                pred_lines.append(
                    f"- {who} predicted {j['expected_direction'] or '?'} for "
                    f"the edit at node {j['node']}; not yet measured.")
        if pred_lines:
            parts.append("## Predictions made by past edit proposals\n"
                         + "\n".join(pred_lines))

        if directives:
            parts.append("## Representation directives from your last "
                         "reflection pass (apply where they serve content)\n"
                         + "\n".join(f"- {d}" for d in directives))
        return "\n\n".join(parts)

    def _maybe_reflect(self, state: dict[str, Any], doc: str,
                       n_updates: int) -> list[str]:
        """Run the reflection call when due; its directives feed THIS update.
        Pending directives persist in the sidecar until consumed."""
        pending = list(state.pop("pending_directives", []) or [])
        if (self.reflect_every <= 0 or n_updates == 0
                or n_updates % self.reflect_every != 0
                or state.get("reflected_at") == n_updates or not doc):
            return pending
        stats = json.dumps({
            "prediction_joins": state.get("prediction_joins", []),
            "citation_outcomes": state.get("citation_outcomes", {}),
            "verification": state.get("verification", {}),
        }, indent=1, default=str)[:8000]
        user = (f"## Belief document (update {n_updates})\n{doc}\n\n"
                f"## Machine statistics\n{stats}")
        got = self._call(REFLECT_SYSTEM, user, REFLECT_TOOL, "belief reflection")
        directives = [str(d)[:300] for d in ((got or {}).get("directives")
                                             or [])][:8]
        state["reflected_at"] = n_updates
        if directives:
            print(f"[edit_beliefs] reflection at update {n_updates}: "
                  f"{len(directives)} directive(s)", flush=True)
        return pending + directives

    # ------------------------------------------------------------------ #
    # Files
    # ------------------------------------------------------------------ #
    def _load_doc(self, experiment_dir: Path, *,
                  with_appendix: bool = False) -> str:
        path = experiment_dir / BELIEFS_NAME
        try:
            text = path.read_text(encoding="utf-8") if path.exists() else ""
        except (OSError, UnicodeDecodeError):
            return ""
        return text.rstrip("\n") if with_appendix else strip_machine_section(text)

    def _archive(self, experiment_dir: Path, state: dict[str, Any],
                 doc_body: str, n_updates: int) -> None:
        try:
            dest_dir = experiment_dir / BELIEFS_ARCHIVE_DIR
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / f"beliefs_{n_updates:04d}.md"
            src = experiment_dir / BELIEFS_NAME
            if src.exists():
                shutil.copyfile(src, dest)  # archive WITH its appendix
            else:
                _atomic_write(dest, doc_body + "\n")
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
    def _call(self, system: str, user: str, tool: dict,
              tag: str) -> Optional[dict]:
        kwargs: dict[str, Any] = {
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "tools": [tool],
        }
        if self.model:
            kwargs["model"] = self.model
        if self.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort
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
