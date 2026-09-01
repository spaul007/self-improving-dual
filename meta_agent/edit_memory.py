"""Per-node edit memory — a tree-global record of what edits were attempted
and what they did to the score.

Complements (and is independent of) the behavior summarizer: that describes how
an agent *behaved at runtime*; this describes *what was changed and whether it
paid off*, across every branch rather than one lineage.

Design points that are load-bearing rather than incidental:

* **One LLM call per node**, plus one per run for setup. Outcomes are refreshed
  continuously but always deterministically — an LLM in that path would cost
  ~8x the per-node calls to do arithmetic, and would end reproducibility.
* **Two category stores.** ``edit_memory_candidates.json`` holds the setup
  pass's proxy categories and is *tagger-only*; ``edit_memory_registry.json``
  holds only categories some edit actually used, and is the one safe to show
  the agent editor. A candidate is promoted at first use, so the registry never
  contains an unused entry — otherwise the editor would read hypothesised moves
  as though they were tried history and bias the search toward them.
* **Presence-based idempotency.** A node's diff is immutable once written, so
  "a record already exists" is a sufficient key; no content hash is needed.
* Every entry point is best-effort: a failure prints and returns, never
  propagating into the manager's round.
"""
from __future__ import annotations

import json
import os
import re
import tempfile
import traceback
from hashlib import blake2b
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional

from . import edit_usage, verbose_log
from .edit_diff import changed_mutable_files, diff_mutable_files
from .edit_outcome import (
    MIN_SHARED_FOR_VERDICT,
    NEUTRAL_BAND,
    TOP_K_CHECKS,
    compute_outcome,
    seen_split,
    validate_recipe,
)
from .registry import register

RECORD_NAME = "edit_memory.md"
PROMPT_NAME = "edit_memory_prompt.txt"
REGISTRY_NAME = "edit_memory_registry.json"
CANDIDATES_NAME = "edit_memory_candidates.json"

# Machine bookkeeping for the refresh skip-guard (case sigs, threshold,
# min_shared, fmt, analysis_sig). Lives in a sidecar, NOT in the record's
# frontmatter — the record is for reading; only node/parent/depth/lineage
# belong there.
STATE_NAME = "edit_memory_state.json"

# Record layout version. Bumping it makes the refresh skip-guard rewrite every
# existing record exactly once (the sig guard then resumes its job).
# v2: performance-first Outcome line. v3: machine keys moved out of the
# frontmatter into the state sidecar. v4: cumulative (all-evaluated) score
# leads the performance line; the shared-set comparison moves to the
# parenthetical. v5: generalization (seen-vs-unseen) line + scorer cross-tab
# on usage lines.
RECORD_FORMAT = 5

# Legacy machine keys: pre-v3 records carried these in frontmatter. Read once
# as a fallback when no sidecar exists, stripped from the frontmatter on the
# migration rewrite.
_STATE_KEYS = ("child_case_sig", "parent_case_sig", "threshold", "min_shared",
               "fmt", "analysis_sig")

# How similar a model-invented category id must be to an existing one before an
# undefined id is folded into it (Jaccard over hyphen-separated tokens). 0.5
# keeps genuine variants together — "add-verifier-retry" -> "add-verifier" is
# 0.67 — while refusing a match that rests on one generic shared word. Ids that
# clear no target are kept and defined from the edit instead.
MIN_FIT_SIMILARITY = 0.5

_SLUG_RE = re.compile(r"[^a-z0-9]+")
# Legacy (pre-fmt) Outcome tally, preserved verbatim across the fmt-2 rewrite.
_CHECKS_MOVED_RE = re.compile(r"^- \*\*checks moved\*\*.*$", re.M)
# Strips file paths / extensions from model prose. Deliberately keyed on
# extension-bearing tokens only: an earlier version also stripped bare "/" and
# mangled word lists like "transport/hotel/restaurant".
_PATH_RE = re.compile(r"\b[\w.-]*[\w-]+\.(?:py|json|ya?ml|md|txt)\b")


def slug(x: Any) -> str:
    return _SLUG_RE.sub("-", str(x).lower()).strip("-")


def _clean(t: Optional[str], cap: int) -> str:
    return re.sub(r" {2,}", " ", _PATH_RE.sub("", t or "")).strip()[:cap]


def _atomic_write(path: Path, text: str) -> None:
    """tmp-in-same-dir -> fsync -> replace, so a reader never sees a torn file."""
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


def case_sig(cases: Iterable[Any]) -> str:
    """Content hash of exactly the inputs to ``compute_outcome`` — lets a
    refresh skip untouched records without opening them for writing."""
    rows = []
    for c in cases or ():
        cid = c["case_id"] if isinstance(c, dict) else getattr(c, "case_id", None)
        sc = c["score"] if isinstance(c, dict) else getattr(c, "score", 0.0)
        rows.append(f"{cid}:{float(sc or 0.0):.6f}")
    return blake2b("\n".join(sorted(rows)).encode("utf-8"), digest_size=8).hexdigest()


# --------------------------------------------------------------------------- #
# Markdown record: fixed front matter + fixed sections, so a refresh can
# replace the Outcome section without ever re-parsing model prose.
# --------------------------------------------------------------------------- #
def split_record(text: str) -> tuple[dict[str, str], str]:
    """-> (front matter, body-above-Outcome). Never raises."""
    fm: dict[str, str] = {}
    body = text
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            for line in text[3:end].strip().splitlines():
                if ":" in line:
                    k, _, v = line.partition(":")
                    fm[k.strip()] = v.strip()
            body = text[end + 4:]
    idx = body.find("\n## Outcome")
    if idx != -1:
        body = body[:idx]
    return fm, body.strip("\n")


def _load_state(round_dir: Path) -> dict[str, Any]:
    """The refresh sidecar; ``{}`` when absent or unreadable. Never raises."""
    path = Path(round_dir) / STATE_NAME
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _save_state(round_dir: Path, state: Mapping[str, Any]) -> None:
    _atomic_write(Path(round_dir) / STATE_NAME,
                  json.dumps(dict(state), indent=1) + "\n")


def render_generalization(gs: Optional[Mapping[str, Any]]) -> str:
    """The Outcome section's seen-vs-unseen line, or ``""``. Improvement-
    positive convention; a leg's Δ shows "unmeasured" when the parent lacks
    coverage on that subset."""
    if not gs:
        return ""
    parts = []
    for leg in ("seen", "unseen"):
        v = gs.get(leg)
        if not v:
            continue
        d = (f"Δ {v['delta']:+.4f}" if v.get("delta") is not None
             else "Δ unmeasured")
        parts.append(f"{leg} {v['mean']:.4f}/{v['n']} ({d})")
    if not parts:
        return ""
    gap = ""
    if gs.get("seen") and gs.get("unseen"):
        gap = f" · gap {gs['unseen']['mean'] - gs['seen']['mean']:+.4f}"
    return "- **generalization**: " + " · ".join(parts) + gap


def extract_analysis(text: str) -> str:
    """The ``## Analysis`` section of a record (it sits after ``## Outcome``),
    or ``""``. Kept verbatim across refreshes that do not re-run the LLM."""
    idx = text.find("\n## Analysis")
    return text[idx + 1:].rstrip("\n") + "\n" if idx != -1 else ""


def render_record(fm: Mapping[str, Any], body: str, outcome: Any,
                  usage: Optional[list[str]] = None,
                  analysis_md: str = "",
                  generalization: str = "") -> str:
    # Per-check tallies (absolute and delta) are computed but deliberately NOT
    # rendered: they reach the analysis LLM via its prompt, and the record
    # carries only the digested target/collateral findings. The Outcome line
    # is performance-first and node-local — no run-best/seed here, those go
    # stale under the radius-1 refresh and are injected at render time.
    # Frontmatter is human keys ONLY; machine state lives in STATE_NAME.
    order = ["node", "parent", "depth", "lineage"]
    lines = ["---"]
    for k in order:
        if fm.get(k) not in (None, ""):
            lines.append(f"{k}: {fm[k]}")
    lines += ["---", "", body, "", "## Outcome"]
    if outcome.n_shared:
        lines.append(
            "- **performance**: child %.4f over %d evaluated cases "
            "(vs parent on %d shared: child %.4f, parent %.4f, Δ %+.4f)"
            % (outcome.child_mean_all, outcome.child_n_all, outcome.n_shared,
               outcome.child_mean_shared, outcome.parent_mean_shared,
               outcome.delta_shared))
    elif outcome.child_n_all:
        lines.append(
            "- **performance**: child %.4f over %d evaluated cases "
            "(no cases shared with parent yet)"
            % (outcome.child_mean_all, outcome.child_n_all))
    else:
        lines.append("- **performance**: not yet measured")
    if generalization:
        lines.append(generalization)
    lines.extend(usage or [])
    if analysis_md:
        lines += ["", analysis_md.rstrip("\n")]
    return "\n".join(lines) + "\n"


def render_edits(sub: list[dict[str, str]]) -> str:
    out: list[str] = []
    for i, e in enumerate(sub, 1):
        out += [f"## Edit {i}",
                f"- **name**: `{e['name']}`",
                f"- **category level 1 (strategy)**: `{e['strategy']}`",
                f"- **category level 2 (area)**: `{e['area']}`",
                f"- **what**: {e['what']}",
                f"- **why**: {e['why']}", ""]
    return "\n".join(out).rstrip("\n")


# --------------------------------------------------------------------------- #
SETUP_SYSTEM = """You are preparing to catalogue the edits that a self-improving agent will
make to itself over many rounds. You see the agent's STARTING code and its FIRST evaluation
results. No edits exist yet.

Produce three things.

1. strategies: 6-12 candidate ids for the KINDS OF MOVE a future edit could plausibly make
   to THIS agent, inferred from what its architecture affords. kebab-case, verb-first.
   GRANULARITY TEST - each id must plausibly describe MANY future edits in DIFFERENT problem
   areas. An id that could only ever match one specific edit is too specific.

2. areas: 5-10 candidate ids for the PROBLEM AREAS edits will target, inferred from the
   failures present in the evaluation results. kebab-case.
   GRANULARITY TEST - an area matching nearly every failure is too broad; one matching a
   single case is too narrow.

3. per_check_recipe: where a per-case result records WHICH individual criteria failed.
   mode is one of: list | dict_keys | dict_flags | none. Give the dotted path from the
   per-case record root.

Every id needs a one-line definition."""

SETUP_TOOL = {"type": "function", "function": {"name": "submit_setup", "parameters": {
    "type": "object", "properties": {
        "strategies": {"type": "array", "items": {"type": "object", "properties": {
            "id": {"type": "string"}, "definition": {"type": "string"}},
            "required": ["id", "definition"]}},
        "areas": {"type": "array", "items": {"type": "object", "properties": {
            "id": {"type": "string"}, "definition": {"type": "string"}},
            "required": ["id", "definition"]}},
        "per_check_recipe": {"type": "object", "properties": {
            "mode": {"type": "string"}, "path": {"type": "string"}},
            "required": ["mode", "path"]}},
    "required": ["strategies", "areas", "per_check_recipe"]}}}

NODE_SYSTEM_TMPL = """You catalogue edits made to a self-improving agent. One NODE per call.

A node is one submission by the agent editor and may bundle more than one distinct change.
Split it into 1-{max_sub} sub-edits.

SPLITTING RULES
- Split by WHAT WAS DONE, never by which area it touches. Several components doing the same
  kind of job are ONE sub-edit.
- Two sub-edits MUST NOT share a strategy. Repeating a strategy means you split by area -
  merge them instead.
- Most nodes do ONE thing. Returning the maximum is almost always wrong.

PER SUB-EDIT
- name: a 2-4 word kebab-case label naming THIS specific change. It identifies the edit;
  it is not a category.
- what: ONE sentence, the action taken.
- why: ONE sentence, the specific failure THIS sub-edit targets. Must differ between
  sub-edits; never restate the node's overall rationale.
- strategy: the reusable KIND OF MOVE. kebab-case, verb-first.
- area: the problem area addressed. kebab-case.

CATEGORY RULES
- Two lists are given. ESTABLISHED lists categories already used by earlier edits in this
  run, with definitions and usage counts. SUGGESTED lists categories not yet used by any
  edit - hints only, with no evidence behind them.
- Reuse an id when its definition genuinely describes this change. A poorly-fitting reuse
  is WORSE than a new id; never force-fit.
- GRANULARITY TEST for a new strategy: would it plausibly describe a future edit in a
  DIFFERENT problem area? If not, it is too specific - generalise it.
- Supply a one-line definition for EVERY id you introduce that appears in neither list.

STYLE
- You MAY name tools, functions and identifiers. Do NOT include file paths, file extensions
  or code fragments.
- You are NOT told the score. Never guess or mention it."""

NODE_TOOL = {"type": "function", "function": {
    "name": "submit_node_edits",
    "description": "Catalogue the distinct changes made by one node.",
    "parameters": {"type": "object", "properties": {
        "edits": {"type": "array", "minItems": 1, "items": {
            "type": "object", "properties": {
                "name": {"type": "string"}, "what": {"type": "string"},
                "why": {"type": "string"}, "strategy": {"type": "string"},
                "area": {"type": "string"}},
            "required": ["name", "what", "why", "strategy", "area"]}},
        "new_category_defs": {"type": "object"}},
        "required": ["edits"]}}}


@register("edit_memory", "default")
class EditMemory:
    """Writes one ``edit_memory.md`` per node and maintains the run-global
    category registry.

    Args:
        llm_caller: the project's ``call_llm``; injected by ``build_components``.
        model / reasoning_effort / base_url: LLM routing, as for the summarizer.
        steering: whether the manager should inject the block into the editor.
        steering_token_budget: cap on that block (~4 chars/token).
        verdict_threshold / min_shared: outcome classification, see edit_outcome.
        max_strategies: ceiling on the level-1 vocabulary.
        max_subedits: ceiling on the per-node split.
        diff_char_cap: cap on the diff shown to the tagger.
        setup_pass: run the one-per-run candidate/recipe call.
        per_check_recipe: pin the recipe instead of letting setup choose it.
        usage_tracking: capture runtime usage (tool calls + mutable_log
            events) into a per-round ``edit_usage.json`` at every refresh.
            trace.jsonl is truncated per eval batch, so refresh time is the
            only moment the data still exists.
        usage_max_events: cap on retained raw mutable_log events per node.
        analysis_mode: "refresh" re-runs the per-node analysis LLM call
            whenever the node's own evidence changed (~one call per eval
            batch per node); "final" only during finalize; "off" disables.
        analysis_max_cases / analysis_max_event_lines: evidence caps for
            that call's prompt.
    """

    def __init__(
        self,
        llm_caller: Callable[..., object],
        *,
        model: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        base_url: Optional[str] = None,
        steering: bool = True,
        steering_token_budget: int = 48000,
        verdict_threshold: float = NEUTRAL_BAND,
        min_shared: int = MIN_SHARED_FOR_VERDICT,
        top_k_checks: int = TOP_K_CHECKS,
        max_strategies: int = 18,
        max_subedits: int = 3,
        diff_char_cap: int = 6000,
        setup_pass: bool = True,
        per_check_recipe: Optional[dict] = None,
        usage_tracking: bool = True,
        usage_max_events: int = edit_usage.MAX_EVENTS,
        analysis_mode: str = "refresh",
        analysis_max_cases: int = edit_usage.MAX_ANALYSIS_CASES,
        analysis_max_event_lines: int = edit_usage.MAX_ANALYSIS_EVENT_LINES,
    ) -> None:
        self.llm = llm_caller
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.base_url = base_url
        self.steering = steering
        self.steering_token_budget = max(0, int(steering_token_budget))
        self.verdict_threshold = float(verdict_threshold)
        self.min_shared = int(min_shared)
        self.top_k_checks = int(top_k_checks)
        self.max_strategies = int(max_strategies)
        self.max_subedits = max(1, int(max_subedits))
        self.diff_char_cap = int(diff_char_cap)
        self.setup_pass = bool(setup_pass)
        self._pinned_recipe = per_check_recipe
        self.usage_tracking = bool(usage_tracking)
        self.usage_max_events = int(usage_max_events)
        self.analysis_mode = (analysis_mode if analysis_mode
                              in ("refresh", "final", "off") else "off")
        self.analysis_max_cases = int(analysis_max_cases)
        self.analysis_max_event_lines = int(analysis_max_event_lines)

        self._dir: Optional[Path] = None
        self._reg: dict[str, Any] = {"strategies": {}, "areas": {}}
        self._cand: dict[str, dict[str, str]] = {"strategies": {}, "areas": {}}
        self._recipe: dict[str, str] = {"mode": "none", "path": ""}
        self._ready = False

    # ------------------------------------------------------------------ #
    # Setup — once per run
    # ------------------------------------------------------------------ #
    def setup(self, experiment_dir: Path, seed_round_dir: Path,
              seed_cases: Iterable[Any]) -> None:
        self._dir = Path(experiment_dir)
        cases = list(seed_cases or ())
        try:
            self._load_stores()
            if self._pinned_recipe:
                self._recipe = validate_recipe(self._pinned_recipe, cases[:20])
            if self._ready:
                return
            if self.setup_pass:
                self._run_setup_call(seed_round_dir, cases)
            elif not self._pinned_recipe:
                self._recipe = validate_recipe(None, cases[:20])
            self._ready = True
            self._save_stores()
            print(f"[edit_memory] setup: {len(self._cand['strategies'])} candidate "
                  f"strategies / {len(self._cand['areas'])} areas (tagger-only); "
                  f"per-check recipe={self._recipe}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[edit_memory] setup failed: {exc!r}", flush=True)
            self._ready = True  # degrade to an empty vocabulary, never block the run

    def _run_setup_call(self, seed_round_dir: Path, cases: list[Any]) -> None:
        src = []
        agent = Path(seed_round_dir) / "task_agent"
        for rel in sorted(p.name for p in agent.glob("*") if p.is_file()):
            try:
                src.append(f"--- {rel} ---\n{(agent / rel).read_text(encoding='utf-8')[:6000]}")
            except (OSError, UnicodeDecodeError):
                continue
        sample = [c if isinstance(c, dict) else getattr(c, "model_dump", lambda: {})()
                  for c in cases[:3]]
        user = "\n".join([
            "## The agent as it starts", "\n".join(src), "",
            "## First evaluation - sample per-case records",
            json.dumps(sample, indent=1, default=str)[:9000]])
        got = self._call(SETUP_SYSTEM, user, SETUP_TOOL, "setup")
        if not got:
            self._recipe = validate_recipe(None, cases[:20])
            return
        for s in (got.get("strategies") or [])[:12]:
            if s.get("id"):
                self._cand["strategies"][slug(s["id"])] = s.get("definition", "")
        for s in (got.get("areas") or [])[:10]:
            if s.get("id"):
                self._cand["areas"][slug(s["id"])] = s.get("definition", "")
        if not self._pinned_recipe:
            self._recipe = validate_recipe(got.get("per_check_recipe"), cases[:20])

    # ------------------------------------------------------------------ #
    # Per node — the single LLM call
    # ------------------------------------------------------------------ #
    def record_node(self, *, round_dir: Path, parent_round_dir: Path,
                    node_id: int, parent_id: int, ancestors: list[int],
                    goal: str = "", proposed: str = "") -> Optional[Path]:
        try:
            dest = Path(round_dir) / RECORD_NAME
            if dest.exists() and "## Edit 1" in dest.read_text(encoding="utf-8"):
                return dest  # presence-based freeze: the diff cannot have changed

            files = changed_mutable_files(parent_round_dir, round_dir)
            diff = diff_mutable_files(parent_round_dir, round_dir,
                                      char_cap=self.diff_char_cap)
            if not files:
                return None

            if self.usage_tracking:
                try:
                    edit_usage.ensure_store(round_dir, parent_round_dir,
                                            node_id, parent_id)
                except Exception as exc:  # noqa: BLE001
                    print(f"[edit_memory] node {node_id}: usage store init "
                          f"failed: {exc!r}", flush=True)

            system = NODE_SYSTEM_TMPL.format(max_sub=self.max_subedits)
            user = "\n".join([
                f"## Node {node_id} (edited from node {parent_id})", "",
                "## Editor intent", f"goal: {goal}",
                f"proposed_changes: {(proposed or '')[:1500]}", "",
                "## Changed files",
                ", ".join(sorted(f.split("/")[-1] for f in files)), "",
                "## Categories", self._render_categories(), "",
                "## Diff vs parent", diff])
            try:
                _atomic_write(Path(round_dir) / PROMPT_NAME,
                              f"### SYSTEM\n{system}\n\n### USER\n{user}")
            except Exception:  # noqa: BLE001
                pass

            got = self._call(system, user, NODE_TOOL, f"node {node_id}")
            if not got:
                return None
            sub = self._absorb(got, node_id)
            if not sub:
                return None

            fm = {"node": node_id, "parent": parent_id, "depth": len(ancestors),
                  "lineage": " > ".join(str(a) for a in list(ancestors) + [node_id])}
            from .edit_outcome import EditOutcome
            _atomic_write(dest, render_record(fm, render_edits(sub), EditOutcome()))
            self._save_stores()  # record first, registry second
            if verbose_log.is_enabled():
                verbose_log.write_json(Path(round_dir), "edit_memory_response.json", got)
            print(f"[edit_memory] node {node_id}: "
                  f"{', '.join(e['strategy'] for e in sub)}", flush=True)
            return dest
        except Exception as exc:  # noqa: BLE001
            print(f"[edit_memory] node {node_id} failed: {exc!r}", flush=True)
            if verbose_log.is_enabled():
                print(traceback.format_exc(limit=3), flush=True)
            return None

    def _absorb(self, got: Mapping[str, Any], node_id: int) -> list[dict[str, str]]:
        defs = {slug(k): v for k, v in (got.get("new_category_defs") or {}).items()}
        sub: list[dict[str, str]] = []
        seen: set[str] = set()
        for e in (got.get("edits") or [])[:self.max_subedits]:
            s, area = slug(e.get("strategy")), slug(e.get("area")) or "general"
            if not s or s in seen:
                continue
            idx = len(sub) + 1
            name = slug(e.get("name")) or f"edit-{idx}"
            # Last-resort definition for a genuinely novel id: the edit's own
            # description. Thin, but it still tells the next tagger and the
            # editor what the category covers — a bare id tells them nothing.
            derived = _clean(e.get("what"), 160) or _clean(e.get("why"), 160)
            if not derived:
                # Neither what nor why: nothing to record and nothing to define
                # a novel category with. Skip before anything is admitted, so
                # no registry row is left pointing at a sub-edit that isn't in
                # the record.
                continue
            fitted = self._fit("strategies", s, defs, node_id, idx, name, seen,
                               derived)
            if fitted is None:
                continue
            s = fitted
            fitted_area = self._fit("areas", area, defs, node_id, idx, name,
                                    set(), derived)
            if fitted_area is None:
                continue     # never tag a record with an unregistered category
            area = fitted_area
            seen.add(s)
            sub.append({"name": name, "strategy": s, "area": area,
                        "what": _clean(e.get("what"), 240),
                        "why": _clean(e.get("why"), 200)})
        return sub

    def _fit(self, axis: str, key: str, defs: Mapping[str, str], node_id: int,
             edit_index: int, name: str, exclude: set[str],
             derived_def: str = "") -> Optional[str]:
        """Record the sub-edit under ``key``, or under the nearest related id
        when ``key`` is refused. Returns the id actually used, or ``None``.

        Force-fitting rather than dropping matters: a dropped sub-edit can take
        a whole node's categorisation with it. But an *unrelated* target is
        worse than no target — it corrupts the ledger — so how close a match
        has to be depends on why ``key`` was refused:

        * **at the cap** there is no room for anything new, so the choice is a
          loose fit or losing the sub-edit; any shared token wins.
        * **undefined** ids can instead be kept and defined from the edit
          itself, so the bar is similarity, not mere overlap. One token in
          common is not enough: ``pre-plan-validation``,
          ``validation-observability`` and ``validation-infrastructure`` all
          share exactly ``validation`` with ``transfer-time-validation`` and
          would otherwise collapse into it, filing three unrelated edits under
          commute-time buffers.
        """
        ok, why = self._admit(axis, key, defs, node_id, edit_index, name)
        if ok:
            return key
        # Candidates are eligible targets as well as already-registered ids:
        # they carry definitions, and promoting one on first real use is
        # exactly what the registry is for. Sorted, so ties break the same way
        # on every run.
        pool = sorted(set(self._reg[axis]) | set(self._cand[axis]))
        toks = set(key.split("-"))
        scored = []
        for c in pool:
            if c in exclude:
                continue
            other = set(c.split("-"))
            shared = len(toks & other)
            # Jaccard, not raw overlap: it discounts a token the two ids merely
            # both happen to contain, which is what a generic word like
            # "validation" or "add" always is.
            scored.append((shared / len(toks | other) if shared else 0.0,
                           shared, c))
        best = max(scored, default=(0.0, 0, None))
        floor = 0.0 if why == "cap" else MIN_FIT_SIMILARITY
        if best[1] > 0 and best[0] >= floor:
            print(f"[edit_memory] node {node_id}: {axis} {why}, "
                  f"'{key}' -> '{best[2]}' (similarity {best[0]:.2f})", flush=True)
            self._admit(axis, best[2], defs, node_id, edit_index, name)
            return best[2]
        if why == "cap":
            return None          # no room and nothing related: drop the sub-edit
        # Novel and unrelated to anything known: keep it, but defined. With no
        # description to derive one from, the sub-edit carries no information
        # worth registering — drop it rather than register a bare id.
        if not derived_def:
            print(f"[edit_memory] node {node_id}: {axis} '{key}' undefined and "
                  f"the edit has no description; dropped", flush=True)
            return None
        print(f"[edit_memory] node {node_id}: {axis} '{key}' novel and "
              f"undefined; defining it from the edit description", flush=True)
        self._admit(axis, key, defs, node_id, edit_index, name,
                    derived_def=derived_def)
        return key

    def _admit(self, axis: str, key: str, defs: Mapping[str, str],
               node_id: int, edit_index: int, name: str,
               derived_def: str = "") -> tuple[bool, str]:
        """Promote a category into the registry **at first use**. A definition
        may come from the model or from the candidate pool; either way the
        registry never holds an entry no edit used.

        Returns ``(admitted, reason_if_refused)``.
        """
        bucket = self._reg[axis]
        if key not in bucket:
            if axis == "strategies" and len(bucket) >= self.max_strategies:
                return False, "cap"
            definition = defs.get(key) or self._cand[axis].get(key) or ""
            if not definition:
                # A bare id teaches neither the tagger nor the editor anything —
                # rendering the registry *with* definitions is what keeps
                # categorisation stable across nodes. Refuse it so the caller
                # can map the edit onto an id that does carry a definition, or
                # come back with one derived from the edit itself.
                if not derived_def:
                    return False, "no definition"
                definition = derived_def
            bucket[key] = {"definition": definition,
                           "first_node": node_id, "edits": []}
        entry = bucket[key]
        # Idempotent: re-processing a node replaces its rows rather than adding.
        entry["edits"] = [r for r in entry["edits"] if r["node"] != node_id
                          or r["edit_index"] != edit_index]
        entry["edits"].append({"node": node_id, "edit_index": edit_index, "name": name})
        entry["edits"].sort(key=lambda r: (r["node"], r["edit_index"]))
        return True, ""

    # ------------------------------------------------------------------ #
    # Outcome refresh — deterministic, no LLM
    # ------------------------------------------------------------------ #
    def refresh_outcomes(self, tree: Any, node_id: int) -> int:
        """Recompute the outcome of ``node_id`` and of **its children**.

        Delta is measured over cases the parent and child both ran, so a node's
        own batch moves its children's numbers too. The invalidation radius is
        exactly 1 downward: grandchildren depend on their own parent's cases,
        which did not change.
        """
        written = 0
        try:
            x = tree[node_id]
            # Fold the just-evaluated node's trace into its usage sidecar NOW —
            # the evaluator truncates trace.jsonl at the start of the next
            # batch, so refresh time is the only moment this data still exists.
            self._consume_trace(x, tree)
            dirty = ([x] if x.parent_id is not None else [])
            dirty += [tree[c] for c in getattr(x, "children", []) if c in tree.nodes]
            for child in dirty:
                if getattr(child, "edit_failed", False) or child.parent_id is None:
                    continue
                written += int(self._refresh_one(tree[child.parent_id], child))
        except Exception as exc:  # noqa: BLE001
            print(f"[edit_memory] refresh around node {node_id} failed: {exc!r}",
                  flush=True)
        return written

    def _consume_trace(self, node: Any, tree: Any) -> None:
        """Best-effort, idempotent: batches are keyed by content hash."""
        if not self.usage_tracking or getattr(node, "edit_failed", False):
            return
        try:
            store = edit_usage.load_store(node.round_dir)
            if store is None:
                # No sidecar yet: the seed node (never tracked), or a record
                # written before usage tracking existed (a resumed run). The
                # latter gets a store now and counts forward batches only —
                # the renderer distinguishes no-data from zero, so a late
                # store never fabricates a "0 calls".
                if (node.parent_id is None
                        or node.parent_id not in getattr(tree, "nodes", {})
                        or not (Path(node.round_dir) / RECORD_NAME).exists()):
                    return
                store = edit_usage.ensure_store(
                    node.round_dir, tree[node.parent_id].round_dir,
                    node.node_id, node.parent_id, capture_seen=False)
            if edit_usage.consume_trace(
                    store, Path(node.round_dir) / "logs" / "trace.jsonl",
                    max_events=self.usage_max_events):
                edit_usage.save_store(node.round_dir, store)
        except Exception as exc:  # noqa: BLE001
            print(f"[edit_memory] trace consume failed for node "
                  f"{getattr(node, 'node_id', '?')}: {exc!r}", flush=True)

    def _refresh_one(self, parent: Any, child: Any, *,
                     force_analysis: bool = False) -> bool:
        path = Path(child.round_dir) / RECORD_NAME
        if not path.exists():
            return False
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return False
        fm, body = split_record(text)
        state = _load_state(child.round_dir)
        if not state:
            # Pre-v3 record: the machine keys lived in the frontmatter. Read
            # them once as the fallback so the migration rewrite is exactly
            # one write, not one per refresh.
            state = {k: fm[k] for k in _STATE_KEYS if fm.get(k)}
        c_sig, p_sig = case_sig(child.case_results), case_sig(parent.case_results)
        if (not force_analysis
                and state.get("child_case_sig") == c_sig
                and state.get("parent_case_sig") == p_sig
                and str(state.get("threshold")) == f"{self.verdict_threshold}"
                and str(state.get("min_shared")) == f"{self.min_shared}"
                and str(state.get("fmt")) == str(RECORD_FORMAT)):
            return False  # nothing moved: do not open the file for writing
        oc = compute_outcome(list(parent.case_results), list(child.case_results),
                             threshold=self.verdict_threshold,
                             top_k_checks=self.top_k_checks,
                             min_shared=self.min_shared, recipe=self._recipe)

        store = edit_usage.load_store(child.round_dir) if self.usage_tracking else None
        usage = (edit_usage.usage_lines(store, child_cases=child.case_results)
                 if store else [])
        gs = (seen_split(parent.case_results, child.case_results,
                         store.get("seen_case_ids") or [])
              if store and store.get("seen_case_ids") else None)
        # A legacy record's "checks moved" tally lives in the old Outcome
        # section, which the rewrite regenerates — carry it forward verbatim.
        # It is the only per-check history for a record whose analysis cannot
        # be re-run (no usage sidecar), and it costs one line.
        legacy = _CHECKS_MOVED_RE.search(text)
        if legacy and legacy.group(0) not in usage:
            usage = list(usage) + [legacy.group(0)]
        analysis_md = extract_analysis(text)  # keep the last one by default
        a_sig = state.get("analysis_sig", "")
        if store and store.get("batches") and self.analysis_mode != "off" \
                and (self.analysis_mode == "refresh" or force_analysis) \
                and oc.n_shared >= self.min_shared:
            new_sig = edit_usage.analysis_sig(c_sig, store["batches"])
            if new_sig != a_sig:
                payload = self._analyze(parent, child, oc, store, body, usage)
                if payload:
                    analysis_md = edit_usage.render_analysis(payload)
                    a_sig = new_sig  # failure keeps the old sig -> retried next refresh

        # Strip legacy machine keys so a pre-v3 record's frontmatter is
        # cleaned by the migration rewrite; the state sidecar takes over.
        for k in _STATE_KEYS:
            fm.pop(k, None)
        _atomic_write(path, render_record(fm, body, oc, usage=usage,
                                          analysis_md=analysis_md,
                                          generalization=render_generalization(gs)))
        _save_state(child.round_dir, {
            "child_case_sig": c_sig, "parent_case_sig": p_sig,
            "threshold": self.verdict_threshold, "min_shared": self.min_shared,
            "fmt": RECORD_FORMAT, "analysis_sig": a_sig})
        return True

    def _analyze(self, parent: Any, child: Any, oc: Any, store: dict,
                 body: str, usage: list[str]) -> Optional[dict]:
        """One LLM call inferring verifier-vs-scorer agreement, targeted-check
        remaining failures, collateral, and per-component likely cause from
        the captured evidence. The parent->child code diff is included so the
        cause can point at the implementation, not just the behavior.
        No generalization evidence here: the seen-vs-unseen split's parent
        baseline keeps moving after this node's analysis_sig freezes, so it
        renders only in Outcome, from live data. Best-effort; None on any
        failure."""
        try:
            node_id = getattr(child, "node_id", store.get("node", "?"))
            try:
                diff = diff_mutable_files(parent.round_dir, child.round_dir,
                                          char_cap=self.diff_char_cap)
            except Exception:  # noqa: BLE001
                diff = ""
            prompt = edit_usage.build_analysis_prompt(
                node_id=node_id, record_body=body, outcome=oc, store=store,
                u_lines=usage,
                parent_cases=list(parent.case_results),
                child_cases=list(child.case_results),
                recipe=self._recipe,
                code_diff=diff,
                max_event_lines=self.analysis_max_event_lines,
                max_cases=self.analysis_max_cases)
            try:
                _atomic_write(Path(child.round_dir) / edit_usage.ANALYSIS_PROMPT_NAME,
                              f"### SYSTEM\n{edit_usage.ANALYSIS_SYSTEM}"
                              f"\n\n### USER\n{prompt}")
            except Exception:  # noqa: BLE001
                pass
            return self._call(edit_usage.ANALYSIS_SYSTEM, prompt,
                              edit_usage.ANALYSIS_TOOL, f"analysis node {node_id}")
        except Exception as exc:  # noqa: BLE001
            print(f"[edit_memory] analysis failed for node "
                  f"{getattr(child, 'node_id', '?')}: {exc!r}", flush=True)
            return None

    def finalize(self, tree: Any) -> None:
        """Full sweep — a no-op under the skip guard when nothing moved. In
        ``analysis_mode="final"`` this is also where every node's analysis
        call runs."""
        try:
            for nid in list(tree.nodes):
                self.refresh_outcomes(tree, nid)
            if self.analysis_mode == "final":
                for nid in list(tree.nodes):
                    child = tree[nid]
                    if (child.parent_id is None
                            or getattr(child, "edit_failed", False)):
                        continue
                    self._refresh_one(tree[child.parent_id], child,
                                      force_analysis=True)
            self._save_stores()
        except Exception as exc:  # noqa: BLE001
            print(f"[edit_memory] finalize failed: {exc!r}", flush=True)

    # ------------------------------------------------------------------ #
    def _render_categories(self) -> str:
        """ESTABLISHED (evidence-backed) then SUGGESTED (proxy, unused).

        Definitions are always shown: with bare names the model matches on the
        name alone and picks plausible-sounding wrong ids.
        """
        L = [f"ESTABLISHED strategies (used by earlier edits; "
             f"{len(self._reg['strategies'])} of max {self.max_strategies}):"]
        if not self._reg["strategies"]:
            L.append("  (none yet - this is the first node)")
        for k, v in sorted(self._reg["strategies"].items(),
                           key=lambda kv: -len({r["node"] for r in kv[1]["edits"]})):
            L.append("  - %-36s %2d nodes - %s"
                     % (k, len({r["node"] for r in v["edits"]}), v["definition"]))
        L.append("ESTABLISHED areas:")
        if not self._reg["areas"]:
            L.append("  (none yet)")
        for k, v in sorted(self._reg["areas"].items(),
                           key=lambda kv: -len({r["node"] for r in kv[1]["edits"]})):
            L.append("  - %-28s %2d nodes - %s"
                     % (k, len({r["node"] for r in v["edits"]}), v["definition"]))
        unused = [(k, d) for k, d in self._cand["strategies"].items()
                  if k not in self._reg["strategies"]]
        unused += [(k, d) for k, d in self._cand["areas"].items()
                   if k not in self._reg["areas"]]
        if unused:
            L.append("SUGGESTED (not yet used by any edit - hints only, no evidence):")
            for k, d in unused:
                L.append("  - %-36s - %s" % (k, d))
        return "\n".join(L)

    # ------------------------------------------------------------------ #
    def _store_paths(self) -> tuple[Path, Path]:
        base = self._dir or Path(".")
        return base / REGISTRY_NAME, base / CANDIDATES_NAME

    def _load_stores(self) -> None:
        rp, cp = self._store_paths()
        if rp.exists():
            try:
                old = json.loads(rp.read_text(encoding="utf-8"))
                for ax in ("strategies", "areas"):
                    self._reg[ax] = {
                        k: {"definition": v.get("definition", ""),
                            "first_node": v.get("first_node"),
                            "edits": v.get("edits", [])}
                        for k, v in (old.get(ax) or {}).items()}
                self._recipe = old.get("per_check_recipe") or self._recipe
                self._ready = True
            except (OSError, json.JSONDecodeError, TypeError) as exc:
                print(f"[edit_memory] registry unreadable ({exc!r}); starting fresh",
                      flush=True)
        if cp.exists():
            try:
                old = json.loads(cp.read_text(encoding="utf-8"))
                self._cand["strategies"] = old.get("strategies") or {}
                self._cand["areas"] = old.get("areas") or {}
            except (OSError, json.JSONDecodeError, TypeError):
                pass

    def _save_stores(self) -> None:
        if self._dir is None:
            return
        rp, cp = self._store_paths()

        def _axis(ax: str) -> dict:
            return {k: {"definition": v["definition"], "first_node": v["first_node"],
                        "n_nodes": len({r["node"] for r in v["edits"]}),
                        "edits": v["edits"]}
                    for k, v in sorted(self._reg[ax].items(),
                                       key=lambda kv: -len({r["node"] for r in kv[1]["edits"]}))}
        _atomic_write(rp, json.dumps({
            "note": "Every category here was used by at least one edit in this run. "
                    "Safe to show the agent editor.",
            "per_check_recipe": self._recipe,
            "strategies": _axis("strategies"), "areas": _axis("areas"),
        }, indent=2))
        _atomic_write(cp, json.dumps({
            "note": "Setup-pass proxy categories. TAGGER-ONLY - never shown to the "
                    "agent editor. Entries also in the registry were actually used.",
            "strategies": self._cand["strategies"], "areas": self._cand["areas"],
        }, indent=2))

    # ------------------------------------------------------------------ #
    def _call(self, system: str, user: str, tool: dict, tag: str) -> Optional[dict]:
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
            print(f"[edit_memory] {tag}: llm call failed: {exc!r}", flush=True)
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
        print(f"[edit_memory] {tag}: no structured output", flush=True)
        return None
