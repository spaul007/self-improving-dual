"""Runtime usage of an edit's components, and the LLM analysis built on it.

Two layers, split by determinism:

* **Capture (deterministic, no LLM).** What did this node's edit add — new
  mutable tools and new ``trace.log`` instrumentation points — and how often
  did each actually run? ``trace.jsonl`` is truncated at the start of every
  eval batch (``meta_agent/evaluator.py``), so the raw events are accumulated
  here, batch by batch, into a per-round sidecar (``edit_usage.json``) at
  refresh time. Consumption is idempotent: each batch is keyed by a content
  hash and never double-counted.
* **Analysis (LLM, prompt assembly only here).** The inference — "when this
  verifier said *pass*, did the agent actually produce a correct output for
  that component per the scorer, and did the edit help its targeted
  constraints?" — is delegated to one LLM call per node. This module builds
  the evidence prompt and renders the structured answer; the call itself is
  made by ``EditMemory`` (which owns LLM routing).

Semantic rule carried from ``edit_outcome.extract_checks``: **"no usage data"
and "0 calls" are opposite statements** and are never conflated. A store with
no consumed batches renders nothing; a consumed batch where a component never
ran renders an explicit "0 calls" / "never fired".
"""
from __future__ import annotations

import json
import os
import re
import tempfile
from collections import defaultdict
from hashlib import blake2b
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from .edit_diff import changed_mutable_files
from .edit_outcome import TOP_K_CHECKS, compact_check, extract_checks, se_of
from .perf_text import EFFECTS, EVIDENCE_LEVELS

USAGE_NAME = "edit_usage.json"
ANALYSIS_PROMPT_NAME = "edit_analysis_prompt.txt"

# Retained raw mutable_log events per sidecar. A cap, not a sample rate:
# events are trimmed round-robin per (label, name, verdict) group so verdict
# diversity survives even when one chatty check dominates.
MAX_EVENTS = 400
# Evidence caps for the analysis prompt.
MAX_ANALYSIS_EVENT_LINES = 120
MAX_ANALYSIS_CASES = 10
# One row per evaluated case (score, failed checks, which of this edit's
# components fired with what verdict) — the overlap-free evidence the judge
# reasons from. Two batches fit; older cases beyond the cap are counted.
MAX_ANALYSIS_CASE_ROWS = 32

LABEL_RE = re.compile(r"^(?:label\s*=\s*)?f?['\"]([^'\"]+)['\"]")
LABEL_KW_RE = re.compile(r"label\s*=\s*f?['\"]([^'\"]+)['\"]")
NAME_KW_RE = re.compile(r"(?<![\w_])name\s*=\s*f?['\"]([^'\"]+)['\"]")
ALIAS_IMPORT_RE = re.compile(
    r"from\s+platform_core\.trace\s+import\s+log(?:\s+as\s+(\w+))?")


def _atomic_write(path: Path, text: str) -> None:
    """Local copy of the tmp->fsync->replace dance (importing it from
    ``edit_memory`` would create an import cycle)."""
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


# --------------------------------------------------------------------------- #
# Surface — what did this edit add? Deterministic, computed once.
# --------------------------------------------------------------------------- #
def _tool_stems(root: Path) -> set[str]:
    d = Path(root) / "task_agent" / "mutable_tools"
    if not d.exists():
        return set()
    return {p.stem for p in d.glob("*.py") if p.name != "__init__.py"}


def _schema_names(root: Path) -> set[str]:
    p = Path(root) / "task_agent" / "tools_schema.json"
    try:
        js = json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return set()
    names: set[str] = set()
    for t in js if isinstance(js, list) else []:
        fn = t.get("function", t) if isinstance(t, dict) else {}
        n = fn.get("name")
        if n:
            names.add(n)
    return names


def _log_call_re(file_text: str) -> re.Pattern:
    """Regex matching this file's trace-log call sites, whatever the editor
    named the import — ``log``, ``trace.log``, or an alias like ``trace_log``
    (editors alias it routinely, so matching bare ``log(`` misses most sites)."""
    names = {"log", "trace.log"}
    for m in ALIAS_IMPORT_RE.finditer(file_text):
        names.add(m.group(1) or "log")
    alt = "|".join(re.escape(n) for n in sorted(names))
    # The 300-char window is a lookahead, not part of the match: a consumed
    # window swallowed any call site that followed within it, so a second
    # log point 2–3 lines after the first was never detected.
    return re.compile(r"(?<![\w.])(?:%s)\(\s*(?=(.{0,300}))" % alt, re.S)


def _log_pairs(text: str, alias_text: Optional[str] = None) -> set[tuple[str, Optional[str]]]:
    """(label, name) pairs at trace-log call sites in ``text``. The label may
    be positional or a ``label=`` kwarg; ``alias_text`` (default: ``text``) is
    where import aliases are looked up — pass the full file when scanning only
    its added lines, since the import statement may not be among them."""
    pairs: set[tuple[str, Optional[str]]] = set()
    for m in _log_call_re(alias_text or text).finditer(text):
        window = m.group(1)
        lm = LABEL_RE.match(window.strip()) or LABEL_KW_RE.search(window)
        if not lm:
            continue
        nm = NAME_KW_RE.search(window)
        pairs.add((lm.group(1), nm.group(1) if nm else None))
    return pairs


def added_surface(parent_round_dir: Path, round_dir: Path) -> dict:
    """New tools and new instrumentation points vs the parent snapshot.

    A ``(label, name)`` pair is attributed to this node only when the pair is
    absent from the parent's mutable files — a new ``name`` under a label the
    parent already used still counts as new. F-string placeholders in either
    part are kept verbatim and matched against runtime values by
    :func:`_dyn_eq`.
    """
    tools_added = sorted((_tool_stems(round_dir) - _tool_stems(parent_round_dir))
                         | (_schema_names(round_dir) - _schema_names(parent_round_dir)))
    tools_removed = sorted((_tool_stems(parent_round_dir) - _tool_stems(round_dir))
                           | (_schema_names(parent_round_dir) - _schema_names(round_dir)))
    parent_pairs: set[tuple[str, Optional[str]]] = set()
    added_pairs: set[tuple[str, Optional[str]]] = set()
    for rel in changed_mutable_files(parent_round_dir, round_dir):
        p_path = Path(parent_round_dir) / "task_agent" / rel
        c_path = Path(round_dir) / "task_agent" / rel
        p_text = (p_path.read_text(encoding="utf-8", errors="replace")
                  if p_path.exists() else "")
        c_text = (c_path.read_text(encoding="utf-8", errors="replace")
                  if c_path.exists() else "")
        p_lines = set(p_text.splitlines())
        added_text = "\n".join(l for l in c_text.splitlines() if l not in p_lines)
        parent_pairs |= _log_pairs(p_text)
        added_pairs |= _log_pairs(added_text, alias_text=c_text)
    # `name` is Optional: the same label logged with and without a name
    # would make the tuple sort compare str with None (node 6 of the
    # 2026-09-07 run lost its usage store — and every later verdict — that way).
    _key = lambda p: (p[0], p[1] or "")  # noqa: E731
    new_pairs = sorted((p for p in added_pairs if p not in parent_pairs), key=_key)
    inherited = sorted(added_pairs & parent_pairs, key=_key)
    return {
        "tools": tools_added,
        "removed_tools": tools_removed,
        "labels": [{"label": l, "name": n} for l, n in new_pairs],
        "inherited_labels": [{"label": l, "name": n} for l, n in inherited],
    }


# --------------------------------------------------------------------------- #
# Sidecar store — created at record time, fed at every refresh.
# --------------------------------------------------------------------------- #
def load_store(round_dir: Path) -> Optional[dict]:
    p = Path(round_dir) / USAGE_NAME
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def save_store(round_dir: Path, store: dict) -> None:
    _atomic_write(Path(round_dir) / USAGE_NAME,
                  json.dumps(store, indent=1, ensure_ascii=False, sort_keys=True))


def _seen_case_ids(parent_round_dir: Path) -> list[str]:
    """The parent's evaluated case ids at THIS moment — called at record time
    (i.e. child creation), so this is exactly the set the editor's feedback
    was computed from. Best-effort: ``[]`` when unreadable."""
    try:
        per_case = json.loads(
            (Path(parent_round_dir) / "eval_result.json").read_text(
                encoding="utf-8")).get("per_case") or []
        return sorted({str(c.get("case_id")) for c in per_case
                       if c.get("case_id") is not None})
    except Exception:  # noqa: BLE001
        return []


def ensure_store(round_dir: Path, parent_round_dir: Path,
                 node_id: int, parent_id: int, *,
                 capture_seen: bool = True) -> dict:
    """Load the sidecar, creating it (surface + empty counters) if absent.
    Safe on runs resumed mid-flight: a late-created store simply counts only
    forward batches, and the renderer distinguishes no-data from zero.

    ``seen_case_ids`` is captured only on CREATION: it must reflect the
    parent's evaluated set at edit time, so it is never refreshed later (a
    late-created store on a resumed run records the ids current at that
    moment — an upper bound on the true seen set, flagged by version)."""
    store = load_store(round_dir)
    if store is not None:
        return store
    store = {
        "version": 2, "node": node_id, "parent": parent_id,
        "surface": added_surface(parent_round_dir, round_dir),
        "seen_case_ids": (_seen_case_ids(parent_round_dir)
                          if capture_seen else []),
        "batches": [], "case_ids": [],
        "tools": {}, "events": [], "events_truncated": False,
    }
    save_store(round_dir, store)
    return store


def consume_trace(store: dict, trace_path: Path, *,
                  max_events: int = MAX_EVENTS) -> bool:
    """Fold one surviving ``trace.jsonl`` into the store. Idempotent: the
    batch is keyed by a content hash, so re-consuming the same bytes (a
    re-refresh, ``finalize``'s full sweep) is a no-op. A missing or empty
    trace (crashed batch) is skipped and never counted. Returns True when the
    store changed."""
    trace_path = Path(trace_path)
    if not trace_path.exists() or trace_path.stat().st_size == 0:
        return False
    raw = trace_path.read_bytes()
    sig = blake2b(raw, digest_size=8).hexdigest()
    if sig in store.get("batches", []):
        return False

    tools = store.setdefault("tools", {})
    events = store.setdefault("events", [])
    case_ids = set(store.get("case_ids", []))
    for line in raw.decode("utf-8", errors="replace").splitlines():
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        kind, payload = ev.get("kind"), ev.get("payload", {})
        if not isinstance(payload, dict):
            continue
        cid = payload.get("case_id")
        if cid is not None:
            case_ids.add(str(cid))
        if kind == "tool_call":
            name = payload.get("name")
            if name:
                entry = tools.setdefault(name, {"calls": 0, "case_ids": []})
                entry["calls"] += 1
                if cid is not None and str(cid) not in entry["case_ids"]:
                    entry["case_ids"].append(str(cid))
                    entry["case_ids"].sort()
        elif kind == "mutable_log":
            events.append(payload)

    if len(events) > max_events:
        # Trim round-robin per (label, name, verdict) so verdict diversity
        # survives a chatty check; flag it rather than trimming silently.
        groups: dict[tuple, list[dict]] = defaultdict(list)
        for e in events:
            groups[(e.get("label"), e.get("name"),
                    str(e.get("verdict", "-")))].append(e)
        kept: list[dict] = []
        depth = 0
        order = sorted(groups, key=str)
        while len(kept) < max_events and any(len(groups[g]) > depth for g in order):
            for g in order:
                if len(kept) >= max_events:
                    break
                if len(groups[g]) > depth:
                    kept.append(groups[g][depth])
            depth += 1
        store["events"] = kept
        store["events_truncated"] = True

    store.setdefault("batches", []).append(sig)
    store["case_ids"] = sorted(case_ids)
    return True


# --------------------------------------------------------------------------- #
# Usage lines — the deterministic output, appended to the Outcome section.
# --------------------------------------------------------------------------- #
def _dyn_eq(pattern: Optional[str], value: Any) -> bool:
    """Exact match, except f-string placeholders in extracted patterns
    (``tool_{tc.name}``) match any runtime substring."""
    if pattern is None:
        return True
    if "{" in pattern:
        parts = re.split(r"\{[^{}]*\}", pattern)
        rx = ".*".join(re.escape(p) for p in parts)
        return re.fullmatch(rx, str(value or "")) is not None
    return pattern == value


def match_events(events: Sequence[Mapping], label: str,
                 name: Optional[str]) -> list[dict]:
    return [dict(e) for e in events
            if _dyn_eq(label, e.get("label")) and _dyn_eq(name, e.get("name"))]


def agreement_counts(events: Sequence[Mapping],
                     child_cases: Sequence[Any]) -> dict:
    """Deterministic verifier-vs-scorer cross-tab: for the given events,
    per verdict -> (distinct cases, scorer passes, scorer fails), joined on
    ``case_id`` against the child's per-case results. Events whose case is
    unknown (not in ``child_cases``) are ignored. Pure; no LLM."""
    passed: dict[str, bool] = {}
    for c in child_cases or ():
        cid = c["case_id"] if isinstance(c, dict) else getattr(c, "case_id", None)
        ok = c["passed"] if isinstance(c, dict) else getattr(c, "passed", None)
        if cid is not None:
            passed[str(cid)] = bool(ok)
    by: dict[str, set] = defaultdict(set)
    for e in events or ():
        cid = e.get("case_id")
        if cid is not None and str(cid) in passed:
            by[str(e.get("verdict", "-"))].add(str(cid))
    return {v: (len(ids), sum(1 for i in ids if passed[i]),
                sum(1 for i in ids if not passed[i]))
            for v, ids in by.items()}


def usage_lines(store: Mapping[str, Any],
                child_cases: Optional[Sequence[Any]] = None) -> list[str]:
    """Render the store into Outcome-section bullet lines. Returns ``[]`` when
    no batch has been consumed yet — absence of data must render as nothing,
    never as "0 calls".

    With ``child_cases`` given, each fired log-point line gains the
    deterministic scorer cross-tab ("-> scorer on those cases: X pass /
    Y fail") from :func:`agreement_counts`, and a component whose ``pass``
    verdicts land mostly on scorer-failed cases is flagged
    ``· SUSPECT VERIFIER`` — the machine-computed candidate list for the
    steering block's REPAIR move."""
    batches = store.get("batches") or []
    if not batches:
        return []
    surface = store.get("surface") or {}
    tools = store.get("tools") or {}
    events = store.get("events") or []
    basis = f"{len(batches)} batches, {len(store.get('case_ids') or [])} cases"
    lines: list[str] = []

    tool_bits = []
    for t in surface.get("tools") or []:
        info = tools.get(t)
        if info:
            tool_bits.append(f"`{t}` {info['calls']} calls / "
                             f"{len(info['case_ids'])} cases")
        else:
            tool_bits.append(f"`{t}` **0 calls**")
    if tool_bits:
        lines.append(f"- **new tools** ({basis}): " + " · ".join(tool_bits))

    for lab in surface.get("labels") or []:
        evs = match_events(events, lab["label"], lab.get("name"))
        key = (f"{lab['label']}/{lab['name']}" if lab.get("name")
               else lab["label"])
        if not evs:
            lines.append(f"- **new log point `{key}`**: **never fired** ({basis})")
            continue
        verdicts: dict[str, int] = defaultdict(int)
        for e in evs:
            verdicts[str(e.get("verdict", "-"))] += 1
        vs = " / ".join(f"{k} {v}" for k, v in sorted(verdicts.items()))
        approx = "≥" if store.get("events_truncated") else ""
        line = (f"- **new log point `{key}`**: fired {approx}{len(evs)}x "
                f"({vs}) ({basis})")
        if child_cases:
            agree = agreement_counts(evs, child_cases)
            tot = sum(n for n, _, _ in agree.values())
            sp = sum(a for _, a, _ in agree.values())
            sf = sum(b for _, _, b in agree.values())
            if tot:
                line += f" -> scorer on those cases: {sp} pass / {sf} fail"
            pv = agree.get("pass")
            if pv and pv[2] > pv[1]:
                line += " · SUSPECT VERIFIER"
        lines.append(line)

    if not (surface.get("tools") or surface.get("labels")):
        lines.append("- **usage**: edit added no new tools or instrumented log "
                     f"points (prompt/logic-only change); {basis}")
    return lines


# --------------------------------------------------------------------------- #
# Analysis — prompt assembly and answer rendering. The call is EditMemory's.
# --------------------------------------------------------------------------- #
ANALYSIS_SYSTEM = """You analyze one edit made by a self-improving agent. You are the JUDGE: your verdicts
are the outcome labels the run learns from, so every verdict must rest on evidence you can
point at, never on the edit's description or on a noisy score difference.
You receive: the edit's description (one `## Edit N` block per sub-edit), its
implementation (added lines per definition vs the parent agent, full source of new
definitions), performance numbers WITH standard errors (each side's mean over its own
cases, the unpaired Δ, the paired Δ when enough cases are shared), a PAIRED per-check
failure table (shared cases only, when any) and an UNPAIRED one (the child's own cases vs
the parent's own cases), deterministic usage counts WITH scorer agreement ("-> scorer on
those cases: X pass / Y fail"), raw runtime verifier logs, one row per evaluated case
(score, failed checks, which of this edit's components fired on it with what verdict —
parent rows for the shared cases), and per-case ground-truth excerpts.

In the component / target / collateral findings report evidence, not judgment labels. For
EVERY finding, explain WHY — including why something HELPED (name the confirmed
mechanism), not only why it failed.

Evidence hierarchy (strongest first):
1. within-case: a component changed the output on a case and the corresponding check then
   passed — or a detector fired and the final output still failed (remediation broken);
2. component-level counts joined to the scorer over ≥ 8 cases (agreement lines, per-case
   rows: did the cases where it fired pass the checks it targets?);
3. per-check movement on the TARGETED checks — paired on shared cases when ≥ 8 are shared,
   else unpaired (different case samples: say so, and weigh it less);
4. the score Δ: ~16-case batches give a standard error near ±0.1, so a Δ smaller than 2×SE
   is NOISE and is evidence of nothing, in either direction;
5. the implementation alone — never enough for `improved`.

Component analysis rules:
- Classify each component's role: GATE (its pass releases the plan), DETECTOR (its
  fail flags a problem), or OTHER.
- GATES: when it passed, did the scorer accept the output? Use the agreement counts.
- DETECTORS: check BOTH directions — false alarms (fired on cases the scorer accepts)
  AND misses (cases failing the corresponding check where it never fired) — AND the
  remediation join: was the flagged problem actually FIXED in the final output
  (compare with the target lines)? "Detects correctly but final plans still fail
  (k->k) — remediation broken" is a required phrasing when that is what the numbers
  show; never call a detector "worked as intended" on detection alone.
- likely cause: 1-3 SHORT lines, each anchored to a count, case id, or diff line.
  Consider this checklist and use only what the evidence supports:
  not-implemented-as-claimed (intent vs diff) · checks a subset -> too lenient ·
  too strict -> retry exhaustion / no-plan / iteration-budget burn · name/key lookup
  misses or missing-data-treated-as-pass (silent skips) · dead code / never wired /
  verdict unused · detects-but-remediation-fails · regeneration side-effects
  (narrowed output -> collateral) · component crash/exception · draft-vs-final
  timing · prompt-instruction ignored vs followed · improvement-not-attributable-
  to-component · seen-case-specific tuning (case-specific constants/names in the
  diff).

Implementation verdict (required): decide whether this edit's IMPLEMENTATION is SOUND —
it does what its description claims, its components actually ran on the observed cases,
and their verdicts agree with the scorer — or UNSOUND (dead code / never fired, suspect
verifier, crash, not-implemented-as-claimed, remediation broken, silent skips). Give one
line of reason anchored to a count, case id, or diff line. When no runtime evidence
exists, judge from the diff and the per-check movement and say so in the reason. This
verdict decides whether the edit counts as evidence about its STRATEGY (sound) or only
about its implementation (unsound), so commit to one answer.
Give the verdict PER SUB-EDIT too (`sub_edit_verdicts`, one entry per `## Edit N` block
in the description, judged on that sub-edit's own components and target checks) as well
as for the node as a whole: a bundled edit with one broken component must not hide a
sound one, and a sound sub-edit next to a broken one still counts as strategy evidence.

Effect verdict (required, per sub-edit, in `sub_edit_verdicts`): did this mechanism IMPROVE
the behaviour it targets (name the checks in `targeted_checks`) on the observed cases?
  improved  — the targeted checks fail less and the mechanism is the reason (evidence
              level 1–3, anchored to the cases where it fired);
  no_effect — the mechanism never fired, fired without changing outputs, or the targeted
              checks did not move; "never fired" is ALWAYS no_effect, never unclear;
  regressed — targeted or collateral checks fail more and the mechanism is the reason;
  unclear   — only when the evidence genuinely cuts both ways or rests on < 4 cases.
Grade the evidence behind the verdict: strong = level 1 or 2; moderate = level 3 (paired or
unpaired per-check movement); weak = level 4–5 only. `effect_reason`: one line anchored to
counts, case ids, or a component's agreement line. The description states the author's
intent, not what happened, and the score Δ is noise at this batch size: neither decides
the effect. Also list `regressions`: checks (targeted or not) whose failures increased with
the edit, "name j->k fails (d)" from the paired table when available, else the unpaired
one; or "none observed".

Targets: absolute failure count after the edit vs parent; remaining_failures is the
bare "k/n" only; the parent count goes in `was`.
Collateral: non-targeted checks whose failure counts changed, worst degradations
first, "name j->k fails (d)", d signed improvement-positive.

Rules: cite case ids for every agreement claim; never invent numbers; quote failure
counts as k/n; "unmeasured in the observed batches" when evidence is thin. Counts
are lower bounds (traces are sampled per batch). Stay compact: at most 3 cause
lines per component."""

ANALYSIS_TOOL = {"type": "function", "function": {
    "name": "submit_edit_analysis",
    "description": "Submit the grounded analysis of this node's edit.",
    "parameters": {"type": "object", "properties": {
        "components": {"type": "array", "items": {"type": "object", "properties": {
            "component": {"type": "string",
                          "description": "tool name or label/name of the verifier/checker"},
            "activated": {"type": "string",
                          "description": "how often it actually ran, per the evidence"},
            "verdict_behavior": {"type": "string",
                                 "description": "pass/fail pattern when it ran"},
            "agreement": {"type": "string",
                          "description": "when it passed, did the agent's output satisfy "
                                         "that component per the scorer? cite case ids"},
            "role": {"type": "string", "enum": ["gate", "detector", "other"]},
            "cause": {"type": "string",
                      "description": "1-3 short lines, newline-separated, each "
                                     "anchored to a count, case id, or diff line; "
                                     "detectors must include the remediation join"}},
            "required": ["component", "role", "activated", "verdict_behavior",
                         "agreement", "cause"]}},
        "targeted_constraints": {"type": "array", "items": {"type": "object", "properties": {
            "constraint": {"type": "string"},
            "remaining_failures": {"type": "string",
                                   "description": "failures AFTER the edit, bare 'k/n' "
                                                  "only, e.g. '17/29'"},
            "was": {"type": "string",
                    "description": "parent failures, bare 'j/n' only, e.g. '20/29'"},
            "evidence": {"type": "string"}},
            "required": ["constraint", "remaining_failures", "was"]}},
        "collateral": {"type": "string",
                       "description": "non-targeted checks whose failure counts changed, "
                                      "e.g. 'essential_attraction_coverage 10->13 fails "
                                      "(-3)'; or 'none observed'"},
        "implementation_sound": {
            "type": "boolean",
            "description": "true when the edit does what it claims, its components "
                           "ran and their verdicts agree with the scorer; false for "
                           "dead code, suspect verifiers, crashes, broken remediation, "
                           "not-implemented-as-claimed"},
        "implementation_reason": {
            "type": "string",
            "description": "one line anchored to a count, case id, or diff line"},
        "sub_edit_verdicts": {
            "type": "array",
            "description": "one entry per `## Edit N` block of the description",
            "items": {"type": "object", "properties": {
                "edit": {"type": "integer", "description": "the N of `## Edit N`"},
                "sound": {"type": "boolean"},
                "reason": {"type": "string",
                           "description": "one line anchored to a count, case id, "
                                          "or diff line"},
                "targeted_checks": {
                    "type": "array", "items": {"type": "string"},
                    "description": "the benchmark checks this sub-edit aims at "
                                   "(names as in the per-check tables)"},
                "effect": {
                    "type": "string",
                    "enum": ["improved", "no_effect", "regressed", "unclear"],
                    "description": "did the mechanism improve its targeted "
                                   "behaviour on the observed cases; 'never "
                                   "fired' is no_effect"},
                "evidence": {
                    "type": "string", "enum": ["strong", "moderate", "weak"],
                    "description": "strong = within-case or component counts "
                                   "over >= 8 cases; moderate = per-check "
                                   "movement (paired or unpaired); weak = score "
                                   "delta or implementation only"},
                "effect_reason": {
                    "type": "string",
                    "description": "one line anchored to counts, case ids, or a "
                                   "component's agreement line"}},
                "required": ["edit", "sound", "reason", "targeted_checks",
                             "effect", "evidence", "effect_reason"]}},
        "regressions": {
            "type": "string",
            "description": "checks whose failures increased with the edit, "
                           "'name j->k fails (d)' each, worst first; or "
                           "'none observed'"}},
        "required": ["components", "targeted_constraints", "collateral",
                     "implementation_sound", "implementation_reason",
                     "sub_edit_verdicts", "regressions"]}}}


# Bumping this re-buys every node's analysis exactly once (on its next
# refresh/finalize) under the current question set — the version is salted
# into the sig, so cached answers from an older ANALYSIS_SYSTEM never satisfy
# the staleness check. v2: evidence-only records (no judgment labels), signed
# improvement-positive convention, per-component likely-cause, code diff in
# the prompt. v3: rigorous multi-line causes (gate/detector roles, both
# agreement directions, remediation join, cause checklist) + seen-vs-unseen
# generalization evidence. v4: generalization removed from the analysis
# entirely — the split's parent-side delta goes stale under this child-only
# sig (the parent keeps accruing evals after the child's evidence freezes),
# so the seen-vs-unseen line lives ONLY in the Outcome section, which is
# recomputed deterministically on every refresh. v5: the implementation
# verdict (sound/unsound + reason) that gates belief scoring. v6: the same
# verdict per sub-edit, so one broken component in a bundled edit no longer
# vetoes strategy evidence for the others. v7: the analysis is the JUDGE —
# an effect verdict per sub-edit (improved / no_effect / regressed / unclear
# + evidence grade + targeted checks) is the strategy label the belief layer
# scores against; the prompt gains standard errors, an unpaired per-check
# table and per-case rows so no parent overlap is needed, and shows the
# implementation view instead of a truncated diff.
ANALYSIS_VERSION = 7


def analysis_sig(child_case_sig: str, batches: Sequence[str], *,
                 version: int = ANALYSIS_VERSION) -> str:
    """Key deciding whether the analysis is stale. Own evidence only (child
    cases + consumed batches): a parent's new batch shifts delta slightly but
    does not change what this node's verifiers did, and keying on the parent
    would re-buy the LLM call on every sibling refresh."""
    raw = f"v{version}|" + child_case_sig + "|" + "|".join(batches)
    return blake2b(raw.encode("utf-8"), digest_size=8).hexdigest()


def _failed_messages(obj: Any, depth: int = 0, cap: int = 5) -> list[str]:
    """Schema-agnostic: collect message strings from dicts shaped like
    ``{"passed": False, "message": ...}`` anywhere in the case details."""
    out: list[str] = []
    if depth > 4 or len(out) >= cap:
        return out
    if isinstance(obj, dict):
        if obj.get("passed") is False and isinstance(obj.get("message"), str):
            out.append(obj["message"][:150])
        for v in obj.values():
            if len(out) >= cap:
                break
            out.extend(_failed_messages(v, depth + 1, cap - len(out)))
    elif isinstance(obj, list):
        for v in obj[:20]:
            if len(out) >= cap:
                break
            out.extend(_failed_messages(v, depth + 1, cap - len(out)))
    return out


def _case_excerpt(case: Any, recipe: Optional[Mapping[str, str]]) -> dict:
    mode = (recipe or {}).get("mode")
    path = (recipe or {}).get("path")
    out: dict[str, Any] = {
        "case_id": str(case["case_id"] if isinstance(case, dict)
                       else getattr(case, "case_id", "?")),
        "passed": case.get("passed") if isinstance(case, dict)
        else getattr(case, "passed", None),
        "score": case.get("score") if isinstance(case, dict)
        else getattr(case, "score", None),
    }
    err = (case.get("error") if isinstance(case, dict)
           else getattr(case, "error", None))
    if err:
        out["error"] = str(err)[:200]
    checks = extract_checks(case, mode, path)
    if checks is not None:
        out["failed_checks"] = sorted(compact_check(c) for c in checks)
    details = (case.get("details") if isinstance(case, dict)
               else getattr(case, "details", None))
    msgs = _failed_messages(details)
    if msgs:
        out["failed_check_messages"] = msgs
    return out


def _clamped_scores(cases: Sequence[Any]) -> list[float]:
    out = []
    for c in cases or ():
        raw = c.get("score") if isinstance(c, dict) else getattr(c, "score", 0.0)
        out.append(min(1.0, max(0.0, float(raw or 0.0))))
    return out


def _own_check_counts(cases: Mapping[str, Any],
                      recipe: Optional[Mapping[str, str]]) -> tuple[int, dict[str, int]]:
    """``(cases with check data, {check: failures})`` over one side's OWN
    cases — the unpaired table's raw material."""
    mode = (recipe or {}).get("mode")
    path = (recipe or {}).get("path")
    n = 0
    fails: dict[str, int] = defaultdict(int)
    for case in cases.values():
        checks = extract_checks(case, mode, path)
        if checks is None:
            continue
        n += 1
        for t in checks:
            fails[t] += 1
    return n, dict(fails)


def _case_key(cid: str) -> tuple[int, Any]:
    return (0, int(cid)) if cid.isdigit() else (1, cid)


def _fired_by_case(events: Sequence[Mapping]) -> dict[str, dict[str, dict[str, int]]]:
    """``{case_id: {label/name: {verdict: count}}}`` from the retained events."""
    out: dict[str, dict[str, dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(int)))
    for e in events or ():
        cid = e.get("case_id")
        if cid is None:
            continue
        key = (f"{e.get('label')}/{e.get('name')}" if e.get("name")
               else str(e.get("label")))
        out[str(cid)][key][str(e.get("verdict", "-"))] += 1
    return out


def _case_row(cid: str, case: Any, recipe: Optional[Mapping[str, str]], *,
              fired: Optional[Mapping[str, Mapping[str, int]]] = None,
              side: str = "") -> str:
    ex = _case_excerpt(case, recipe)
    score = ex.get("score")
    status = ("PASS" if ex.get("passed") else "FAIL") if ex.get("passed") is not None else "?"
    bits = [f"{float(score):.4f} {status}" if isinstance(score, (int, float))
            else f"score ? {status}"]
    checks = ex.get("failed_checks")
    if checks is None:
        bits.append("failed: (no check data)")
    elif not checks:
        bits.append("failed: none")
    else:
        shown = ", ".join(checks[:8])
        more = f", +{len(checks) - 8} more" if len(checks) > 8 else ""
        bits.append(f"failed: {shown}{more}")
    if fired:
        comps = []
        for key in sorted(fired)[:6]:
            vs = fired[key]
            comps.append(f"{key}=" + ",".join(
                f"{v}×{n}" if n > 1 else v for v, n in sorted(vs.items())))
        if len(fired) > 6:
            comps.append(f"+{len(fired) - 6} more")
        bits.append("fired: " + " ".join(comps))
    if ex.get("error"):
        bits.append("error: " + " ".join(str(ex["error"]).split())[:80])
    tag = f" ({side})" if side else ""
    return f"- case {cid}{tag}: " + " · ".join(bits)


def build_analysis_prompt(
    *,
    node_id: int,
    record_body: str,
    outcome: Any,
    store: Mapping[str, Any],
    u_lines: Sequence[str],
    parent_cases: Sequence[Any],
    child_cases: Sequence[Any],
    recipe: Optional[Mapping[str, str]],
    code_diff: str = "",
    code_kind: str = "diff",
    max_event_lines: int = MAX_ANALYSIS_EVENT_LINES,
    max_cases: int = MAX_ANALYSIS_CASES,
    max_case_rows: int = MAX_ANALYSIS_CASE_ROWS,
) -> str:
    """Assemble the evidence for one node's analysis call. Deterministic.

    ``code_kind`` names what ``code_diff`` holds: ``"view"`` for the
    implementation view (added lines per definition, whole units) or
    ``"diff"`` for a unified diff (the fallback when sources are missing)."""
    events = store.get("events") or []
    # Stratified sample: up to 3 per (label, name, verdict) group, so every
    # verdict of every component is represented before any group repeats.
    by_group: dict[tuple, list[dict]] = defaultdict(list)
    for e in events:
        by_group[(e.get("label"), e.get("name"),
                  str(e.get("verdict", "-")))].append(e)
    sampled: list[dict] = []
    for _, evs in sorted(by_group.items(), key=lambda kv: str(kv[0])):
        sampled.extend(evs[:3])
    sampled = sampled[:max_event_lines]
    ev_lines = [json.dumps(e, ensure_ascii=False, default=str)[:300]
                for e in sampled]

    child_by_id = {str(c["case_id"] if isinstance(c, dict)
                       else getattr(c, "case_id", None)): c
                   for c in child_cases or ()}
    parent_by_id = {str(c["case_id"] if isinstance(c, dict)
                        else getattr(c, "case_id", None)): c
                    for c in parent_cases or ()}
    cids: list[str] = []
    for e in sampled:
        c = e.get("case_id")
        if c is not None and str(c) not in cids and str(c) in child_by_id:
            cids.append(str(c))
    gt_basis = "cases the components touched"
    if not cids:
        # Uninstrumented edit: no events to select cases by. Fall back to the
        # shared cases where per-check outcomes MOVED (largest symmetric
        # difference first), so the targeted/collateral findings can still
        # cite case-level evidence.
        mode = (recipe or {}).get("mode")
        path = (recipe or {}).get("path")
        moved: list[tuple[int, str]] = []
        for cid, child_case in child_by_id.items():
            if cid not in parent_by_id:
                continue
            ps = extract_checks(parent_by_id[cid], mode, path)
            cs = extract_checks(child_case, mode, path)
            if ps is None or cs is None:
                continue
            d = len(ps ^ cs)
            if d:
                moved.append((d, cid))
        moved.sort(key=lambda t: (-t[0], t[1]))
        cids = [cid for _, cid in moved]
        if cids:
            gt_basis = ("shared cases where check outcomes changed "
                        "(fallback: this edit produced no runtime events)")
    cases = []
    for cid in cids[:max_cases]:
        entry: dict[str, Any] = {"child": _case_excerpt(child_by_id[cid], recipe)}
        if cid in parent_by_id:
            pe = _case_excerpt(parent_by_id[cid], recipe)
            entry["parent_same_case"] = {"passed": pe.get("passed"),
                                         "failed_checks": pe.get("failed_checks")}
        cases.append(entry)

    # Absolute per-check tallies ground the targeted/collateral answers. They
    # are supplied here (and only here) — the record itself never carries the
    # table; it carries the LLM's digested findings.
    pc = getattr(outcome, "per_check", None) or {}
    n_check = getattr(outcome, "n_check_cases", 0)
    tbl_lines = [f"`{compact_check(k)}` {a}->{b} fails / {n_check} cases ({a - b:+d})"
                 for k, (a, b) in pc.items()] or ["(none measured)"]

    # Performance with standard errors: the judge must see how little a
    # 16-case Δ can say before it reads the tables.
    # Tolerate partial outcome objects (older sidecars, test stubs): fall
    # back from the own-case fields to the shared-set ones.
    n_sh = int(getattr(outcome, "n_shared", 0) or 0)
    n_c = int(getattr(outcome, "child_n_all", None) or n_sh)
    n_p = int(getattr(outcome, "parent_n_all", None) or n_sh)
    child_mean = float(getattr(outcome, "child_mean_all", None)
                       or getattr(outcome, "child_mean_shared", 0.0) or 0.0)
    parent_mean = float(getattr(outcome, "parent_mean_all", None)
                        or getattr(outcome, "parent_mean_shared", 0.0) or 0.0)
    delta_all = getattr(outcome, "delta_all", None)
    if delta_all is None:
        delta_all = float(getattr(outcome, "delta_shared", 0.0) or 0.0)
    delta_shared = float(getattr(outcome, "delta_shared", 0.0) or 0.0)
    min_sh = int(getattr(outcome, "min_shared", 8) or 8)
    se_c = se_of(_clamped_scores(child_cases))
    se_p = se_of(_clamped_scores(parent_cases))
    se_all = getattr(outcome, "se_all", None)
    se_sh = getattr(outcome, "se_shared", None)

    def _pm(x: Optional[float]) -> str:
        return f"±{x:.4f}" if x is not None else "n/a"

    perf_lines: list[str] = []
    if n_c:
        perf_lines.append(
            f"- child {child_mean:.4f} over {n_c} own cases (SE {_pm(se_c)})"
            f" · parent {parent_mean:.4f} over {n_p} own cases (SE {_pm(se_p)})")
        perf_lines.append(
            f"- unpaired Δ (own cases, different samples): {float(delta_all):+.4f} "
            f"{_pm(se_all)}")
        if n_sh >= min_sh:
            perf_lines.append(f"- paired Δ over {n_sh} shared cases: "
                              f"{delta_shared:+.4f} {_pm(se_sh)}")
        elif n_sh:
            perf_lines.append(f"- paired Δ over {n_sh} shared cases: "
                              f"{delta_shared:+.4f} — fewer than the "
                              f"{min_sh}-shared minimum, not a usable estimate")
        else:
            perf_lines.append("- no shared cases yet: no paired estimate")
    else:
        perf_lines.append("- not yet measured")

    # Unpaired per-check table: each side over its OWN cases with check data.
    n_pc, p_fails = _own_check_counts(parent_by_id, recipe)
    n_cc, c_fails = _own_check_counts(child_by_id, recipe)
    unpaired: list[str] = []
    if n_pc and n_cc:
        keys = sorted(set(p_fails) | set(c_fails),
                      key=lambda k: (-c_fails.get(k, 0), -p_fails.get(k, 0), k))
        for k in keys[:TOP_K_CHECKS]:
            pf, cf = p_fails.get(k, 0), c_fails.get(k, 0)
            unpaired.append(
                f"`{compact_check(k)}` parent {pf}/{n_pc} -> child {cf}/{n_cc} fails "
                f"(rate {100 * pf / n_pc:.0f}% -> {100 * cf / n_cc:.0f}%)")
        if len(keys) > TOP_K_CHECKS:
            unpaired.append(f"(+{len(keys) - TOP_K_CHECKS} more checks)")
    if not unpaired:
        unpaired = ["(no check data on one side)"]

    # Per-case rows: shared cases first (paired evidence, parent row beside
    # the child's), then the rest by id, up to the cap.
    fired = _fired_by_case(events)
    shared_ids = sorted((c for c in child_by_id if c in parent_by_id), key=_case_key)
    other_ids = sorted((c for c in child_by_id if c not in parent_by_id), key=_case_key)
    row_ids = (shared_ids + other_ids)[:max_case_rows]
    rows: list[str] = []
    for cid in row_ids:
        rows.append(_case_row(cid, child_by_id[cid], recipe, fired=fired.get(cid)))
        if cid in parent_by_id:
            rows.append(_case_row(cid, parent_by_id[cid], recipe, side="parent"))
    if not rows:
        rows = ["(no evaluated cases yet)"]
    shown_note = (f"{len(row_ids)} of {len(child_by_id)} child cases shown"
                  if len(child_by_id) > len(row_ids) else
                  f"all {len(child_by_id)} child cases")
    sampled_note = (" — fired lists come from sampled logs and are lower bounds"
                    if store.get("events_truncated") else "")

    code_header = ("# Implementation vs parent (added lines per definition; full "
                   "source of new definitions)" if code_kind == "view"
                   else "# Code diff vs parent (the edit itself)")
    parts = [
        f"# Node {node_id} — edit description",
        record_body.strip(),
        "\n" + code_header,
        code_diff or "(no diff supplied)",
        "\n# Performance (standard errors: ~16-case batches put every SE near ±0.1; "
        "a Δ smaller than 2×SE is noise, not evidence)",
        "\n".join(perf_lines),
        f"\n# Per-check failure counts, PAIRED (parent -> child, over {n_check} shared "
        "cases with check data; signed value = improvement, + means fewer "
        "failures)",
        "\n".join(tbl_lines),
        f"\n# Per-check failure counts, UNPAIRED (parent over its {n_pc} own cases with "
        f"check data vs child over its {n_cc}; different case samples — weaker "
        "evidence than the paired table, but it needs no overlap)",
        "\n".join(unpaired),
        "\n# Deterministic usage",
        "\n".join(u_lines) if u_lines else "(no runtime data captured)",
        "\n# Surface detected (what this edit added vs parent)",
        json.dumps(store.get("surface") or {}, indent=1),
        f"\n# Per-case results ({shown_note}; shared cases first with the parent's "
        f"row beside; fired = this edit's components on that case{sampled_note})",
        "\n".join(rows),
        f"\n# Runtime mutable_log events (sampled {len(sampled)} of "
        f"{len(events)} retained)",
        "\n".join(ev_lines) if ev_lines else
        "(none)\nNo runtime events exist for this edit's components: report "
        'every component\'s "agreement" as unmeasured; do not infer firing '
        "behavior. Likely causes may still be read from the implementation.",
        f"\n# Ground truth for {gt_basis} ({len(cases)} "
        "cases; parent_same_case = same case on the PARENT agent)",
        json.dumps(cases, indent=1, default=str) if cases
        else "(no overlapping cases)",
    ]
    return "\n".join(parts)


_KN_RE = re.compile(r"(\d+)\s*/\s*(\d+)")


def render_analysis(payload: Mapping[str, Any]) -> str:
    """The ``## Analysis`` section. Compact by design — a component gets its
    role tag, one agreement line, and at most three cause lines — because
    ~100 of these must fit the steering budget.

    Evidence only, no judgment labels. Tolerates older payload shapes
    (``assessment``/``constraint_effect``, no ``role``/``generalization``) so
    a ``"final"``-mode sweep over a resumed run never crashes on a cached old
    answer."""
    lines = ["## Analysis"]
    for c in payload.get("components") or []:
        if not isinstance(c, Mapping) or not c.get("component"):
            continue
        role = f" ({c['role']})" if c.get("role") else ""
        lines.append(f"- **`{c.get('component')}`**{role} — "
                     f"{c.get('activated', '?')}; {c.get('verdict_behavior', '?')}")
        lines.append(f"  - agreement: {c.get('agreement', '?')}")
        # Old cached payloads carry `assessment` instead of `cause`.
        cause = str(c.get("cause") or c.get("assessment") or "?")
        for cl in cause.splitlines()[:3]:
            if cl.strip():
                lines.append(f"  - likely cause: {cl.strip()}")
    for t in payload.get("targeted_constraints") or []:
        if not isinstance(t, Mapping) or not t.get("constraint"):
            continue
        now = str(t.get("remaining_failures", "?"))
        was = str(t.get("was", "") or "")
        # Guard against the model stuffing "(was ...)" into the field itself —
        # only append our computed context when the field is a bare k/n.
        wastxt = ""
        if was and "(" not in now:
            m_now, m_was = _KN_RE.search(now), _KN_RE.search(was)
            if m_now and m_was:
                d = int(m_was.group(1)) - int(m_now.group(1))
                wastxt = f" (was {was}, {d:+d})"
            else:
                wastxt = f" (was {was})"
        ev = f" — {t['evidence']}" if t.get("evidence") else ""
        lines.append(f"- **target `{t['constraint']}`** — remaining {now}{wastxt}{ev}")
    collateral = payload.get("collateral") or payload.get("constraint_effect")
    lines.append(f"- **collateral**: {collateral or '?'}")
    # v5 verdict. Older cached payloads carry no field and render no line, so
    # a node analysed under v4 stays "no verdict" until its re-analysis.
    impl = payload.get("implementation_sound")
    if isinstance(impl, bool):
        reason = " ".join(str(payload.get("implementation_reason") or "").split())[:240]
        lines.append(f"- **implementation**: {'sound' if impl else 'unsound'}"
                     + (f" — {reason}" if reason else ""))
        # v6: per-sub-edit verdicts, in edit order. Malformed entries skipped.
        subs = [v for v in (payload.get("sub_edit_verdicts") or [])
                if isinstance(v, Mapping) and isinstance(v.get("sound"), bool)
                and str(v.get("edit", "")).strip().isdigit()]
        for v in sorted(subs, key=lambda v: int(v["edit"])):
            rs = " ".join(str(v.get("reason") or "").split())[:200]
            lines.append(f"- **implementation (edit {int(v['edit'])})**: "
                         f"{'sound' if v['sound'] else 'unsound'}"
                         + (f" — {rs}" if rs else ""))
        # v7: the effect verdict per sub-edit — the judge's label the belief
        # layer scores strategy predictions against. Older payloads have no
        # `effect` and render no line (the node stays "no verdict" until its
        # re-analysis, exactly like the v5 implementation line).
        for v in sorted(subs, key=lambda v: int(v["edit"])):
            eff = str(v.get("effect") or "").strip().lower()
            if eff not in EFFECTS:
                continue
            ev = str(v.get("evidence") or "").strip().lower()
            ev = ev if ev in EVIDENCE_LEVELS else "weak"
            targets = []
            for t in (v.get("targeted_checks") or [])[:4]:
                t = compact_check(str(t)).replace("(", "").replace(")", "").strip()
                if t:
                    targets.append(t)
            tgt = f"; targets: {', '.join(targets)}" if targets else ""
            rs = " ".join(str(v.get("effect_reason") or "").split())[:240]
            lines.append(f"- **effect (edit {int(v['edit'])})**: {eff} ({ev}{tgt})"
                         + (f" — {rs}" if rs else ""))
        reg = payload.get("regressions")
        if isinstance(reg, str) and reg.strip():
            lines.append("- **regressions**: " + " ".join(reg.split())[:300])
    # `generalization` from older cached payloads is deliberately dropped:
    # its parent-side delta goes stale (see ANALYSIS_VERSION v4 note); the
    # seen-vs-unseen line is rendered only in Outcome, from live data.
    return "\n".join(lines) + "\n"
