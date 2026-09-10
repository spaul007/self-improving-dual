"""Render the accumulated edit memory into one block for the agent editor.

Pure read: no LLM, no writes, and deterministic — the same tree state yields
byte-identical output, so a run stays reproducible.

Two inputs, both real history only: the **registry** (categories that some edit
actually used) and the **records** of every node generated so far. The setup
pass's proxy categories are deliberately *not* here — showing the editor
hypothesised moves as though they were tried history would bias the search.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from statistics import median
from typing import Any, Mapping, Optional

from .edit_memory import REGISTRY_NAME, RECORD_NAME, split_record

_CHARS_PER_TOKEN = 4
# Both record generations parse. The fmt-2 performance line is tried first;
# the legacy delta-shared line already carries the child absolute in its
# parenthetical, so old records feed the absolute ledger without a rewrite.
# fmt-4: cumulative score leads, shared comparison in the parenthetical.
_PERF4_RE = re.compile(
    r"\*\*performance\*\*: child ([\d.]+) over (\d+) evaluated cases "
    r"\(vs parent on (\d+) shared: child ([\d.]+), parent ([\d.]+), "
    r"Δ ([-+][\d.]+)\)")
_PERF4_NS_RE = re.compile(
    r"\*\*performance\*\*: child ([\d.]+) over (\d+) evaluated cases "
    r"\(no cases shared with parent yet\)")
_PERF_RE = re.compile(
    r"\*\*performance\*\*: child ([\d.]+) over (\d+) shared cases "
    r"\(parent ([\d.]+), Δ ([-+][\d.]+)\)")
_DELTA_RE = re.compile(
    r"\*\*delta shared\*\*: ([-+][\d.]+) over (\d+) shared cases "
    r"\(parent ([\d.]+) -> child ([\d.]+)\)")
# Usage lines and the Analysis section live in/after the Outcome section,
# which split_record cuts from `body` to protect the refresh contract — so
# both are read from the FULL text. (Reading from `body` is how per-check
# data got silently dropped from the steering block once already.)
_USAGE_RE = re.compile(r"^- \*\*(?:usage|new tools|new log point)[^\n]*$", re.M)
_ANALYSIS_RE = re.compile(r"\n## Analysis\n(.+)$", re.S)
# The analysis call's implementation verdict (edit_usage.render_analysis):
# one node-level line (v5) and, from v6, one per sub-edit.
_IMPL_RE = re.compile(r"^- \*\*implementation\*\*: (sound|unsound)(?: — (.*))?$", re.M)
_IMPL_EDIT_RE = re.compile(
    r"^- \*\*implementation \(edit (\d+)\)\*\*: (sound|unsound)(?: — (.*))?$", re.M)
# fmt-6: the unpaired comparison (each side over its OWN cases) with its
# standard error, plus the paired SE when any cases are shared.
_UNPAIRED_RE = re.compile(
    r"\*\*unpaired\*\*: child ([\d.]+)/(\d+) vs parent ([\d.]+)/(\d+) · "
    r"Δ ([-+][\d.]+) ± ([\d.]+|n/a)(?: · paired SE ±([\d.]+))?")
# Analysis v7: the judge's effect verdict per sub-edit, and the node-level
# regressions line.
_EFFECT_EDIT_RE = re.compile(
    r"^- \*\*effect \(edit (\d+)\)\*\*: (improved|no_effect|regressed|unclear) "
    r"\((strong|moderate|weak)(?:; targets: ([^)]*))?\)(?: — (.*))?$", re.M)
_REGRESSIONS_RE = re.compile(r"^- \*\*regressions\*\*: (.*)$", re.M)
_EDIT_HDR_RE = re.compile(r"^## Edit (\d+)\s*$")
_FIELD_RE = re.compile(r"^- \*\*([^*]+)\*\*: (.*)$")
_FIELD_KEYS = {"name": "name", "category level 1 (strategy)": "strategy",
               "category level 2 (area)": "area", "what": "what", "why": "why",
               "fit": "fit"}


def record_tags(body: str) -> list[dict[str, Any]]:
    """The ``## Edit N`` blocks of a record body as
    ``[{edit, name, strategy, area, what, why, fit}]`` (registry ids
    unquoted; ``fit`` is ``exact`` / ``folded`` / ``forced`` — see
    ``edit_memory.render_edits``). Tolerant of missing fields."""
    out: list[dict[str, Any]] = []
    cur: Optional[dict[str, Any]] = None
    for line in (body or "").split("\n"):
        m = _EDIT_HDR_RE.match(line)
        if m:
            cur = {"edit": int(m.group(1)), "name": "", "strategy": "",
                   "area": "", "what": "", "why": "", "fit": "exact"}
            out.append(cur)
            continue
        if cur is None:
            continue
        f = _FIELD_RE.match(line)
        if f and f.group(1).strip() in _FIELD_KEYS:
            key = _FIELD_KEYS[f.group(1).strip()]
            val = f.group(2).strip().strip("`")
            if key == "fit":
                low = val.lower()
                val = ("forced" if low.startswith("forced")
                       else "folded" if low.startswith("folded") else "exact")
            cur[key] = val
    return out


def _load_records(experiment_dir: Path) -> dict[int, dict[str, Any]]:
    """Every parseable ``round_*/edit_memory.md``, keyed by node id."""
    out: dict[int, dict[str, Any]] = {}
    for d in sorted(Path(experiment_dir).glob("round_*")):
        path = d / RECORD_NAME
        if not path.is_dir() and path.exists():
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            fm, body = split_record(text)
            try:
                nid = int(fm.get("node", ""))
            except (TypeError, ValueError):
                continue
            delta = parent_abs = child_abs = None
            n_shared = 0
            n_abs = 0
            m = _PERF4_RE.search(text)
            if m:
                child_abs, n_abs = float(m.group(1)), int(m.group(2))
                n_shared = int(m.group(3))
                parent_abs, delta = float(m.group(5)), float(m.group(6))
            elif (m := _PERF4_NS_RE.search(text)):
                child_abs, n_abs = float(m.group(1)), int(m.group(2))
            elif (m := _PERF_RE.search(text)):
                # fmt-2/3: only the shared-set score is on record
                child_abs, n_shared = float(m.group(1)), int(m.group(2))
                parent_abs, delta = float(m.group(3)), float(m.group(4))
                n_abs = n_shared
            else:
                m = _DELTA_RE.search(text)
                if m:
                    delta, n_shared = float(m.group(1)), int(m.group(2))
                    parent_abs, child_abs = float(m.group(3)), float(m.group(4))
                    n_abs = n_shared
            am = _ANALYSIS_RE.search(text)
            im = _IMPL_RE.search(text)
            per_edit = list(_IMPL_EDIT_RE.finditer(text))
            um = _UNPAIRED_RE.search(text)
            delta_all = se_all = se_shared = parent_abs_all = None
            parent_n_all = 0
            if um:
                parent_abs_all, parent_n_all = float(um.group(3)), int(um.group(4))
                delta_all = float(um.group(5))
                se_all = float(um.group(6)) if um.group(6) != "n/a" else None
                se_shared = float(um.group(7)) if um.group(7) else None
            effects = list(_EFFECT_EDIT_RE.finditer(text))
            effect_by_edit = {int(x.group(1)): x.group(2) for x in effects}
            first_edit = min(effect_by_edit) if effect_by_edit else None
            rm = _REGRESSIONS_RE.search(text)
            out[nid] = {
                "fm": fm, "body": body, "text": text,
                "usage": "\n".join(_USAGE_RE.findall(text)),
                "suspect": "SUSPECT VERIFIER" in text,
                "analysis": am.group(1).strip("\n") if am else "",
                "delta": delta,
                "n_shared": n_shared,
                "n_abs": n_abs,
                "parent_abs": parent_abs,
                "child_abs": child_abs,
                # fmt-6: the unpaired comparison over each side's own cases
                # (always available after one batch) with its SE.
                "delta_all": delta_all,
                "se_all": se_all,
                "se_shared": se_shared,
                "parent_abs_all": parent_abs_all,
                "parent_n_all": parent_n_all,
                # Belief-layer inputs: the registry tags of each sub-edit and
                # the analysis LLM's implementation verdict (None = no verdict
                # yet, which keeps the node unscorable rather than "sound").
                "tags": record_tags(body),
                "impl_sound": (im.group(1) == "sound") if im else None,
                "impl_reason": (im.group(2) or "").strip() if im else "",
                # Per-sub-edit verdicts (analysis v6): a bundled edit with
                # one broken component no longer hides a sound one.
                "impl_by_edit": {int(x.group(1)): x.group(2) == "sound"
                                 for x in per_edit},
                "impl_reason_by_edit": {int(x.group(1)): (x.group(3) or "").strip()
                                        for x in per_edit},
                # The judge's effect verdict per sub-edit (analysis v7). The
                # node-level `effect` is the first sub-edit's — the primary
                # mechanism — used when a prediction matched no sub-edit.
                "effect_by_edit": effect_by_edit,
                "evidence_by_edit": {int(x.group(1)): x.group(3) for x in effects},
                "targets_by_edit": {int(x.group(1)): [t.strip() for t in
                                                      (x.group(4) or "").split(",")
                                                      if t.strip()]
                                    for x in effects},
                "effect_reason_by_edit": {int(x.group(1)): (x.group(5) or "").strip()
                                          for x in effects},
                "effect": (effect_by_edit.get(first_edit)
                           if first_edit is not None else None),
                "evidence": (next(x.group(3) for x in effects
                                  if int(x.group(1)) == first_edit)
                             if first_edit is not None else None),
                "effect_reason": (next((x.group(5) or "").strip() for x in effects
                                       if int(x.group(1)) == first_edit)
                                  if first_edit is not None else ""),
                "regressions": rm.group(1).strip() if rm else "",
            }
    return out


def _verdict(delta: Optional[float], n_shared: int, threshold: float,
             min_shared: int) -> str:
    """Derived at render, never stored — so changing the threshold takes effect
    immediately instead of requiring every record to be rewritten."""
    if delta is None or n_shared == 0:
        return "unmeasured"
    if n_shared < min_shared:
        return "inconclusive"
    if delta >= threshold - 1e-9:
        return "helped"
    if delta <= -threshold + 1e-9:
        return "hurt"
    return "neutral"


def build_ledger(registry: Mapping[str, Any], records: Mapping[int, Any],
                 *, threshold: float, min_shared: int) -> list[dict[str, Any]]:
    """Per level-1 strategy: attempts, Δ stats, verdict tally, level-2 split."""
    rows = []
    for sid, entry in (registry.get("strategies") or {}).items():
        nodes = sorted({r["node"] for r in entry.get("edits", [])})
        deltas = [records[n]["delta"] for n in nodes
                  if n in records and records[n]["delta"] is not None]
        if not nodes:
            continue
        verdicts = [_verdict(records[n]["delta"], records[n]["n_shared"],
                             threshold, min_shared) for n in nodes if n in records]
        bundled = sum(
            1 for n in nodes
            if sum(1 for e in (registry.get("strategies") or {}).values()
                   if any(r["node"] == n for r in e.get("edits", []))) > 1)
        absv = [(records[n]["child_abs"], records[n]["n_abs"]) for n in nodes
                if n in records and records[n]["child_abs"] is not None]
        suspect = sum(1 for n in nodes
                      if n in records and records[n].get("suspect"))
        # The judge's verdicts for THIS strategy's sub-edits (a bundled node
        # contributes the verdict of the sub-edit tagged with this id), and
        # the unpaired Δ that needs no shared cases.
        effects: dict[str, int] = {}
        targets: dict[str, int] = {}
        sound = unsound = n_regressed = 0
        for n in nodes:
            rec = records.get(n)
            if not rec:
                continue
            mine = [t for t in (rec.get("tags") or []) if t.get("strategy") == sid]
            for t in mine:
                e = (rec.get("effect_by_edit") or {}).get(t.get("edit"))
                if e:
                    effects[e] = effects.get(e, 0) + 1
                for chk in (rec.get("targets_by_edit") or {}).get(t.get("edit")) or []:
                    targets[chk] = targets.get(chk, 0) + 1
                s = (rec.get("impl_by_edit") or {}).get(t.get("edit"))
                if s is None:
                    s = rec.get("impl_sound")
                if s is True:
                    sound += 1
                elif s is False:
                    unsound += 1
            reg = " ".join(str(rec.get("regressions") or "").split()).lower()
            if mine and reg and reg.rstrip(".") != "none observed":
                n_regressed += 1
        unpaired = [records[n]["delta_all"] for n in nodes
                    if n in records and records[n].get("delta_all") is not None]
        rows.append({
            "id": sid, "definition": entry.get("definition", ""),
            "nodes": nodes, "n_nodes": len(nodes), "bundled": bundled,
            "median": median(deltas) if deltas else None,
            "best": max(deltas) if deltas else None,
            "worst": min(deltas) if deltas else None,
            "effects": effects, "sound": sound, "unsound": unsound,
            "targets": targets, "n_regressed": n_regressed,
            "unpaired_median": median(unpaired) if unpaired else None,
            # Absolute child scores (primary signal); n_range keeps every
            # absolute honest about its case sample.
            "abs_median": median(a for a, _ in absv) if absv else None,
            "abs_best": max(a for a, _ in absv) if absv else None,
            "n_range": ((min(n for _, n in absv), max(n for _, n in absv))
                        if absv else None),
            # Machine-computed REPAIR candidates: nodes whose pass-verdicts
            # landed mostly on scorer-failed cases (see usage_lines).
            "suspect": suspect,
            "tally": {v: verdicts.count(v) for v in
                      ("helped", "hurt", "neutral", "inconclusive", "unmeasured")
                      if verdicts.count(v)},
        })
    # Sort stays by attempt count: it is stable across eval batches (a score
    # sort would reorder the block every eval and rank a lucky single-node
    # strategy above a well-tested one); the absolute best shows on each row.
    rows.sort(key=lambda r: (-r["n_nodes"], r["id"]))
    return rows


_EFFECT_ORDER = ("improved", "no_effect", "regressed", "unclear")


def judge_ledger_lines(ledger: list[dict[str, Any]]) -> list[str]:
    """One judge-first row per strategy for the belief maintainer (and the
    example renderer): the judge's verdict tally for this strategy's
    sub-edits, the checks they targeted, how many nodes regressed, the
    implementation verdicts, and the score Δ medians as trailing context."""
    out: list[str] = []
    for r in ledger:
        effects = r.get("effects") or {}
        eff = ", ".join(f"{k} {effects[k]}" for k in sorted(
            effects, key=lambda k: (_EFFECT_ORDER.index(k)
                                    if k in _EFFECT_ORDER else 9, k)))
        bits = [f"judge: {eff or 'not judged yet'}"]
        targets = r.get("targets") or {}
        if targets:
            top = sorted(targets.items(), key=lambda kv: (-kv[1], kv[0]))[:4]
            bits.append("targets: " + ", ".join(f"{k} ×{v}" for k, v in top))
        if r.get("n_regressed"):
            bits.append(f"regressions on {r['n_regressed']} node(s)")
        if r.get("sound") or r.get("unsound"):
            bits.append(f"implementation: sound {r.get('sound', 0)} / "
                        f"unsound {r.get('unsound', 0)}")
        else:
            bits.append("implementation: no verdict")
        ctx = []
        if r.get("median") is not None:
            ctx.append("paired Δ median %+.4f" % r["median"])
        if r.get("unpaired_median") is not None:
            ctx.append("unpaired Δ median %+.4f" % r["unpaired_median"])
        bits.append("score (context): " + (" · ".join(ctx) or "no score Δ yet"))
        out.append(f"- `{r['id']}` — {r['n_nodes']} node(s) "
                   f"({', '.join(str(n) for n in r['nodes'])}) · "
                   + " · ".join(bits) + f" — {r['definition']}")
    return out


def _areas_for(registry: Mapping[str, Any], nodes: list[int]) -> list[tuple[str, int]]:
    counts: dict[str, int] = {}
    for aid, entry in (registry.get("areas") or {}).items():
        n = len({r["node"] for r in entry.get("edits", [])} & set(nodes))
        if n:
            counts[aid] = n
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))


def _ledger_lines(ledger: list[dict[str, Any]], registry: Mapping[str, Any],
                  level2_min_nodes: int) -> list[str]:
    """The per-strategy ledger rows, shared verbatim by both render modes."""
    out: list[str] = []
    for r in ledger:
        if r["abs_median"] is not None:
            lo, hi = r["n_range"]
            nrange = f"n {lo}" if lo == hi else f"n {lo}–{hi}"
            stat = ("child median %.3f · best %.3f (%s)"
                    % (r["abs_median"], r["abs_best"], nrange))
            if r["median"] is not None:
                stat += " · Δ median %+.4f" % r["median"]
        elif r["median"] is not None:
            stat = "Δ median %+.4f · best %+.4f · worst %+.4f" % (
                r["median"], r["best"], r["worst"])
        else:
            stat = "no measured outcome yet"
        tally = ", ".join(f"{k} {v}" for k, v in r["tally"].items())
        bundle = (f" ({r['bundled']} bundled with other strategies)"
                  if r["bundled"] else "")
        flag = (f" · suspect-verifier in {r['suspect']} node(s)"
                if r.get("suspect") else "")
        out.append(f"- **`{r['id']}`** — {r['n_nodes']}×{bundle} · {stat} · "
                   f"{tally}{flag}")
        out.append(f"  - {r['definition']}")
        # A dominant bucket's median sits near the run mean and says little, so
        # its level-2 split is what carries the signal — always show it there.
        if r["n_nodes"] >= level2_min_nodes:
            areas = _areas_for(registry, r["nodes"])[:5]
            if areas:
                out.append("  - aimed at: "
                           + ", ".join(f"{a} ×{n}" for a, n in areas))
        out.append(f"  - nodes: {', '.join(str(n) for n in r['nodes'])}")
    return out


def _focus_lines(records: Mapping[int, Any], focus_node_id: Optional[int],
                 threshold: float, min_shared: int) -> list[str]:
    """The 'edits already tried off this parent' block, shared by both modes."""
    if focus_node_id is None:
        return []
    kids = [n for n, rec in sorted(records.items())
            if rec["fm"].get("parent") == str(focus_node_id)]
    if not kids:
        return []
    out = ["", f"### Edits already tried directly off node "
               f"{focus_node_id} (the parent being edited now)"]
    for n in kids:
        rec = records[n]
        v = _verdict(rec["delta"], rec["n_shared"], threshold, min_shared)
        if rec["child_abs"] is not None and rec["delta"] is not None:
            out.append(
                f"- node {n}: child {rec['child_abs']:.4f}/"
                f"{rec['n_abs']} (Δ {rec['delta']:+.4f} vs parent "
                f"on {rec['n_shared']} shared, {v})")
        elif rec["child_abs"] is not None:
            out.append(
                f"- node {n}: child {rec['child_abs']:.4f}/"
                f"{rec['n_abs']} (Δ vs parent unmeasured)")
        elif rec["delta"] is not None:
            out.append(f"- node {n} (Δ {rec['delta']:+.4f}, {v})")
        else:
            out.append(f"- node {n} (unmeasured)")
        block = rec["body"]
        if rec["usage"]:
            block += "\n" + rec["usage"]
        if rec["analysis"]:
            block += "\n" + rec["analysis"]
        out.append("  " + block.replace("\n", "\n  "))
    return out


def render_edit_memory(
    experiment_dir: Path,
    *,
    token_budget: int = 48000,
    threshold: float = 0.02,
    min_shared: int = 8,
    focus_node_id: Optional[int] = None,
    level2_min_nodes: int = 6,
    run_context: Optional[Mapping[str, Any]] = None,
    mode: str = "full",
    belief_block: str = "",
) -> str:
    """The editor-facing block for ``steering_mode: "full"`` — the legacy
    layout, byte-identical to before the belief layer existed. ``""`` when
    there is nothing to show.

    Belief-mode steering is not rendered here (and ``mode="belief"`` raises):
    it is authored in ``meta_agent/steering.py`` so its text has one owner.
    ``belief_block`` is accepted for signature compatibility and ignored.
    """
    if mode != "full":
        raise ValueError(f"render_edit_memory: unsupported mode {mode!r}; "
                         "belief-mode steering lives in meta_agent/steering.py")
    experiment_dir = Path(experiment_dir)
    reg_path = experiment_dir / REGISTRY_NAME
    if token_budget <= 0 or not reg_path.exists():
        return ""
    try:
        registry = json.loads(reg_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    records = _load_records(experiment_dir)
    if not records:
        return ""

    budget = token_budget * _CHARS_PER_TOKEN
    ledger = build_ledger(registry, records, threshold=threshold, min_shared=min_shared)

    head = [
        "\n## Edit memory — the run's global edit history: what was tried, "
        "what worked, what did not",
    ]
    if run_context:
        head.append(
            "Run context: seed %.4f/%d · best so far %.4f/%d (node %d). "
            "The goal is the highest ABSOLUTE score."
            % (run_context.get("seed_mean", 0.0), run_context.get("seed_n", 0),
               run_context.get("best_mean", 0.0), run_context.get("best_n", 0),
               run_context.get("best_node", -1)))
    head += [
        "",
        "How to read each node record below:",
        '- "node / parent / depth / lineage": the record\'s place in the edit '
        "tree — node is the agent this edit produced, parent is the agent it "
        "was edited from (the Δ baseline), depth counts edits since the seed, "
        'and lineage is the full edit chain from the seed to this node (e.g. '
        '"0 > 2" = seed edited into node 2).',
        '- "Edit N" blocks: what was changed and why (tagged with a reusable '
        "strategy and problem area).",
        '- "performance": the node\'s ABSOLUTE score over ALL cases evaluated '
        "so far (scale [0,1], higher is better). The parenthesis compares "
        "child vs parent on the cases BOTH ran — that Δ is the causal effect "
        "of the single edit, free of case-mix; a Δ over few shared cases is "
        "noisy until coverage grows.",
        '- "new tools"/"new log point" lines: whether code this edit added '
        'actually ran. "never fired" / "0 calls" = dead code that shipped but '
        "never executed.",
        "- Analysis component bullets: each added verifier/tool with its role "
        "(gate = its pass releases the output; detector = its fail flags a "
        "problem), whether its verdicts agreed with the benchmark scorer in "
        "both directions, and 1-3 evidence-anchored likely-cause lines — "
        "including, for detectors, whether the flagged problem was actually "
        'fixed in the final output. On a usage line, "-> scorer on those '
        'cases: X pass / Y fail" is that component\'s measured agreement, and '
        '"SUSPECT VERIFIER" marks one whose passes land mostly on scorer-'
        "failed cases.",
        '- "target" lines: benchmark checks the edit aimed at — '
        '"remaining k/n (was j/n, +d)" are FAILURE counts; 0/n remaining means '
        "that problem is solved on the observed cases.",
        '- "collateral": non-targeted checks whose failure counts changed.',
        '- "generalization": the child\'s score split into SEEN cases (the '
        "parent's evaluated cases at edit time — exactly what the editor's "
        "feedback was computed from) vs UNSEEN cases. A clearly better seen "
        "side means the edit likely overfits the feedback it saw — discount "
        "its Δ accordingly.",
        "",
        "Conventions (identical everywhere): per-check numbers are failure "
        "counts, fewer is better; every signed value is improvement-positive "
        "(+ = better, − = worse), for scores and checks alike. Absolute "
        "scores from different nodes may rest on different case samples — "
        "each carries its n.",
        "",
        "You can potentially utilize this edit history to guide the next "
        "edit — for example:",
        "1. BUILD ON an influential edit: extend what the numbers show "
        "already works.",
        "2. REPAIR a promising category: when a strategy's intent is sound "
        "but the analyses show its implementations are broken — gates passing "
        "outputs the scorer rejects, detectors whose flagged problems never "
        "get fixed, dead components, gains only on seen cases — fix the "
        "implementation instead of abandoning the idea or repeating it "
        "unchanged.",
        "3. DIVERSIFY: try something different from everything recorded here.",
        "When you draw on the history, weight the measured evidence rather "
        "than how often something was tried.",
        "",
        "### What has been tried, by strategy",
    ]
    head += _ledger_lines(ledger, registry, level2_min_nodes)

    focus_block = _focus_lines(records, focus_node_id, threshold, min_shared)

    detail = ["", "### Every edit, oldest first"]
    for n in sorted(records):
        rec = records[n]
        v = _verdict(rec["delta"], rec["n_shared"], threshold, min_shared)
        if rec["child_abs"] is not None and rec["delta"] is not None:
            perf = (f"child {rec['child_abs']:.4f}/{rec['n_abs']} · "
                    f"Δ {rec['delta']:+.4f} vs parent on {rec['n_shared']} "
                    f"shared, {v}")
        elif rec["child_abs"] is not None:
            perf = f"child {rec['child_abs']:.4f}/{rec['n_abs']} · Δ unmeasured"
        elif rec["delta"] is not None:
            perf = f"Δ {rec['delta']:+.4f} over {rec['n_shared']} shared, {v}"
        else:
            perf = "unmeasured"
        detail.append("")
        detail.append(f"#### node {n} ← {rec['fm'].get('parent', '?')}  "
                      f"(lineage {rec['fm'].get('lineage', '?')})  " + perf)
        detail.append(rec["body"])
        # Runtime usage ("this verifier never fired") and the analysis
        # ("when it passed, the plan was still wrong on X") — what separates
        # "this did nothing" from "it fixed X and broke Y".
        if rec["usage"]:
            detail.append(rec["usage"])
        if rec["analysis"]:
            detail.append(rec["analysis"])

    def _fits(*parts: list[str]) -> Optional[str]:
        text = "\n".join(p for chunk in parts for p in chunk)
        return text if len(text) <= budget else None

    full = _fits(head, focus_block, detail)
    if full is not None:
        return full
    # Over budget (large trees): collapse the per-node detail to one line each,
    # and say so rather than truncating silently.
    compact = ["", "### Every edit, oldest first (compact — full records omitted for space)"]
    for n in sorted(records):
        rec = records[n]
        v = _verdict(rec["delta"], rec["n_shared"], threshold, min_shared)
        cats = re.findall(r"\(strategy\)\*\*: `([^`]+)`", rec["body"])
        first_what = re.search(r"\*\*what\*\*: (.+)", rec["body"])
        detail_line = (first_what.group(1)[:100] if first_what else "")
        detail_line = re.sub(r"\s+", " ", detail_line)
        flags = ""
        if "0 calls" in rec["usage"] or "never fired" in rec["usage"]:
            flags += " | has-unused-component"
        if rec["child_abs"] is not None and rec["delta"] is not None:
            perf = (f"{rec['child_abs']:.3f}/{rec['n_abs']} "
                    f"Δ{rec['delta']:+.3f} {v}")
        elif rec["child_abs"] is not None:
            perf = f"{rec['child_abs']:.3f}/{rec['n_abs']}"
        elif rec["delta"] is not None:
            perf = f"Δ{rec['delta']:+.4f}/{rec['n_shared']} {v}"
        else:
            perf = "unmeasured"
        compact.append(f"- n{n} ← {rec['fm'].get('parent', '?')} " + perf
                       + f" | {', '.join(cats)} | {detail_line}{flags}")
    squeezed = _fits(head, focus_block, compact)
    if squeezed is not None:
        return squeezed
    print("[edit_memory] steering block over budget even compacted; "
          "ledger + local context only", flush=True)
    return "\n".join(head + focus_block)
