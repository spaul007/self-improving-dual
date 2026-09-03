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
        rows.append({
            "id": sid, "definition": entry.get("definition", ""),
            "nodes": nodes, "n_nodes": len(nodes), "bundled": bundled,
            "median": median(deltas) if deltas else None,
            "best": max(deltas) if deltas else None,
            "worst": min(deltas) if deltas else None,
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


BELIEF_PREAMBLE = """How to read the belief document below:
- It is maintained by a belief-maintainer LLM from this run's MEASURED edit
  history and rewritten after every evaluation batch; its structure is that
  maintainer's own choosing.
- `### belief:<slug>` sections are its beliefs; the slug is the id to name in
  a prediction when an edit relies on that belief.
- `[node N: Δx/y]` citations are machine-verified against the actual records.
  When the machine appendix at the bottom reports a mismatch, trust the
  appendix's numbers over the body text.
- The machine appendix is code-generated fact tables (citation checks,
  evidence bases, proposal outcomes) — not the belief maintainer's opinion.
- Beliefs are judgments over noisy evidence and can be wrong: weigh each
  against its cited evidence base. `unproven` means untested — an invitation
  to try, never a rejection.
- Next-move lines are suggestions to build on, repair, or avoid; you may
  override them with better judgment grounded in the code you see."""


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
    """The editor-facing block. ``""`` when there is nothing to show.

    ``mode="full"`` (default) renders the legacy layout, byte-identical to
    before the belief layer existed. ``mode="belief"`` replaces the per-node
    record dump with the belief document: head guidance + interpretation
    preamble + ``belief_block`` + the per-strategy ledger + the focus block.
    """
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

    if mode == "belief":
        # Belief-led steering: the maintained belief document replaces the
        # per-node record dump entirely; full records reach the editor only
        # through the retrieval stage.
        out = ["\n## Edit memory — the run's digested edit history: beliefs "
               "over what was tried, plus the deterministic ledger"]
        if run_context:
            out.append(
                "Run context: seed %.4f/%d · best so far %.4f/%d (node %d). "
                "The goal is the highest ABSOLUTE score."
                % (run_context.get("seed_mean", 0.0), run_context.get("seed_n", 0),
                   run_context.get("best_mean", 0.0), run_context.get("best_n", 0),
                   run_context.get("best_node", -1)))
        out += [
            "",
            "You can use this to guide the next edit — for example:",
            "1. BUILD ON an influential edit: extend what the numbers show "
            "already works.",
            "2. REPAIR a promising category: when a strategy's intent is sound "
            "but its implementations are broken — gates passing outputs the "
            "scorer rejects, detectors whose flagged problems never get "
            "fixed, dead components — fix the implementation instead of "
            "abandoning the idea or repeating it unchanged.",
            "3. DIVERSIFY: try something different from everything recorded "
            "here.",
            "When you draw on the history, weight the measured evidence "
            "rather than how often something was tried.",
        ]
        if belief_block:
            out += ["", BELIEF_PREAMBLE, "", "### Belief document",
                    belief_block]
        out += ["", "### What has been tried, by strategy (deterministic "
                    "ledger)"]
        out += _ledger_lines(ledger, registry, level2_min_nodes)
        out += _focus_lines(records, focus_node_id, threshold, min_shared)
        text = "\n".join(out)
        if len(text) > budget:
            from .edit_diff import truncate_middle
            text = truncate_middle(text, budget)
        return text

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
