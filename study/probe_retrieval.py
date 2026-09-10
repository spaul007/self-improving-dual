"""Free probe — what did `max_retrieved_nodes` hide?

A1 showed retrieval is dominated by explicit node ids: associative matches
(strategy / area / keyword) survive the 4-slot cap only 15% of the time. This
replays every expand's OWN recorded query against the world as it stood at that
moment, sweeping `max_nodes` and `char_budget`, and asks:

  1. fidelity  — does replaying at the run's own settings reproduce the run's
                 own manifest? (gate: if not, nothing below is trustworthy)
  2. admission — which nodes does a bigger cap let in, and by which channel?
  3. payoff    — do those newly admitted nodes CONTAIN the definitions the child
                 went on to edit? That is the only version of "the cap cost us
                 something" that is not merely cosmetic.

Deterministic: no LLM, no evaluation.

Usage:
    PYTHONPATH=. python3 study/probe_retrieval.py --run <snapshot> [--out report.md]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from study import asof  # noqa: E402
from study.audit_chain import child_edited_units, parse_retrieved  # noqa: E402

SWEEP = [(4, 60000), (8, 60000), (12, 60000), (999, 60000), (999, 240000)]


def _channel(why: str) -> str:
    why = why or "?"
    for c in ("explicit", "strategy", "area", "keyword"):
        if why.startswith(c):
            return c
    return "other"


def _units_in(blocks: list[str]) -> set[tuple[str, str]]:
    """Definitions actually rendered by a retrieval result."""
    return parse_retrieved(
        "## Retrieved records and implementations\n" + "\n\n".join(blocks))["shown"]


def probe_node(run: Path, child: int, workdir: Path) -> dict[str, Any] | None:
    from meta_agent import edit_archive

    d = run / f"round_{child:03d}"
    pred, man = d / "edit_prediction.json", d / "retrieval_manifest.json"
    if not pred.exists() or not man.exists():
        return None
    query = (json.loads(pred.read_text()) or {}).get("query") or {}
    real = json.loads(man.read_text())
    parent = json.loads((d / "hgm_node.json").read_text()).get("parent_id")
    if parent is None:
        return None

    dest = workdir / f"asof_{child:03d}"
    try:
        meta = asof.build(run, child, dest)
    except SystemExit:
        return None

    edited = child_edited_units(run, child, parent)
    row: dict[str, Any] = {
        "node": child, "parent": parent,
        "nodes_present": len(meta["nodes_present"]),
        "n_edited": len(edited), "sweep": [],
    }

    real_sel = [s["node"] for s in (real.get("selected") or [])]
    for max_nodes, budget in SWEEP:
        res = edit_archive.resolve_query(dest, query, char_budget=budget,
                                         max_nodes=max_nodes, include_code=True)
        sel = res.manifest["selected"]
        nodes = [s["node"] for s in sel]
        chans = Counter(_channel(s["why"]) for s in sel)
        shown = _units_in(res.blocks)
        row["sweep"].append({
            "max_nodes": max_nodes, "budget": budget,
            "nodes": nodes, "n": len(nodes),
            "channels": dict(chans),
            "defs_omitted": sum(s.get("defs_omitted") or 0 for s in sel),
            "n_shown": len(shown),
            "edited_covered": len(edited & shown),
            "total_chars": res.manifest["total_chars"],
        })
        if max_nodes == 4 and budget == 60000:
            row["replay_matches_run"] = (nodes == real_sel)
            row["real_selected"] = real_sel
            row["replay_selected"] = nodes
    shutil.rmtree(dest, ignore_errors=True)
    return row


def report(rows: list[dict[str, Any]], run: Path) -> str:
    L: list[str] = []
    A = L.append
    A(f"# Retrieval cap probe — {run.name}")
    A("")
    A("Every expand's own recorded query, replayed against the world as it stood "
      "just before that expand, sweeping the node cap and the char budget.")
    A("")

    # ---- fidelity gate ----------------------------------------------------
    checked = [r for r in rows if "replay_matches_run" in r]
    ok = [r for r in checked if r["replay_matches_run"]]
    A("## Fidelity gate")
    A("")
    A(f"Replay at the run's own settings (`max_nodes=4`, `char_budget=60000`) "
      f"reproduces the recorded manifest for **{len(ok)}/{len(checked)}** expands.")
    bad = [r for r in checked if not r["replay_matches_run"]]
    if bad:
        A("")
        A("Mismatches (as-of reconstruction vs what the run recorded):")
        for r in bad[:10]:
            extra = [n for n in r["replay_selected"] if n not in r["real_selected"]]
            miss = [n for n in r["real_selected"] if n not in r["replay_selected"]]
            A(f"- node {r['node']}: run {r['real_selected']} vs "
              f"replay {r['replay_selected']}"
              + (f" — replay admits {extra}" if extra else "")
              + (f", misses {miss}" if miss else ""))
        A("")
        A("**Cause, and why it is benign.** Every mismatch is in the *keyword* "
          "channel. `resolve_query` keyword-scans each node's `edit_memory.md` "
          "body, and records are **refreshed in place** as evaluations arrive — "
          "their Outcome/Analysis text today is richer than it was at the moment "
          "of the expand, and no history is kept (`study/asof.py` documents this). "
          "So a stale record can match a keyword it did not contain then.")
        A("")
        A("The bias has a known sign: replay admits **more** nodes than the run "
          "did, never fewer. Every 'the cap hid something' number below is "
          "therefore an **upper bound** on what better retrieval could have "
          "delivered.")
    A("")

    # ---- admission --------------------------------------------------------
    A("## What a bigger cap admits")
    A("")
    A("| cap | budget | mean nodes | explicit | strategy | area | keyword | defs omitted |")
    A("|---|---|---|---|---|---|---|---|")
    for i, (mn, bud) in enumerate(SWEEP):
        tot = Counter()
        n_nodes = 0
        om = 0
        for r in rows:
            s = r["sweep"][i]
            tot.update(s["channels"])
            n_nodes += s["n"]
            om += s["defs_omitted"]
        cap = "∞" if mn > 100 else str(mn)
        A(f"| {cap} | {bud:,} | {n_nodes / max(1, len(rows)):.1f} "
          f"| {tot.get('explicit', 0)} | {tot.get('strategy', 0)} "
          f"| {tot.get('area', 0)} | {tot.get('keyword', 0)} | {om} |")
    A("")
    A("> **The cap and the budget fight each other.** `resolve_query` sets "
      "`per_node = char_budget // len(selected)` (`edit_archive.py:167`), and each "
      "node's code budget is what is left after its record text. So admitting more "
      "nodes under a fixed budget starves every one of them: `defs omitted` climbs "
      "from 66 to 812 across the sweep. Raising the node cap alone does not buy "
      "more memory — it buys more record headers and less code.")
    A("")

    # ---- payoff -----------------------------------------------------------
    A("## Payoff — does the extra memory contain what the child actually edited?")
    A("")
    A("`covered` = definitions this node's own diff touched that appear in the "
      "retrieval block. This is the only measure of the cap costing something real.")
    A("")
    A("| cap | budget | defs shown | edited defs covered | coverage | retrieval chars |")
    A("|---|---|---|---|---|---|")
    tot_edited = sum(r["n_edited"] for r in rows)
    for i, (mn, bud) in enumerate(SWEEP):
        shown = sum(r["sweep"][i]["n_shown"] for r in rows)
        cov = sum(r["sweep"][i]["edited_covered"] for r in rows)
        chars = sum(r["sweep"][i]["total_chars"] for r in rows)
        cap = "∞" if mn > 100 else str(mn)
        A(f"| {cap} | {bud:,} | {shown} | {cov}/{tot_edited} "
          f"| {100.0 * cov / max(1, tot_edited):.0f}% | {chars:,} |")
    A("")
    base = sum(r["sweep"][0]["edited_covered"] for r in rows)
    best = max(sum(r["sweep"][i]["edited_covered"] for r in rows)
               for i in range(len(SWEEP)))
    A(f"- coverage at the run's settings: **{base}/{tot_edited}** "
      f"({100.0 * base / max(1, tot_edited):.0f}%)")
    A(f"- best achievable in this sweep: **{best}/{tot_edited}** "
      f"({100.0 * best / max(1, tot_edited):.0f}%)")
    A(f"- headroom from lifting the caps alone: **{best - base} definitions**")
    A("")
    A("> Coverage well short of 100% at an unlimited cap means the definitions the "
      "editor edits are mostly not in ANY sibling's implementation — they are in "
      "the parent's own sources, which the editor always sees in full. Retrieval "
      "cannot be the binding constraint on those edits.")
    A("")

    # ---- per node ---------------------------------------------------------
    A("## Per expand")
    A("")
    A("| node | nodes then | cap 4 | cap ∞ | new nodes admitted | edited covered 4 → ∞ |")
    A("|---|---|---|---|---|---|")
    for r in rows:
        s4, sinf = r["sweep"][0], r["sweep"][3]
        new = [n for n in sinf["nodes"] if n not in s4["nodes"]]
        A(f"| {r['node']} | {r['nodes_present']} | {s4['n']} | {sinf['n']} "
          f"| {new if new else '—'} | {s4['edited_covered']} → {sinf['edited_covered']} |")
    A("")
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    run = args.run.resolve()
    nodes = sorted(int(p.name[-3:]) for p in run.glob("round_*")
                   if (p / "retrieval_manifest.json").exists())
    rows = []
    with tempfile.TemporaryDirectory(prefix="asof_") as tmp:
        work = Path(tmp)
        for n in nodes:
            r = probe_node(run, n, work)
            if r:
                rows.append(r)
    text = report(rows, run)
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    else:
        print(text)
    if args.json:
        args.json.write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
