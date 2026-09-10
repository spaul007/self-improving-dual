"""A1 — does the memory chain deliver the right thing?

Six links, of which this script measures the four that leave machine-readable
traces in a finished run:

  L4  the planning pass asks for the right thing   edit_prediction.json -> query
  L5a retrieval SELECTS it                          retrieval_manifest.json
  L5b retrieval SHOWS it, uncut                     the rendered editor prompt
  L6  the editor USES it                            parent->child diff

Everything is parsed from what the models were actually shown
(`verbose/editor_attempt_1_user.txt`), not re-rendered, so it cannot disagree
with the run.

Usage:
    PYTHONPATH=. python3 study/audit_chain.py --run <snapshot dir> [--out report.md]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Optional

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

RETRIEVED_HDR = "## Retrieved records and implementations"
NODE_HDR_RE = re.compile(r"^### Retrieved node (\d+) \(([^)]*)\)", re.M)
# "### workflow.py :: _repair_plan (function, changed)  +28/-0"
UNIT_RE = re.compile(
    r"^### ([^\s:]+\.[A-Za-z0-9]+) :: ([^\s(]+) \(([^)]*)\)(?:\s+\+(\d+)/-(\d+))?", re.M)
OMITTED_RE = re.compile(r"^omitted \(budget\): (.+)$", re.M)
# "workflow.py :: _build_day_audit_report (full source 8027 chars)"
OMIT_UNIT_RE = re.compile(r"([^\s;]+\.[A-Za-z0-9]+) :: ([^\s(]+)")


def _read(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def parse_retrieved(prompt: str) -> dict[str, Any]:
    """What the editor was actually shown from memory, by node."""
    i = prompt.find(RETRIEVED_HDR)
    if i < 0:
        return {"present": False, "nodes": [], "shown": set(), "omitted": set()}
    block = prompt[i:]
    marks = [(m.start(), int(m.group(1)), m.group(2))
             for m in NODE_HDR_RE.finditer(block)]
    nodes: list[dict[str, Any]] = []
    shown: set[tuple[str, str]] = set()
    omitted: set[tuple[str, str]] = set()
    for k, (pos, nid, why) in enumerate(marks):
        end = marks[k + 1][0] if k + 1 < len(marks) else len(block)
        seg = block[pos:end]
        units = {(m.group(1), m.group(2)) for m in UNIT_RE.finditer(seg)}
        om: set[tuple[str, str]] = set()
        for line in OMITTED_RE.findall(seg):
            for u in line.split("; "):
                m = OMIT_UNIT_RE.search(u)
                if m:
                    om.add((m.group(1), m.group(2)))
        # a unit named on the omitted line is NOT shown, even if a header exists
        units -= om
        shown |= units
        omitted |= om
        nodes.append({"node": nid, "why": why,
                      "n_shown": len(units), "n_omitted": len(om),
                      "shown": sorted(units), "omitted": sorted(om)})
    return {"present": True, "nodes": nodes, "shown": shown, "omitted": omitted}


def child_edited_units(run: Path, node: int, parent: int) -> set[tuple[str, str]]:
    """The top-level definitions this node's own edit actually touched."""
    from meta_agent import edit_code
    from meta_agent.edit_diff import changed_mutable_files, read_text
    pdir, cdir = run / f"round_{parent:03d}", run / f"round_{node:03d}"
    out: set[tuple[str, str]] = set()
    try:
        rels = changed_mutable_files(pdir, cdir)
    except Exception:  # noqa: BLE001
        return out
    for rel in rels:
        child = read_text(cdir / "task_agent" / rel) or ""
        par = read_text(pdir / "task_agent" / rel)
        try:
            for u in edit_code.group_hunks_by_def(rel, child, par):
                out.add((u["file"], u["def"]))
        except Exception:  # noqa: BLE001
            continue
    return out


def audit_node(run: Path, node: int, registry: dict[str, Any]) -> Optional[dict[str, Any]]:
    d = run / f"round_{node:03d}"
    nj = d / "hgm_node.json"
    if not nj.exists():
        return None
    parent = json.loads(_read(nj)).get("parent_id")
    if parent is None:
        return None

    row: dict[str, Any] = {"node": node, "parent": parent}

    # --- L4: did the planning pass name real registry ids? ------------------
    pred = d / "edit_prediction.json"
    if pred.exists():
        q = (json.loads(_read(pred)) or {}).get("query") or {}
        strat = q.get("strategies") or []
        areas = q.get("areas") or []
        known_s = set((registry.get("strategies") or {}))
        known_a = set((registry.get("areas") or {}))
        row["q_strategies"] = strat
        row["q_areas"] = areas
        row["q_bad_strategies"] = [s for s in strat if s not in known_s]
        row["q_bad_areas"] = [a for a in areas if a not in known_a]
        row["q_nodes"] = q.get("nodes") or []

    # --- L5a: what retrieval selected, and what it starved ------------------
    man = d / "retrieval_manifest.json"
    if man.exists():
        m = json.loads(_read(man))
        sel = m.get("selected") or []
        drop = m.get("dropped") or []

        def chan(why: str) -> str:
            why = why or "?"
            if why.startswith("explicit"):
                return "explicit"
            if why.startswith("strategy"):
                return "strategy"
            if why.startswith("area"):
                return "area"
            if why.startswith("keyword"):
                return "keyword"
            return "other"

        row["sel_by_channel"] = Counter(chan(s.get("why")) for s in sel)
        row["drop_by_channel"] = Counter(chan(x.get("why")) for x in drop)

    # --- L5b: what the editor was actually shown ----------------------------
    prompt = _read(d / "verbose" / "editor_attempt_1_user.txt")
    ret = parse_retrieved(prompt)
    row["retrieved_present"] = ret["present"]
    row["units_shown"] = ret["shown"]
    row["units_omitted"] = ret["omitted"]
    row["n_units_shown"] = len(ret["shown"])
    row["n_units_omitted"] = len(ret["omitted"])

    # --- L6: what the editor then edited ------------------------------------
    edited = child_edited_units(run, node, parent)
    row["units_edited"] = edited
    row["n_edited"] = len(edited)
    row["edited_from_shown"] = edited & ret["shown"]
    row["edited_from_omitted"] = edited & ret["omitted"]
    row["edited_novel"] = edited - ret["shown"] - ret["omitted"]
    return row


def audit(run: Path) -> list[dict[str, Any]]:
    reg_path = run / "edit_memory_registry.json"
    registry = json.loads(_read(reg_path)) if reg_path.exists() else {}
    nodes = sorted(int(p.name[-3:]) for p in run.glob("round_*")
                   if (p / "hgm_node.json").exists())
    return [r for r in (audit_node(run, n, registry) for n in nodes) if r]


def report(rows: list[dict[str, Any]], run: Path) -> str:
    L: list[str] = []
    A = L.append
    A(f"# A1 chain audit — {run.name}")
    A("")
    A(f"{len(rows)} expand events with a parent.")
    A("")

    # ---- L4 ---------------------------------------------------------------
    bad = [r for r in rows if r.get("q_bad_strategies") or r.get("q_bad_areas")]
    A("## L4 — did the planning pass name real registry ids?")
    A("")
    A(f"- invented ids in **{len(bad)}/{len(rows)}** expands")
    for r in bad:
        A(f"  - node {r['node']}: strategies {r['q_bad_strategies']} "
          f"areas {r['q_bad_areas']}")
    if not bad:
        A("  - (the `## Registry ids for the memory query` block appears to be working)")
    A("")

    # ---- L5a --------------------------------------------------------------
    A("## L5a — retrieval selection: which channel survives the node cap?")
    A("")
    sel = Counter()
    drop = Counter()
    for r in rows:
        sel.update(r.get("sel_by_channel") or {})
        drop.update(r.get("drop_by_channel") or {})
    chans = ["explicit", "strategy", "area", "keyword", "other"]
    A("| channel | selected | dropped `over max_nodes` | survival |")
    A("|---|---|---|---|")
    for c in chans:
        s, dd = sel.get(c, 0), drop.get(c, 0)
        if not (s or dd):
            continue
        tot = s + dd
        A(f"| {c} | {s} | {dd} | {100.0 * s / tot:.0f}% |")
    tot_s, tot_d = sum(sel.values()), sum(drop.values())
    A(f"| **all** | **{tot_s}** | **{tot_d}** | "
      f"**{100.0 * tot_s / max(1, tot_s + tot_d):.0f}%** |")
    A("")
    assoc_s = sum(sel.get(c, 0) for c in ("strategy", "area", "keyword"))
    assoc_d = sum(drop.get(c, 0) for c in ("strategy", "area", "keyword"))
    A(f"> Explicit ids (nodes the planning pass already named) take precedence in "
      f"`edit_archive.resolve_query`. Associative matches — the part that could "
      f"surface something the planner did NOT think of — were selected "
      f"{assoc_s} times and dropped {assoc_d} times "
      f"({100.0 * assoc_s / max(1, assoc_s + assoc_d):.0f}% survival).")
    A("")

    # ---- L5b / L6 ---------------------------------------------------------
    A("## L5b / L6 — was the code shown, and did the editor use it?")
    A("")
    A("`shown` = top-level definitions rendered in the retrieval block. "
      "`omitted` = named on an `omitted (budget):` line, i.e. announced and withheld. "
      "`edited` = definitions this node's own diff touched.")
    A("")
    A("| node | shown | omitted | edited | edited∩shown | edited∩omitted | novel |")
    A("|---|---|---|---|---|---|---|")
    for r in rows:
        A(f"| {r['node']} | {r['n_units_shown']} | {r['n_units_omitted']} "
          f"| {r['n_edited']} | {len(r['edited_from_shown'])} "
          f"| {len(r['edited_from_omitted'])} | {len(r['edited_novel'])} |")
    A("")
    tot_ed = sum(r["n_edited"] for r in rows)
    tot_shown = sum(len(r["edited_from_shown"]) for r in rows)
    tot_om = sum(len(r["edited_from_omitted"]) for r in rows)
    tot_nov = sum(len(r["edited_novel"]) for r in rows)
    A(f"- definitions edited across all expands: **{tot_ed}**")
    A(f"  - also shown in retrieved memory: **{tot_shown}** "
      f"({100.0 * tot_shown / max(1, tot_ed):.0f}%)")
    A(f"  - named on an `omitted (budget):` line — announced but withheld: "
      f"**{tot_om}** ({100.0 * tot_om / max(1, tot_ed):.0f}%)")
    A(f"  - not in the retrieval block at all: **{tot_nov}** "
      f"({100.0 * tot_nov / max(1, tot_ed):.0f}%)")
    A("")
    A("> **Read this carefully.** \"not in the retrieval block\" does NOT mean the "
      "editor was flying blind: `AgentEditor._format_current_sources` always shows "
      "the parent's mutable files in full. Retrieval only adds OTHER nodes' "
      "implementations. So this row means the edit landed on code the editor could "
      "already see, and memory contributed nothing to that particular definition.")
    A("")
    A(f"> Only **{tot_om}** of {tot_ed} edited definitions had been withheld by the "
      f"char budget. So the `omitted (budget):` truncation — frequent as it is — "
      f"mostly withheld code the editor was not going to touch. The char budget is "
      f"**not** the load-bearing failure; the node cap and the explicit-echo "
      f"behaviour in L5a are.")
    A("")
    hit = [r for r in rows if r["edited_from_omitted"]]
    if hit:
        A("### Definitions the editor edited that memory had withheld")
        A("")
        A("These are the cleanest evidence that the char budget cost something: the "
          "editor was told the definition existed, was not shown it, and edited it anyway.")
        A("")
        for r in hit:
            A(f"- **node {r['node']}**: "
              + ", ".join(f"`{f} :: {d}`" for f, d in sorted(r["edited_from_omitted"])))
        A("")
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    rows = audit(args.run)
    text = report(rows, args.run)
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
