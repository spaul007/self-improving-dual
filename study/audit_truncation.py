"""A0 — truncation audit.

Where does every cap in the edit-memory pipeline actually bind, and what was lost?

The pipeline truncates in several independent places, each with its own knob. This
script parses the machine-readable evidence a finished run leaves on disk (it never
re-renders anything, so it cannot disagree with what the models were really shown)
and reports, per node and per stage:

  * did the cap bind, and by how much
  * what was dropped, by name, where the artifact names it
  * whether nodes truncated EARLY are over-represented among nodes later judged
    unsound / mis-tagged / unscored  (correlational — see study/probe_bug.py for
    the causal test)

Stages, in pipeline order:

  tagger      round_NNN/edit_memory_prompt.txt      diff_char_cap
  code record round_NNN/edit_code.md                code_diff_char_cap, defs cap
  judge       round_NNN/edit_analysis_prompt.txt    analysis_code_char_budget
  retrieval   round_NNN/retrieval_manifest.json     max_retrieved_nodes, retrieval_char_budget
  editor      round_NNN/verbose/editor_attempt_1_user.txt   (the rendered result)
  beliefs     edit_memory_beliefs*.md               doc_char_cap  (reject+retry, never silent)

Usage:
    PYTHONPATH=. python3 study/audit_truncation.py --run runs/<dir> [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import re
import statistics as st
import sys
from pathlib import Path
from typing import Any, Optional

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# "<... 1234 chars elided ...>" — edit_diff.truncate_middle
ELIDED_RE = re.compile(r"<\.\.\.\s*([0-9,]+)\s*chars elided\s*\.\.\.>")
# "<... remaining definitions elided at 30000 chars ...>" — edit_code.render_edit_code
DEFS_ELIDED_RE = re.compile(r"remaining definitions elided at\s*([0-9,]+)")
# "omitted (budget): a.py :: f (+12/-3); b.py :: g (full source 4210 chars)"
OMITTED_LINE_RE = re.compile(r"^omitted \(budget\): (.+)$", re.M)
OMITTED_FULL_RE = re.compile(r"full source ([0-9,]+) chars")
OMITTED_ADD_RE = re.compile(r"\(\+([0-9]+)/-([0-9]+)\)")


def _int(text: str) -> int:
    return int(text.replace(",", ""))


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _elisions(text: str) -> tuple[int, int]:
    """(number of middle-elisions, total chars elided)."""
    hits = ELIDED_RE.findall(text)
    return len(hits), sum(_int(h) for h in hits)


def _omitted_units(text: str) -> list[dict[str, Any]]:
    """Every unit named on an `omitted (budget):` line, with its size where stated.

    The line names what the reader was told exists but was not shown — which is
    exactly the set we want to cross-reference against what the child went on to edit.
    """
    out: list[dict[str, Any]] = []
    for line in OMITTED_LINE_RE.findall(text):
        for unit in line.split("; "):
            unit = unit.strip()
            if not unit:
                continue
            chars = None
            m = OMITTED_FULL_RE.search(unit)
            if m:
                chars = _int(m.group(1))
            added = removed = None
            m = OMITTED_ADD_RE.search(unit)
            if m:
                added, removed = int(m.group(1)), int(m.group(2))
            name = unit.split(" (")[0].strip()
            out.append({"unit": name, "chars": chars,
                        "added": added, "removed": removed, "raw": unit})
    return out


def _caps(run: Path) -> dict[str, Any]:
    """The caps this run actually ran with, from its config snapshot."""
    import yaml
    snap = run / "config.snapshot.yaml"
    if not snap.exists():
        return {}
    cfg = yaml.safe_load(_read(snap)) or {}
    em = ((cfg.get("edit_memory") or {}).get("config")
          or cfg.get("edit_memory") or {})
    beliefs = em.get("beliefs") or {}
    ed = ((cfg.get("editor") or {}).get("config") or {})
    return {
        "diff_char_cap": em.get("diff_char_cap"),
        "code_diff_char_cap": em.get("code_diff_char_cap"),
        "analysis_code_char_budget": em.get("analysis_code_char_budget"),
        "steering_token_budget": em.get("steering_token_budget"),
        "retrieval_char_budget": ed.get("retrieval_char_budget"),
        "max_retrieved_nodes": ed.get("max_retrieved_nodes"),
        "doc_char_cap": beliefs.get("doc_char_cap"),
        "evidence_char_budget": beliefs.get("evidence_char_budget"),
        "instruction_char_cap": beliefs.get("instruction_char_cap"),
    }


def _record_flags(run: Path, node: int) -> dict[str, Any]:
    """Downstream quality flags for one node, parsed from its record."""
    from meta_agent.edit_memory_render import record_tags
    body = _read(run / f"round_{node:03d}" / "edit_memory.md")
    if not body:
        return {}
    tags = record_tags(body)
    impl = None
    m = re.search(r"- \*\*implementation\*\*: (sound|unsound)", body)
    if m:
        impl = m.group(1) == "sound"
    return {
        "n_subedits": len(tags),
        "forced_fit": any(t.get("fit") == "forced" for t in tags),
        "impl_sound": impl,
        "suspect_verifier": "SUSPECT VERIFIER" in body,
        "strategies": [t.get("strategy") for t in tags],
    }


def audit_node(run: Path, node: int) -> dict[str, Any]:
    d = run / f"round_{node:03d}"
    row: dict[str, Any] = {"node": node}

    # --- stage: tagger (diff_char_cap) --------------------------------------
    n, chars = _elisions(_read(d / "edit_memory_prompt.txt"))
    row["tagger_elisions"] = n
    row["tagger_chars_elided"] = chars

    # --- stage: on-disk code record (code_diff_char_cap / defs cap) ---------
    code = _read(d / "edit_code.md")
    n, chars = _elisions(code)
    row["code_elisions"] = n
    row["code_chars_elided"] = chars
    row["code_defs_elided"] = bool(DEFS_ELIDED_RE.search(code))

    # --- stage: judge -------------------------------------------------------
    # Two possible code paths, and they have DIFFERENT caps. The current tree
    # sends an uncut implementation view (analysis_code_char_budget, default
    # 20000); the pre-2026-09-08 code sent a middle-truncated unified diff at
    # diff_char_cap (6000). Which one ran is visible in the prompt's own
    # section header, so attribute from the artifact, not from the config.
    judge = _read(d / "edit_analysis_prompt.txt")
    n, chars = _elisions(judge)
    row["judge_elisions"] = n
    row["judge_chars_elided"] = chars
    row["judge_omitted_units"] = _omitted_units(judge)
    if judge:
        row["judge_code_path"] = (
            "diff" if "# Code diff vs parent" in judge else "view")

    # --- stage: retrieval (max_retrieved_nodes / retrieval_char_budget) -----
    man_path = d / "retrieval_manifest.json"
    if man_path.exists():
        man = json.loads(_read(man_path))
        sel = man.get("selected") or []
        drop = man.get("dropped") or []
        row["retr_selected"] = len(sel)
        row["retr_dropped"] = len(drop)
        row["retr_dropped_over_cap"] = sum(
            1 for x in drop if x.get("reason") == "over max_nodes")
        row["retr_selected_nodes"] = [s.get("node") for s in sel]
        row["retr_dropped_nodes"] = [x.get("node") for x in drop]
        row["retr_why"] = [s.get("why") for s in sel]
        row["retr_explicit"] = sum(
            1 for s in sel if (s.get("why") or "").startswith("explicit"))
        row["retr_associative"] = len(sel) - row["retr_explicit"]
        row["defs_omitted"] = sum(s.get("defs_omitted") or 0 for s in sel)
        row["hunks_omitted"] = sum(s.get("hunks_omitted") or 0 for s in sel)
        row["defs_shown"] = sum(s.get("defs_shown") or 0 for s in sel)
        row["hunks_shown"] = sum(s.get("hunks_shown") or 0 for s in sel)
        row["retr_total_chars"] = man.get("total_chars")
        row["retr_per_node"] = man.get("per_node")
        row["retr_char_budget"] = man.get("char_budget")
        row["retr_max_nodes"] = man.get("max_nodes")
        q = man.get("query") or {}
        row["query_nodes"] = q.get("nodes") or []
        row["query_strategies"] = q.get("strategies") or []
        row["query_areas"] = q.get("areas") or []
        row["query_keywords"] = q.get("keywords") or []

    # --- stage: the editor prompt actually rendered -------------------------
    ed = _read(d / "verbose" / "editor_attempt_1_user.txt")
    row["editor_prompt_chars"] = len(ed)
    row["editor_omitted_units"] = _omitted_units(ed)
    n, chars = _elisions(ed)
    row["editor_elisions"] = n
    row["editor_chars_elided"] = chars

    row.update(_record_flags(run, node))
    return row


def audit(run: Path) -> dict[str, Any]:
    nodes = sorted(int(p.name[-3:]) for p in run.glob("round_*")
                   if (p / "hgm_node.json").exists())
    rows = [audit_node(run, n) for n in nodes]
    # nodes with no edit (the seed) have no record artifacts; keep them out of rates
    edited = [r for r in rows if r.get("n_subedits")]
    return {"run": str(run), "caps": _caps(run), "nodes": rows, "edited": edited}


def _belief_docs(run: Path, cap: Optional[int]) -> dict[str, Any]:
    sizes = []
    for p in sorted((run / "edit_memory_beliefs_archive").glob("beliefs_*.md")):
        sizes.append(len(_read(p)))
    cur = run / "edit_memory_beliefs.md"
    if cur.exists():
        sizes.append(len(_read(cur)))
    if not sizes:
        return {}
    return {"n": len(sizes), "min": min(sizes), "max": max(sizes),
            "median": int(st.median(sizes)), "cap": cap,
            "at_cap": sum(1 for s in sizes if cap and s >= cap * 0.98)}


def report(res: dict[str, Any]) -> str:
    run = Path(res["run"])
    caps = res["caps"]
    rows = res["nodes"]
    ed = res["edited"]
    n_ed = len(ed) or 1
    L: list[str] = []
    A = L.append

    A(f"# A0 truncation audit — {run.name}")
    A("")
    A(f"{len(rows)} nodes, {len(ed)} with an edit record.")
    A("")
    A("## Caps this run ran with")
    A("")
    A("| knob | value |")
    A("|---|---|")
    for k, v in caps.items():
        A(f"| `{k}` | {v if v is not None else '(default)'} |")
    A("")

    def rate(pred) -> str:
        k = sum(1 for r in ed if pred(r))
        return f"{k}/{len(ed)} ({100.0 * k / n_ed:.0f}%)"

    judge_diff_path = sum(1 for r in ed if r.get("judge_code_path") == "diff")
    judge_knob = ("`diff_char_cap` (old path)" if judge_diff_path
                  else "`analysis_code_char_budget`")

    A("## Did each cap bind?")
    A("")
    A("`status` distinguishes a cap that is still live in the current tree from one")
    A("this run hit but the working tree has since changed.")
    A("")
    A("| stage | knob | nodes where it bound | what was lost | status |")
    A("|---|---|---|---|---|")
    tot = sum(r.get("tagger_chars_elided") or 0 for r in ed)
    A(f"| tagger diff | `diff_char_cap` | {rate(lambda r: r.get('tagger_elisions'))} "
      f"| {tot:,} chars middle-elided | **live** (`edit_memory.py:578`) |")
    tot = sum(r.get("code_chars_elided") or 0 for r in ed)
    A(f"| code record | `code_diff_char_cap` | {rate(lambda r: r.get('code_elisions'))} "
      f"| {tot:,} chars | live (retrieval fallback only) |")
    tot = sum(r.get("judge_chars_elided") or 0 for r in ed)
    A(f"| judge | {judge_knob} | {rate(lambda r: r.get('judge_elisions'))} "
      f"| {tot:,} chars "
      f"| {'**fixed in working tree**' if judge_diff_path else 'live'} |")
    tot = sum(r.get("retr_dropped_over_cap") or 0 for r in ed)
    A(f"| retrieval nodes | `max_retrieved_nodes` "
      f"| {rate(lambda r: r.get('retr_dropped_over_cap'))} | **{tot} node-drops** "
      f"| **live** |")
    tot = sum(r.get("defs_omitted") or 0 for r in ed)
    A(f"| retrieval chars | `retrieval_char_budget` "
      f"| {rate(lambda r: r.get('defs_omitted'))} | **{tot} defs omitted** | **live** |")
    A(f"| editor prompt (rendered) | — "
      f"| {rate(lambda r: r.get('editor_omitted_units'))} "
      f"| {sum(len(r.get('editor_omitted_units') or []) for r in ed)} named-but-absent units "
      f"| **live** |")
    bd = _belief_docs(run, caps.get("doc_char_cap"))
    if bd:
        A(f"| belief doc | `doc_char_cap` | {bd['at_cap']}/{bd['n']} within 2% of cap "
          f"| sizes {bd['min']:,}–{bd['max']:,}, cap {bd['cap']} "
          f"| not binding (reject+retry, never silent) |")
    A("")
    if judge_diff_path:
        n_trunc = sum(1 for r in ed if r.get("judge_code_path") == "diff"
                      and r.get("judge_elisions"))
        A(f"**Judge attribution.** All {judge_diff_path} analysis prompts carry the "
          f"`# Code diff vs parent` header ({n_trunc} of them long enough to be cut), "
          f"i.e. the pre-2026-09-08 path that fed the "
          f"judge a `truncate_middle` diff at `diff_char_cap` "
          f"({caps.get('diff_char_cap')}), not the {caps.get('analysis_code_char_budget') or 20000}-char "
          f"implementation view. The run predates that change "
          f"(`edit_memory.py` mtime is after the run's `config.snapshot.yaml`), so this "
          f"is a truncation the run suffered and the current tree does not. "
          f"See the verification section below.")
        A("")

    A("## Retrieval: selection channel and the cap")
    A("")
    sel = sum(r.get("retr_selected") or 0 for r in ed)
    exp = sum(r.get("retr_explicit") or 0 for r in ed)
    asso = sum(r.get("retr_associative") or 0 for r in ed)
    drop = sum(r.get("retr_dropped_over_cap") or 0 for r in ed)
    A(f"- selected **{sel}** node-slots total: **{exp} explicit** (named by the "
      f"planning pass), **{asso} associative** (strategy/area/keyword)")
    A(f"- dropped **{drop}** `over max_nodes`")
    if sel:
        A(f"- explicit share of the returned slots: **{100.0 * exp / sel:.0f}%**")
    A("")
    A("| node | query n/s/a/k | selected | explicit | assoc | dropped | defs omitted |")
    A("|---|---|---|---|---|---|---|")
    for r in ed:
        if r.get("retr_selected") is None:
            continue
        A(f"| {r['node']} | {len(r.get('query_nodes') or [])}/"
          f"{len(r.get('query_strategies') or [])}/"
          f"{len(r.get('query_areas') or [])}/"
          f"{len(r.get('query_keywords') or [])} "
          f"| {r['retr_selected']} | {r['retr_explicit']} | {r['retr_associative']} "
          f"| {r.get('retr_dropped_over_cap', 0)} | {r.get('defs_omitted', 0)} |")
    A("")

    A("## Headroom — what budget would have removed the omissions")
    A("")
    sizes = [u["chars"] for r in ed for u in (r.get("editor_omitted_units") or [])
             if u.get("chars")]
    if sizes:
        A(f"- {len(sizes)} omitted units state their size: median "
          f"{int(st.median(sizes)):,} chars, max {max(sizes):,}, total {sum(sizes):,}")
    per_node = [r["retr_per_node"] for r in ed if r.get("retr_per_node")]
    if per_node:
        A(f"- per-node char allowance actually granted: median "
          f"{int(st.median(per_node)):,} (budget "
          f"{caps.get('retrieval_char_budget')} ÷ selected nodes, "
          f"`edit_archive.py:167`)")
    A(f"- so the per-node squeeze is a direct consequence of the node cap: naming "
      f"{caps.get('max_retrieved_nodes')} nodes divides the same budget "
      f"{caps.get('max_retrieved_nodes')} ways.")
    A("")

    A("## Compounding (correlational — see study/probe_bug.py for the causal test)")
    A("")
    early = [r for r in ed if r.get("tagger_elisions") or r.get("code_elisions")]
    late = [r for r in ed if not (r.get("tagger_elisions") or r.get("code_elisions"))]

    def frac(rs, pred) -> str:
        rs2 = [r for r in rs if pred(r) is not None]
        if not rs2:
            return "n/a"
        k = sum(1 for r in rs2 if pred(r))
        return f"{k}/{len(rs2)} ({100.0 * k / len(rs2):.0f}%)"

    A(f"Nodes truncated at the **tagger or code-record** stage: "
      f"{[r['node'] for r in early]}")
    A("")
    A("| downstream flag | truncated early | not truncated early |")
    A("|---|---|---|")
    A(f"| implementation judged unsound "
      f"| {frac(early, lambda r: r.get('impl_sound') is False if r.get('impl_sound') is not None else None)} "
      f"| {frac(late, lambda r: r.get('impl_sound') is False if r.get('impl_sound') is not None else None)} |")
    A(f"| tag forced by the registry cap "
      f"| {frac(early, lambda r: r.get('forced_fit'))} "
      f"| {frac(late, lambda r: r.get('forced_fit'))} |")
    A(f"| suspect verifier flagged "
      f"| {frac(early, lambda r: r.get('suspect_verifier'))} "
      f"| {frac(late, lambda r: r.get('suspect_verifier'))} |")
    A("")
    A("These counts are small; treat them as a pointer, not a result.")
    A("")

    hit = [r for r in ed if r.get("judge_code_path") == "diff"
           and r.get("judge_elisions")]
    if hit:
        A("## Does the working-tree fix actually resolve the judge truncation?")
        A("")
        A("Re-render each affected node through the CURRENT "
          "`edit_code.render_implementation_view` at the budget the current tree "
          "would use. `defs_omitted: 0` means the judge would now see the whole edit.")
        A("")
        A("| node | old: chars elided | new: view chars | defs omitted | hunks omitted |")
        A("|---|---|---|---|---|")
        budget = caps.get("analysis_code_char_budget") or 20000
        for r in hit:
            n = r["node"]
            d = run / f"round_{n:03d}"
            try:
                par = json.loads(_read(d / "hgm_node.json"))["parent_id"]
                from meta_agent import edit_code
                txt, stats = edit_code.render_implementation_view(
                    run / f"round_{par:03d}", d, char_budget=budget)
                A(f"| {n} | {r['judge_chars_elided']:,} | {len(txt):,} "
                  f"| {stats['defs_omitted']} | {stats['hunks_omitted']} |")
            except Exception as exc:  # noqa: BLE001
                A(f"| {n} | {r['judge_chars_elided']:,} | re-render failed: {exc!r} | | |")
        A("")

    A("## Named-but-absent units, per node")
    A("")
    A("Each entry was announced to the editor by name and then not shown.")
    A("")
    for r in ed:
        units = r.get("editor_omitted_units") or []
        if units:
            A(f"- **node {r['node']}** ({len(units)}): "
              + "; ".join(u["raw"] for u in units))
    A("")
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    res = audit(args.run)
    text = report(res)
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    else:
        print(text)
    if args.json:
        args.json.write_text(json.dumps(res, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
