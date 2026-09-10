"""Part B — counterfactual editor ablation.

Replay a real expand, rebuild the world as of that moment, vary ONLY what memory
says, and resample the editor. Everything else — system prompt, feedback digest,
project context, the parent's own sources — is byte-identical across arms.

Arms:
  M0  no edit memory at all      `_edit_memory = None`, single-call editor
  M3  production                 belief steering + planning pass + retrieval
  P   placebo                    M3's structure, outcome labels permuted
  D   oracle (A2)                M3 with a forced query and a generous budget

`--verify` reconstructs M3 and diffs it, section by section, against what the run
actually sent (`verbose/editor_propose_user.txt`, `verbose/editor_attempt_1_user.txt`).
That gate runs before any spending: if the assembly is wrong, nothing downstream
is worth paying for.

Usage:
    PYTHONPATH=. python3 study/gen_edits.py --run <snapshot> --verify --children 10,26
    PYTHONPATH=. python3 study/gen_edits.py --run <snapshot> --generate \\
        --children 5,8,10,... --arms M0,M3,P --samples 6 --out-dir study/out/edits
"""
from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Optional

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from study import asof  # noqa: E402


# --------------------------------------------------------------------------- #
# edit-memory stubs: only what HGMManager._render_expand_context touches
# --------------------------------------------------------------------------- #
class BeliefStub:
    """Belief-mode edit memory backed by an as-of directory."""

    def __init__(self, asof_dir: Path, calibration_line: str = "") -> None:
        self.asof_dir = Path(asof_dir)
        self.steering = True
        self.steering_mode = "belief"
        self.steering_token_budget = 48000
        self.verdict_threshold = 0.02
        self.min_shared = 8
        self._cal = calibration_line

    def render_belief_block(self) -> str:
        from meta_agent.edit_beliefs import BeliefStore
        return BeliefStore(lambda **kw: None).render_block(self.asof_dir)

    def belief_calibration_line(self) -> str:
        return self._cal


def _pre_expand_tallies(run: Path, child: int) -> dict[int, dict[str, Any]]:
    """Node tallies as of the snapshot immediately before `child` appeared.

    hgm_node.json is rewritten as evaluations arrive, so reading it directly
    would show the editor scores from its own future.
    """
    path = run / "snapshots" / "tree_snapshots.jsonl"
    prev: dict[int, dict[str, Any]] = {}
    if not path.exists():
        return prev
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            snap = json.loads(line)
        except json.JSONDecodeError:
            continue
        nodes = {n["node_id"]: n for n in snap.get("nodes") or []}
        if child in nodes:
            return prev
        prev = nodes
    return prev


def rehydrate(run: Path, asof_dir: Path, child: int, arm: str):
    """An HGMManager whose tree, feedback and memory are as of this expand."""
    from meta_agent.managers.hgm import HGMManager
    from meta_agent.managers.hgm_tree import HGMNode, HGMTree
    from meta_agent.models import AgentFeedback, CaseResult

    mgr = HGMManager()
    mgr._experiment_dir = asof_dir
    mgr._tree = HGMTree()
    mgr._summarizer = None

    tallies = _pre_expand_tallies(run, child)
    for p in sorted(asof_dir.glob("round_*/hgm_node.json")):
        d = json.loads(p.read_text())
        nid = d["node_id"]
        if tallies and nid not in tallies:
            continue
        t = tallies.get(nid, d)
        n = HGMNode(node_id=nid, parent_id=d["parent_id"],
                    round_dir=p.parent.resolve())
        n.children = [c for c in (d.get("children") or [])
                      if not tallies or c in tallies]
        n.edit_failed = bool(t.get("edit_failed", False))
        n.n_success = float(t.get("n_success", 0.0) or 0.0)
        n.n_failure = float(t.get("n_failure", 0.0) or 0.0)
        k = int(t.get("n_evals", 0) or 0)
        # only count and mean are read downstream; a flat list reproduces both
        n.utility_measures = [float(t.get("mean_utility", 0.0) or 0.0)] * k
        ids = (d.get("evaluated_case_ids") or [])[:k]
        n.evaluated_case_ids = set(ids)
        n.case_results = [CaseResult(case_id=cid, score=s, passed=s > 0)
                          for cid, s in zip(d.get("evaluated_case_ids") or [],
                                            d.get("utility_measures") or [])]
        mgr._tree.add(n)

    for p in sorted(asof_dir.glob("round_*/feedback.json")):
        try:
            fb = AgentFeedback.model_validate_json(p.read_text())
        except Exception:  # noqa: BLE001
            continue
        mgr._feedback[fb.round_number] = fb

    mgr._edit_memory = None if arm == "M0" else BeliefStub(asof_dir)
    return mgr


def project_context(project: str = "travel") -> dict[str, Any]:
    from meta_agent.config import _read_db_schema, _read_tools_source
    base = REPO / "projects" / project
    return {"tools_source": _read_tools_source(base),
            "db_schema": _read_db_schema(base),
            "scorer_source": None}   # eval_visibility: blackbox in these runs


def _editor(arm: str, llm, **pc):
    from meta_agent.agent_editor import AgentEditor
    from meta_agent.agent_editor_two_stage import TwoStageEditor
    if arm == "M0":
        return AgentEditor(llm, [], **pc)
    return TwoStageEditor(llm, [], **pc)


# --------------------------------------------------------------------------- #
# prompt reconstruction (no LLM) — used by --verify
# --------------------------------------------------------------------------- #
def reconstruct(run: Path, asof_dir: Path, child: int,
                arm: str = "M3") -> dict[str, str]:
    """Rebuild the exact user prompts this expand sent, from stored artifacts."""
    from meta_agent import edit_archive
    from meta_agent.agent_editor_two_stage import TwoStageEditor

    mgr = rehydrate(run, asof_dir, child, arm)
    parent_id = json.loads(
        (run / f"round_{child:03d}" / "hgm_node.json").read_text())["parent_id"]
    parent = mgr._tree[parent_id]
    context = mgr._render_expand_context(parent)

    pc = project_context()
    ed = TwoStageEditor(lambda **kw: None, [], **pc)
    fb = mgr._feedback.get(parent_id)
    sources = ed._read_mutable_sources(parent.round_dir / "task_agent")

    out = {"context": context}

    # ---- stage 1 (planning) ------------------------------------------------
    parts: list[str] = []
    if context:
        parts.append(f"## Steering context\n{context}\n")
    reg = ed._render_registry_ids(asof_dir, base_dir=parent.round_dir)
    if reg:
        parts.append("## Registry ids for the memory query (strategy / "
                     "area ids with the nodes that used them)\n" + reg + "\n")
    if fb is not None:
        parts.append(ed._format_feedback(fb))
    parts.append(ed._format_current_sources(sources))
    out["stage1_user"] = "\n".join(parts)

    # ---- stage 2 (the edit) ------------------------------------------------
    # Use the proposal the run actually produced, so this validates ASSEMBLY,
    # not the model's sampling.
    resp = run / f"round_{child:03d}" / "verbose" / "editor_propose_response.json"
    context2 = context or ""
    if resp.exists():
        calls = (json.loads(resp.read_text()) or {}).get("tool_calls") or []
        proposal = (calls[0].get("arguments") or {}) if calls else {}
        context2 += ("\n\n## Planning-pass proposal (advisory — override it "
                     "if the code says otherwise)\n"
                     + TwoStageEditor._render_proposal(proposal))
        res = edit_archive.resolve_query(
            asof_dir, proposal.get("memory_query") or {},
            char_budget=60000, max_nodes=4)
        retrieved = edit_archive.render_retrieved(res)
        if retrieved:
            context2 += ("\n\n## Retrieved records and implementations "
                         "(what the planning pass asked for)\n" + retrieved)

    p2: list[str] = []
    if context2:
        p2.append(f"## Steering context\n{context2}\n")
    if fb is not None:
        p2.append(ed._format_feedback(fb))
    p2.extend(ed._format_project_context())
    p2.append(ed._format_current_sources(sources))
    out["stage2_user"] = "\n".join(p2)
    return out


# --------------------------------------------------------------------------- #
# section-aware diffing for the fidelity gate
# --------------------------------------------------------------------------- #
SECTION_RE = re.compile(r"^## (.+)$", re.M)


def sections(text: str) -> dict[str, str]:
    marks = [(m.start(), m.group(1)) for m in SECTION_RE.finditer(text)]
    out: dict[str, str] = {}
    for i, (pos, name) in enumerate(marks):
        end = marks[i + 1][0] if i + 1 < len(marks) else len(text)
        out.setdefault(name, "")
        out[name] += text[pos:end]
    return out


def compare(actual: str, rebuilt: str) -> dict[str, Any]:
    sa, sr = sections(actual), sections(rebuilt)
    names = sorted(set(sa) | set(sr))
    rows = []
    for n in names:
        a, r = sa.get(n, ""), sr.get(n, "")
        rows.append({
            "section": n, "match": a == r,
            "actual_chars": len(a), "rebuilt_chars": len(r),
            "only_in": ("actual" if a and not r else
                        "rebuilt" if r and not a else None),
        })
    return {"identical": actual == rebuilt,
            "actual_chars": len(actual), "rebuilt_chars": len(rebuilt),
            "sections": rows}


def _diff_snippet(a: str, b: str, n: int = 12) -> str:
    d = list(difflib.unified_diff(a.splitlines(), b.splitlines(),
                                  "run", "rebuilt", lineterm="", n=1))
    return "\n".join(d[:n])


def verify(run: Path, children: list[int], workdir: Path) -> str:
    L: list[str] = []
    A = L.append
    A("# Part B fidelity gate")
    A("")
    A("Reconstructed M3 prompts vs what the run actually sent. Sections that "
      "differ are listed with their sizes so a mismatch can be attributed to "
      "assembly (a bug) or to refreshed state (a documented as-of limitation).")
    A("")
    for child in children:
        d = run / f"round_{child:03d}" / "verbose"
        dest = workdir / f"asof_{child:03d}"
        asof.build(run, child, dest)
        rb = reconstruct(run, dest, child, "M3")
        A(f"## node {child}")
        A("")
        for stage, fname in (("stage1_user", "editor_propose_user.txt"),
                             ("stage2_user", "editor_attempt_1_user.txt")):
            real_path = d / fname
            if not real_path.exists():
                A(f"- `{stage}`: no stored prompt ({fname})")
                continue
            actual = real_path.read_text(encoding="utf-8", errors="replace")
            cmp = compare(actual, rb[stage])
            status = "IDENTICAL" if cmp["identical"] else "differs"
            A(f"- **`{stage}`** — {status} "
              f"(run {cmp['actual_chars']:,} chars, rebuilt "
              f"{cmp['rebuilt_chars']:,})")
            if not cmp["identical"]:
                bad = [r for r in cmp["sections"] if not r["match"]]
                for r in bad:
                    note = f", only in {r['only_in']}" if r["only_in"] else ""
                    A(f"  - `## {r['section']}` — run {r['actual_chars']:,} vs "
                      f"rebuilt {r['rebuilt_chars']:,}{note}")
        A("")
    return "\n".join(L)


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--children", required=True,
                    help="comma-separated child node ids")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--workdir", type=Path,
                    default=Path("study/out/asof"))
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    run = args.run.resolve()
    children = [int(c) for c in args.children.split(",") if c.strip()]
    args.workdir.mkdir(parents=True, exist_ok=True)

    if args.verify:
        os.environ.setdefault("META_AGENT_VERBOSE", "0")
        text = verify(run, children, args.workdir)
        if args.out:
            args.out.write_text(text, encoding="utf-8")
        else:
            print(text)
        return
    raise SystemExit("only --verify is implemented so far")


if __name__ == "__main__":
    main()
