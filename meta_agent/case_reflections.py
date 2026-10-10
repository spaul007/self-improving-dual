"""Per-test-case reflection files: every node's reflections on ONE case, in one file.

``<run_dir>/case_reflections/<case>.md`` (one per test case) and ``INDEX.md`` (every case, hardest
first) are a PURE REBUILD from two sources, so they are correct after any resume or aborted round:

* the tree's per-node ``case_results`` (pass/fail and score of every evaluation; infra-excluded
  and errored evaluations are left out of the pass rate, the same rule as ``_record_batch``);
* every ``round_*/reflections/*.json`` record (meta_agent/reflector.py), rendered under the
  reflector's ``exposure`` and REDACTED with the record's own terms plus the union of every term
  recorded for that case (a solved run can quote what another run was graded on).

The raw records stay outside every readable root; the meta-agent reads these files through
``log_access`` as ``cases/<file>``. Written only by the manager's main thread, atomically.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from .log_access import CASES_DIR
from .models import normalize_probes
from .reflector import _atomic_write, _field, _oneline, _safe, load_records, redact, render_record

INDEX = "INDEX.md"


def case_file_name(case_id: Any) -> str:
    return f"{_safe(case_id)}.md"


def _json(p: Path) -> dict:
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else {}
    except (OSError, ValueError):
        return {}


def _depth(nid: int, parents: dict[int, Optional[int]]) -> int:
    d, seen = 0, set()
    while parents.get(nid) is not None and nid not in seen:
        seen.add(nid)
        nid = parents[nid]
        d += 1
    return d


def _usable(c: Any) -> bool:
    return not (getattr(c, "details", None) or {}).get("excluded") and not getattr(c, "error", None)


def collect(run_dir: Path, nodes: Iterable[Any],
            limited_fn: Optional[Callable[[Any], Optional[str]]] = None) -> dict[str, dict]:
    """``{case_id: {"evals": [(node_id, passed, score)], "runs": [...], "records": [...]}}``
    over every node. ``runs`` carries each scored evaluation's node, parent, per-node
    evaluation index ``k`` (the reflector's ``eval_index``: the k-th scored evaluation of
    that case on that node) and ``limited`` -- why the run was cut short by a budget, per
    the project's optional ``limited_by_budget`` hook (``None`` = a behavioural outcome)."""
    run_dir = Path(run_dir)
    cases: dict[str, dict] = {}
    for n in nodes:
        seen: dict[str, int] = {}
        for c in getattr(n, "case_results", None) or []:
            ent = cases.setdefault(c.case_id, {"evals": [], "runs": [], "records": []})
            if _usable(c):
                ent["evals"].append((n.node_id, bool(c.passed), float(c.score or 0.0)))
                seen[c.case_id] = seen.get(c.case_id, 0) + 1
                try:
                    limited = limited_fn(c) if limited_fn is not None else None
                except Exception:  # noqa: BLE001 -- a project hook must never break the rebuild
                    limited = None
                ent["runs"].append({"node": n.node_id, "parent": n.parent_id, "k": seen[c.case_id],
                                    "passed": bool(c.passed), "score": float(c.score or 0.0),
                                    "limited": limited or None})
        for rec in load_records(Path(n.round_dir)):
            rec.setdefault("node_id", n.node_id)
            rec.setdefault("parent_id", n.parent_id)
            cid = rec.get("case_id")
            if cid is not None:
                cases.setdefault(cid, {"evals": [], "runs": [], "records": []})["records"].append(rec)
    return cases


def _node_rows(nodes: list[Any]) -> dict[int, dict]:
    parents = {n.node_id: n.parent_id for n in nodes}
    rows = {}
    for n in nodes:
        s = _json(Path(n.round_dir) / "strategy.json")
        rows[n.node_id] = {"parent": n.parent_id, "depth": _depth(n.node_id, parents),
                           "block": s.get("block") or ("seed" if n.parent_id is None else ""),
                           "goal": _oneline(s.get("optimization_goal") or
                                            ("seed agent" if n.parent_id is None else ""))[:160]}
    return rows


def find_contrast(runs: list[dict], min_gap: float = 0.2) -> Optional[dict]:
    """The best pass/fail contrast among one case's scored runs, or ``None``.

    A pair (high, low) qualifies when the high run passed and the low did not, or their scores
    differ by at least ``min_gap``. A low run that was cut short by a budget (``limited``) is
    never the low side -- its outcome says nothing about behaviour. Preference: the SAME node
    (repeats), then PARENT -> CHILD (one edit apart), then any two nodes; within a tier the
    largest gap, then the newest nodes. No extra evaluations are needed."""
    best, best_key = None, None
    for hi in runs:
        for lo in runs:
            if hi is lo or lo.get("limited"):
                continue
            gap = hi["score"] - lo["score"]
            if not ((hi["passed"] and not lo["passed"]) or gap >= min_gap) or gap < 0:
                continue
            if hi["node"] == lo["node"]:
                tier, rel = 0, "same node"
            elif hi.get("parent") == lo["node"] or lo.get("parent") == hi["node"]:
                tier, rel = 1, "parent -> child"
            else:
                tier, rel = 2, "different nodes -- the edit may explain the difference"
            key = (tier, -gap, -max(hi["node"], lo["node"]))
            if best_key is None or key < best_key:
                best_key, best = key, {"high": hi, "low": lo, "gap": gap, "relation": rel, "tier": tier}
    return best


def _run_label(run: dict) -> str:
    par = run.get("parent")
    return (f"node {run['node']} (parent {par if par is not None else '-'}) · eval {run['k']} · "
            f"score {run['score']:.2f} {'PASSED' if run['passed'] else 'FAILED'}")


def render_contrast(c: dict, recs: list[dict], rows: dict[int, dict], terms: list[str],
                    max_field_chars: Optional[int]) -> list[str]:
    """Lines of the ``## Pass/fail contrast`` section: the two runs, how they relate, and per
    role the high run's ESSENTIAL / CLOSE CALLS / KEEP beside the low run's WHERE / WHY / WHAT
    WOULD HAVE CAUGHT IT (redacted with the case's union of terms, each field clipped)."""
    hi, lo = c["high"], c["low"]

    def mine(run: dict) -> dict[str, dict]:
        return {str(r.get("role")): (r.get("parsed") or {}) for r in recs
                if r.get("node_id") == run["node"] and (r.get("eval_index") or 1) == run["k"]}

    rh, rl = mine(hi), mine(lo)
    rel = c["relation"]
    if c["tier"] == 1:
        child = hi if hi.get("parent") == lo["node"] else lo
        goal = redact(rows.get(child["node"], {}).get("goal", ""), terms)
        rel = f"parent -> child; node {child['node']}'s edit: {goal or '-'}"
    out = ["## Pass/fail contrast", "",
           f"high: {_run_label(hi)}", f"low:  {_run_label(lo)}", f"relation: {rel} · gap {c['gap']:.2f}", ""]

    def cell(p: dict, keys: tuple) -> str:
        parts = [f"{k.replace('_', ' ')}: {_field(p[k], terms, max_field_chars)}" for k in keys if p.get(k)]
        return " / ".join(parts).replace("|", "/") or "-"

    roles = sorted(set(rh) | set(rl))
    if roles:
        out += ["| role | high run: essential / close calls / keep | low run: where / why / would have caught it |",
                "|---|---|---|"]
        for role in roles:
            h = cell(rh.get(role, {}), ("essential", "close_calls", "keep")) if rh.get(role) else "-"
            if h == "-" and rh.get(role):
                h = cell(rh[role], ("lesson",))
            lo_cell = cell(rl.get(role, {}), ("where", "why", "catch")) if rl.get(role) else "-"
            out.append(f"| {role} | {h} | {lo_cell} |")
        ch = [p.get("overall_confidence") for p in rh.values() if isinstance(p.get("overall_confidence"), (int, float))]
        cl = [p.get("overall_confidence") for p in rl.values() if isinstance(p.get("overall_confidence"), (int, float))]
        if ch or cl:
            fmt = lambda v: f"{sum(v) / len(v):.0f}" if v else "-"  # noqa: E731
            out.append(f"\nblind confidence: high {fmt(ch)}, low {fmt(cl)}")
    else:
        out.append("(no reflections recorded for these two runs)")
    return out


def render_case(case_id: str, ent: dict, rows: dict[int, dict], exposure: str, max_chars: int,
                max_field_chars: Optional[int] = 400, contrast_min_gap: float = 0.2) -> str:
    evals = ent["evals"]
    recs = sorted(ent["records"], key=lambda r: (float(r.get("ts") or 0), r.get("node_id") or 0,
                                                  r.get("eval_index") or 0, str(r.get("role"))))
    terms = sorted({t for r in recs for t in (r.get("redact_terms") or [])})
    n_ev, n_pass = len(evals), sum(p for _, p, _ in evals)
    node_ids = sorted({nid for nid, _, _ in evals} | {r.get("node_id") for r in recs if r.get("node_id") is not None})
    head = [f"# Test case {case_id}", ""]
    if n_ev:
        mean = sum(s for _, _, s in evals) / n_ev
        head.append(f"pass rate: {n_pass}/{n_ev} evaluations ({100 * n_pass / n_ev:.0f}%) across "
                    f"{len({nid for nid, _, _ in evals})} node(s) · mean score {mean:.3f}")
    else:
        head.append("pass rate: no scored evaluations (excluded / errored only)")
    confs = [r["parsed"]["overall_confidence"] for r in recs
             if isinstance((r.get("parsed") or {}).get("overall_confidence"), (int, float))]
    if confs and n_ev:
        head.append(f"calibration: mean blind confidence {sum(confs) / len(confs):.0f}/100 over "
                    f"{len(confs)} reflection(s) vs actual pass rate {100 * n_pass / n_ev:.0f}%")
    head += ["", "## Evaluations by node", "",
             "| node | parent | depth | block | edit goal | evals | passed | scores |",
             "|---|---|---|---|---|---|---|---|"]
    for nid in node_ids:
        r = rows.get(nid, {})
        mine = [(p, s) for n, p, s in evals if n == nid]
        goal = redact(r.get("goal", ""), terms).replace("|", "/")
        head.append(f"| {nid} | {r.get('parent') if r.get('parent') is not None else '-'} | {r.get('depth', '?')} "
                    f"| {r.get('block') or '-'} | {goal or '-'} | {len(mine)} | {sum(p for p, _ in mine)} "
                    f"| {', '.join(f'{s:.2f}' for _, s in mine) or '-'} |")
    limited = [r for r in ent.get("runs") or [] if r.get("limited")]
    if limited:
        head += ["", "runs cut short by a budget (not counted as behavioural lows): " + "; ".join(
            f"node {r['node']} eval {r['k']}: {_oneline(r['limited'])[:120]}" for r in limited[:6])]
    c = find_contrast(ent.get("runs") or [], contrast_min_gap)
    if c is not None and exposure != "off":
        head += [""] + render_contrast(c, recs, rows, terms, max_field_chars)
    header = "\n".join(head)
    if exposure == "off" or not recs:
        return header + ("\n" if exposure == "off" else "\n\n## Reflections\n\n(none recorded yet)\n")
    # One block per (node, eval index): its roles' entries together, oldest first.
    blocks: list[str] = []
    groups: dict[tuple, list[dict]] = {}
    for r in recs:
        groups.setdefault((r.get("node_id"), r.get("eval_index") or 1), []).append(r)
    for (nid, k), grp in groups.items():
        g0 = grp[0]
        score = g0.get("score")
        lines = [f"### node {nid} (parent {g0.get('parent_id') if g0.get('parent_id') is not None else '-'}) "
                 f"· eval {k} · score {float(score):.2f} · {'PASSED' if g0.get('passed') else 'FAILED'}"
                 if isinstance(score, (int, float)) else
                 f"### node {nid} · eval {k} · {'PASSED' if g0.get('passed') else 'FAILED'}"]
        qs = normalize_probes(next((r.get("probe_questions") for r in grp if r.get("probe_questions")), None))
        if qs and exposure != "full":   # the node's probes, shown once with whom each was asked
            lines.append("probe questions (written by the editor for this node's edit): " + " ".join(
                f"[{i}] ({', '.join(p['roles']) or 'all roles'}) {redact(_oneline(p['q']), terms)}"
                for i, p in enumerate(qs, 1)))
        for r in grp:
            body = render_record(r, exposure, terms, max_field_chars, probe_questions=not qs)
            body = [redact(x, terms) for x in body]
            status = r.get("status") or ""
            lines.append(f"**{r.get('role')}**" + (f" ({status})" if status not in ("", "ok") else ""))
            lines += [f"- {x}" for x in body] or ["- (no usable answer)"]
        blocks.append("\n".join(lines))
    budget = max(0, max_chars - len(header) - 80)
    kept: list[str] = []
    used = 0
    for b in reversed(blocks):  # newest first, so the cap drops the OLDEST entries
        if kept and used + len(b) + 2 > budget:
            break
        kept.append(b[:budget] if not kept else b)
        used += len(kept[-1]) + 2
    kept.reverse()
    omitted = len(blocks) - len(kept)
    note = f"[{omitted} older reflection block(s) omitted to fit {max_chars} chars]\n\n" if omitted else ""
    return f"{header}\n\n## Reflections (oldest first; the task agent's own account -- fallible)\n\n{note}" \
           + "\n\n".join(kept) + "\n"


def build_case_files(run_dir: Path, nodes: Iterable[Any], *, exposure: str = "lessons_only",
                     max_chars_per_case: int = 60000,
                     max_field_chars: Optional[int] = 400,
                     limited_fn: Optional[Callable[[Any], Optional[str]]] = None,
                     contrast_min_gap: float = 0.2) -> dict[str, dict]:
    """Rebuild ``<run_dir>/case_reflections/``. Returns ``{case_id: {"evals", "passes",
    "nodes", "reflections", "file", "spread", "unstable", "contrast"}}`` -- ``unstable``: the
    case has a pass/fail contrast whose low run was not cut short by a budget (the reliability
    focus's candidates; meta_agent/focus.py). Never raises past an OSError on the directory."""
    nodes = list(nodes)
    out = Path(run_dir) / CASES_DIR
    out.mkdir(parents=True, exist_ok=True)
    cases = collect(Path(run_dir), nodes, limited_fn)
    rows = _node_rows(nodes)
    summary: dict[str, dict] = {}
    written: set[str] = set()
    for cid, ent in cases.items():
        name = case_file_name(cid)
        _atomic_write(out / name, render_case(cid, ent, rows, exposure, max_chars_per_case,
                                                       max_field_chars, contrast_min_gap))
        written.add(name)
        scores = [s for _, _, s in ent["evals"]]
        c = find_contrast(ent["runs"], contrast_min_gap)
        summary[cid] = {"evals": len(ent["evals"]), "passes": sum(p for _, p, _ in ent["evals"]),
                        "nodes": len({n for n, _, _ in ent["evals"]}), "reflections": len(ent["records"]),
                        "file": name, "spread": (max(scores) - min(scores)) if scores else 0.0,
                        "mean": (sum(scores) / len(scores)) if scores else 0.0,
                        "unstable": c is not None,
                        "contrast": None if c is None else {"high": (c["high"]["node"], c["high"]["k"]),
                                                            "low": (c["low"]["node"], c["low"]["k"]),
                                                            "relation": c["relation"], "gap": round(c["gap"], 3)}}
    idx = ["# Per-test-case reflections", "",
           "One file per test case: its pass rate across every evaluated node, a per-node table (which edit",
           "each node made), and the task agent's reflections on each evaluation, linked to the node.",
           "Hardest cases first. Read one with read_file('cases/<file>').", "",
           "`unstable` = the case has both a high and a low run (see its Pass/fail contrast); `spread` = max - min score.",
           "",
           "| case | file | evals | passed | pass rate | nodes | reflections | unstable | spread |",
           "|---|---|---|---|---|---|---|---|---|"]
    order = sorted(summary.items(), key=lambda kv: ((kv[1]["passes"] / kv[1]["evals"]) if kv[1]["evals"] else 2.0,
                                                    -kv[1]["evals"], str(kv[0])))
    for cid, s in order:
        rate = f"{100 * s['passes'] / s['evals']:.0f}%" if s["evals"] else "-"
        idx.append(f"| {cid} | {s['file']} | {s['evals']} | {s['passes']} | {rate} | {s['nodes']} | {s['reflections']} "
                   f"| {'yes' if s['unstable'] else '-'} | {s['spread']:.2f} |")
    _atomic_write(out / INDEX, "\n".join(idx) + "\n")
    written.add(INDEX)
    for f in out.glob("*.md"):
        if f.name not in written:
            f.unlink(missing_ok=True)
    return summary


def excerpt_for_cases(run_dir: Path, case_ids: Iterable[str], max_chars: int = 6000) -> str:
    """Bounded text of the case files of ``case_ids`` (for prompt-only consumers like the
    failure summarizer): each case's pass-rate header plus its newest reflection block(s)."""
    out = Path(run_dir) / CASES_DIR
    ids = list(dict.fromkeys(case_ids))
    if not ids:
        return ""
    per = max(400, max_chars // len(ids))
    parts = []
    for cid in ids:
        try:
            text = (out / case_file_name(cid)).read_text(encoding="utf-8")
        except OSError:
            continue
        head, _, refl = text.partition("\n## Reflections")
        head = re.sub(r"\n## Evaluations by node\n.*", "", head, flags=re.S).strip()
        blocks = re.split(r"\n(?=### node )", refl)
        tail = ""
        for b in reversed(blocks[1:]):
            if len(tail) + len(b) > per - len(head):
                break
            tail = b.strip() + ("\n\n" + tail if tail else "")
        parts.append(head + ("\n" + tail if tail else ""))
        if sum(len(p) for p in parts) > max_chars:
            break
    text = "\n\n".join(parts)
    return text[:max_chars]
