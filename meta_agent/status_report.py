"""Human-readable live status of an HGM run: ``STATUS.md`` + ``status.json``.

A pure function of the experiment dir -- it reads only what the manager already
persists (``loop_state.json``, ``round_NNN/hgm_node.json``, ``feedback.json``,
``strategy.json``, ``config.snapshot.yaml``) -- so it works for a live run (the
manager calls :func:`write` after every step), a paused one, a killed one and a
finished one::

    python -m meta_agent.status_report <run_dir>

Enabled per run with ``manager.config.status_report: true`` (default off). Per node
it reports the utility (mean score), the pass count (``CaseResult.passed``), the LCB
the final pick uses, and a PAIRED comparison with the root on the tasks both have
run -- the only comparison that is not confounded by which tasks a node happened to
draw. Small or insignificant differences are flagged: single-run task outcomes can be
very noisy (on one coding benchmark identical code scored 18 vs 25 of 60).
"""
from __future__ import annotations

import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

CONTROL_HELP = {
    "pause": "touch {run}/PAUSE        # stops after the current step; exits rc 3, no finalize",
    "resume": "python main_loop.py --resume {run}",
    "finalize": "touch {run}/FINALIZE_NOW # stop spending budget; finalize (top-up + LCB pick)",
}


def _read(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _sign_p(k: int, n: int) -> float:
    if n == 0:
        return 1.0
    k = min(k, n - k)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def _lcb(s: float, f: float, eps: float) -> float:
    draws = np.random.default_rng(0).beta(1.0 + s, 1.0 + f, 8192)
    return float(np.quantile(draws, eps))


def _one_line(text: Optional[str], limit: int = 110) -> str:
    t = " ".join((text or "").split())
    return t if len(t) <= limit else t[: limit - 1] + "…"


def _epsilon(run: Path) -> float:
    try:
        import yaml
        cfg = yaml.safe_load((run / "config.snapshot.yaml").read_text())
        return float(((cfg.get("manager") or {}).get("config") or {}).get("epsilon", 0.25))
    except Exception:  # noqa: BLE001
        return 0.25


def collect(run: Path) -> dict[str, Any]:
    run = Path(run)
    state = _read(run / "loop_state.json") or {}
    eps = _epsilon(run)
    nodes: list[dict[str, Any]] = []
    per_case: dict[int, dict[str, dict[str, Any]]] = {}
    for rd in sorted(d for d in run.iterdir() if d.is_dir() and re.fullmatch(r"round_\d{3}", d.name)):
        side = _read(rd / "hgm_node.json")
        fb = _read(rd / "feedback.json") or {}
        if side is None:
            nodes.append({"node_id": int(rd.name[6:]), "status": "being created", "round": rd.name})
            continue
        strat = _read(rd / "strategy.json") or fb.get("strategy") or {}
        cases = {}
        for c in ((fb.get("eval_result") or {}).get("per_case") or []):
            d = c.get("details") or {}
            if d.get("excluded"):
                continue
            cases[c["case_id"]] = {"score": float(c.get("score") or 0.0),
                                   "passed": bool(c.get("passed"))}
        per_case[side["node_id"]] = cases
        resolved = sum(1 for v in cases.values() if v["passed"])
        nodes.append({
            "node_id": side["node_id"], "parent_id": side.get("parent_id"), "round": rd.name,
            "block": strat.get("block"),
            # meta_agent/focus.py: None when the focus axis is off; targets_delta after the first batch.
            "focus": strat.get("focus"),
            "targets_delta": ((_read(rd / "focus.json") or {}).get("targets_delta") or {}).get("mean_delta"),
            "targets_delta_case": ((_read(rd / "focus.json") or {}).get("targets_delta") or {}).get("mean_delta_vs_case"),
            "change": _one_line(strat.get("optimization_goal")),
            "edit_failed": side.get("edit_failed", False),
            "n": side.get("n_evals", 0), "n_excluded": side.get("n_excluded", 0),
            "mean": side.get("mean_utility", 0.0),
            "resolved": resolved,
            "lcb": _lcb(side.get("n_success", 0.0), side.get("n_failure", 0.0), eps)
                   if side.get("n_evals") else None,
            "children": side.get("children", []),
        })
    root = per_case.get(0, {})
    for nd in nodes:
        nid = nd.get("node_id")
        if nid in (None, 0) or nid not in per_case or nd.get("status"):
            continue
        shared = sorted(set(per_case[nid]) & set(root))
        if not shared:
            nd["paired"] = None
            continue
        a, b = per_case[nid], root
        diffs = [a[c]["score"] - b[c]["score"] for c in shared]
        better = sum(x > 1e-9 for x in diffs)
        worse = sum(x < -1e-9 for x in diffs)
        ra = sum(1 for c in shared if a[c]["passed"])
        rb = sum(1 for c in shared if b[c]["passed"])
        nd["paired"] = {
            "n": len(shared), "d_mean": sum(diffs) / len(shared),
            "better": better, "worse": worse, "same": len(shared) - better - worse,
            "p": _sign_p(better, better + worse), "resolved": ra, "root_resolved": rb,
        }
    n_train = state.get("n_train") or max(
        [nd.get("n", 0) + nd.get("n_excluded", 0) for nd in nodes] or [0])
    for nd in nodes:
        if nd.get("status"):
            continue
        if nd["edit_failed"]:
            nd["status"] = "edit-failed"
        elif nd["n"] == 0:
            nd["status"] = "not yet evaluated"
        elif n_train and nd["n"] + nd["n_excluded"] >= n_train:
            nd["status"] = "fully evaluated"
        else:
            nd["status"] = "partially evaluated"
        p = nd.get("paired")
        nd["noisy"] = bool(nd["node_id"] != 0 and (nd["n"] < 30 or not p or p["p"] > 0.2))
    return {"run": str(run), "state": state, "epsilon": eps, "nodes": nodes,
            "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def _eta(state: dict[str, Any], rate_per_h: Optional[float]) -> str:
    left = (state.get("eval_budget") or 0) - (state.get("budget_spent") or 0)
    if not rate_per_h or left <= 0:
        return "n/a"
    return f"~{left / rate_per_h:.0f} h of evaluation left (at {rate_per_h:.1f} tasks/h), plus editor time"


def _focus_note(nd: dict[str, Any]) -> str:
    """`` · reliability (targets Δ +0.120 vs case mean, -0.083 vs parent)`` for a reliability-focus
    node, else "". The case-mean delta comes first: it is the steadier reference."""
    if nd.get("focus") != "reliability":
        return ""
    parts = [f"{v:+.3f} vs {lbl}" for v, lbl in ((nd.get("targets_delta_case"), "case mean"),
                                                 (nd.get("targets_delta"), "parent")) if v is not None]
    return " · reliability" + (f" (targets Δ {', '.join(parts)})" if parts else "")


def render(info: dict[str, Any], repo: str = "<worktree>") -> str:
    st, run = info["state"], info["run"]
    rate = None
    try:
        t0 = time.mktime(time.strptime(st["started"], "%Y-%m-%dT%H:%M:%SZ"))
        t1 = time.mktime(time.strptime(st["updated"], "%Y-%m-%dT%H:%M:%SZ"))
        done = (st.get("budget_spent") or 0) + (st.get("n_train") or 0)  # + root pre-eval
        rate = done / max((t1 - t0) / 3600.0, 1e-6) if t1 > t0 else None
    except (KeyError, ValueError, TypeError):
        pass
    L = [f"# Run status -- {Path(run).name}", "",
         f"_updated {st.get('updated', '?')} (report generated {info['generated']})_", "",
         f"- **Budget:** {st.get('budget_spent', '?')} / {st.get('eval_budget', '?')} task evaluations "
         f"(root pre-eval and finalize are free); nodes: {st.get('n_nodes', '?')}",
         f"- **Now:** {st.get('current_action') or '?'}  (last completed step: {st.get('last_event') or '-'})",
         f"- **Control file:** {st.get('control') or 'none'};  resumes so far: {st.get('resume_count', 0)}",
         f"- **ETA:** {_eta(st, rate)}", "",
         "## Nodes", "",
         "Utility = mean per-task score used by the search; passed = tasks the scorer marked passed. "
         "*Paired vs root* compares only tasks both ran: Δ = mean score difference, "
         "better/worse/same task counts, sign-test p. ⚠ = too few tasks (n<30) or p>0.2 -- "
         "treat as noise.", "",
         "| node | parent | block | n | utility | passed | LCB | paired vs root (Δ, +/−/=, p, passed) | status | change |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for nd in sorted(info["nodes"], key=lambda x: x.get("node_id", 0)):
        if nd.get("status") == "being created":
            L.append(f"| {nd['node_id']} | | | | | | | | being created | |")
            continue
        p = nd.get("paired")
        ps = ("—" if nd["node_id"] == 0 else "no shared tasks yet" if not p else
              f"{p['d_mean']:+.3f}, {p['better']}/{p['worse']}/{p['same']}, p={p['p']:.2f}, "
              f"{p['resolved']} vs {p['root_resolved']} of {p['n']}")
        warn = " ⚠" if nd.get("noisy") else ""
        lcb = f"{nd['lcb']:.3f}" if nd.get("lcb") is not None else "—"
        L.append(
            f"| {nd['node_id']}{warn} | {'' if nd['parent_id'] is None else nd['parent_id']} | "
            f"{nd.get('block') or ('seed' if nd['node_id'] == 0 else '?')}{_focus_note(nd)} | {nd['n']}"
            f"{'+' + str(nd['n_excluded']) + 'x' if nd.get('n_excluded') else ''} | {nd['mean']:.3f} | "
            f"{nd['resolved']}/{nd['n']} | {lcb} | {ps} | {nd['status']} | {nd.get('change', '')} |")
    fr = repo
    L += ["", "## Controls (copy-paste)", ""]
    for k, v in CONTROL_HELP.items():
        L.append(f"- **{k}:** `{v.format(run=run, repo=fr)}`")
    L += ["", "The final pick is only a candidate: claim an improvement only from a held-out paired "
          "evaluation, never from the train numbers above."]
    return "\n".join(L) + "\n"


def write(run: Path, repo: Optional[str] = None) -> Path:
    from .atomic_io import atomic_write_text

    run = Path(run)
    info = collect(run)
    atomic_write_text(run / "status.json", json.dumps(info, indent=1, default=str))
    out = run / "STATUS.md"
    atomic_write_text(out, render(info, repo or "<worktree>"))
    return out


if __name__ == "__main__":
    print(write(Path(sys.argv[1])).read_text())
