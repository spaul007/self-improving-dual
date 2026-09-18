"""Compare view: best-so-far curves and summary table across several runs."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from meta_agent import run_inspect as ri

from . import components as C
from . import loaders as L


def _short(name: str) -> str:
    """Run names share a long common middle; keep the timestamp and the
    trailing variant so labels stay readable in legends/tables."""
    parts = name.split("_")
    if len(parts) > 4:
        return f"{parts[0]}_{parts[1]} …{'_'.join(parts[-2:])}"
    return name


def _default_selection(experiments: list[Path]) -> list[str]:
    names = [p.name for p in experiments]
    # The pair launched together (same yyyymmdd_hhmm prefix) is the natural
    # default when one exists; otherwise the two newest runs.
    by_prefix: dict[str, list[str]] = {}
    for n in names:
        by_prefix.setdefault(n[:13], []).append(n)
    for prefix in sorted(by_prefix, reverse=True):
        if len(by_prefix[prefix]) >= 2:
            return by_prefix[prefix][:4]
    return names[:2]


def render(experiments: list[Path]) -> bool:
    """Returns True when any selected run is still active."""
    st.title("Compare runs")
    names = [p.name for p in experiments]
    by_name = {p.name: p for p in experiments}
    selected = st.multiselect("Runs", names, default=_default_selection(experiments), key="cmp_runs",
                              max_selections=len(C.SERIES))
    if not selected:
        st.info("Pick at least one run.")
        return False

    summaries = {n: L.cached_experiment_summary(by_name[n]) for n in selected}
    overlay = {n: L.cached_eval_at_budget(by_name[n]) for n in selected}
    any_active = any(not s.finished and ri.run_is_active(s.path) for s in summaries.values())

    st.subheader("Best train mean vs budget")
    C.best_so_far_chart({n: s.curve for n, s in summaries.items() if s.curve}, overlay)
    missing = [n for n, s in summaries.items() if not s.has_snapshots]
    if missing:
        st.caption("No tree snapshots (omitted from the curve): " + ", ".join(f"`{n}`" for n in missing))

    st.subheader("Summary")
    rows = []
    for n, s in summaries.items():
        rows.append(
            {
                "run": n,
                "status": "finished" if s.finished else ("live" if ri.run_is_active(s.path) else "stopped"),
                "nodes": s.n_nodes,
                "edit_failed": s.n_edit_failed,
                "best_node": s.best_node_id,
                "best_mean": s.best_mean,
                "budget": f"{s.budget_spent}/{s.budget_total}" if s.budget_total else s.budget_spent,
                "edit_memory": "yes" if s.has_edit_memory else "no",
                "pulls with/without": f"{s.pulls.get('with', 0)}/{s.pulls.get('without', 0)}" if s.pulls else "",
                "memory_v": s.memory_version,
            }
        )
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    st.subheader("Whole-tree Beta posterior per run")
    st.caption(
        "Pool every node's HGM tallies across the tree: `Beta(ΣS + 1, ΣF + 1)` where each evaluated case adds "
        "its score to S and 1 − score to F, so S + F is the number of evaluations spent on that run's nodes. "
        "This is the posterior over *the average quality of an expansion the run produces*, not the best node. "
        "Runs with more evaluations get narrower curves; the mean is the eval-weighted train mean of the tree."
    )
    include_root = st.checkbox("Include the seed (round_000) in the pool", value=False, key="cmp_beta_root",
                               help="Off by default: the seed isn't produced by the run and is often shared across runs.")
    tallies = {n: ri.run_beta_tally(L.cached_rounds(by_name[n]), include_root=include_root) for n in selected}
    entries = [
        {
            "name": _short(n),
            "label": f"{_short(n)}  (nodes={t['n_nodes']}, evals={t['n_evals']:.0f})",
            "a": t["S"] + 1.0, "b": t["F"] + 1.0, "color": C.SERIES[i],
        }
        for i, (n, t) in enumerate(tallies.items()) if t["n_evals"] > 0
    ]
    left, right = st.columns([3, 2])
    with left:
        C.beta_curves_chart(entries, x_title="expansion success rate θ (score-weighted)", height=300)
    with right:
        rows = []
        for n, t in tallies.items():
            s = ri.beta_summary(t["S"] + 1.0, t["F"] + 1.0)
            rows.append({"run": _short(n), "nodes": t["n_nodes"], "evaluated": t["n_evaluated"],
                         "evals": int(round(t["n_evals"])), "S": round(t["S"], 2), "F": round(t["F"], 2),
                         "post. mean": round(s["mean"], 3), "90% interval": f"{s['lo90']:.3f} – {s['hi90']:.3f}"})
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        if len(tallies) >= 2:
            names_t = list(tallies)
            mat = pd.DataFrame(index=[_short(n) for n in names_t], columns=[_short(n) for n in names_t], dtype=object)
            for a_name in names_t:
                for b_name in names_t:
                    if a_name == b_name:
                        mat.loc[_short(a_name), _short(b_name)] = "—"
                        continue
                    ta, tb = tallies[a_name], tallies[b_name]
                    p = ri.prob_beta_greater(ta["S"] + 1, ta["F"] + 1, tb["S"] + 1, tb["F"] + 1)
                    mat.loc[_short(a_name), _short(b_name)] = f"{p:.1%}"
            st.markdown("**P(row > column)** — probability the row run's θ exceeds the column run's")
            st.dataframe(mat, width="stretch")

    st.subheader("Best node — dimension means")
    dims = sorted({d for s in summaries.values() for d in s.best_dimension_means})
    if dims:
        wide = pd.DataFrame({n: [s.best_dimension_means.get(d) for d in dims] for n, s in summaries.items()}, index=dims)
        wide.index.name = "dimension"
        C.grouped_dimension_bar(wide, list(summaries))
        st.dataframe(wide.round(3), width="stretch")
    else:
        st.info("No dimension scores available for the selected runs' best nodes.")
    return any_active
