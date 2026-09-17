"""Compare view: best-so-far curves and summary table across several runs."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from meta_agent import run_inspect as ri

from . import components as C
from . import loaders as L


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
