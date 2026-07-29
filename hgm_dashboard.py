"""Streamlit dashboard for watching/analyzing an HGM (or HGM-dual) run.

    streamlit run hgm_dashboard.py

All data comes from ``meta_agent.run_inspect`` (pure Python, no Streamlit
dependency) -- this file is purely presentation. Generic across any project
using the exclude-list ``mutable_exclude`` + ``seed_dir_name`` convention
(db_mas, math_mas, ...); travel/shopping/math's legacy include-list
convention isn't given dedicated UI (see the plan's Addendum 7).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from meta_agent import run_inspect as ri  # noqa: E402
from meta_agent.feedback_gatherer import render_metrics  # noqa: E402

st.set_page_config(page_title="HGM Run Dashboard", layout="wide")


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #

st.sidebar.title("HGM Run Dashboard")
runs_root = Path(st.sidebar.text_input("Runs root", value="runs"))
experiments = ri.list_experiments(runs_root)

if not experiments:
    st.error(f"No experiment directories found under `{runs_root}`.")
    st.stop()

exp_names = [p.name for p in experiments]
selected_name = st.sidebar.selectbox("Experiment (newest first)", exp_names, index=0)
experiment_dir = runs_root / selected_name

auto_refresh = st.sidebar.checkbox("Auto-refresh", value=True)
refresh_interval = st.sidebar.slider("Refresh interval (s)", 2, 30, 5)
if st.sidebar.button("Refresh now"):
    st.rerun()


# --------------------------------------------------------------------------- #
# Load everything (cheap -- pure file reads)
# --------------------------------------------------------------------------- #

cfg = ri.load_config_snapshot(experiment_dir)
rounds = ri.discover_rounds(experiment_dir)
snapshots = ri.load_tree_snapshots(experiment_dir)
is_active = ri.run_is_active(experiment_dir)
budget = ri.budget_progress(cfg, snapshots, rounds)
alerts = ri.extract_diagnostics(rounds)
run_summary = ri.load_run_summary(experiment_dir)

rounds_by_id = {r.node_id: r for r in rounds}

# Per-node diff cache (against each node's actual parent) -- computed once,
# reused by both the tree diagram and the nodes table.
diffs_by_node: dict[int, tuple[dict[str, ri.FileDiff], int, int]] = {}
mutable_exclude = cfg.get("mutable_exclude")
for r in rounds:
    if r.parent_id is None or r.parent_id not in rounds_by_id:
        continue
    parent = rounds_by_id[r.parent_id]
    diffs = ri.diff_round_files(parent.round_dir, r.round_dir, mutable_exclude)
    added, removed = ri.diff_totals(diffs)
    diffs_by_node[r.node_id] = (diffs, added, removed)


# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #

badge = "🟢 LIVE" if is_active else "⚪ STOPPED"
st.title(f"{cfg.get('experiment_name', selected_name)} — {badge}")

hcols = st.columns(5)
hcols[0].metric("Project", cfg.get("project", "?"))
hcols[1].metric("Manager", (cfg.get("manager") or {}).get("type", "?"))
hcols[2].metric("Task-agent model", ri.task_agent_model(cfg) or "?")
hcols[3].metric("Meta-agent model", ri.meta_agent_model(cfg) or "?")
hcols[4].metric("Max rounds", (cfg.get("loop") or {}).get("max_rounds", "?"))

if budget.total:
    spent = budget.spent or 0
    frac = max(0.0, min(1.0, spent / budget.total))
    caption = f"Budget: {spent}/{budget.total} evals"
    if not budget.exact:
        caption += "  (approximate — enable `manager.config.snapshot_tree: true` for exact tracking)"
    st.progress(frac, text=caption)


# --------------------------------------------------------------------------- #
# Diagnostics panel
# --------------------------------------------------------------------------- #

st.subheader("Diagnostics")
if not alerts:
    st.success("No issues detected.")
else:
    icon = {"error": "🔴", "warning": "🟡", "info": "🔵"}
    for a in alerts:
        st.markdown(f"{icon.get(a.severity, '⚪')} **node {a.node_id}** — {a.message}")


# --------------------------------------------------------------------------- #
# Tree diagram
# --------------------------------------------------------------------------- #

st.subheader("Tree")


def _color_for(mean_utility: Optional[float], edit_failed: bool) -> str:
    if edit_failed or mean_utility is None:
        return "#bdbdbd"
    mu = max(0.0, min(1.0, mean_utility))
    r = int(220 * (1 - mu) + 20)
    g = int(180 * mu + 40)
    return f"#{r:02x}{g:02x}3a"


dot_lines = [
    "digraph tree {",
    'node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=11];',
]
for r in rounds:
    mu_str = f"{r.mean_utility:.3f}" if r.mean_utility is not None else "n/a"
    label = f"node {r.node_id}\\nmean={mu_str}"
    if r.node_id in diffs_by_node:
        _, added, removed = diffs_by_node[r.node_id]
        label += f"\\n+{added}/-{removed}"
    if r.edit_failed:
        label += "\\nEDIT FAILED"
    color = _color_for(r.mean_utility, r.edit_failed)
    dot_lines.append(f'  n{r.node_id} [label="{label}", fillcolor="{color}"];')
    if r.parent_id is not None and r.parent_id in rounds_by_id:
        dot_lines.append(f"  n{r.parent_id} -> n{r.node_id};")
dot_lines.append("}")

try:
    st.graphviz_chart("\n".join(dot_lines))
except Exception as exc:  # noqa: BLE001 -- graphviz binary may be absent
    st.warning(
        f"Tree diagram couldn't render ({exc}); falling back to the nodes "
        "table below (still fully functional, just less visual)."
    )


# --------------------------------------------------------------------------- #
# Nodes table + score trend
# --------------------------------------------------------------------------- #

st.subheader("Nodes")

rows: list[dict[str, Any]] = []
for r in rounds:
    added = removed = 0
    if r.node_id in diffs_by_node:
        _, added, removed = diffs_by_node[r.node_id]
    rows.append(
        {
            "node_id": r.node_id,
            "parent": r.parent_id,
            "edit_failed": r.edit_failed,
            "mean_utility": r.mean_utility,
            "n_evals": r.n_evals,
            "cmp": r.cmp,
            "+added": added,
            "-removed": removed,
            "optimization_goal": r.optimization_goal[:100],
        }
    )
nodes_df = pd.DataFrame(rows)
st.dataframe(nodes_df, width="stretch", hide_index=True)

trend_df = nodes_df[["node_id", "mean_utility"]].dropna()
if not trend_df.empty:
    st.line_chart(trend_df.set_index("node_id"))


# --------------------------------------------------------------------------- #
# Round drill-down
# --------------------------------------------------------------------------- #

st.subheader("Round drill-down")

node_ids = [r.node_id for r in rounds]
selected_node = st.selectbox("Node", node_ids, index=len(node_ids) - 1)
round_ = rounds_by_id[selected_node]

tab_strategy, tab_eval, tab_diff, tab_feedback, tab_memory = st.tabs(
    ["Strategy", "Evaluation", "Diff vs parent", "Feedback", "Behavior memory"]
)

with tab_strategy:
    if round_.strategy:
        st.json(round_.strategy)
    else:
        st.info("No strategy.json (seed round, or not written yet).")

with tab_eval:
    er = round_.eval_result or {}
    if er.get("_synthesized_from_case_logs"):
        st.warning(
            "eval_result.json is missing — this round likely crashed "
            "mid-evaluation. Showing per-case results reconstructed from "
            "logs/case_*.json instead."
        )
    per_case = er.get("per_case") or []
    if per_case:
        case_rows = [
            {
                "case_id": c.get("case_id"),
                "passed": c.get("passed"),
                "score": c.get("score"),
                "error": c.get("error"),
            }
            for c in per_case
        ]
        st.dataframe(pd.DataFrame(case_rows), width="stretch", hide_index=True)
        case_ids = [c.get("case_id") for c in per_case]
        sel_case_id = st.selectbox("Inspect case details", case_ids)
        case_obj = next(c for c in per_case if c.get("case_id") == sel_case_id)
        st.json(case_obj.get("details") or {})
    else:
        st.info("No per-case results yet for this round.")

with tab_diff:
    if round_.parent_id is None:
        st.info("Root round — no parent to diff against.")
    elif round_.node_id not in diffs_by_node:
        st.info("No diff available (edit failed before producing a workspace, or parent missing).")
    else:
        diffs, added, removed = diffs_by_node[round_.node_id]
        if not diffs:
            st.info("No changed files.")
        else:
            st.markdown(f"**{len(diffs)} file(s) changed, +{added}/-{removed} lines total**")
            for path, d in diffs.items():
                with st.expander(f"{path}  (+{d.lines_added}/-{d.lines_removed}, {d.status})"):
                    st.code(d.diff_text, language="diff")

with tab_feedback:
    fb = round_.feedback or {}
    if fb:
        project_metrics = fb.get("project_metrics") or {}
        if project_metrics:
            st.markdown("**Project metrics**")
            st.code("\n".join(render_metrics(project_metrics, cap=20)))
        runtime_exceptions = fb.get("runtime_exceptions") or []
        if runtime_exceptions:
            st.markdown("**Runtime exceptions**")
            for exc in runtime_exceptions:
                st.code(exc)
        edit_errors = fb.get("edit_errors") or []
        if edit_errors:
            st.markdown("**Edit errors**")
            for err in edit_errors:
                st.code(err)
        failure_report = fb.get("failure_report") or {}
        if failure_report:
            st.markdown("**Failure report**")
            st.json(failure_report)
        if not (project_metrics or runtime_exceptions or edit_errors or failure_report):
            st.info("feedback.json present but has nothing to show (clean round).")
    else:
        st.info("No feedback.json yet for this round.")

with tab_memory:
    if round_.behavior_memory:
        st.markdown(round_.behavior_memory)
    else:
        st.info(
            "No behavior_memory.md — root round (nothing to diff against), "
            "no summarizer configured, or not written yet."
        )


# --------------------------------------------------------------------------- #
# run_summary.md, when the run has finished
# --------------------------------------------------------------------------- #

if run_summary:
    st.subheader("Run summary")
    st.markdown(run_summary)


# --------------------------------------------------------------------------- #
# Auto-refresh (no extra dependency: sleep-then-rerun at the end of the
# script). Only loops while the run looks live -- a finished run has nothing
# left to refresh toward.
# --------------------------------------------------------------------------- #

if auto_refresh and is_active:
    time.sleep(refresh_interval)
    st.rerun()
