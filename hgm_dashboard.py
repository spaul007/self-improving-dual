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

# For now: only visualize the currently-running experiment, not a picker
# across every run under `runs/` -- auto-select the newest LIVE experiment
# (falling back to the newest experiment overall if nothing looks active).
# To restore the full picker, swap this block back for the commented-out
# selectbox below.
#
# ri.run_is_active() does a full recursive file scan (rglob + stat on every
# file) to check staleness -- fine for one experiment, but with 50+ historical
# run dirs under `runs/`, calling it across the WHOLE list on every render
# (every `refresh_interval` seconds via auto-refresh) turns into a massive,
# unnecessary filesystem walk. `experiments` is already newest-mtime-first,
# so a genuinely active run is always near the front -- only check a handful.
_CANDIDATES_TO_CHECK = 5
active = [p for p in experiments[:_CANDIDATES_TO_CHECK] if ri.run_is_active(p)]
default_dir = active[0] if active else experiments[0]
selected_name = default_dir.name
st.sidebar.markdown(f"**Experiment (auto):** `{selected_name}`")
experiment_dir = default_dir

# exp_names = [p.name for p in experiments]
# selected_name = st.sidebar.selectbox("Experiment (newest first)", exp_names, index=0)
# experiment_dir = runs_root / selected_name

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
alerts = ri.extract_diagnostics(rounds, is_active=is_active)
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
# Block reward distributions (adaptive block_selection_strategy only)
# --------------------------------------------------------------------------- #

adaptive = ri.latest_adaptive_strategy(rounds)
if adaptive:
    st.subheader("Block reward distributions")
    st.caption(
        "Beta posterior per block, as of the most recent EXPAND (node "
        f"{max(r.node_id for r in rounds if r.adaptive_strategy is adaptive)}). "
        "Width = uncertainty, not just the mean -- a wide curve means the "
        "search hasn't ruled that block out yet, even if its mean looks low."
    )
    try:
        import matplotlib.pyplot as plt

        beta_prior = adaptive.get("beta_prior", 1.0)
        fig, ax = plt.subplots(figsize=(8, 3.2))
        for block, post in sorted(adaptive.get("posteriors", {}).items()):
            a = beta_prior + post["n_success"]
            b = beta_prior + post["n_failure"]
            xs, ys = ri.beta_pdf_curve(a, b)
            ax.plot(xs, ys, label=f"{block}  (n={post['n_evals']}, mean={post['mean']:.2f})")
            ax.axvline(post["mean"], linestyle=":", linewidth=1, alpha=0.5)
        ax.set_xlim(0, 1)
        ax.set_xlabel("reward")
        ax.set_ylabel("density")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, fontsize=8)
        fig.tight_layout()
        st.pyplot(fig)
    except Exception as exc:  # noqa: BLE001 -- matplotlib may be absent
        st.warning(f"Couldn't render the Beta curves ({exc}); showing raw posteriors instead.")
        st.json(adaptive.get("posteriors", {}))

    post_rows = [
        {
            "block": block,
            "mean": post["mean"],
            "n_evals": post["n_evals"],
            "n_success": round(post["n_success"], 2),
            "n_failure": round(post["n_failure"], 2),
            "last_sample": round(post["sampled_value"], 3),
        }
        for block, post in sorted(adaptive.get("posteriors", {}).items())
    ]
    st.dataframe(pd.DataFrame(post_rows), width="stretch", hide_index=True)


# --------------------------------------------------------------------------- #
# Diagnostics panel
# --------------------------------------------------------------------------- #

st.subheader("Diagnostics")
if not alerts:
    st.success("No issues detected.")
else:
    icon = {"error": "🔴", "warning": "🟡", "info": "🔵"}
    # A single crashed/broken node can produce one alert per case (up to
    # eval_batch_size each) -- e.g. one bad edit failing all 32 cases in a
    # batch floods this panel with 32 near-identical lines. Collapse to one
    # representative line per (node, severity), with a "(x N)" count, so the
    # panel reads as "which nodes have problems" rather than a raw error log.
    grouped: dict[tuple[int, str], list[str]] = {}
    for a in alerts:
        grouped.setdefault((a.node_id, a.severity), []).append(a.message)

    _MAX_GROUPS_SHOWN = 20
    group_items = sorted(
        grouped.items(),
        key=lambda kv: ({"error": 0, "warning": 1, "info": 2}.get(kv[0][1], 3), kv[0][0]),
    )
    st.caption(
        f"{len(alerts)} raw alert(s) across {len(grouped)} node(s) -- "
        "showing one representative message per node/severity."
    )
    for (node_id, severity), messages in group_items[:_MAX_GROUPS_SHOWN]:
        suffix = f"  *(×{len(messages)})*" if len(messages) > 1 else ""
        st.markdown(f"{icon.get(severity, '⚪')} **node {node_id}** — {messages[0]}{suffix}")
    if len(group_items) > _MAX_GROUPS_SHOWN:
        st.caption(f"... and {len(group_items) - _MAX_GROUPS_SHOWN} more node/severity group(s) not shown.")


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
    # mean_utility is 0.0 (not None) for a node with zero evals so far --
    # showing "mean=0.000" would misleadingly read as a failing score
    # rather than "hasn't been evaluated yet".
    display_mean = r.mean_utility if r.n_evals > 0 else None
    mu_str = f"{display_mean:.3f}" if display_mean is not None else "in-progress"
    label = f"node {r.node_id}\\nmean={mu_str}"
    if r.node_id in diffs_by_node:
        _, added, removed = diffs_by_node[r.node_id]
        label += f"\\n+{added}/-{removed}"
    if r.edit_failed:
        label += "\\nEDIT FAILED"
    color = _color_for(display_mean, r.edit_failed)
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
            "block": (r.strategy or {}).get("block"),
            "edit_failed": r.edit_failed,
            # mean_utility is 0.0 (not None) for a node with zero evals so
            # far -- "0.000" would misleadingly read as a failing score
            # rather than "hasn't been evaluated yet". NaN (not the string
            # "--") so the column stays numeric -- Arrow can't serialize a
            # float column with a string mixed in, and st.dataframe renders
            # NaN as a blank cell, which reads the same way to a viewer.
            "mean_utility": r.mean_utility if r.n_evals > 0 else float("nan"),
            "n_evals": r.n_evals,
            "cmp": r.cmp,
            "+added": added,
            "-removed": removed,
            "optimization_goal": r.optimization_goal[:100],
        }
    )
nodes_df = pd.DataFrame(rows)
st.dataframe(nodes_df, width="stretch", hide_index=True)

trend_df = nodes_df[nodes_df["n_evals"] > 0][["node_id", "mean_utility"]]
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
    agg = round_.behavior_aggregate or {}
    check_reliability = agg.get("check_reliability") or {}
    vs_parent = agg.get("check_reliability_vs_parent") or {}

    if vs_parent:
        st.markdown(
            "**Check/constraint reliability vs. parent** — significance "
            "via Fisher's exact test (pre-computed, not LLM judgment)"
        )
        vs_parent_rows = [
            {
                "check": check,
                "parent (failed/n)": f"{e['parent_failed_in_n_cases']}/{e['parent_n_cases']}",
                "child (failed/n)": f"{e['child_failed_in_n_cases']}/{e['child_n_cases']}",
                "direction": e["direction"],
                "significance": "✅ significant" if e["significant"] else "— within noise",
            }
            for check, e in vs_parent.items()
        ]
        st.dataframe(pd.DataFrame(vs_parent_rows), width="stretch", hide_index=True)

    if check_reliability.get("checks"):
        st.markdown(
            f"**Check/constraint reliability, this round alone** "
            f"(n={check_reliability.get('n_cases')} cases; rarest failures first)"
        )
        check_rows = [
            {
                "check": check,
                "failed_in_n_cases": e["failed_in_n_cases"],
                "sample_failing_case_ids": ", ".join(e["sample_failing_case_ids"]),
            }
            for check, e in check_reliability["checks"].items()
        ]
        st.dataframe(pd.DataFrame(check_rows), width="stretch", hide_index=True)

    if vs_parent or check_reliability.get("checks"):
        st.markdown("---")

    if round_.behavior_memory:
        st.markdown(round_.behavior_memory)
    elif not (vs_parent or check_reliability.get("checks")):
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
