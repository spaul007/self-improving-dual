"""Single-run view: header, budget, diagnostics, tree, nodes, round drill-down."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from meta_agent import run_inspect as ri
from meta_agent import run_inspect_agentic as ra
from meta_agent.feedback_gatherer import render_metrics

from . import components as C
from . import loaders as L


def _header(cfg: dict[str, Any], name: str, is_active: bool) -> None:
    badge = "🟢 LIVE" if is_active else "⚪ STOPPED"
    st.title(f"{cfg.get('experiment_name', name)} — {badge}")
    st.caption(name)
    em_cfg = ri.edit_memory_config(cfg)
    cols = st.columns(7)
    cols[0].metric("Project", cfg.get("project", "?"))
    cols[1].metric("Manager", (cfg.get("manager") or {}).get("type", "?"))
    cols[2].metric("Task-agent model", (ri.task_agent_model(cfg) or "?").split("/")[-1])
    cols[3].metric("Editor model", (ri.editor_model(cfg) or "?").split("/")[-1])
    cols[4].metric("read_scope", ri.editor_read_scope(cfg) or "—")
    cols[5].metric("Eval budget", ((cfg.get("manager") or {}).get("config") or {}).get("eval_budget", "?"))
    cols[6].metric("Edit memory", f"on ({em_cfg.get('selection', '?')})" if em_cfg else "off")


def _evaluation_tab(round_: ri.RoundInfo) -> None:
    er = round_.eval_result or {}
    if er.get("_synthesized_from_case_logs"):
        st.warning(
            "eval_result.json is missing or behind — showing per-case results reconstructed "
            "from logs/case_*.json (normal while a batch is still running)."
        )
    per_case = er.get("per_case") or []
    if not per_case:
        st.info("No per-case results yet for this round.")
        return
    n_err = sum(1 for c in per_case if c.get("error"))
    c = st.columns(4)
    c[0].metric("Cases", len(per_case))
    c[1].metric("Passed", er.get("passed", sum(1 for x in per_case if x.get("passed"))))
    c[2].metric("Errors / timeouts", n_err)
    scores = [float(x.get("score") or 0) for x in per_case]
    c[3].metric("Mean score", f"{sum(scores)/len(scores):.3f}" if scores else "—")

    left, right = st.columns([3, 2])
    with left:
        st.markdown("**Dimension means** (non-error cases)")
        C.dimension_bar(ri.dimension_means(er))
    with right:
        hard = ri.hard_constraint_failure_counts(er)
        checks = ri.failed_check_counts(er)
        if hard:
            st.markdown("**Hard-constraint failures**")
            st.dataframe(pd.DataFrame(hard, columns=["constraint", "cases"]), width="stretch", hide_index=True, height=180)
        if checks:
            st.markdown("**Most-failed checks**")
            st.dataframe(pd.DataFrame(checks[:15], columns=["check", "cases"]), width="stretch", hide_index=True, height=220)

    st.markdown("**Per case**")
    st.dataframe(pd.DataFrame(ri.per_case_dimension_rows(er)), width="stretch", hide_index=True)
    case_ids = [c.get("case_id") for c in per_case]
    sel = st.selectbox("Inspect case", case_ids, key=f"case_{round_.node_id}")
    case_obj = next(c for c in per_case if c.get("case_id") == sel)
    details = dict(case_obj.get("details") or {})
    plan = details.pop("raw_result", None)
    query = details.pop("query", None)
    if query:
        st.markdown("**Query**")
        st.markdown(str(query))
    if plan:
        with st.expander("Agent output (raw_result)"):
            st.text(str(plan))
    st.json({k: v for k, v in case_obj.items() if k != "details"} | {"details": details}, expanded=1)


def _diff_tab(round_: ri.RoundInfo, diffs_by_node: dict[int, tuple[dict[str, ri.FileDiff], int, int]]) -> None:
    if round_.parent_id is None:
        st.info("Root round — no parent to diff against.")
    elif round_.node_id not in diffs_by_node:
        st.info("No diff available (edit failed before producing a workspace, or parent missing).")
    else:
        diffs, added, removed = diffs_by_node[round_.node_id]
        if not diffs:
            st.info("No changed files on the mutable surface.")
        else:
            st.markdown(f"**{len(diffs)} file(s) changed, +{added}/-{removed} lines**")
            for path, d in diffs.items():
                with st.expander(f"{path}  (+{d.lines_added}/-{d.lines_removed}, {d.status})", expanded=len(diffs) == 1):
                    st.code(d.diff_text, language="diff")


def _feedback_tab(round_: ri.RoundInfo) -> None:
    fb = round_.feedback or {}
    if not fb:
        st.info("No feedback.json yet for this round.")
        return
    c = st.columns(4)
    c[0].metric("LLM calls (last batch)", fb.get("llm_calls", "—"))
    c[1].metric("Cases traced", fb.get("trace_n_cases", "—"))
    tu = fb.get("tool_usage") or {}
    c[2].metric("Tool calls", sum(tu.values()) if tu else "—")
    ter = fb.get("tool_error_rate") or {}
    worst = max(ter.items(), key=lambda kv: kv[1]) if ter else None
    c[3].metric("Worst tool error rate", f"{worst[0]}: {worst[1]:.0%}" if worst else "—")
    st.caption("Trace-derived numbers (tool usage, LLM calls, failure report) cover only the most recent evaluation batch; scores are cumulative.")
    if tu:
        st.dataframe(pd.DataFrame([{"tool": k, "calls": v, "error_rate": ter.get(k)} for k, v in sorted(tu.items(), key=lambda kv: -kv[1])]),
                     width="stretch", hide_index=True)
    pm = fb.get("project_metrics") or {}
    if pm:
        st.markdown("**Project metrics**")
        st.code("\n".join(render_metrics(pm, cap=20)))
    for key, title in (("runtime_exceptions", "Runtime exceptions"), ("edit_errors", "Edit errors")):
        items = fb.get(key) or []
        if items:
            st.markdown(f"**{title}** ({len(items)})")
            for x in items[:30]:
                st.code(str(x))
            if len(items) > 30:
                st.caption(f"... {len(items) - 30} more")
    fr = fb.get("failure_report") or {}
    if fr:
        st.markdown("**Failure report**")
        summ = fr.get("summary") or {}
        if summ:
            st.json(summ, expanded=1)
        cats = fr.get("categories") or []
        if cats:
            st.dataframe(pd.DataFrame(cats), width="stretch", hide_index=True)
        with st.expander("Examples / hard cases"):
            st.json({"examples": fr.get("examples") or [], "hard_cases": fr.get("hard_cases") or []}, expanded=1)


def render(experiment_dir: Path) -> bool:
    """Returns the run's liveness so the entry point can decide on refresh."""
    cfg = L.cached_config(experiment_dir)
    rounds = L.cached_rounds(experiment_dir)
    snapshots = L.cached_snapshots(experiment_dir)
    is_active = ri.run_is_active(experiment_dir)
    _header(cfg, experiment_dir.name, is_active)

    if not rounds:
        st.info("No round_* directories yet.")
        return is_active

    rounds_by_id = {r.node_id: r for r in rounds}
    diffs_by_node: dict[int, tuple[dict[str, ri.FileDiff], int, int]] = {}
    for r in rounds:
        if r.parent_id is None or r.parent_id not in rounds_by_id or not r.has_task_agent:
            continue
        diffs = L.cached_diff(rounds_by_id[r.parent_id].round_dir, r.round_dir)
        diffs_by_node[r.node_id] = (diffs, *ri.diff_totals(diffs))

    budget = ri.budget_progress(cfg, snapshots, rounds)
    if budget.total:
        spent = budget.spent or 0
        text = f"Budget: {spent}/{budget.total} evals"
        if not budget.exact:
            text += "  (approximate — set `manager.config.snapshot_tree: true` for exact tracking)"
        st.progress(max(0.0, min(1.0, spent / budget.total)), text=text)

    best_node_id = snapshots[-1].get("best_node_id") if snapshots else None
    show_arms = any(r.memory_arm != "none" for r in rounds)

    st.subheader("Diagnostics")
    C.diagnostics_panel(ri.extract_diagnostics(rounds, is_active=is_active))

    st.subheader("Tree")
    C.tree_chart(rounds, diffs_by_node, show_arms=show_arms, best_node_id=best_node_id)

    st.subheader("Nodes")
    df = C.nodes_table(rounds, diffs_by_node, show_arms=show_arms)
    st.dataframe(df, width="stretch", hide_index=True)
    trend = df[df["n_evals"] > 0][["node", "mean_utility"]]
    if len(trend) > 1:
        cols = st.columns([3, 2]) if show_arms else [st.container()]
        with cols[0]:
            st.line_chart(trend, x="node", y="mean_utility", x_label="node", y_label="train mean",
                          color=C.SERIES[0], height=220)
        if show_arms:
            with cols[1]:
                summ = ri.arm_utility_summary(rounds)
                C.arm_bar(summ)
                st.dataframe(pd.DataFrame([{"arm": k, **v} for k, v in summ.items()]), width="stretch", hide_index=True)

    st.subheader("Round drill-down")
    node_ids = [r.node_id for r in rounds]
    sel = st.selectbox("Node", node_ids, index=len(node_ids) - 1, key="drill_node",
                       format_func=lambda n: f"node {n}" + (" (best)" if n == best_node_id else ""))
    round_ = rounds_by_id[sel]
    t_strategy, t_agentic, t_eval, t_diff, t_feedback = st.tabs(
        ["Strategy", "Agentic session", "Evaluation", "Diff vs parent", "Feedback"]
    )
    with t_strategy:
        if round_.strategy:
            s = round_.strategy
            st.markdown(f"**Goal:** {s.get('optimization_goal') or '—'}")
            st.markdown(f"**Target files:** {', '.join(s.get('target_files') or []) or '—'}")
            st.markdown("**Proposed changes**")
            st.markdown(str(s.get("proposed_changes") or "—"))
            st.markdown("**Rationale**")
            st.markdown(str(s.get("rationale") or "—"))
        else:
            st.info("No strategy.json (seed round, or the editor hasn't submitted yet).")
    with t_agentic:
        if round_.is_root:
            st.info("Seed round — no editor session.")
        elif not round_.has_agentic and round_.agentic_session is None:
            st.info("No agentic/ directory yet for this round.")
        else:
            tr = L.cached_transcript(ra.transcript_path(round_.round_dir / "agentic")) if round_.has_agentic else None
            prompts = ra.load_verbose_prompts(round_.round_dir) if round_.has_verbose else None
            C.transcript_view(tr, round_.agentic_session, key_prefix=f"r{round_.node_id}", prompts=prompts)
    with t_eval:
        _evaluation_tab(round_)
    with t_diff:
        _diff_tab(round_, diffs_by_node)
    with t_feedback:
        _feedback_tab(round_)

    summary_md = ri.load_run_summary(experiment_dir)
    if summary_md:
        st.subheader("Run summary")
        st.markdown(summary_md)
    return is_active
