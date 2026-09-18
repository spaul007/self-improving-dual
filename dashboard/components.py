"""Widgets shared by more than one view: diagnostics panel, tree diagram,
nodes table, the agentic transcript viewer, and the altair charts.

Colors follow one rule set: categorical hues are assigned in a fixed order
(``SERIES`` -- run 1 is always blue, run 2 orange, ...), magnitude (node
train-mean) is one blue ramp light->dark, and identity of the memory arm is
carried by node border/shape in the tree so it never rests on color alone.
"""
from __future__ import annotations

from typing import Any, Optional

import altair as alt
import pandas as pd
import streamlit as st

from meta_agent import run_inspect as ri
from meta_agent import run_inspect_agentic as ra

# Categorical slots, fixed order (blue, orange, aqua, yellow, magenta, green,
# violet, red). Compare view maps run k -> SERIES[k].
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
ARM_COLOR = {"with": SERIES[0], "without": SERIES[1], "none": "#8a8985"}
NEUTRAL_FILL = "#e2e1dd"
_SEV_ICON = {"error": "🔴", "warning": "🟡", "info": "🔵"}


# --------------------------------------------------------------------------- #
# Diagnostics
# --------------------------------------------------------------------------- #


def diagnostics_panel(alerts: list[ri.Alert], *, max_groups: int = 20) -> None:
    if not alerts:
        st.success("No issues detected.")
        return
    # One bad node can produce one alert per case (up to eval_batch_size) --
    # collapse to one representative line per (node, severity) with a count,
    # so the panel reads as "which nodes have problems", not a raw error log.
    grouped: dict[tuple[int, str], list[str]] = {}
    for a in alerts:
        grouped.setdefault((a.node_id, a.severity), []).append(a.message)
    items = sorted(
        grouped.items(),
        key=lambda kv: ({"error": 0, "warning": 1, "info": 2}.get(kv[0][1], 3), kv[0][0]),
    )
    st.caption(
        f"{len(alerts)} raw alert(s) across {len(grouped)} node/severity group(s) -- "
        "one representative message per group."
    )
    for (node_id, severity), messages in items[:max_groups]:
        suffix = f"  *(×{len(messages)})*" if len(messages) > 1 else ""
        st.markdown(f"{_SEV_ICON.get(severity, '⚪')} **node {node_id}** — {messages[0]}{suffix}")
    if len(items) > max_groups:
        st.caption(f"... and {len(items) - max_groups} more group(s) not shown.")


# --------------------------------------------------------------------------- #
# Tree
# --------------------------------------------------------------------------- #


def _lerp(a: tuple[int, int, int], b: tuple[int, int, int], t: float) -> str:
    return "#%02x%02x%02x" % tuple(int(round(a[k] + (b[k] - a[k]) * t)) for k in range(3))


_RAMP_LO = (0xE3, 0xEE, 0xFB)  # blue-100
_RAMP_HI = (0x14, 0x3F, 0x7C)  # blue-700


def utility_fill(mean_utility: Optional[float]) -> tuple[str, str]:
    """(fill, font color) for a node given its train mean -- one blue ramp,
    light->dark; ink flips to white once the fill gets dark."""
    if mean_utility is None:
        return NEUTRAL_FILL, "#0b0b0b"
    mu = max(0.0, min(1.0, float(mean_utility)))
    return _lerp(_RAMP_LO, _RAMP_HI, mu), ("#ffffff" if mu > 0.55 else "#0b0b0b")


def tree_dot(
    rounds: list[ri.RoundInfo],
    diffs_by_node: dict[int, tuple[dict[str, ri.FileDiff], int, int]],
    *,
    show_arms: bool,
    best_node_id: Optional[int] = None,
) -> str:
    ids = {r.node_id for r in rounds}
    lines = [
        "digraph tree {",
        'rankdir=TB; bgcolor="transparent";',
        'node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=11, color="#b8b7b2", penwidth=1];',
        'edge [color="#8a8985"];',
    ]
    for r in rounds:
        # mean_utility is 0.0 (not None) for a node with zero evals so far --
        # "mean=0.000" would read as a failing score rather than "not yet".
        display_mean = r.mean_utility if r.n_evals > 0 else None
        mu_str = f"{display_mean:.3f}" if display_mean is not None else "in-progress"
        label = f"node {r.node_id}\\nmean={mu_str}  n={r.n_evals}"
        if r.node_id in diffs_by_node:
            _, added, removed = diffs_by_node[r.node_id]
            label += f"\\n+{added}/-{removed}"
        if show_arms and not r.is_root:
            label += f"\\n{r.memory_arm}" + (f" v{r.memory_version}" if r.memory_version is not None else "")
        if r.edit_failed:
            label += "\\nEDIT FAILED"
        fill, ink = utility_fill(None if r.edit_failed else display_mean)
        attrs = [f'label="{label}"', f'fillcolor="{fill}"', f'fontcolor="{ink}"']
        if show_arms and not r.is_root:
            if r.memory_arm == "with":
                attrs += [f'color="{ARM_COLOR["with"]}"', "penwidth=3"]
            elif r.memory_arm == "without":
                attrs += [f'color="{ARM_COLOR["without"]}"', "penwidth=3", 'style="filled,rounded,dashed"']
        if best_node_id is not None and r.node_id == best_node_id:
            attrs += ["peripheries=2"]
        lines.append(f"  n{r.node_id} [{', '.join(attrs)}];")
        if r.parent_id is not None and r.parent_id in ids:
            lines.append(f"  n{r.parent_id} -> n{r.node_id};")
    lines.append("}")
    return "\n".join(lines)


def tree_chart(
    rounds: list[ri.RoundInfo],
    diffs_by_node: dict[int, tuple[dict[str, ri.FileDiff], int, int]],
    *,
    show_arms: bool,
    best_node_id: Optional[int] = None,
) -> None:
    caption = "Fill = train mean (light → dark). Grey = not evaluated / edit failed. Double border = current best."
    if show_arms:
        caption += "  Memory arm: **with** = solid blue border, **without** = dashed orange border, none = plain."
    st.caption(caption)
    try:
        st.graphviz_chart(tree_dot(rounds, diffs_by_node, show_arms=show_arms, best_node_id=best_node_id))
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Tree diagram couldn't render ({exc}); the nodes table below has the same information.")


# --------------------------------------------------------------------------- #
# Nodes table
# --------------------------------------------------------------------------- #


def nodes_table(
    rounds: list[ri.RoundInfo],
    diffs_by_node: dict[int, tuple[dict[str, ri.FileDiff], int, int]],
    *,
    show_arms: bool,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for r in rounds:
        added = removed = 0
        if r.node_id in diffs_by_node:
            _, added, removed = diffs_by_node[r.node_id]
        sess = r.agentic_session or {}
        row: dict[str, Any] = {
            "node": r.node_id,
            "parent": r.parent_id,
            "edit_failed": r.edit_failed,
            # NaN (not a string) so the column stays numeric for Arrow; it
            # renders as a blank cell, which reads as "not yet evaluated".
            "mean_utility": r.mean_utility if r.n_evals > 0 else float("nan"),
            "n_evals": r.n_evals,
            "cmp": r.cmp,
        }
        if show_arms:
            row["arm"] = "" if r.is_root else r.memory_arm
            row["mem_v"] = r.memory_version
        row.update(
            {
                "+": added,
                "-": removed,
                "files": ", ".join(r.changed_files),
                "session_end": r.session_end_reason,
                "llm_calls": sess.get("n_llm_calls"),
                "editor_s": round(float(sess["elapsed_s"])) if sess.get("elapsed_s") is not None else None,
                "goal": r.optimization_goal[:120],
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Agentic transcript
# --------------------------------------------------------------------------- #


def _fmt_tokens(n: Any) -> str:
    try:
        n = int(n)
    except (TypeError, ValueError):
        return "—"
    return f"{n/1000:.1f}k" if n >= 1000 else str(n)


def _session_metrics(session: dict[str, Any], tr: Optional[ra.Transcript]) -> None:
    usage = session.get("usage") or {}
    counts = session.get("n_tool_calls") or (ra.tool_call_counts(tr) if tr else {})
    c = st.columns(6)
    ok = session.get("success")
    c[0].metric("Outcome", ("✅ " if ok else "❌ ") + str(session.get("end_reason") or "?"))
    c[1].metric("LLM calls", session.get("n_llm_calls", len(tr.steps) if tr else "—"))
    c[2].metric("Tool calls", " · ".join(f"{k} {v}" for k, v in counts.items()) or "—")
    c[3].metric("Tokens in / out", f"{_fmt_tokens(usage.get('input_tokens'))} / {_fmt_tokens(usage.get('output_tokens'))}",
                help=f"reasoning tokens: {_fmt_tokens(usage.get('reasoning_tokens'))}")
    elapsed = session.get("elapsed_s")
    c[4].metric("Wall time", f"{float(elapsed)/60:.1f} min" if elapsed is not None else "—")
    extra = []
    if session.get("read_scope"):
        extra.append(f"read_scope={session['read_scope']}")
    if "memory_path" in session:
        extra.append("memory: " + ("with" if session.get("memory_path") else "without"))
    if session.get("output_file"):
        extra.append(f"output={session['output_file']}")
    c[5].metric("Config", " · ".join(extra) or session.get("sandbox_mode", "—"))
    if session.get("errors"):
        for e in session["errors"]:
            st.error(str(e)[:500])
    if session.get("summary"):
        st.markdown("**Curator summary**")
        st.markdown(str(session["summary"]))


def _tool_call_block(tc: ra.ToolCallRec) -> None:
    head = f"`{tc.name}`"
    if tc.name == "editor":
        head += f" — {ra.editor_call_summary(tc.input)}"
    elif tc.name == "bash":
        cmd = str(tc.input.get("command") or "")
        first = cmd.strip().splitlines()[0] if cmd.strip() else ""
        head += f" — `{first[:100]}{'…' if len(first) > 100 else ''}`"
    trunc = "" if tc.result_chars <= len(tc.result) else f", showing {len(tc.result)}"
    head += f"  <small>({tc.elapsed_s:.2f}s, {tc.result_chars} chars{trunc})</small>"
    with st.container(border=True):
        st.markdown(head, unsafe_allow_html=True)
        if tc.name == "bash":
            st.code(str(tc.input.get("command") or ""), language="bash")
            if tc.result:
                st.code(tc.result, language="text")
        elif tc.name == "editor":
            diff = ra.editor_call_as_diff(tc.input)
            if diff:
                st.code(diff, language="diff")
            if tc.result:
                st.code(tc.result, language="text")
        elif tc.name in ("submit_self_improvement", "submit_curation"):
            st.json(tc.input)
            if tc.result:
                st.caption(tc.result[:300])
        else:  # validate & anything new
            if tc.input:
                st.json(tc.input)
            if tc.result:
                st.code(tc.result, language="text")


def _note_line(ev: dict[str, Any]) -> None:
    kind = ev.get("kind")
    if kind == "validation":
        errs = ev.get("errors") or []
        msg = f"validation round {ev.get('round')}: changed {ev.get('changed_files') or []}"
        if errs:
            st.error(msg + "\n\n" + "\n".join(f"- {str(e)[:300]}" for e in errs))
        else:
            st.success(msg + " — all validators passed")
    elif kind == "budget_reminder":
        st.info(f"budget reminder: {ev.get('used')}/{ev.get('total')} LLM calls used")
    elif kind == "wrap_up":
        st.warning(f"wrap-up requested: {ev.get('reason')}")
    elif kind == "nudge":
        st.info("nudge: model returned no tool call, asked to continue")
    else:
        st.info(f"{kind}: " + ", ".join(f"{k}={v}" for k, v in ev.items() if k not in ("kind", "t")))


def transcript_view(
    tr: Optional[ra.Transcript],
    session: Optional[dict[str, Any]],
    *,
    key_prefix: str,
    prompts: Optional[dict[str, str]] = None,
) -> None:
    """Session summary + per-LLM-call timeline. Rendered for both the editor
    session of a round and the curator sessions inside edit_memory/, so
    every widget key carries ``key_prefix``."""
    if session:
        _session_metrics(session, tr)
    if tr is None:
        st.info("No transcript.jsonl yet.")
        return
    if not tr.steps:
        st.info("Transcript has no LLM calls yet.")
        return

    st.caption(
        f"{len(tr.steps)} LLM call(s), {sum(len(s.tool_calls) for s in tr.steps)} tool call(s), "
        f"{ra.fmt_time(tr.t_start)} → {ra.fmt_time(tr.t_end)}"
    )
    if tr.end is None or tr.truncated_tail:
        st.warning("Transcript has no `end` event yet — the editor session is still running (or was killed).")

    curve = pd.DataFrame(ra.token_curve(tr))
    if len(curve) > 1:
        st.line_chart(curve, x="i", y=["cum_input", "cum_output"], x_label="LLM call", y_label="cumulative tokens",
                      color=[SERIES[0], SERIES[1]], height=180)

    if prompts:
        with st.expander("Prompts (system + instruction)", expanded=False):
            for k, v in prompts.items():
                st.markdown(f"**{k}**")
                st.code(v, language="markdown")

    only_edits = st.toggle("Only steps with editor/validate/submit calls", value=False, key=f"{key_prefix}_only_edits")
    n_last_open = 2
    for idx, s in enumerate(tr.steps):
        names = [tc.name for tc in s.tool_calls]
        if only_edits and not any(n in ("editor", "validate", "submit_self_improvement", "submit_curation") for n in names):
            continue
        title = f"#{s.i}"
        if s.error:
            title += "  ·  ❌ llm_error"
        else:
            title += f"  ·  {s.stop_reason or '…'}"
        if names:
            counts: dict[str, int] = {}
            for n in names:
                counts[n] = counts.get(n, 0) + 1
            title += "  ·  " + ", ".join(f"{k}×{v}" if v > 1 else k for k, v in counts.items())
        if s.elapsed_s is not None:
            title += f"  ·  {s.elapsed_s:.1f}s"
        if s.usage:
            title += f"  ·  in {_fmt_tokens(s.usage.get('input_tokens'))} / out {_fmt_tokens(s.usage.get('output_tokens'))}"
        with st.expander(title, expanded=idx >= len(tr.steps) - n_last_open):
            if s.content:
                with st.container(border=True):
                    st.markdown(s.content)
            if s.error:
                st.error(s.error[:1000])
            for tc in s.tool_calls:
                _tool_call_block(tc)
            for ev in s.notes:
                _note_line(ev)

    if tr.end is not None:
        msg = (f"session ended: **{tr.end.get('reason')}** — {tr.end.get('n_llm_calls')} LLM calls, "
               f"{tr.end.get('validation_rounds')} validation round(s)")
        (st.success if tr.end.get("success") else st.error)(msg)


# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #


def dimension_bar(means: dict[str, float], *, title: str = "") -> None:
    if not means:
        st.info("No dimension scores available.")
        return
    df = pd.DataFrame({"dimension": list(means), "mean": list(means.values())})
    chart = (
        alt.Chart(df, title=title)
        .mark_bar(color=SERIES[0], cornerRadiusEnd=4, size=14)
        .encode(
            x=alt.X("mean:Q", scale=alt.Scale(domain=[0, 1]), title="mean score"),
            y=alt.Y("dimension:N", sort="-x", title=None),
            tooltip=[alt.Tooltip("dimension:N"), alt.Tooltip("mean:Q", format=".3f")],
        )
        .properties(height=max(120, 22 * len(df)))
    )
    text = chart.mark_text(align="left", dx=4, color="#52514e").encode(text=alt.Text("mean:Q", format=".2f"))
    st.altair_chart(chart + text, width="stretch")


def best_so_far_chart(
    series: dict[str, list[ri.CurvePoint]],
    overlay: Optional[dict[str, list[dict[str, Any]]]] = None,
) -> None:
    """Best train-mean vs budget spent, one step line per run (fixed hue by
    run order) with optional held-out points from ``eval_at_budget_*``."""
    names = list(series)
    rows = [
        {"run": name, "budget_spent": p.budget_spent, "best_mean": p.best_mean_utility, "best_node": p.best_node_id}
        for name, pts in series.items()
        for p in pts
    ]
    if not rows:
        st.info("No tree snapshots in the selected runs (needs `manager.config.snapshot_tree: true`).")
        return
    df = pd.DataFrame(rows)
    color = alt.Color("run:N", scale=alt.Scale(domain=names, range=SERIES[: len(names)]),
                      legend=alt.Legend(orient="bottom", columns=1, title=None))
    line = (
        alt.Chart(df)
        .mark_line(interpolate="step-after", strokeWidth=2)
        .encode(
            x=alt.X("budget_spent:Q", title="evaluations spent"),
            y=alt.Y("best_mean:Q", title="best train mean", scale=alt.Scale(zero=False)),
            color=color,
            tooltip=["run:N", "budget_spent:Q", alt.Tooltip("best_mean:Q", format=".3f"), "best_node:Q"],
        )
    )
    pts = (
        alt.Chart(df)
        .mark_point(size=40, filled=True)
        .encode(x="budget_spent:Q", y="best_mean:Q", color=color,
                tooltip=["run:N", "budget_spent:Q", alt.Tooltip("best_mean:Q", format=".3f"), "best_node:Q"])
    )
    layers = [line, pts]
    orows = [
        {"run": name, "budget_spent": e.get("snapshot_budget", e.get("requested_budget")),
         "held_out": e.get("composite_score"), "node": e.get("node_id"), "case_set": e.get("case_set")}
        for name, evs in (overlay or {}).items()
        for e in evs
        if e.get("composite_score") is not None
    ]
    if orows:
        odf = pd.DataFrame(orows)
        layers.append(
            alt.Chart(odf)
            .mark_point(shape="diamond", size=110, strokeWidth=2)
            .encode(x="budget_spent:Q", y=alt.Y("held_out:Q"), color=color,
                    tooltip=["run:N", "budget_spent:Q", alt.Tooltip("held_out:Q", format=".3f"), "node:Q", "case_set:N"])
        )
    st.altair_chart(alt.layer(*layers).properties(height=340).interactive(), width="stretch")
    if orows:
        st.caption("Lines: best train mean at each budget. Diamonds: held-out composite of the node selected at that budget (`snapshot_eval.py`).")


def grouped_dimension_bar(wide: pd.DataFrame, run_names: list[str]) -> None:
    """``wide``: index = dimension, one column per run."""
    long = wide.reset_index().melt(id_vars="dimension", var_name="run", value_name="mean").dropna()
    if long.empty:
        return
    chart = (
        alt.Chart(long)
        .mark_bar(cornerRadiusEnd=3)
        .encode(
            y=alt.Y("dimension:N", title=None),
            x=alt.X("mean:Q", scale=alt.Scale(domain=[0, 1]), title="best node's mean score"),
            yOffset="run:N",
            color=alt.Color("run:N", scale=alt.Scale(domain=run_names, range=SERIES[: len(run_names)]),
                            legend=alt.Legend(orient="bottom", columns=1, title=None)),
            tooltip=["run:N", "dimension:N", alt.Tooltip("mean:Q", format=".3f")],
        )
        .properties(height=max(160, 22 * len(wide) * max(1, len(run_names))))
    )
    st.altair_chart(chart, width="stretch")


def beta_curves_chart(
    entries: list[dict[str, Any]],
    *,
    x_title: str = "success rate θ",
    height: int = 260,
    x_domain: Optional[tuple[float, float]] = None,
) -> None:
    """Density curves for several Beta posteriors with a dotted line at each
    mean. ``entries``: ``[{"name", "label", "a", "b", "color"}]`` -- ``name``
    is the short id used in tooltips, ``label`` the legend text."""
    rows, means = [], []
    for e in entries:
        xs, ys = ri.beta_pdf_curve(e["a"], e["b"])
        rows += [{"name": e["name"], "label": e["label"], "x": x, "density": y} for x, y in zip(xs, ys)]
        means.append({"label": e["label"], "mean": e["a"] / (e["a"] + e["b"])})
    if not rows:
        st.info("Nothing to plot yet.")
        return
    df, mdf = pd.DataFrame(rows), pd.DataFrame(means)
    order = [e["label"] for e in entries]
    color = alt.Color("label:N", scale=alt.Scale(domain=order, range=[e["color"] for e in entries]),
                      legend=alt.Legend(orient="bottom", columns=1, title=None))
    if x_domain is None:
        # Zoom to where the mass is: the union of the curves' 0.5%–99.5%
        # ranges, padded, so sharp posteriors don't become a spike.
        lo = min(ri.beta_summary(e["a"], e["b"])["lo90"] for e in entries)
        hi = max(ri.beta_summary(e["a"], e["b"])["hi90"] for e in entries)
        pad = max(0.05, (hi - lo) * 0.6)
        x_domain = (max(0.0, lo - pad), min(1.0, hi + pad))
    curves = (
        alt.Chart(df)
        .mark_line(strokeWidth=2, clip=True)
        .encode(
            x=alt.X("x:Q", scale=alt.Scale(domain=list(x_domain)), title=x_title),
            y=alt.Y("density:Q", title="posterior density"),
            color=color,
            tooltip=["name:N", alt.Tooltip("x:Q", format=".3f"), alt.Tooltip("density:Q", format=".2f")],
        )
    )
    rules = alt.Chart(mdf).mark_rule(strokeDash=[3, 3], strokeWidth=1.5).encode(x="mean:Q", color=color)
    st.altair_chart(alt.layer(curves, rules).properties(height=height).interactive(bind_y=False), width="stretch")


def arm_beta_chart(tallies: dict[str, dict[str, float]], beta_prior: float) -> None:
    """The bandit's two posteriors, Beta(S+prior, F+prior) per arm."""
    entries = []
    for arm in ("with", "without"):
        t = tallies.get(arm)
        if not t:
            continue
        entries.append({
            "name": arm,
            "label": f"{arm}  (nodes={int(t['n_nodes'])}, S={t['S']:.1f}, F={t['F']:.1f})",
            "a": t["S"] + beta_prior, "b": t["F"] + beta_prior, "color": ARM_COLOR[arm],
        })
    beta_curves_chart(entries, x_title="expansion success rate θ", x_domain=(0.0, 1.0))


def arm_bar(summary: dict[str, dict[str, Any]]) -> None:
    rows = [
        {"arm": arm, "mean_of_means": e["mean_of_means"], "n": e["n_evaluated"]}
        for arm, e in summary.items()
        if e.get("mean_of_means") is not None
    ]
    if not rows:
        return
    df = pd.DataFrame(rows)
    order = [a for a in ("with", "without", "none") if a in set(df["arm"])]
    chart = (
        alt.Chart(df)
        .mark_bar(size=22, cornerRadiusEnd=4)
        .encode(
            x=alt.X("arm:N", sort=order, title=None),
            y=alt.Y("mean_of_means:Q", scale=alt.Scale(domain=[0, 1]), title="mean of node train means"),
            color=alt.Color("arm:N", scale=alt.Scale(domain=order, range=[ARM_COLOR[a] for a in order]), legend=None),
            tooltip=["arm:N", alt.Tooltip("mean_of_means:Q", format=".3f"), alt.Tooltip("n:Q", title="evaluated nodes")],
        )
        .properties(height=220)
    )
    text = chart.mark_text(dy=-8, color="#52514e").encode(text=alt.Text("mean_of_means:Q", format=".3f"))
    st.altair_chart(chart + text, width="stretch")
