"""Streamlit views for the agentic editor and the edit-memory layer, used by
``hgm_dashboard.py``: the editor-session transcript viewer, the assignment
panel, and the Edit memory section (arm posteriors, memory / instruction
versions, curation windows). Every panel renders only when the run has the
data, so dashboards of older runs are unchanged.

Ported from the sep18 dashboard (``dashboard/components.py`` and
``dashboard/editmem_view.py`` of edit-memory-sep18-agentic); data comes from
the Streamlit-free ``meta_agent.run_inspect`` / ``run_inspect_agentic``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import altair as alt
import pandas as pd
import streamlit as st

from meta_agent import run_inspect as ri
from meta_agent import run_inspect_agentic as ra

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
ARM_COLOR = {"with": SERIES[0], "without": SERIES[1], "none": "#8a8985"}
NEUTRAL_FILL = "#e2e1dd"


# --------------------------------------------------------------------------- #
# Memory arm in the tree diagram and the nodes table
# --------------------------------------------------------------------------- #

SEED_COLOR = "#52514e"
PENDING_COLOR = "#3d3c3a"
_ARM_LABEL = {"with": "with memory", "without": "without memory", "none": "before memory",
              "pending": "editing…"}


def _arm_of(r: ri.RoundInfo) -> str:
    """"with" / "without" / "none", or "pending" while the node's editor
    session is still running: its arm is recorded only when the edit ends
    (hgm_node.json for a success, feedback.json + the layer's state for a
    failure), so it isn't shown as "before memory" in the meantime."""
    if (not r.is_root and r.hgm_node is None and r.feedback is None
            and r.state_arm is None and r.agentic_session is None):
        return "pending"
    arm = r.memory_arm
    return arm if arm in ("with", "without") else "none"


def _arm_badge(r: ri.RoundInfo) -> tuple[str, str]:
    """(band text, band color) for a node: the arm its edit ran under."""
    if r.is_root:
        return "SEED", SEED_COLOR
    arm = _arm_of(r)
    if arm == "with":
        v = r.memory_version
        return f"MEMORY v{v}" if v is not None else "MEMORY", ARM_COLOR["with"]
    if arm == "without":
        return "NO MEMORY", ARM_COLOR["without"]
    if arm == "pending":
        return "editing…", PENDING_COLOR
    return "before memory", ARM_COLOR["none"]


def tree_node_attrs(r: ri.RoundInfo, lines: list[str], fill: str) -> str:
    """Graphviz attributes of a tree node in a run with edit memory: an HTML
    label with a header band naming the arm (SEED / MEMORY vN / NO MEMORY /
    before memory) above the usual lines, the score-colored fill, and a
    border in the arm's color (solid for with, dashed for without)."""
    from html import escape

    band, band_color = _arm_badge(r)
    rows = [f'<TR><TD BGCOLOR="{band_color}"><FONT COLOR="white" POINT-SIZE="10">'
            f"<B>{escape(band)}</B></FONT></TD></TR>"]
    for i, text in enumerate(lines):
        cell = f"<B>{escape(text)}</B>" if i == 0 else escape(text)
        rows.append(f"<TR><TD>{cell}</TD></TR>")
    label = ('<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0" CELLPADDING="2">'
             + "".join(rows) + "</TABLE>>")
    attrs = [f"label={label}", f'fillcolor="{fill}"']
    arm = "none" if r.is_root else _arm_of(r)
    if arm == "with":
        attrs += [f'color="{ARM_COLOR["with"]}"', "penwidth=3"]
    elif arm == "without":
        attrs += [f'color="{ARM_COLOR["without"]}"', "penwidth=3", 'style="filled,rounded,dashed"']
    return ", ".join(attrs)


def tree_edge_attrs(r: ri.RoundInfo) -> str:
    """Attributes of the parent -> r edge: the edit that created r, drawn in
    the style of its arm ("" for the default edge)."""
    arm = _arm_of(r)
    if arm == "with":
        return f'color="{ARM_COLOR["with"]}", penwidth=2.5'
    if arm == "without":
        return f'color="{ARM_COLOR["without"]}", penwidth=2, style=dashed'
    return ""


def render_tree_arm_legend(rounds: list[ri.RoundInfo]) -> None:
    """Legend for the tree's arm styling, with each arm's node count and the
    average of its nodes' mean scores (paired-eval batches are small, so
    read the averages as a rough signal)."""
    groups: dict[str, list[ri.RoundInfo]] = {"with": [], "without": [], "none": [], "pending": []}
    for r in rounds:
        if not r.is_root:
            groups[_arm_of(r)].append(r)
    colors = {**ARM_COLOR, "pending": PENDING_COLOR}
    chips = []
    for arm in ("with", "without", "none", "pending"):
        rs = groups[arm]
        if arm == "pending" and not rs:
            continue
        scored = [r.mean_utility for r in rs if r.n_evals > 0 and not r.edit_failed]
        stat = f"{len(rs)} node{'s' if len(rs) != 1 else ''}"
        if scored:
            stat += f" · avg node mean {sum(scored) / len(scored):.3f}"
        border = "dashed" if arm == "without" else "solid"
        width = 3 if arm in ("with", "without") else 1
        chips.append(
            f'<span style="display:inline-block;border:{width}px {border} {colors[arm]};'
            f'border-radius:6px;padding:2px 8px;margin:0 10px 6px 0">'
            f'<span style="background:{colors[arm]};color:white;border-radius:3px;'
            f'padding:0 6px;font-weight:600">{_ARM_LABEL[arm]}</span>&nbsp;{stat}</span>'
        )
    st.markdown("".join(chips), unsafe_allow_html=True)
    st.caption(
        "Each node's header band, border and incoming edge show the arm its edit ran under: "
        "**with memory** (the editor read memory vN), **without memory** (a memory existed but "
        "was withheld: the control), **before memory** (expanded before the first memory "
        "existed), **editing…** (session still running; the arm is recorded when it ends). "
        "The fill is still the node's mean score (red → green)."
    )


def arm_cell(r: ri.RoundInfo) -> str:
    """The nodes table's `arm` value, with the tree's color as a marker."""
    if r.is_root:
        return "seed"
    return {"with": "🔵 with", "without": "🟠 without",
            "pending": "⏳ editing"}.get(_arm_of(r), "⚪ before memory")


def _bottom_legend() -> "alt.Legend":
    """Legend below the chart, one entry per line, with NO label truncation:
    run names are long and share a prefix, so Vega's default labelLimit
    (~160px) would cut off exactly the part that distinguishes them."""
    return alt.Legend(orient="bottom", columns=1, title=None, labelLimit=0, symbolLimit=50)


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
                      legend=_bottom_legend())
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


def _version_browser(label: str, versions: list[tuple[int, Path]], *, key: str) -> None:
    if not versions:
        st.info(f"No {label} versions written yet.")
        return
    nums = [v for v, _ in versions]
    sel = st.selectbox(f"{label} version", nums, index=len(nums) - 1, key=f"{key}_sel", format_func=lambda v: f"v{v:03d}")
    idx = nums.index(sel)
    text = ra._read_text(versions[idx][1]) or ""
    show_diff = idx > 0 and st.toggle(f"Diff vs v{nums[idx-1]:03d}", value=False, key=f"{key}_diff")
    st.caption(f"{versions[idx][1].name} — {len(text)} chars")
    if show_diff:
        prev = ra._read_text(versions[idx - 1][1]) or ""
        st.code(ra.diff_markdown(prev, text, old_label=f"v{nums[idx-1]:03d}", new_label=f"v{sel:03d}") or "(identical)", language="diff")
    elif text.strip():
        with st.container(border=True):
            st.markdown(text)
    else:
        st.info("(empty)")


def _call_record(call: Optional[dict[str, Any]], title: str) -> None:
    if not call:
        st.info(f"No {title} record yet.")
        return
    ok = call.get("accepted")
    (st.success if ok else st.error)(f"{title}: {'accepted' if ok else 'rejected'} after {len(call.get('attempts') or [])} attempt(s)")
    rows = ra.memory_call_rows(call)
    if rows:
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


def _nodes_table(nodes: list[dict[str, Any]]) -> None:
    if not nodes:
        st.info("No nodes recorded.")
        return
    cols = ["node_id", "parent_id", "memory_arm", "memory_version", "block", "implementation_strategy",
            "n_evals", "mean_utility", "edit_failed", "changed_files"]
    df = pd.DataFrame(nodes)
    df = df[[c for c in cols if c in df.columns]]
    if "changed_files" in df.columns:
        df["changed_files"] = df["changed_files"].apply(lambda v: ", ".join(v) if isinstance(v, list) else v)
    st.dataframe(df, width="stretch", hide_index=True)


def _agentic_expander(agentic_dir: Path, *, key: str, title: str) -> None:
    with st.expander(title, expanded=False):
        tr = ra.load_transcript(ra.transcript_path(agentic_dir))
        session = ra.load_session(ra.session_path(agentic_dir))
        transcript_view(tr, session, key_prefix=key)


def _arms_tab(state: dict, rounds: list, layer_cfg: dict) -> None:
    pulls = state.get("pulls") or {}
    st.subheader("Arms")
    st.caption(
        "Each expansion draws an arm: **with** = editor was handed the current memory file, "
        "**without** = same editor, no memory; **none** = before the first memory version existed."
    )
    beta_prior = float(layer_cfg.get("beta_prior", 1.0))
    tallies = ri.arm_beta_tallies(rounds)
    st.markdown("**Bandit posteriors** — what `choose_arm` samples from")
    st.caption(
        f"Beta(S + {beta_prior:g}, F + {beta_prior:g}) per arm, where S/F pool the HGM `n_success`/`n_failure` "
        "of every node pulled under that arm (each case adds its score in [0, 1] to S and 1 − score to F). "
        "Pre-memory *none* nodes (before the first window closed) are excluded, so both posteriors start at the "
        f"prior; edit-failed nodes carry no mass. Selection: `{layer_cfg.get('selection', '?')}`, first "
        f"{layer_cfg.get('arm_min_pulls', '?')} pulls of each arm are forced. Width = uncertainty; the next "
        "expansion picks the arm whose draw is larger."
    )
    left, right = st.columns([3, 2])
    with left:
        arm_beta_chart(tallies, beta_prior)
    with right:
        rows = []
        for arm in ("with", "without"):
            t = tallies[arm]
            s = ri.beta_summary(t["S"] + beta_prior, t["F"] + beta_prior)
            rows.append({"arm": arm, "nodes": int(t["n_nodes"]), "S": round(t["S"], 2), "F": round(t["F"], 2),
                         "post. mean": round(s["mean"], 3), "90% interval": f"{s['lo90']:.3f} – {s['hi90']:.3f}",
                         "pulls": pulls.get(arm, 0)})
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        p = ri.prob_beta_greater(tallies["with"]["S"] + beta_prior, tallies["with"]["F"] + beta_prior,
                                 tallies["without"]["S"] + beta_prior, tallies["without"]["F"] + beta_prior)
        st.metric("P(next draw picks *with*)", f"{p:.1%}",
                  help="P(θ_with > θ_without) under the two posteriors above — the Thompson-sampling probability of choosing the with-memory arm on the next expansion.")

    st.markdown("**Node train means by arm** (each node's own mean, not pooled)")
    summ = ri.arm_utility_summary(rounds)
    left, right = st.columns([2, 3])
    with left:
        arm_bar(summ)
    with right:
        st.dataframe(pd.DataFrame([{"arm": k, **v} for k, v in summ.items()]), width="stretch", hide_index=True)
        st.dataframe(pd.DataFrame(ra.node_arm_rows(state)), width="stretch", hide_index=True, height=200)

    st.subheader("Events")
    ev_rows = ra.events_rows(state)
    if ev_rows:
        st.dataframe(pd.DataFrame(ev_rows), width="stretch", hide_index=True, height=min(400, 40 + 35 * len(ev_rows)))
    else:
        st.info("No events yet.")


def _memory_tab(em: ra.EditMemoryInfo) -> None:
    st.caption("`edit_memory_vNNN.md` — what the *with* arm's editor reads as `$EDIT_MEMORY_FILE`. "
               "Written by the memory curator at the close of each window.")
    _version_browser("memory", em.memory_versions, key="mem")


def _instruction_tab(em: ra.EditMemoryInfo) -> None:
    st.caption("The instruction addendum (`instruction_vNNN.md`) is appended to the memory curator's brief; "
               "it is rewritten every `instruction_every` memory versions by the instruction curator, whose "
               "analysis is `q.md`. v000 is always empty.")
    st.markdown("**Addendum versions**")
    _version_browser("instruction", em.instruction_versions, key="instr")
    st.markdown("---")
    st.markdown("**Instruction updates**")
    if not em.updates:
        st.info("No instruction update yet.")
        return
    idxs = [u.index for u in em.updates]
    sel = st.selectbox("Update", idxs, index=len(idxs) - 1, key="upd_sel", format_func=lambda i: f"instruction_update_{i:03d}")
    u = next(x for x in em.updates if x.index == sel)
    st.caption("Nodes the instruction curator was shown (with-memory nodes since the previous update):")
    _nodes_table(u.nodes)
    _call_record(u.update_call, "instruction update call")
    if u.q_md:
        with st.expander("q.md (instruction curator output)", expanded=True):
            st.markdown(u.q_md)
    else:
        st.info("No q.md yet.")
    if u.has_agentic:
        _agentic_expander(u.dir / "agentic", key=f"u{u.index}", title="Instruction-curator session transcript")


def _windows_tab(em: ra.EditMemoryInfo) -> None:
    if not em.windows:
        st.info("No window closed yet.")
        return
    idxs = [w.index for w in em.windows]
    sel = st.selectbox("Window", idxs, index=len(idxs) - 1, key="win_sel", format_func=lambda i: f"window_{i:03d}")
    w = next(x for x in em.windows if x.index == sel)
    meta = w.window
    if meta:
        c = st.columns(4)
        c[0].metric("Memory before", f"v{meta.get('memory_version_before', '?')}")
        c[1].metric("Instruction", f"v{meta.get('instruction_version', '?')}")
        c[2].metric("Nodes", len(meta.get("nodes") or []))
        c[3].metric("Closed at", ra.fmt_time(meta.get("t")) or "—")
        _nodes_table(meta.get("nodes") or [])
    else:
        st.warning("window.json not written yet — the curator is probably still running.")
    _call_record(w.memory_call, "memory generation call")
    if w.curation_md:
        with st.expander("curation.md (memory curator output)", expanded=True):
            st.markdown(w.curation_md)
    else:
        st.info("No curation.md yet.")
    if w.has_agentic:
        _agentic_expander(w.dir / "agentic", key=f"w{w.index}", title="Curator session transcript")


# --------------------------------------------------------------------------- #
# Entry points used by hgm_dashboard.py
# --------------------------------------------------------------------------- #


def render_edit_memory(experiment_dir: Path, rounds: list, cfg: dict) -> None:
    """The Edit memory section; nothing when the run has no edit_memory/."""
    em = ra.load_edit_memory(experiment_dir)
    if em is None:
        return
    state = em.state
    layer_cfg = ri.edit_memory_config(cfg) or state.get("config") or {}
    ri.attach_state_arms(rounds, state)

    st.subheader("Edit memory")
    st.caption(
        "One memory document for the whole run (all blocks). Per expansion a Thompson bandit over HGM's "
        "own pooled tallies picks the arm -- independently of the block. " + str(em.dir)
    )
    pulls = state.get("pulls") or {}
    c = st.columns(6)
    c[0].metric("Memory version", state.get("memory_version", 0))
    c[1].metric("Instruction version", state.get("instruction_version", 0))
    c[2].metric("Windows closed", state.get("window_index", 0))
    c[3].metric("Pulls with / without", f"{pulls.get('with', 0)} / {pulls.get('without', 0)}")
    c[4].metric("Open window", ", ".join(str(n) for n in state.get("window") or []) or "—",
                help="nodes collected toward the next window; window_failed: " + str(state.get("window_failed") or []))
    c[5].metric("Since last instruction", state.get("versions_since_instruction", "—"),
                help="memory versions written since the last instruction update")
    with st.expander("Layer config"):
        st.json(layer_cfg)

    n_mem, n_upd = len(em.memory_versions), len(em.updates)
    tab_arms, tab_mem, tab_instr, tab_win = st.tabs([
        "Arms & events",
        f"Memory ({n_mem} version{'s' if n_mem != 1 else ''})",
        f"Instruction ({n_upd} update{'s' if n_upd != 1 else ''})",
        f"Curation windows ({len(em.windows)})",
    ])
    with tab_arms:
        _arms_tab(state, rounds, layer_cfg)
    with tab_mem:
        _memory_tab(em)
    with tab_instr:
        _instruction_tab(em)
    with tab_win:
        _windows_tab(em)


def render_assignment(round_: Any) -> None:
    """What the manager assigned this expansion (assignment.json)."""
    a = round_.assignment
    if not a:
        st.info("No assignment.json -- the root, or a round edited by the default editor.")
        return
    cols = st.columns(3)
    cols[0].metric("Block", a.get("block") or "—")
    cols[1].metric("Implementation strategy", a.get("implementation_strategy") or "off")
    cols[2].metric("Memory arm", round_.memory_arm)
    st.markdown("**Block scope**")
    st.markdown(a.get("block_scope") or "")
    if a.get("implementation_strategy_body"):
        with st.expander("Implementation strategy text"):
            st.markdown(a["implementation_strategy_body"])
    if a.get("curriculum_directive"):
        with st.expander("Curriculum focus"):
            st.markdown(a["curriculum_directive"])
    if a.get("suggestion"):
        with st.expander("Advisory suggestion (block suggester)"):
            st.markdown(a["suggestion"])


def render_editor_session(round_: Any) -> None:
    """The agentic editor session that produced this round."""
    agentic_dir = round_.round_dir / "agentic"
    if not agentic_dir.is_dir():
        st.info("No agentic/ session -- the root, or a round edited by the default editor.")
        return
    transcript_view(
        ra.load_transcript(ra.transcript_path(agentic_dir)),
        ra.load_session(ra.session_path(agentic_dir)),
        key_prefix=f"r{round_.node_id}",
        prompts=ra.load_verbose_prompts(round_.round_dir),
    )
