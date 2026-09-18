"""Edit-memory view: the layer's state, arm outcomes, memory/instruction
versions, curation windows and instruction updates for one experiment."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import pandas as pd
import streamlit as st

from meta_agent import run_inspect as ri
from meta_agent import run_inspect_agentic as ra

from . import components as C
from . import loaders as L


def _version_browser(label: str, versions: list[tuple[int, Path]], *, key: str) -> None:
    if not versions:
        st.info(f"No {label} versions written yet.")
        return
    nums = [v for v, _ in versions]
    sel = st.selectbox(f"{label} version", nums, index=len(nums) - 1, key=f"{key}_sel", format_func=lambda v: f"v{v:03d}")
    idx = nums.index(sel)
    text = L.cached_text(versions[idx][1]) or ""
    show_diff = idx > 0 and st.toggle(f"Diff vs v{nums[idx-1]:03d}", value=False, key=f"{key}_diff")
    st.caption(f"{versions[idx][1].name} — {len(text)} chars")
    if show_diff:
        prev = L.cached_text(versions[idx - 1][1]) or ""
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
    cols = ["node_id", "parent_id", "memory_arm", "memory_version", "n_evals", "mean_utility", "edit_failed", "changed_files"]
    df = pd.DataFrame(nodes)
    df = df[[c for c in cols if c in df.columns]]
    if "changed_files" in df.columns:
        df["changed_files"] = df["changed_files"].apply(lambda v: ", ".join(v) if isinstance(v, list) else v)
    st.dataframe(df, width="stretch", hide_index=True)


def _agentic_expander(agentic_dir: Path, *, key: str, title: str) -> None:
    with st.expander(title, expanded=False):
        tr = L.cached_transcript(ra.transcript_path(agentic_dir))
        session = ra.load_session(ra.session_path(agentic_dir))
        C.transcript_view(tr, session, key_prefix=key)


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
        C.arm_beta_chart(tallies, beta_prior)
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
        C.arm_bar(summ)
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


def render(experiment_dir: Path) -> None:
    em = L.cached_edit_memory(experiment_dir)
    if em is None:
        st.info("This run has no edit_memory/ directory.")
        return
    state = em.state
    rounds = L.cached_rounds(experiment_dir)
    cfg = L.cached_config(experiment_dir)
    layer_cfg = ri.edit_memory_config(cfg) or state.get("config") or {}

    st.title("Edit memory")
    st.caption(str(em.dir))
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
    tab_arms, tab_mem, tab_instr, tab_win = st.tabs(
        [
            "Arms & events",
            f"Memory ({n_mem} version{'s' if n_mem != 1 else ''})",
            f"Instruction ({n_upd} update{'s' if n_upd != 1 else ''})",
            f"Curation windows ({len(em.windows)})",
        ]
    )
    with tab_arms:
        _arms_tab(state, rounds, layer_cfg)
    with tab_mem:
        _memory_tab(em)
    with tab_instr:
        _instruction_tab(em)
    with tab_win:
        _windows_tab(em)
