"""Streamlit dashboard for watching/analyzing HGM runs in this repo.

    python -m streamlit run hgm_dashboard.py --server.port 8502 --server.address 0.0.0.0

Three views (sidebar): **Run** (tree, diagnostics, nodes, per-round
drill-down incl. the agentic editor's transcript), **Edit memory** (only for
runs with an ``edit_memory/`` dir) and **Compare** (best-so-far curves across
runs). All data comes from ``meta_agent.run_inspect`` /
``meta_agent.run_inspect_agentic`` (pure Python); the ``dashboard/`` package
is presentation only. Needs ``streamlit`` + ``pandas`` (+ ``altair``, which
streamlit bundles) -- see README "Dashboard".
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dashboard import compare_view, editmem_view, run_view  # noqa: E402
from meta_agent import run_inspect as ri  # noqa: E402
from meta_agent import run_inspect_agentic as ra  # noqa: E402

st.set_page_config(page_title="HGM Run Dashboard", page_icon="🌳", layout="wide")

# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #

st.sidebar.title("HGM Run Dashboard")
runs_root = Path(st.sidebar.text_input("Runs root", value="runs"))
experiments = ri.list_experiments(runs_root)
if not experiments:
    st.error(f"No experiment directories (config.snapshot.yaml + round_*/) found under `{runs_root}`.")
    st.stop()

# Default to the newest LIVE experiment. ri.run_is_active is a bounded stat
# set (no recursive walk) but there's still no reason to probe 50 historical
# dirs -- `experiments` is newest-first, so a live run is near the front.
exp_names = [p.name for p in experiments]
_active_default = next((i for i, p in enumerate(experiments[:5]) if ri.run_is_active(p)), 0)
selected_name = st.sidebar.selectbox("Experiment (newest first)", exp_names, index=_active_default)
experiment_dir = runs_root / selected_name

views = ["Run"]
if ra.edit_memory_dir(experiment_dir) is not None:
    views.append("Edit memory")
views.append("Compare")
view = st.sidebar.radio("View", views, index=0)

st.sidebar.markdown("---")
# HGM_DASHBOARD_AUTOREFRESH=0 turns the default off (used by the headless
# AppTest smoke check, which would otherwise sit in the sleep loop).
auto_refresh = st.sidebar.checkbox(
    "Auto-refresh while live", value=os.environ.get("HGM_DASHBOARD_AUTOREFRESH", "1") != "0"
)
refresh_interval = st.sidebar.slider("Refresh interval (s)", 5, 120, 20)
if st.sidebar.button("Refresh now"):
    st.cache_data.clear()
    st.rerun()
st.sidebar.caption("Times are shown in America/Los_Angeles.")

# --------------------------------------------------------------------------- #
# Views
# --------------------------------------------------------------------------- #

if view == "Run":
    is_active = run_view.render(experiment_dir)
elif view == "Edit memory":
    editmem_view.render(experiment_dir)
    is_active = ri.run_is_active(experiment_dir)
else:
    is_active = compare_view.render(experiments)

# Auto-refresh: sleep-then-rerun at the end of the script (no extra
# dependency). Only loops while something on screen is still being written.
if auto_refresh and is_active:
    time.sleep(refresh_interval)
    st.rerun()
