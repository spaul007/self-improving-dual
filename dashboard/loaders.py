"""``st.cache_data`` wrappers around the run_inspect loaders.

Every wrapper takes ``(path_str, signature)`` where ``signature`` is a tuple
of ``os.stat`` results for the files that loader reads -- computed OUTSIDE
the cache on every render (cheap: a handful of stats) so a changed file
invalidates exactly its own entry and an unchanged round is never re-parsed
on auto-refresh. Return values are plain dataclasses/dicts (they get
pickled).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import streamlit as st

from meta_agent import run_inspect as ri
from meta_agent import run_inspect_agentic as ra


def _sig(*paths: Path) -> tuple:
    out = []
    for p in paths:
        try:
            s = p.stat()
            out.append((s.st_mtime, s.st_size))
        except OSError:
            out.append((None, None))
    return tuple(out)


@st.cache_data(show_spinner=False)
def _config(path: str, sig: tuple) -> dict[str, Any]:
    return ri.load_config_snapshot(Path(path))


def cached_config(experiment_dir: Path) -> dict[str, Any]:
    return _config(str(experiment_dir), _sig(experiment_dir / "config.snapshot.yaml"))


@st.cache_data(show_spinner=False)
def _round(path: str, sig: tuple) -> Optional[ri.RoundInfo]:
    return ri.load_round(Path(path))


def cached_rounds(experiment_dir: Path) -> list[ri.RoundInfo]:
    rounds: list[ri.RoundInfo] = []
    for d in sorted(experiment_dir.glob("round_*")):
        if not d.is_dir():
            continue
        r = _round(str(d), ri.round_signature(d))
        if r is not None:
            rounds.append(r)
    rounds.sort(key=lambda r: r.node_id)
    return rounds


@st.cache_data(show_spinner=False)
def _snapshots(path: str, sig: tuple) -> list[dict[str, Any]]:
    return ri.load_tree_snapshots(Path(path))


def cached_snapshots(experiment_dir: Path) -> list[dict[str, Any]]:
    return _snapshots(str(experiment_dir), _sig(experiment_dir / "snapshots" / "tree_snapshots.jsonl"))


@st.cache_data(show_spinner=False)
def _diff(parent: str, child: str, sig: tuple) -> dict[str, ri.FileDiff]:
    return ri.diff_round_files(Path(parent), Path(child))


def cached_diff(parent_dir: Path, round_dir: Path) -> dict[str, ri.FileDiff]:
    # task_agent/ is written once per round and never mutated afterwards, so
    # the two directory mtimes are a sufficient signature.
    return _diff(str(parent_dir), str(round_dir), _sig(parent_dir / "task_agent", round_dir / "task_agent"))


@st.cache_data(show_spinner=False)
def _transcript(path: str, sig: tuple) -> Optional[ra.Transcript]:
    return ra.load_transcript(Path(path))


def cached_transcript(path: Path) -> Optional[ra.Transcript]:
    return _transcript(str(path), _sig(path))


@st.cache_data(show_spinner=False)
def _edit_memory(path: str, sig: tuple) -> Optional[ra.EditMemoryInfo]:
    return ra.load_edit_memory(Path(path))


def cached_edit_memory(experiment_dir: Path) -> Optional[ra.EditMemoryInfo]:
    em_dir = ra.edit_memory_dir(experiment_dir)
    if em_dir is None:
        return None
    return _edit_memory(str(experiment_dir), ra.edit_memory_signature(em_dir))


@st.cache_data(show_spinner=False)
def _summary(path: str, sig: tuple) -> ri.ExperimentSummary:
    return ri.experiment_summary(Path(path))


def cached_experiment_summary(experiment_dir: Path) -> ri.ExperimentSummary:
    return _summary(
        str(experiment_dir),
        _sig(
            experiment_dir / "snapshots" / "tree_snapshots.jsonl",
            experiment_dir / "edit_memory" / "state.json",
            experiment_dir / "run_summary.md",
            experiment_dir,
        ),
    )


@st.cache_data(show_spinner=False)
def _eval_at_budget(path: str, sig: tuple) -> list[dict[str, Any]]:
    return ri.load_eval_at_budget(Path(path))


def cached_eval_at_budget(experiment_dir: Path) -> list[dict[str, Any]]:
    return _eval_at_budget(str(experiment_dir), _sig(experiment_dir / "snapshots"))


@st.cache_data(show_spinner=False)
def _text(path: str, sig: tuple) -> Optional[str]:
    p = Path(path)
    try:
        return p.read_text(encoding="utf-8") if p.exists() else None
    except OSError:
        return None


def cached_text(path: Path) -> Optional[str]:
    return _text(str(path), _sig(path))
