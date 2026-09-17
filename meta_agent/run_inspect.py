"""Read-only data layer for inspecting HGM run directories.

Pure Python, no Streamlit import -- keeps this testable/reusable independent
of the UI (``hgm_dashboard.py`` + ``dashboard/`` are the Streamlit front-end
built on top of this module; ``run_inspect_agentic.py`` is the sibling that
parses agentic-editor transcripts and the ``edit_memory/`` directory). Every
function here only reads files already written by ``meta_agent.managers.hgm``
/ ``meta_agent.feedback_gatherer`` / ``meta_agent.tree_snapshot`` -- nothing
is inferred by re-running any part of a round.

Ported from multi-agent-setup/self-improving-dual and adapted to this repo's
allow-list mutable surface (``editor_validators.MUTABLE_FILES``/``MUTABLE_DIRS``
via ``edit_diff.changed_mutable_files``) instead of the exclude-list
``mutable_exclude`` convention.

Deliberately never read here (too large for an interactive viewer, and not
needed for any panel): ``logs/trace.jsonl`` (~20 MB per round),
``feedback.json``'s ``log_excerpt`` (~460 KB, popped right after parse) and
``verbose/editor_agentic_messages.json``.
"""
from __future__ import annotations

import difflib
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

from .edit_diff import changed_mutable_files

_ROUND_DIR_RE = re.compile(r"round_(\d+)$")

# Sidecars the manager writes under a round; their mtimes together identify
# "has this round changed since the dashboard last parsed it".
_ROUND_SIGNATURE_FILES = (
    "hgm_node.json",
    "strategy.json",
    "eval_result.json",
    "feedback.json",
    "agentic/session.json",
    "agentic/transcript.jsonl",
    "logs",
)


def _round_number(round_dir: Path) -> Optional[int]:
    m = _ROUND_DIR_RE.match(round_dir.name)
    return int(m.group(1)) if m else None


def _stat_sig(path: Path) -> tuple:
    try:
        st = path.stat()
    except OSError:
        return (None, None)
    return (st.st_mtime, st.st_size)


# --------------------------------------------------------------------------- #
# Experiment / config discovery
# --------------------------------------------------------------------------- #


def list_experiments(runs_root: Path) -> list[Path]:
    """``runs/*`` directories, newest-mtime first. Skips anything that isn't
    a directory (console logs) and doesn't crash if ``runs_root`` is missing.

    Requires BOTH a top-level ``config.snapshot.yaml`` AND at least one
    ``round_NNN`` child. ``evaluate.py`` writes ``runs/eval_<stamp>_<agent>/``
    with a ``config.snapshot.yaml`` but only a ``round_eval/`` dir -- those are
    standalone evals, not experiments, and would otherwise show up in (and,
    being freshly written, often outrank) the experiment picker."""
    if not runs_root.is_dir():
        return []
    dirs = [
        p for p in runs_root.iterdir()
        if p.is_dir()
        and (p / "config.snapshot.yaml").is_file()
        and any(_round_number(c) is not None for c in p.glob("round_*") if c.is_dir())
    ]
    dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return dirs


def load_config_snapshot(experiment_dir: Path) -> dict[str, Any]:
    path = experiment_dir / "config.snapshot.yaml"
    if not path.exists():
        return {}
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError:
        return {}


def task_agent_model(cfg: dict[str, Any]) -> Optional[str]:
    """``TaskAgentSpec`` is flat here (``task_agent.model``); the nested
    ``task_agent.config.model`` shape is accepted as a fallback for snapshots
    from the sibling repos."""
    ta = cfg.get("task_agent") or {}
    if ta.get("model"):
        return str(ta["model"])
    nested = ta.get("config") or {}
    return str(nested["model"]) if nested.get("model") else None


def editor_model(cfg: dict[str, Any]) -> Optional[str]:
    editor = (cfg.get("editor") or {}).get("config") or {}
    return str(editor["model"]) if editor.get("model") else None


def editor_read_scope(cfg: dict[str, Any]) -> Optional[str]:
    editor = (cfg.get("editor") or {}).get("config") or {}
    return str(editor["read_scope"]) if editor.get("read_scope") else None


def edit_memory_config(cfg: dict[str, Any]) -> Optional[dict[str, Any]]:
    """The ``edit_memory.config`` block, or ``None`` for runs without the
    edit-memory layer."""
    em = cfg.get("edit_memory")
    if not em:
        return None
    return dict(em.get("config") or {})


def load_tree_snapshots(experiment_dir: Path) -> list[dict[str, Any]]:
    path = experiment_dir / "snapshots" / "tree_snapshots.jsonl"
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _latest_round_dirs(experiment_dir: Path, n: int) -> list[Path]:
    dirs = [d for d in experiment_dir.glob("round_*") if d.is_dir() and _round_number(d) is not None]
    dirs.sort(key=lambda d: _round_number(d) or 0, reverse=True)
    return dirs[:n]


def run_is_active(experiment_dir: Path, *, staleness_s: float = 4500.0) -> bool:
    """Best-effort liveness heuristic (documented as such -- no PID/`ps`
    involved, so a killed-but-not-cleaned-up run can look briefly "live").

    ``staleness_s`` needs to comfortably exceed the evaluator's
    ``wall_time_s_per_case`` (1800s here, 3600s in some configs): a single
    slow straggler case within a batch legitimately blocks any new file write
    until it finishes or times out, and a short threshold makes a healthy run
    falsely show as "STOPPED".

    A finished run always has ``run_summary.md`` -- checked first and
    authoritative. Otherwise "active" means one of a bounded set of files was
    modified more recently than ``staleness_s`` ago: the experiment dir,
    the tree snapshot log, the edit-memory state, and (for the two
    highest-numbered rounds) the round dir, its agentic transcript, logs dir
    and result sidecars. No recursive walk -- a live run dir already holds
    dozens of rounds with 20 MB traces each, and the dashboard calls this on
    every auto-refresh."""
    if (experiment_dir / "run_summary.md").exists():
        return False
    candidates: list[Path] = [
        experiment_dir,
        experiment_dir / "snapshots" / "tree_snapshots.jsonl",
        experiment_dir / "edit_memory" / "state.json",
    ]
    for rd in _latest_round_dirs(experiment_dir, 2):
        candidates += [
            rd,
            rd / "agentic" / "transcript.jsonl",
            rd / "logs",
            rd / "eval_result.json",
            rd / "hgm_node.json",
        ]
    newest = 0.0
    for p in candidates:
        try:
            mtime = p.stat().st_mtime
        except OSError:
            continue
        newest = max(newest, mtime)
    if newest == 0.0:
        return False
    return (time.time() - newest) < staleness_s


# --------------------------------------------------------------------------- #
# Round discovery
# --------------------------------------------------------------------------- #


@dataclass
class RoundInfo:
    round_dir: Path
    node_id: int
    hgm_node: Optional[dict[str, Any]] = None
    strategy: Optional[dict[str, Any]] = None
    eval_result: Optional[dict[str, Any]] = None
    feedback: Optional[dict[str, Any]] = None  # log_excerpt already popped
    # agentic/session.json -- the editor session's summary (tiny). The
    # transcript itself is loaded lazily by run_inspect_agentic.
    agentic_session: Optional[dict[str, Any]] = None
    has_task_agent: bool = False
    has_agentic: bool = False
    has_verbose: bool = False

    @property
    def parent_id(self) -> Optional[int]:
        if self.hgm_node is not None:
            return self.hgm_node.get("parent_id")
        if self.feedback is not None:
            return self.feedback.get("base_round")
        return None

    @property
    def edit_failed(self) -> bool:
        if self.hgm_node is not None:
            return bool(self.hgm_node.get("edit_failed"))
        # hgm_node.json is only ever skipped (until finalize) on a failed edit
        # (see hgm.py::_expand) -- its absence alongside a present feedback
        # with edit_errors is the authoritative signal.
        if self.feedback is not None and self.feedback.get("edit_errors"):
            return True
        return False

    @property
    def mean_utility(self) -> Optional[float]:
        return self.hgm_node.get("mean_utility") if self.hgm_node else None

    @property
    def n_evals(self) -> int:
        return int(self.hgm_node.get("n_evals", 0)) if self.hgm_node else 0

    @property
    def cmp(self) -> Optional[float]:
        return self.hgm_node.get("cmp") if self.hgm_node else None

    @property
    def optimization_goal(self) -> str:
        if self.strategy is not None:
            return str(self.strategy.get("optimization_goal") or "")
        return ""

    @property
    def edit_errors(self) -> list[str]:
        if self.feedback is not None:
            return list(self.feedback.get("edit_errors") or [])
        return []

    @property
    def is_root(self) -> bool:
        """The seed round. ``parent_id`` alone can't tell: a freshly created
        round with no sidecars yet also has no parent."""
        if self.hgm_node is not None:
            return self.hgm_node.get("parent_id") is None
        if self.feedback is not None:
            return self.feedback.get("base_round") is None
        return self.node_id == 0

    @property
    def memory_arm(self) -> str:
        """"none" | "with" | "without". ``hgm_node.json`` is authoritative;
        before it exists (edit-failed node mid-run) fall back to whether the
        editor session was handed a memory file."""
        if self.hgm_node is not None and self.hgm_node.get("memory_arm"):
            return str(self.hgm_node["memory_arm"])
        if self.agentic_session is not None and self.agentic_session.get("memory_path"):
            return "with"
        return "none"

    @property
    def memory_version(self) -> Optional[int]:
        if self.hgm_node is not None and self.hgm_node.get("memory_version") is not None:
            return int(self.hgm_node["memory_version"])
        return None

    @property
    def session_end_reason(self) -> Optional[str]:
        if self.agentic_session is not None:
            return self.agentic_session.get("end_reason")
        return None

    @property
    def changed_files(self) -> list[str]:
        if self.agentic_session is not None:
            return list(self.agentic_session.get("changed_files") or [])
        return []


def _read_json(path: Path) -> Optional[dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _read_case_logs(round_dir: Path) -> list[dict[str, Any]]:
    """Raw per-case result files (``logs/case_<id>.json``), written by
    ``SubprocessEvaluator`` independently of the aggregated
    ``eval_result.json`` -- these survive even when the manager process
    crashes mid-batch and never gets to persist the aggregate."""
    logs_dir = round_dir / "logs"
    if not logs_dir.is_dir():
        return []
    out: list[dict[str, Any]] = []
    for p in sorted(logs_dir.glob("case_*.json")):
        d = _read_json(p)
        if d is not None:
            out.append(d)
    return out


def round_signature(round_dir: Path) -> tuple:
    """Cheap change-detection key for a round: (mtime, size) of every sidecar
    the dashboard reads. Used by the UI's ``st.cache_data`` wrappers so an
    unchanged round is never re-parsed on auto-refresh."""
    return tuple(_stat_sig(round_dir / rel) for rel in _ROUND_SIGNATURE_FILES)


def load_round(round_dir: Path) -> Optional[RoundInfo]:
    """One ``round_NNN`` dir -> ``RoundInfo``. Missing per-round files are
    ``None``, not an error -- a round that's mid-EXPAND (edit not yet
    validated) or mid-EVALUATE legitimately has only some of the files.

    When ``eval_result.json`` is missing/empty but raw ``logs/case_*.json``
    files exist (a process crash mid-batch, or an in-progress batch), the
    ``per_case`` list is backfilled from those files and the reconstructed
    dict is flagged ``_synthesized_from_case_logs: True`` so callers can show
    an honest caveat instead of silently looking clean."""
    if not round_dir.is_dir():
        return None
    node_id = _round_number(round_dir)
    if node_id is None:
        return None
    hgm_node = _read_json(round_dir / "hgm_node.json")
    strategy = _read_json(round_dir / "strategy.json")
    eval_result = _read_json(round_dir / "eval_result.json")
    if not eval_result or not eval_result.get("per_case"):
        case_logs = _read_case_logs(round_dir)
        if case_logs:
            eval_result = dict(eval_result or {})
            eval_result["per_case"] = case_logs
            eval_result["_synthesized_from_case_logs"] = True
    feedback = _read_json(round_dir / "feedback.json")
    if feedback is not None:
        # ~460 KB of raw log text per round; nothing in the dashboard shows
        # it and it would dominate the cache.
        feedback.pop("log_excerpt", None)
    agentic_dir = round_dir / "agentic"
    return RoundInfo(
        round_dir=round_dir,
        node_id=node_id,
        hgm_node=hgm_node,
        strategy=strategy,
        eval_result=eval_result,
        feedback=feedback,
        agentic_session=_read_json(agentic_dir / "session.json"),
        has_task_agent=(round_dir / "task_agent").is_dir(),
        has_agentic=(agentic_dir / "transcript.jsonl").is_file(),
        has_verbose=(round_dir / "verbose").is_dir(),
    )


def discover_rounds(experiment_dir: Path) -> list[RoundInfo]:
    """Walk ``round_*`` dirs, oldest first (``round_NNN`` == node id ``NNN``
    for the HGM manager)."""
    rounds: list[RoundInfo] = []
    for round_dir in sorted(experiment_dir.glob("round_*")):
        r = load_round(round_dir)
        if r is not None:
            rounds.append(r)
    rounds.sort(key=lambda r: r.node_id)
    return rounds


# --------------------------------------------------------------------------- #
# Diffing a round's task_agent/ against its parent's
# --------------------------------------------------------------------------- #


@dataclass
class FileDiff:
    path: str
    status: str  # "added" | "removed" | "modified"
    diff_text: str
    lines_added: int
    lines_removed: int


def diff_round_files(parent_dir: Path, round_dir: Path) -> dict[str, FileDiff]:
    """Per-file unified diffs between ``parent_dir/task_agent`` and
    ``round_dir/task_agent``, restricted to the mutable surface the editor
    and validators operate on. Candidate selection is delegated to
    ``edit_diff.changed_mutable_files`` (which imports ``MUTABLE_FILES`` /
    ``MUTABLE_DIRS`` from ``editor_validators``) so the three can't drift."""
    parent_root = parent_dir / "task_agent"
    child_root = round_dir / "task_agent"
    out: dict[str, FileDiff] = {}
    for rel in changed_mutable_files(parent_dir, round_dir):
        p_path, c_path = parent_root / rel, child_root / rel
        p_exists, c_exists = p_path.exists(), c_path.exists()
        try:
            old_lines = p_path.read_text(encoding="utf-8").splitlines() if p_exists else []
        except (OSError, UnicodeDecodeError):
            old_lines = []
        try:
            new_lines = c_path.read_text(encoding="utf-8").splitlines() if c_exists else []
        except (OSError, UnicodeDecodeError):
            new_lines = []
        diff_lines = list(
            difflib.unified_diff(
                old_lines, new_lines,
                fromfile=f"parent/{rel}", tofile=f"child/{rel}",
                n=3, lineterm="",
            )
        )
        added = sum(1 for l in diff_lines if l.startswith("+") and not l.startswith("+++"))
        removed = sum(1 for l in diff_lines if l.startswith("-") and not l.startswith("---"))
        status = "added" if not p_exists else ("removed" if not c_exists else "modified")
        out[rel] = FileDiff(
            path=rel,
            status=status,
            diff_text="\n".join(diff_lines),
            lines_added=added,
            lines_removed=removed,
        )
    return out


def diff_totals(diffs: dict[str, FileDiff]) -> tuple[int, int]:
    """(total lines_added, total lines_removed) across all changed files."""
    added = sum(d.lines_added for d in diffs.values())
    removed = sum(d.lines_removed for d in diffs.values())
    return added, removed


# --------------------------------------------------------------------------- #
# Diagnostics -- the automated version of the manual grep habit
# --------------------------------------------------------------------------- #

_SEVERITY_ORDER = {"error": 0, "warning": 1, "info": 2}


@dataclass
class Alert:
    severity: str  # "error" | "warning" | "info"
    node_id: int
    message: str


def extract_diagnostics(rounds: list[RoundInfo], *, is_active: bool) -> list[Alert]:
    """Automated, severity-ranked (error > warning > info) sweep for the
    problems one otherwise finds by hand: edit-failed nodes (with their
    validator errors), editor sessions that ended without submitting,
    per-case runtime/agent errors, crashed eval results, zero-mean nodes.

    ``is_active`` disambiguates a case a single filesystem snapshot can't:
    a node whose eval_result.json is behind its logs/case_*.json (flagged
    ``_synthesized_from_case_logs``) looks IDENTICAL whether it's genuinely
    mid-crash or simply still being evaluated. Only the run's liveness can
    tell those apart; a live run's in-progress node is normal."""
    alerts: list[Alert] = []

    for r in rounds:
        if r.edit_failed:
            errs = r.edit_errors
            msg = "edit failed" + (f": {errs[0][:200]}" if errs else "")
            alerts.append(Alert("error", r.node_id, msg))
        elif r.agentic_session is not None and not r.agentic_session.get("success", True):
            alerts.append(
                Alert("warning", r.node_id, f"editor session ended: {r.agentic_session.get('end_reason')}")
            )

        er = r.eval_result or {}
        if er.get("_synthesized_from_case_logs") and not is_active:
            alerts.append(
                Alert(
                    "error", r.node_id,
                    "eval_result.json missing/behind -- likely crashed "
                    "mid-evaluation (per-case results reconstructed from "
                    "logs/case_*.json; run is no longer active, so this "
                    "will never catch up)",
                )
            )
        if er.get("crashed"):
            alerts.append(Alert("error", r.node_id, "evaluation crashed"))

        for case in er.get("per_case") or []:
            case_id = case.get("case_id", "?")
            if case.get("error"):
                alerts.append(
                    Alert("error", r.node_id, f"case {case_id}: {str(case['error'])[:200]}")
                )
                continue
            details = case.get("details") or {}
            agent_meta = details.get("agent_metadata") or {}
            if agent_meta.get("error"):
                alerts.append(
                    Alert("error", r.node_id, f"case {case_id} agent_error: {str(agent_meta['error'])[:200]}")
                )
            if agent_meta.get("validation_error"):
                alerts.append(
                    Alert("warning", r.node_id, f"case {case_id} validation_error: {str(agent_meta['validation_error'])[:200]}")
                )
            if agent_meta.get("forced_fallback"):
                alerts.append(Alert("warning", r.node_id, f"case {case_id}: forced_fallback"))

        if r.hgm_node is not None and r.n_evals > 0 and (r.mean_utility or 0) == 0:
            alerts.append(Alert("warning", r.node_id, f"zero mean utility over {r.n_evals} eval(s)"))

    alerts.sort(key=lambda a: (_SEVERITY_ORDER.get(a.severity, 3), a.node_id))
    return alerts


# --------------------------------------------------------------------------- #
# Budget tracking
# --------------------------------------------------------------------------- #


@dataclass
class BudgetInfo:
    spent: Optional[int]
    total: Optional[int]
    exact: bool  # True when sourced from tree_snapshots.jsonl


def budget_progress(cfg: dict[str, Any], snapshots: list[dict[str, Any]], rounds: list[RoundInfo]) -> BudgetInfo:
    """Exact from ``tree_snapshots.jsonl`` (``budget_spent`` on the last
    line) when the run has ``manager.config.snapshot_tree: true``; otherwise
    an approximate fallback (sum of ``n_evals`` across non-root nodes -- the
    root's pre-evaluation is unbudgeted, see ``hgm.py::_run_seed``)."""
    manager_cfg = (cfg.get("manager") or {}).get("config") or {}
    total = manager_cfg.get("eval_budget")
    if snapshots:
        return BudgetInfo(spent=snapshots[-1].get("budget_spent"), total=total, exact=True)
    spent = sum(r.n_evals for r in rounds if r.hgm_node and r.hgm_node.get("parent_id") is not None)
    return BudgetInfo(spent=spent, total=total, exact=False)


# --------------------------------------------------------------------------- #
# run_summary.md passthrough
# --------------------------------------------------------------------------- #


def load_run_summary(experiment_dir: Path) -> Optional[str]:
    path = experiment_dir / "run_summary.md"
    return path.read_text(encoding="utf-8") if path.exists() else None


# --------------------------------------------------------------------------- #
# Tree-level analytics (snapshots, arms, dimensions)
# --------------------------------------------------------------------------- #


@dataclass
class CurvePoint:
    budget_spent: int
    best_mean_utility: float
    best_node_id: int
    snapshot_idx: int


def best_so_far_curve(snapshots: list[dict[str, Any]]) -> list[CurvePoint]:
    """Best train-mean as a function of budget spent, one point per distinct
    ``budget_spent`` (last snapshot wins when several share a budget -- e.g.
    an EXPAND right after an EVALUATE). ``best_mean_utility`` is already
    best-so-far by construction in the snapshot writer, but a running max is
    enforced anyway so the curve is guaranteed monotone."""
    by_budget: dict[int, CurvePoint] = {}
    for s in snapshots:
        b = s.get("budget_spent")
        if b is None or s.get("best_mean_utility") is None:
            continue
        by_budget[int(b)] = CurvePoint(
            budget_spent=int(b),
            best_mean_utility=float(s["best_mean_utility"]),
            best_node_id=int(s.get("best_node_id", -1)),
            snapshot_idx=int(s.get("snapshot_idx", -1)),
        )
    out: list[CurvePoint] = []
    running = float("-inf")
    for b in sorted(by_budget):
        pt = by_budget[b]
        running = max(running, pt.best_mean_utility)
        out.append(CurvePoint(pt.budget_spent, running, pt.best_node_id, pt.snapshot_idx))
    return out


def latest_snapshot_nodes(snapshots: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return list(snapshots[-1].get("nodes") or []) if snapshots else []


def arm_utility_summary(rounds: list[RoundInfo]) -> dict[str, dict[str, Any]]:
    """Per memory arm ("with" | "without" | "none"): node counts and the mean
    of evaluated nodes' train means. Only arms that actually occur are
    returned, so a run without the edit-memory layer yields just ``{"none":
    ...}`` and the UI can hide the panel."""
    out: dict[str, dict[str, Any]] = {}
    for r in rounds:
        if r.is_root:
            continue  # the seed has no arm
        arm = r.memory_arm
        e = out.setdefault(arm, {
            "n_nodes": 0, "n_evaluated": 0, "n_edit_failed": 0,
            "mean_of_means": None, "best_mean": None, "total_evals": 0, "_means": [],
        })
        e["n_nodes"] += 1
        if r.edit_failed:
            e["n_edit_failed"] += 1
        if r.n_evals > 0 and r.mean_utility is not None:
            e["n_evaluated"] += 1
            e["total_evals"] += r.n_evals
            e["_means"].append(float(r.mean_utility))
    for e in out.values():
        means = e.pop("_means")
        if means:
            e["mean_of_means"] = sum(means) / len(means)
            e["best_mean"] = max(means)
    return out


def _case_details(eval_result: Optional[dict[str, Any]]):
    for c in (eval_result or {}).get("per_case") or []:
        if c.get("error"):
            continue
        details = c.get("details") or {}
        if details:
            yield c, details


def dimension_means(eval_result: Optional[dict[str, Any]]) -> dict[str, float]:
    """Mean of each ``details.dimension_scores`` key across non-error cases."""
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for _, details in _case_details(eval_result):
        for dim, v in (details.get("dimension_scores") or {}).items():
            if v is None:
                continue
            sums[dim] = sums.get(dim, 0.0) + float(v)
            counts[dim] = counts.get(dim, 0) + 1
    return {dim: sums[dim] / counts[dim] for dim in sums}


def per_case_dimension_rows(eval_result: Optional[dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per case with the scorer's headline numbers and one column per
    dimension (``None`` where a case has no details, e.g. a timeout)."""
    dims: list[str] = []
    for _, details in _case_details(eval_result):
        for dim in (details.get("dimension_scores") or {}):
            if dim not in dims:
                dims.append(dim)
    rows: list[dict[str, Any]] = []
    for c in (eval_result or {}).get("per_case") or []:
        details = c.get("details") or {}
        row: dict[str, Any] = {
            "case_id": c.get("case_id"),
            "passed": c.get("passed"),
            "score": c.get("score"),
            "composite": details.get("composite_score"),
            "commonsense": details.get("commonsense_score"),
            "hard": details.get("hard_score"),
        }
        ds = details.get("dimension_scores") or {}
        for dim in dims:
            row[dim] = ds.get(dim)
        row["error"] = c.get("error")
        rows.append(row)
    return rows


def hard_constraint_failure_counts(eval_result: Optional[dict[str, Any]]) -> list[tuple[str, int]]:
    counts: dict[str, int] = {}
    for _, details in _case_details(eval_result):
        for name, entry in (details.get("hard_constraints") or {}).items():
            if isinstance(entry, dict) and entry.get("passed") is False:
                counts[name] = counts.get(name, 0) + 1
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))


def failed_check_counts(eval_result: Optional[dict[str, Any]]) -> list[tuple[str, int]]:
    counts: dict[str, int] = {}
    for _, details in _case_details(eval_result):
        for name in details.get("failed_checks") or []:
            counts[str(name)] = counts.get(str(name), 0) + 1
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))


# --------------------------------------------------------------------------- #
# Cross-run summary (cheap: config + snapshots + edit-memory state + one
# eval_result.json for the best node)
# --------------------------------------------------------------------------- #


@dataclass
class ExperimentSummary:
    name: str
    path: Path
    project: str
    finished: bool
    has_edit_memory: bool
    n_nodes: int
    n_edit_failed: int
    best_node_id: Optional[int]
    best_mean: Optional[float]
    budget_spent: Optional[int]
    budget_total: Optional[int]
    pulls: dict[str, int] = field(default_factory=dict)
    memory_version: Optional[int] = None
    best_dimension_means: dict[str, float] = field(default_factory=dict)
    curve: list[CurvePoint] = field(default_factory=list)
    has_snapshots: bool = False


def experiment_summary(experiment_dir: Path) -> ExperimentSummary:
    """Everything the Compare view needs for one run, without touching the
    per-round sidecars (except the best node's ``eval_result.json``). Falls
    back to walking ``hgm_node.json`` files when the run has no snapshots."""
    cfg = load_config_snapshot(experiment_dir)
    snapshots = load_tree_snapshots(experiment_dir)
    manager_cfg = (cfg.get("manager") or {}).get("config") or {}
    total = manager_cfg.get("eval_budget")

    best_node_id: Optional[int] = None
    best_mean: Optional[float] = None
    best_round_dir: Optional[Path] = None
    if snapshots:
        last = snapshots[-1]
        nodes = last.get("nodes") or []
        n_nodes = int(last.get("n_nodes", len(nodes)))
        n_edit_failed = sum(1 for n in nodes if n.get("edit_failed"))
        budget_spent = last.get("budget_spent")
        best_node_id = last.get("best_node_id")
        best_mean = last.get("best_mean_utility")
        if last.get("best_round_dir"):
            best_round_dir = experiment_dir / str(last["best_round_dir"])
    else:
        rounds = discover_rounds(experiment_dir)
        n_nodes = len(rounds)
        n_edit_failed = sum(1 for r in rounds if r.edit_failed)
        budget_spent = budget_progress(cfg, snapshots, rounds).spent
        evaluated = [r for r in rounds if r.n_evals > 0 and r.mean_utility is not None]
        if evaluated:
            best = max(evaluated, key=lambda r: float(r.mean_utility or 0))
            best_node_id, best_mean, best_round_dir = best.node_id, best.mean_utility, best.round_dir

    best_dims: dict[str, float] = {}
    if best_round_dir is not None:
        best_dims = dimension_means(_read_json(best_round_dir / "eval_result.json"))

    state = _read_json(experiment_dir / "edit_memory" / "state.json")
    return ExperimentSummary(
        name=experiment_dir.name,
        path=experiment_dir,
        project=str(cfg.get("project", "?")),
        finished=(experiment_dir / "run_summary.md").exists(),
        has_edit_memory=(experiment_dir / "edit_memory").is_dir(),
        n_nodes=n_nodes,
        n_edit_failed=n_edit_failed,
        best_node_id=best_node_id,
        best_mean=best_mean,
        budget_spent=budget_spent,
        budget_total=total,
        pulls=dict((state or {}).get("pulls") or {}),
        memory_version=(state or {}).get("memory_version"),
        best_dimension_means=best_dims,
        curve=best_so_far_curve(snapshots),
        has_snapshots=bool(snapshots),
    )


_EVAL_AT_BUDGET_RE = re.compile(r"eval_at_budget_(\d+)\.json$")


def load_eval_at_budget(experiment_dir: Path) -> list[dict[str, Any]]:
    """``snapshots/eval_at_budget_<B>.json`` files written by
    ``snapshot_eval.py`` (held-out re-evaluation of the node selected at a
    given budget), sorted by budget. Empty when none exist."""
    snap_dir = experiment_dir / "snapshots"
    if not snap_dir.is_dir():
        return []
    out: list[dict[str, Any]] = []
    for p in snap_dir.iterdir():
        m = _EVAL_AT_BUDGET_RE.match(p.name)
        if not m:
            continue
        d = _read_json(p)
        if d is None:
            continue
        d.setdefault("requested_budget", int(m.group(1)))
        out.append(d)
    out.sort(key=lambda d: int(d.get("requested_budget", 0)))
    return out
