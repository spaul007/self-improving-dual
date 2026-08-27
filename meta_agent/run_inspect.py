"""Read-only data layer for inspecting HGM (and HGM-dual) run directories.

Pure Python, no Streamlit import -- keeps this testable/reusable independent
of the UI (see ``hgm_dashboard.py``, the Streamlit front-end built on top of
this module). Every function here only reads files already written by
``meta_agent.managers.hgm``/``meta_agent.feedback_gatherer``/
``meta_agent.behavior_summarizer`` -- nothing is inferred by re-running any
part of a round.

Generic across any project using the exclude-list ``mutable_exclude`` +
``seed_dir_name`` convention (db_mas, math_mas, ...). travel/shopping/math's
legacy include-list convention isn't specially supported: diffing still
works (``mutable_exclude=None`` just means nothing is excluded from the
recursive scan), but no dedicated UI exists for their `tool_wrapper.py`/
`tools_schema.json`/`mutable_tools/` shape.

HGM-dual's ``variants/`` subdirectory is tolerated (never crashes anything
here) but not specially parsed -- out of scope for this first version, see
the plan's Addendum 7.
"""
from __future__ import annotations

import difflib
import json
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import yaml

from .editor_validators import is_excluded

# Same noise-skip set every other exclude-mode scanner in this repo uses
# (agent_editor.py, behavior_summarizer.py) -- generated/scratch output and
# Python's own cache, never source, never worth diffing either way.
_ALWAYS_IGNORE_DIRS = {"__pycache__", "results"}

_ROUND_DIR_RE = re.compile(r"round_(\d+)$")


def _round_number(round_dir: Path) -> Optional[int]:
    m = _ROUND_DIR_RE.match(round_dir.name)
    return int(m.group(1)) if m else None


# --------------------------------------------------------------------------- #
# Experiment / config discovery
# --------------------------------------------------------------------------- #


def list_experiments(runs_root: Path) -> list[Path]:
    """``runs/*`` directories, newest-mtime first. Skips anything that isn't
    a directory (stray files) and doesn't crash if ``runs_root`` is missing."""
    if not runs_root.is_dir():
        return []
    dirs = [p for p in runs_root.iterdir() if p.is_dir()]
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
    """Best-effort task-agent model name. Travel/shopping/math-style configs
    put it at ``task_agent.config.model``; db_mas/math_mas-style configs set
    it via an arbitrary ``env.*_MODEL`` var instead (their own code reads it
    directly, bypassing ``platform_core.llm_wrapper`` entirely) -- so when
    the first shape is absent, scan ``env`` for any key ending in
    ``_MODEL``."""
    ta = (cfg.get("task_agent") or {}).get("config") or {}
    if ta.get("model"):
        return str(ta["model"])
    env = cfg.get("env") or {}
    for key, val in env.items():
        if str(key).upper().endswith("_MODEL"):
            return str(val)
    return None


def meta_agent_model(cfg: dict[str, Any]) -> Optional[str]:
    editor = (cfg.get("editor") or {}).get("config") or {}
    return str(editor["model"]) if editor.get("model") else None


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


def run_is_active(experiment_dir: Path, *, staleness_s: float = 4500.0) -> bool:
    """Best-effort liveness heuristic (documented as such -- no PID/`ps`
    involved, so a killed-but-not-cleaned-up run can look briefly "live").

    ``staleness_s`` needs to comfortably exceed the evaluator's own
    ``wall_time_s_per_case`` (3600s in every travel_mas_refactored config):
    a single slow straggler case within a batch legitimately blocks any new
    file write until it finishes OR the evaluator's timeout kills it --
    observed directly, live, growing past 40+ minutes with no sign of
    finishing before hitting that ceiling. A short threshold (this used to
    be 120s, then 1800s -- still not enough) makes a perfectly healthy,
    actively-running process falsely show as "STOPPED". 4500s = the 3600s
    timeout plus real margin for subprocess cleanup and the manager's own
    post-batch processing (writing eval_result.json, failure_summarizer,
    etc.) before the next file write lands.

    A finished run always has ``run_summary.md`` -- that's checked first and
    is authoritative. Absent that, "active" means *some* file under the
    experiment dir was modified more recently than ``staleness_s`` seconds
    ago."""
    if (experiment_dir / "run_summary.md").exists():
        return False
    newest = 0.0
    for p in experiment_dir.rglob("*"):
        if p.is_file():
            try:
                mtime = p.stat().st_mtime
            except OSError:
                continue
            if mtime > newest:
                newest = mtime
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
    feedback: Optional[dict[str, Any]] = None
    behavior_memory: Optional[str] = None
    has_task_agent: bool = False
    has_variants: bool = False  # HGM-dual marker, tolerated not visualized
    # Per-block Beta posterior snapshot at the moment this EXPAND's block was
    # selected (see meta_agent/block_bandit.py::AdaptiveStrategy) -- only
    # present when manager.config.block_selection_strategy == "adaptive".
    adaptive_strategy: Optional[dict[str, Any]] = None

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
        # hgm_node.json is only ever skipped on a failed edit (see
        # hgm.py::_expand) -- its absence alongside a present feedback with
        # edit_errors is the authoritative signal.
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


def _read_json(path: Path) -> Optional[dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _read_case_logs(round_dir: Path) -> list[dict[str, Any]]:
    """Raw per-case result files (``logs/case_<id>.json``), written by
    ``SubprocessEvaluator._finish`` independently of the aggregated
    ``eval_result.json`` -- these survive even when the manager process
    crashes mid-batch and never gets to persist the aggregate (confirmed
    against a real crashed run: 5 ``case_*.json`` files existed with no
    ``eval_result.json``/``hgm_node.json`` anywhere in the round)."""
    logs_dir = round_dir / "logs"
    if not logs_dir.is_dir():
        return []
    out: list[dict[str, Any]] = []
    for p in sorted(logs_dir.glob("case_*.json")):
        d = _read_json(p)
        if d is not None:
            out.append(d)
    return out


def discover_rounds(experiment_dir: Path) -> list[RoundInfo]:
    """Walk ``round_*`` dirs, oldest first (matches node-id order for every
    manager observed so far -- ``round_NNN`` == node id ``NNN``). Missing
    per-round files are ``None``, not an error -- a round that's mid-EXPAND
    (edit not yet validated) or mid-EVALUATE legitimately has only some of
    the possible files.

    When ``eval_result.json`` is missing/empty but raw ``logs/case_*.json``
    files exist (a process crash mid-batch, before the manager could persist
    the aggregate — observed for real in an early db_mas run), ``per_case``
    is backfilled from those files directly and the reconstructed dict is
    flagged ``_synthesized_from_case_logs: True`` so callers can show an
    honest caveat instead of silently looking clean."""
    rounds: list[RoundInfo] = []
    for round_dir in sorted(experiment_dir.glob("round_*")):
        if not round_dir.is_dir():
            continue
        node_id = _round_number(round_dir)
        if node_id is None:
            continue
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
        behavior_path = round_dir / "behavior_memory.md"
        behavior_memory = (
            behavior_path.read_text(encoding="utf-8") if behavior_path.exists() else None
        )
        adaptive_strategy = _read_json(round_dir / "adaptive_strategy.json")
        rounds.append(
            RoundInfo(
                round_dir=round_dir,
                node_id=node_id,
                hgm_node=hgm_node,
                strategy=strategy,
                eval_result=eval_result,
                feedback=feedback,
                behavior_memory=behavior_memory,
                has_task_agent=(round_dir / "task_agent").is_dir(),
                has_variants=(round_dir / "variants").is_dir(),
                adaptive_strategy=adaptive_strategy,
            )
        )
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


def diff_round_files(
    parent_dir: Path, round_dir: Path, mutable_exclude: Optional[list[str]] = None
) -> dict[str, FileDiff]:
    """Per-file diffs between ``parent_dir/task_agent`` and
    ``round_dir/task_agent``, restricted to the same mutable surface the
    editor/validator/summarizer actually operate on (exclude-mode: recurse
    everything not matched by ``mutable_exclude``; ``None`` diffs the whole
    tree). Reuses ``editor_validators.is_excluded`` rather than
    reimplementing the prefix-match rule."""
    parent_root = parent_dir / "task_agent"
    child_root = round_dir / "task_agent"
    if not child_root.is_dir():
        return {}

    candidates: set[str] = set()
    for root in (parent_root, child_root):
        if not root.is_dir():
            continue
        for p in root.rglob("*"):
            if p.is_dir() or p.name == "__init__.py":
                continue
            rel_parts = p.relative_to(root).parts
            if set(rel_parts) & _ALWAYS_IGNORE_DIRS:
                continue
            rel = p.relative_to(root).as_posix()
            if mutable_exclude is not None and is_excluded(rel, mutable_exclude):
                continue
            candidates.add(rel)

    out: dict[str, FileDiff] = {}
    for rel in sorted(candidates):
        p_path, c_path = parent_root / rel, child_root / rel
        p_exists, c_exists = p_path.exists(), c_path.exists()
        if not p_exists and not c_exists:
            continue
        try:
            old_lines = p_path.read_text(encoding="utf-8").splitlines() if p_exists else []
        except (OSError, UnicodeDecodeError):
            old_lines = []
        try:
            new_lines = c_path.read_text(encoding="utf-8").splitlines() if c_exists else []
        except (OSError, UnicodeDecodeError):
            new_lines = []
        if old_lines == new_lines:
            continue
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
# Diagnostics -- the automated version of this session's manual grep habit
# --------------------------------------------------------------------------- #

_SEVERITY_ORDER = {"error": 0, "warning": 1, "info": 2}


@dataclass
class Alert:
    severity: str  # "error" | "warning" | "info"
    node_id: int
    message: str


def extract_diagnostics(rounds: list[RoundInfo], *, is_active: bool) -> list[Alert]:
    """Automated, severity-ranked (error > warning > info) sweep for the
    problems this session repeatedly found by hand: edit-failed nodes (with
    their validator errors), per-case runtime/agent errors, crashed eval
    results, and zero-mean evaluated nodes.

    ``is_active`` disambiguates a case a single filesystem snapshot can't:
    a node whose eval_result.json is behind its logs/case_*.json (backfilled
    by discover_rounds, flagged ``_synthesized_from_case_logs``) looks
    IDENTICAL whether it's genuinely mid-crash or simply still being
    evaluated right now -- the batch just hasn't finished and persisted its
    aggregate yet. Only the run's own liveness (``ri.run_is_active``, based
    on file mtimes) can tell those apart; a live run's in-progress node is
    completely normal and must not be reported as an error."""
    alerts: list[Alert] = []

    for r in rounds:
        if r.edit_failed:
            errs = r.edit_errors
            msg = f"edit failed" + (f": {errs[0][:200]}" if errs else "")
            alerts.append(Alert("error", r.node_id, msg))

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
    line) when the run has ``manager.config.snapshot_tree: true`` set;
    otherwise an approximate fallback (sum of ``n_evals`` across non-root
    nodes -- the root's pre-evaluation is unbudgeted, see
    ``hgm.py::_run_seed``, so it's excluded from the approximation too)."""
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
# Adaptive (Thompson-sampling) block-selection posteriors
# --------------------------------------------------------------------------- #


def latest_adaptive_strategy(rounds: list[RoundInfo]) -> Optional[dict[str, Any]]:
    """The most recent round's ``adaptive_strategy.json`` (see
    meta_agent/block_bandit.py::AdaptiveStrategy) -- the freshest snapshot of
    every block's Beta posterior. ``None`` when the run isn't using
    ``block_selection_strategy: "adaptive"``, or no round has EXPANDed yet."""
    for r in sorted(rounds, key=lambda r: r.node_id, reverse=True):
        if r.adaptive_strategy is not None:
            return r.adaptive_strategy
    return None


def beta_pdf_curve(a: float, b: float, *, n_points: int = 200) -> tuple[list[float], list[float]]:
    """(x, pdf(x)) over (0, 1) exclusive for Beta(a, b), computed from
    ``math.lgamma`` (stdlib only -- no scipy dependency). ``a``/``b`` are
    always >= beta_prior (default 1.0) here, so the density never blows up
    at the boundaries the way it could for a Beta with a shape parameter
    below 1."""
    log_norm = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    xs = [(i + 0.5) / n_points for i in range(n_points)]  # avoid exact 0/1
    ys = [
        math.exp((a - 1) * math.log(x) + (b - 1) * math.log(1 - x) - log_norm)
        for x in xs
    ]
    return xs, ys
