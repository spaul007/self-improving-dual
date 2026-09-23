"""Read ONE finished Pier trial directory into a compact, meta-agent-safe outcome.

Pure functions over files -- no Pier import, runs in the framework venv (py3.12) and in
the offline re-scoring tests. The single source of truth for:

  * the reward (``result.json`` -> ``verifier_result.rewards.reward`` -- NOT
    ``verifier_result.reward``, which is None for every trial; CLAUDE.md trap),
  * the INFRA vs MODEL boundary (``infra_class``): an infrastructure failure is excluded
    from the node's utility (HGM ``exclude_flagged_cases``); a failure caused by the agent
    -- including a crash of mutable seedling code -- is a scored zero,
  * the failure-class taxonomy the categorizer / aggregate() / curriculum read.

Hidden-test isolation: nothing here returns test names, test output or the contents of
``verifier/`` beyond the numeric reward fields.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

# Pier exception types that mean the environment/grader failed, not the agent.
# (pier/trial/execution.py, trial.py, verifier/verifier.py, environments/base.py.)
INFRA_EXCEPTIONS = {
    "EnvironmentStartTimeoutError": "env_start",
    "HealthcheckError": "env_start",
    "AgentSetupTimeoutError": "env_start",
    "AddTestsDirError": "verifier_setup",
    "DownloadVerifierDirError": "verifier_setup",
    "RewardFileNotFoundError": "reward_missing",
    "RewardFileEmptyError": "reward_missing",
    "VerifierOutputParseError": "reward_missing",
    # The agent completed but grading ran past the task's 1800 s verifier budget. Could in
    # principle be caused by a patch that hangs the suite; EXP-028 reported both such trials
    # as "lost to verification". Kept as its OWN class so its rate stays visible.
    "VerifierTimeoutError": "verifier_timeout",
}

FAILURE_CLASSES = (
    "build_break",          # p2p == 0: the whole pre-existing suite is dead (Go/Rust build break)
    "empty_patch",          # no diff produced
    "verify_false_pass",    # VERIFY's final verdict was pass, grader says 0
    "deadline_unverified",  # shipped at the deadline with the last verdict = fail
    "near_miss",            # 0 < f2p < 1 and p2p == 1
    "p2p_regression",       # 0 < p2p < 1
    "f2p_zero",             # f2p == 0 with the suite alive
    "role_wall",            # >= 1 role ended on its wall budget
    "role_zero_edit",       # a PATCH attempt made no edits
    "agent_crash",          # seedling's own run() contained an exception (mutable-code bug)
)


def _load(p: Path) -> Optional[dict]:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def find_trial_dir(job_dir: Path) -> Optional[Path]:
    """The single trial under a one-task Pier job dir (the subdir holding result.json)."""
    job_dir = Path(job_dir)
    if not job_dir.is_dir():
        return None
    trials = sorted(d for d in job_dir.iterdir() if d.is_dir() and (d / "result.json").is_file())
    return trials[-1] if trials else None


def dispatch_outputs(trial_dir: Path) -> list[dict]:
    """Each role's structured report, in order, from trajectory.json's `dispatch` steps:
    [{"role": "verify", "output": {...finish args...}}, ...]. These are the MODEL'S OWN
    reports (safe to show the meta-agent), never grader output."""
    traj = _load(Path(trial_dir) / "agent" / "trajectory.json") or {}
    out = []
    for st in traj.get("steps") or []:
        role = (st.get("extra") or {}).get("role")
        if not role:
            continue
        for res in (st.get("observation") or {}).get("results") or []:
            try:
                payload = json.loads(res.get("content") or "")
            except (TypeError, ValueError):
                payload = {"_raw": str(res.get("content"))[:500]}
            out.append({"role": role, "output": payload if isinstance(payload, dict) else {}})
    return out


def load_outcome(trial_dir: Optional[Path]) -> dict[str, Any]:
    """Everything the scorer needs from one trial. ``reward`` is None when unscored."""
    o: dict[str, Any] = {"trial_dir": str(trial_dir) if trial_dir else None,
                         "reward": None, "infra_class": None}
    if trial_dir is None:
        o["infra_class"] = "no_trial"
        return o
    trial_dir = Path(trial_dir)
    res = _load(trial_dir / "result.json")
    if res is None:
        o["infra_class"] = "no_result_json"
        return o
    rw = ((res.get("verifier_result") or {}).get("rewards")) or {}
    for k in ("reward", "f2p", "p2p", "partial", "f2p_total", "f2p_passed", "p2p_total", "p2p_passed"):
        o[k] = rw.get(k)
    exc = res.get("exception_info") or {}
    o["exception_type"] = exc.get("exception_type")

    rs = _load(trial_dir / "agent" / "run_summary.json")
    o["has_run_summary"] = rs is not None
    rs = rs or {}
    o["agent_outcome"] = rs.get("outcome")
    o["signals"] = rs.get("signals") or {}
    o["llm"] = rs.get("llm") or {}
    o["role_stats"] = [
        {k: r.get(k) for k in ("role", "attempt", "steps", "wall_sec", "planned_wall_sec",
                                "stop_reason", "wall_terminated", "edits", "zero_edit",
                                "tests_run", "compactions", "truncations", "verdict",
                                "n_behaviours", "n_behaviours_tested", "llm_calls")}
        for r in rs.get("role_stats") or []
    ]
    o["verdicts"] = [r.get("verdict") for r in o["role_stats"] if r.get("role") == "verify"]
    patch = trial_dir / "artifacts" / "model.patch"
    o["patch_bytes"] = patch.stat().st_size if patch.is_file() else 0

    if o["reward"] is None:
        o["infra_class"] = INFRA_EXCEPTIONS.get(o["exception_type"] or "", "reward_missing")
    elif _llm_unreachable(o):
        # The server never answered: every role died on llm_error with zero completed calls.
        o["infra_class"] = "llm_unreachable"
    return o


def _llm_unreachable(o: dict) -> bool:
    llm = o.get("llm") or {}
    stops = (o.get("signals") or {}).get("stop_reasons") or {}
    return (llm.get("n_calls") == 0 and (llm.get("n_retries") or 0) > 0) or (
        bool(stops) and stops.get("llm_error", 0) > 0 and sum(stops.values()) == stops.get("llm_error")
        and (llm.get("n_calls") or 0) == 0)


def failure_classes(o: dict) -> list[str]:
    """Failure classes for a SCORED trial (empty when reward == 1). Deterministic."""
    if o.get("reward") is None or o.get("reward") == 1:
        return []
    cls: list[str] = []
    f2p, p2p = o.get("f2p"), o.get("p2p")
    if (o.get("agent_outcome") or "").startswith("contained:"):
        cls.append("agent_crash")
    if not o.get("patch_bytes"):
        cls.append("empty_patch")
    if p2p is not None and p2p == 0 and (o.get("p2p_total") or 0) > 0:
        cls.append("build_break")
    elif p2p is not None and 0 < p2p < 1:
        cls.append("p2p_regression")
    if f2p is not None and 0 < f2p < 1 and p2p == 1:
        cls.append("near_miss")
    elif f2p == 0 and "build_break" not in cls and "empty_patch" not in cls:
        cls.append("f2p_zero")
    v = o.get("verdicts") or []
    if v and v[-1] == "pass":
        cls.append("verify_false_pass")
    elif v and v[-1] == "fail":
        cls.append("deadline_unverified")
    rsx = o.get("role_stats") or []
    if any(r.get("wall_terminated") for r in rsx):
        cls.append("role_wall")
    if any(r.get("role") == "patch" and (r.get("edits") or 0) == 0 for r in rsx):
        cls.append("role_zero_edit")
    return cls


def outcome_line(o: dict) -> str:
    """One-line summary that goes FIRST in the run report (the feedback gatherer keeps only
    ~1000 chars head+tail of raw_result)."""
    if o.get("infra_class"):
        return f"INFRA-EXCLUDED ({o['infra_class']}; pier exception={o.get('exception_type')})"
    def frac(p, t):
        return f"{p}/{t}" if t else "n/a"
    stops = ",".join(f"{r.get('role')}.{r.get('attempt')}={r.get('stop_reason')}"
                     for r in o.get("role_stats") or [])
    return (f"reward={o.get('reward')} f2p={frac(o.get('f2p_passed'), o.get('f2p_total'))} "
            f"p2p={frac(o.get('p2p_passed'), o.get('p2p_total'))} patch={o.get('patch_bytes')}B "
            f"verdicts={o.get('verdicts')} classes={failure_classes(o)} stops=[{stops}]")
