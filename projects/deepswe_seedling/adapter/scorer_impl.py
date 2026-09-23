"""Scorer for the seedling-on-DeepSWE project (``deepswe_seedling_default``).

Runs in the PARENT (evaluator) process. The utility is Pier's own binary grade,
``verifier_result.rewards.reward`` (user decision 2026-09-23: binary reward), re-read from
the trial's result.json -- never trusted from the child's metadata.

Infrastructure failures (``details["excluded"] = True``) are excluded from the node's
utility by HGM ``exclude_flagged_cases``: docker/env start, verifier setup/timeout,
missing reward, unreachable LLM server, adapter/watchdog failures. A crash of the MUTABLE
seedling code is NOT infra -- it scores 0.

Hidden-test isolation: details carry only numeric grade fields and the agent's own
signals -- never test names or test output.
"""
from __future__ import annotations

import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from meta_agent.registry import register

from .trial import FAILURE_CLASSES, failure_classes, load_outcome

JOBS_ROOT_ENV = "SID_PIER_JOBS_ROOT"
DEFAULT_JOBS_ROOT = "/groups/AIC-MV/n.tzou/sid_pier_jobs"


def _trusted(trial_dir: str | None) -> Path | None:
    if not trial_dir:
        return None
    root = Path(os.environ.get(JOBS_ROOT_ENV) or DEFAULT_JOBS_ROOT).resolve()
    p = Path(trial_dir).resolve()
    return p if (root == p or root in p.parents) and (p / "result.json").is_file() else None


@register("scorer", "deepswe_seedling_default")
class DeepSWESeedlingScorer:
    def score(self, case: dict, agent_output: Any) -> dict:
        meta = dict(getattr(agent_output, "metadata", None) or {})
        status = meta.get("status")
        trial = _trusted(meta.get("trial_dir"))
        o = load_outcome(trial)
        if trial is None:
            o["infra_class"] = ("untrusted_trial_dir" if meta.get("trial_dir")
                                else (status or "no_trial"))
        elif status in ("watchdog", "watchdog_sigkill") and o.get("reward") is None:
            o["infra_class"] = "watchdog_kill"
        excluded = bool(o.get("infra_class"))
        reward = o.get("reward")
        score = 0.0 if excluded else float(reward or 0.0)
        rs = o.get("role_stats") or []
        details = {
            "excluded": excluded,
            "infra_class": o.get("infra_class"),
            "language": ((case.get("meta_info") or {}).get("language")),
            "reward": reward,
            **{k: o.get(k) for k in ("f2p", "p2p", "partial", "f2p_total", "f2p_passed",
                                      "p2p_total", "p2p_passed", "patch_bytes",
                                      "agent_outcome", "exception_type", "verdicts")},
            "failure_classes": [] if excluded else failure_classes(o),
            "patch_attempts": sum(1 for r in rs if r.get("role") == "patch"),
            "roles": [{k: r.get(k) for k in ("role", "attempt", "steps", "wall_sec",
                                              "stop_reason", "wall_terminated", "edits",
                                              "tests_run", "compactions", "truncations",
                                              "verdict")} for r in rs],
            "status": status,
            "run_wall_s": meta.get("wall_s"),
            "inflight_start": meta.get("inflight_start"),
            "tree_hash": meta.get("tree_hash"),
            "scratch_dir": meta.get("scratch_dir"),
        }
        return {"score": score, "passed": (not excluded) and reward == 1, "details": details}

    def aggregate(self, per_case: list, trace_events: list) -> dict:
        """Round-level project_metrics over the node's CUMULATIVE cases (the trace is
        per-batch; everything here comes from per-case details)."""
        rows = [(c, c.details or {}) for c in per_case]
        scored = [(c, d) for c, d in rows if not d.get("excluded") and not c.error]
        excluded = [(c, d) for c, d in rows if d.get("excluded") or c.error]
        n = len(scored)
        solved = sum(1 for c, d in scored if d.get("reward") == 1)
        final = [(d.get("verdicts") or [None])[-1] for _, d in scored]
        vpass = [(c, d) for (c, d), v in zip(scored, final) if v == "pass"]
        vfail = [(c, d) for (c, d), v in zip(scored, final) if v == "fail"]
        classes = Counter(k for _, d in scored for k in d.get("failure_classes") or [])
        infra = Counter((d.get("infra_class") or ("evaluator:" + (c.error or "")[:40])) for c, d in excluded)
        by_lang: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        for _, d in scored:
            lang = d.get("language") or "?"
            by_lang[lang][1] += 1
            by_lang[lang][0] += d.get("reward") == 1
        roles = [r for _, d in scored for r in d.get("roles") or []]
        def rate(xs, pred):
            return round(sum(1 for x in xs if pred(x)) / len(xs), 3) if xs else None
        return {
            "n_scored": n,
            "n_excluded_infra": len(excluded),
            "resolve_rate_of_scored": round(solved / n, 3) if n else None,
            "verify_final_pass_rate": round(len(vpass) / n, 3) if n else None,
            "verify_false_pass_rate": rate(vpass, lambda cd: cd[1].get("reward") != 1),
            "verify_false_fail_rate": rate(vfail, lambda cd: cd[1].get("reward") == 1),
            "mean_patch_attempts": round(sum(d.get("patch_attempts") or 0 for _, d in scored) / n, 2) if n else None,
            "patch_role_wall_rate": rate([r for r in roles if r.get("role") == "patch"], lambda r: r.get("wall_terminated")),
            "verify_role_wall_rate": rate([r for r in roles if r.get("role") == "verify"], lambda r: r.get("wall_terminated")),
            "patch_zero_edit_rate": rate([r for r in roles if r.get("role") == "patch"], lambda r: (r.get("edits") or 0) == 0),
            "verify_zero_test_run_rate": rate([r for r in roles if r.get("role") == "verify"], lambda r: (r.get("tests_run") or 0) == 0),
            "truncations_total": sum(r.get("truncations") or 0 for r in roles),
            "resolve_rate_by_language": {k: round(v[0] / v[1], 3) for k, v in sorted(by_lang.items())},
            # Curriculum / categorizer inputs.
            "top_failed_checks": sorted(classes.items(), key=lambda kv: (-kv[1], kv[0])),
            "harness_checks": sorted(infra.items(), key=lambda kv: (-kv[1], kv[0])),
            "no_plan_rate": rate(scored, lambda cd: not cd[1].get("patch_bytes")),
        }


_SCORER = DeepSWESeedlingScorer()


def score(case: dict, agent_output: Any) -> dict:
    return _SCORER.score(case, agent_output)


assert set(FAILURE_CLASSES)  # exported for the categorizer
