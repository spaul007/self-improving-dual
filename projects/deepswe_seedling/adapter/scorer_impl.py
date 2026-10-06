"""Scorer for the seedling-on-DeepSWE project (``deepswe_seedling_default``).

Runs in the PARENT (evaluator) process. The utility is Pier's own binary grade,
``verifier_result.rewards.reward`` (user decision 2026-09-23: binary reward), re-read from
the trial's result.json -- never trusted from the child's metadata. With
``SID_UTILITY=f2p`` the utility is the trial's f2p instead (see ``utility``).

Infrastructure failures (``details["excluded"] = True``) are excluded from the node's
utility by HGM ``exclude_flagged_cases``: docker/env start, verifier setup/timeout,
missing reward, unreachable LLM server, adapter/watchdog failures. A crash of the MUTABLE
seedling code is NOT infra -- it scores 0.

Hidden-test isolation: details carry only numeric grade fields and the agent's own
signals -- never test names or test output.
"""
from __future__ import annotations

import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from meta_agent.registry import register

from .trial import FAILURE_CLASSES, failure_classes, load_outcome

JOBS_ROOT_ENV = "SID_PIER_JOBS_ROOT"
# Utility knob (opt-in; default = binary reward, unchanged). "f2p": the fraction of the
# hidden feature (fail-to-pass) tests passed -- ~2x steadier across identical runs than the
# binary reward on DeepSWE (EXP-035 vs EXP-039). `passed` stays reward == 1 either way.
UTILITY_ENV = "SID_UTILITY"
UTILITIES = ("binary", "f2p")


def utility(reward: Any, f2p: Any) -> float:
    mode = (os.environ.get(UTILITY_ENV) or "binary").strip().lower()
    if mode not in UTILITIES:
        raise ValueError(f"{UTILITY_ENV}={mode!r}: expected one of {UTILITIES}")
    if mode == "binary" or reward == 1:
        return float(reward or 0.0)
    return float(f2p) if isinstance(f2p, (int, float)) else 0.0
DEFAULT_JOBS_ROOT = "sid_pier_jobs"   # same default as pier_case.SID_PIER_JOBS_ROOT


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
        score = 0.0 if excluded else utility(reward, o.get("f2p"))
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


    # ------------------------------------------------------------------ #
    # Task-agent reflection hooks (meta_agent/reflector.py; opt-in via the
    # ``reflector:`` config section). Sessions are pier's own per-role
    # conversation snapshots (agent/conv/<role>.<attempt>.json); the grading
    # outcome is read from the trial's verifier/ -- hidden test names go INTO the
    # task agent's question and are returned as redact terms, so under
    # ``exposure: lessons_only`` they never reach the meta-agent.
    # ------------------------------------------------------------------ #

    REFLECTION_ROLES = ("patch", "verify")
    _WHO = {"patch": "You are the PATCH role; this is your own session and your patch is the one "
                     "that was submitted.",
            "verify": "You are the VERIFY role; this is your own session and your last verdict stands."}

    @staticmethod
    def _trial(case: Any) -> Path | None:
        meta = (case.details or {}).get("agent_metadata") or {}
        return _trusted(meta.get("trial_dir"))

    def reflection_sessions(self, case: Any, round_dir: Path) -> dict[str, dict]:
        trial = self._trial(case)
        if trial is None:
            return {}
        out = {}
        for role in self.REFLECTION_ROLES:
            files = sorted(trial.glob(f"agent/conv/{role}.*.json"),
                           key=lambda p: int(p.stem.split(".")[1]) if p.stem.split(".")[1].isdigit() else -1)
            conv = _read_json(files[-1]) if files else None
            if conv and conv.get("messages"):
                out[role] = {"messages": conv["messages"], "format": "chat",
                             "preamble": self._WHO.get(role, f"You are the {role.upper()} role.")}
        return out

    def grading_outcome(self, case: Any, detail: str) -> dict[str, Any]:
        d = case.details or {}
        f2p = f"{d.get('f2p_passed')}/{d.get('f2p_total')}"
        p2p = f"{d.get('p2p_passed')}/{d.get('p2p_total')}"
        head = (f"Result: NOT resolved (reward 0). Hidden feature tests passed: {f2p}. "
                f"Pre-existing tests passed: {p2p}.")
        trial = self._trial(case)
        if detail == "numeric" or trial is None:
            return {"text": head, "redact": []}
        ctrf = _read_json(trial / "verifier" / "ctrf.json") or {}
        tests = ((ctrf.get("results") or {}).get("tests")) or []
        fails = [str(t.get("name", "?")) for t in tests if t.get("status") != "passed"]
        lines = [head, "", f"Hidden tests that FAILED ({len(fails)} total; up to {_MAX_FAILING} listed):"]
        lines += [f"  - {n}" for n in fails[:_MAX_FAILING]]
        exc = _excerpts(trial, fails)
        if exc:
            lines += ["", "Test-output excerpts for some of them (best effort, may be truncated):"]
            for n, e in list(exc.items())[:8]:
                lines += [f"--- {n}", e.strip(), ""]
        redact = set()
        for n in fails:
            redact.add(n)
            bare = _bare_test_name(n)
            redact.add(bare)
            # The unqualified test/function name too (Go pkg.TestX, pytest a.py::T::test_x,
            # jest "suite > case"): a lesson naming just that must still be redacted.
            # Identifier-like only (has a capital, digit or underscore), so a plain English
            # word from a jest description never gets blanked out of every lesson.
            redact.update(p for p in re.split(r"::|[./#>\s]+", bare)[-2:]
                          if len(p) >= 6 and re.search(r"[A-Z0-9_]", p))
            redact.update(p for p in re.findall(r"[\w./-]+\.(?:go|rs|py|ts|tsx|js|jsx|java|rb)\b", n))
        return {"text": "\n".join(lines), "redact": sorted(x for x in redact if len(x) >= 4)}


_MAX_FAILING = 25
_EXCERPT_CHARS = 600


def _read_json(p: Path) -> Any:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def _bare_test_name(name: str) -> str:
    return re.sub(r"^\[(f2p|p2p)\]\s*", "", name).split(" ")[-1]


def _excerpts(trial: Path, names: list[str]) -> dict[str, str]:
    """Short test-output excerpt per failing test, located by its bare name (best effort)."""
    try:
        out = (trial / "verifier" / "test-stdout.txt").read_text(errors="replace")
    except OSError:
        return {}
    got = {}
    for n in names[:_MAX_FAILING]:
        bare = _bare_test_name(n)
        if len(bare) < 4:
            continue
        i = out.find(bare)
        if i >= 0:
            got[n] = out[max(0, i - 100): i + _EXCERPT_CHARS].replace("\r", "")
    return got

_SCORER = DeepSWESeedlingScorer()


def score(case: dict, agent_output: Any) -> dict:
    return _SCORER.score(case, agent_output)


assert set(FAILURE_CLASSES)  # exported for the categorizer
