"""Error categorizer for the feedback gatherer's failure report
(``gatherer.config.error_categorizer: projects.deepswe_seedling.adapter.categorizer:categorize_errors``).

One category per failure class from ``trial.failure_classes`` (task-level) plus one per
infrastructure class (harness-level). Representative errors carry only the run report's
outcome line and the agent's own last VERIFY/PATCH statements -- never hidden-test output.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

category_type_priority = ["seedling_outcome", "infrastructure"]

_NAMES = {
    "build_break": "Build broken: whole pre-existing suite dead (p2p == 0)",
    "empty_patch": "No patch produced",
    "verify_false_pass": "VERIFY passed a patch the grader failed",
    "deadline_unverified": "Shipped at the deadline with VERIFY's last verdict = fail",
    "near_miss": "Near miss: some required behaviour missing (0 < f2p < 1, p2p == 1)",
    "p2p_regression": "Regression: pre-existing tests broken (0 < p2p < 1)",
    "f2p_zero": "No required behaviour passed (f2p == 0, suite alive)",
    "role_wall": "A role ran out of its wall-clock budget",
    "role_zero_edit": "A PATCH attempt made no edits",
    "agent_crash": "seedling run() contained an exception",
}


def categorize_errors(per_case: list) -> list[dict[str, Any]]:
    total = len(per_case)
    groups: dict[str, dict[str, Any]] = defaultdict(lambda: {"cases": []})
    for c in per_case:
        d = c.details or {}
        if d.get("excluded") or c.error:
            key = "infra__" + str(d.get("infra_class") or "evaluator_crash")
            groups[key]["type"] = "infrastructure"
            groups[key]["name"] = f"Infrastructure: {d.get('infra_class') or c.error}"[:120]
            groups[key]["cases"].append((c, [key[7:]]))
            continue
        for k in d.get("failure_classes") or []:
            key = "seedling__" + k
            groups[key]["type"] = "seedling_outcome"
            groups[key]["name"] = _NAMES.get(k, k)
            groups[key]["cases"].append((c, d.get("failure_classes") or []))
    out = []
    for cid, g in groups.items():
        reps = []
        for c, checks in g["cases"][:5]:
            raw = str((c.details or {}).get("raw_result") or "")
            first = raw.splitlines()[0] if raw else ""
            tail = raw[-400:] if len(raw) > 400 else ""
            reps.append({"sample_id": c.case_id, "checks_failed": list(checks),
                         "messages": [m for m in (first, tail) if m]})
        n = len(g["cases"])
        out.append({
            "category_id": cid, "category_type": g["type"], "category_name": g["name"],
            "slug": cid.split("__", 1)[1], "num_failing_samples": n, "total_samples": total,
            "failure_rate": round(n / total, 4) if total else 0.0,
            "representative_errors": reps,
        })
    out.sort(key=lambda c: (-c["failure_rate"], c["category_id"]))
    return out
