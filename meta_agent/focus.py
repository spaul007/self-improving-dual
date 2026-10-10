"""Per-EXPAND focus: ``default`` (the usual EXPAND) or ``reliability``.

A third, orthogonal steering axis next to block (WHICH part of the agent) and implementation
strategy (HOW MUCH code vs prompt). Focus says WHICH EVIDENCE drives the edit. ``reliability``
targets test cases the agent sometimes solves and sometimes does not -- the per-case files'
**Pass/fail contrast** (meta_agent/case_reflections.py) puts a high-scoring and a low-scoring run
of the same case side by side -- and asks the editor to make the high run's behaviour mandatory
instead of adding new capability. It spends no extra evaluations: the child's first batch merely
includes its target cases (HGMManager).

Steer, don't fence (the same convention as implementation_strategy.py): nothing prevents the
editor from deviating; the body asks it to say so in its rationale.
"""
from __future__ import annotations

from typing import Iterable, Optional

FOCUS_VALUES: tuple[str, ...] = ("default", "reliability")

_RELIABILITY_BODY = (
    "## Reliability focus for this EXPAND\n\n"
    "These test cases are UNSTABLE: some runs scored high and some low ({targets}).\n"
    "For each, read `cases/<file>` and its **Pass/fail contrast** section: one high-scoring and one "
    "low-scoring run of the same case, with each role's own account side by side (what was essential "
    "in the good run; where and why the bad run went wrong).\n"
    "Find what the high-scoring run did that the low-scoring run did not, and make THAT behaviour "
    "reliable -- a mandatory step or check (or code, if the implementation strategy allows) at the "
    "point where the runs diverged. Do not add new capabilities.\n"
    "If the contrast shows the difference is luck, server load or a budget rather than behaviour, say "
    "so in your rationale and make the smallest change. Write a role-targeted probe question for the "
    "behaviour you made mandatory, so the next evaluation shows whether it now happens every time.\n"
)


def focus_body(focus: Optional[str], targets: list[dict]) -> str:
    """The steering text for one EXPAND (``""`` for default / no focus)."""
    if focus != "reliability" or not targets:
        return ""
    desc = "; ".join(
        f"`{t['case_id']}` -> {t['file']}: {t['passes']}/{t['evals']} passed, score spread {t['spread']:.2f}"
        for t in targets)
    return _RELIABILITY_BODY.format(targets=desc)


def reliability_targets(summary: dict[str, dict], *, k: int = 3, min_evals: int = 3,
                        prefer: Iterable[str] = ()) -> list[dict]:
    """Up to ``k`` unstable cases from ``build_case_files``' summary: a pass/fail contrast exists
    whose low run was not cut short by a budget, with at least ``min_evals`` scored evaluations.
    Cases the parent was evaluated on (``prefer``) first, then the widest score spread."""
    pref = set(prefer)
    cands = [dict(case_id=cid, **s) for cid, s in summary.items()
             if s.get("unstable") and s.get("evals", 0) >= min_evals]
    cands.sort(key=lambda t: (t["case_id"] not in pref, -float(t.get("spread") or 0.0), str(t["case_id"])))
    return cands[: max(0, k)]
