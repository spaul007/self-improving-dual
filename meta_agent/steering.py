"""Belief-mode steering context — the ONE place its text is authored.

In ``steering_mode: "belief"`` the manager hands the editor exactly this
block as its ``context``: the objective (fix what the judge found; the score
is one labelled context line), the judge's line for the parent, a scope
rule, the lineage, one judge-first line per sibling already tried off the
same parent, and the belief document verbatim. Nothing else is appended by
the manager, and nothing here is ever truncated: the belief document is
bounded where it is generated (its hard char cap), and the sibling lines are
one line each.

Pure: reads the run's records, no LLM, no writes. The legacy (``full``) mode
and the no-edit-memory control keep their own text in ``managers/hgm.py``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from .edit_memory_render import _load_records
from .perf_text import judge_summary, perf_summary, regressions_summary

OBJECTIVE_TEXT = (
    "Fix what the judge found. Every edit in this run is graded by the per-node "
    "analysis (the judge) on its own runtime traces and per-check results, and "
    "those verdicts — targeted checks that did not move, mechanisms that never "
    "fired or disagreed with the scorer, regressions — are what the run learns "
    "from and what your edit will be judged on. Choose the failure your edit "
    "removes from the judge's findings on this parent and its siblings below "
    "and from the failure analysis in the feedback."
)
SCORE_CONTEXT_LABEL = "Score context (a noisy reference, not the objective): "
SCOPE_TEXT = (
    "Make ONE targeted, coherent change to this parent — one strategy in one "
    "area — small enough to apply correctly in a single pass, and instrument "
    "every new decision point with trace.log — the analysis that judges your "
    "edit reads those logs. Do not bundle independent mechanisms into one edit: "
    "each mechanism is judged on its own, and a broken one next to a working "
    "one costs the whole node its credit."
)
BELIEF_FRAMING = (
    "Maintained from this run's judged edit history: each belief is a scoped "
    "probability whose calibration is tracked in its `- track:` line. Advisory — "
    "verify it against the code you see."
)
SIBLING_FRAMING = (
    "Per sibling: the judge's effect verdict for its primary mechanism "
    "(evidence grade; the checks it targeted), any regressions the judge "
    "listed, then — as context only — the score (paired Δ over shared cases "
    "when ≥ 8 are shared, else unpaired Δ ± SE over own cases; noise unless "
    "|Δ| > 2×SE), the child's mean, the goal, and flags."
)


def parent_judge_line(records: Mapping[int, Any], parent_id: int) -> str:
    """What the judge said about the parent's own edit, in one line — the
    verdict for its primary mechanism with targets, any regressions, the
    implementation verdict and the judge's reason."""
    rec = records.get(parent_id)
    if rec is None:
        return ("(node 0 is the seed — no edit to judge)" if parent_id == 0
                else "(no record for this parent)")
    judge = judge_summary(rec)
    if not judge:
        return ("- not yet judged (the analysis runs after one batch of the "
                "parent's own evaluations)")
    bits = [f"judge {judge}"]
    reg = regressions_summary(rec)
    if reg:
        bits.append(f"regressions: {reg}")
    impl = rec.get("impl_sound")
    if impl is not None:
        bits.append(f"implementation {'sound' if impl else 'unsound'}")
    line = "- " + " · ".join(bits)
    reason = " ".join(str(rec.get("effect_reason") or "").split())[:200]
    if reason:
        line += f' — "{reason}"'
    return line


def compact_sibling_lines(records: Mapping[int, Any],
                          siblings: Sequence[tuple[int, str, bool]], *,
                          threshold: float, min_shared: int) -> list[str]:
    """One line per sibling: judge verdict (with targets), regressions, the
    score as context, absolute score, goal, flags."""
    out: list[str] = []
    for nid, goal, failed in sorted(siblings, key=lambda t: t[0]):
        # First line of the goal only (dual-manager goals are multi-line).
        g = " ".join(str(goal or "").split("\n")[0].split())[:160]
        rec = records.get(nid)
        if rec is None:
            status = "edit failed" if failed else "unmeasured"
            out.append(f'- node {nid}: {status} · "{g}"')
            continue
        child_abs, n_abs = rec.get("child_abs"), rec.get("n_abs") or 0
        bits: list[str] = [f"judge {judge_summary(rec) or 'not yet judged'}"]
        reg = regressions_summary(rec)
        if reg:
            bits.append(f"regressions: {reg}")
        score = perf_summary(rec, threshold=threshold, min_shared=min_shared)
        if score == "unmeasured" and child_abs is not None:
            score = "unmeasured vs parent"
        bits.append(f"score {score}")
        if child_abs is not None:
            bits.append(f"child {child_abs:.4f}/{n_abs}")
        perf = " · ".join(bits)
        flags: list[str] = []
        usage = rec.get("usage") or ""
        if "0 calls" in usage or "never fired" in usage:
            flags.append("dead component")
        if rec.get("suspect"):
            flags.append("suspect verifier")
        if rec.get("impl_sound") is False:
            flags.append("implementation unsound")
        if failed:
            flags.append("edit failed")
        out.append(f'- node {nid}: {perf} · "{g}"'
                   + (f" · flags: {', '.join(flags)}" if flags else ""))
    return out


def render_belief_steering(
    *,
    experiment_dir: Path,
    parent_id: int,
    lineage: Sequence[tuple[int, str]],
    parent_score: Optional[tuple[float, int]],
    run_context: Mapping[str, Any],
    siblings: Sequence[tuple[int, str, bool]],
    belief_doc: str,
    calibration_line: str = "",
    threshold: float,
    min_shared: int,
) -> str:
    """The complete belief-mode steering context.

    ``lineage`` is ``[(depth, goal)]`` root → parent (``hgm._ancestor_goals``);
    ``parent_score`` is ``(mean, n_evals)`` or ``None`` when unevaluated;
    ``siblings`` is ``[(node_id, goal, edit_failed)]`` for the parent's
    existing children.
    """
    records = _load_records(Path(experiment_dir))
    parts: list[str] = ["## Objective", OBJECTIVE_TEXT]
    bits: list[str] = []
    rc = run_context or {}
    if rc:
        bits.append("seed %.4f/%d" % (rc.get("seed_mean", 0.0), rc.get("seed_n", 0)))
        bits.append("best so far %.4f/%d (node %d)"
                    % (rc.get("best_mean", 0.0), rc.get("best_n", 0),
                       rc.get("best_node", -1)))
    if parent_score is not None and parent_score[1] > 0:
        bits.append("this parent (node %d) %.4f/%d"
                    % (parent_id, parent_score[0], parent_score[1]))
    else:
        bits.append(f"this parent (node {parent_id}) not yet evaluated")
    parts.append(SCORE_CONTEXT_LABEL + " · ".join(bits) + ".")

    parts += ["", f"## What the judge found on this parent (node {parent_id})",
              parent_judge_line(records, parent_id)]

    parts += ["", "## Scope of this edit", SCOPE_TEXT]

    lin = [(d, g) for d, g in lineage if d > 0 and g]
    if lin:
        parts += ["", "## Edits already applied along this lineage (root → parent)"]
        for depth, goal in lin:
            lines = str(goal).split("\n")
            parts.append(f"  [depth {depth}] {lines[0][:200]}")
            for cont in lines[1:]:
                if cont.strip():
                    parts.append(f"             {cont[:200]}")

    parts += ["", f"## Edits already tried directly off this parent (node {parent_id})"]
    sib = compact_sibling_lines(records, siblings, threshold=threshold,
                                min_shared=min_shared)
    if sib:
        parts.append(SIBLING_FRAMING)
        parts += sib
        parts.append("(Full records and implementation of any node are "
                     "retrievable by node id.)")
    else:
        parts.append("(none yet)")

    parts += ["", "## Belief document",
              BELIEF_FRAMING + (f" {calibration_line}" if calibration_line else "")]
    doc = (belief_doc or "").strip("\n")
    parts.append(doc if doc.strip() else
                 "(no belief document yet — this is an early edit)")
    return "\n".join(parts)
