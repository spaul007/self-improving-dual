"""Curriculum layer for HGM EXPAND: stay on one failing constraint at a
time instead of hopping between unrelated diagnoses every round.

Built ONCE via ``infer_curriculum`` from the seed node's own
``AgentFeedback.project_metrics`` (see ``meta_agent/managers/hgm.py``'s
post-seed construction) -- and never re-ranked or re-merged across the
tree afterwards. Only the resolution check re-reads live data (the current
best node's own ``project_metrics``) on each call.

Pure logic, no I/O, no LLM calls -- same spirit as
``meta_agent/block_bandit.py``, fully unit-testable offline. Takes only
plain ``dict``/``list``/``float`` arguments, never ``HGMTree``/
``AgentFeedback`` directly, so the manager (``hgm.py``) owns all framework
wiring and this module stays trivially testable.
"""
from __future__ import annotations

import dataclasses
from typing import Any, Optional


def _source_counts(project_metrics: dict, source_key: str) -> dict[str, int]:
    """Validated ``(name, count)`` entries from ONE ``project_metrics`` list
    key (``"top_failed_checks"`` or ``"harness_checks"``), normalized the
    same way regardless of caller. Never raises: a malformed entry is
    skipped, not fatal."""
    counts: dict[str, int] = {}
    for entry in project_metrics.get(source_key) or []:
        try:
            name, count = entry[0], entry[1]
        except (TypeError, IndexError, KeyError):
            continue
        if not isinstance(name, str):
            continue
        try:
            count = int(count)
        except (TypeError, ValueError):
            continue
        counts[name] = counts.get(name, 0) + count
    return counts


def _ranked(counts: dict[str, int]) -> list[tuple[str, int]]:
    """Descending by count, ties broken alphabetically for determinism."""
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))


def _combined_check_counts(project_metrics: dict) -> dict[str, int]:
    """Merge two independent (name, count) rankings a project's scorer may
    emit in ``project_metrics`` into one combined name -> count dict:

    - ``top_failed_checks`` -- task-level constraint failures (a specific
      check or constraint the generated output failed), counted only
      among cases that actually reached real evaluation.
    - ``harness_checks`` -- harness-level crash CAUSES, tallied from
      whatever boolean flags the agent's own (project-specific) workflow
      happens to set on crashed (no-plan) cases specifically. See
      ``projects/travel_mas_refactored/adapter/scorer_impl.py::aggregate``
      for one concrete example of how both are actually computed -- this
      function itself makes no assumption about which project a given
      ``project_metrics`` dict came from, and degrades to an empty merge
      for a project whose
      scorer emits neither key.

    Both lists happen to already share the same ``[(name, count), ...]``
    shape, so no per-source special-casing is needed here -- a name
    appearing in both (shouldn't happen in practice, since the two
    namespaces are disjoint by construction) has its counts summed rather
    than one silently overwriting the other."""
    combined: dict[str, int] = {}
    for source_key in ("top_failed_checks", "harness_checks"):
        for name, count in _source_counts(project_metrics, source_key).items():
            combined[name] = combined.get(name, 0) + count
    return combined


# A harness crash rate at or above this is treated as "the harness itself
# is unreliable" -- see infer_curriculum's harness_priority_threshold.
DEFAULT_HARNESS_PRIORITY_THRESHOLD = 0.15


def infer_curriculum(
    project_metrics: dict, *, k: int = 15,
    harness_priority_threshold: float = DEFAULT_HARNESS_PRIORITY_THRESHOLD,
) -> list[tuple[str, int]]:
    """Infer an ordered curriculum goal list from a single ``project_metrics``
    snapshot (in practice the seed node's own, evaluated once for free
    before any EXPAND spends budget -- see ``hgm.py``'s post-seed
    construction, which calls this exactly ONCE per run).

    Below ``harness_priority_threshold``: merges task-level constraint
    failures (``top_failed_checks``) and harness-level crash causes
    (``harness_checks``) into ONE combined ranking by raw occurrence count
    (ties broken alphabetically for determinism) and returns the top
    ``k`` -- competing fairly against each other, since both are, at
    bottom, the same kind of thing: a reason SOME cases didn't get full
    credit.

    At or above it: ``project_metrics["no_plan_rate"]`` -- the harness's
    OWN error rate, the fraction of cases that crashed before any
    task-level check could even run -- is high enough that the harness
    itself, not any specific task behavior, is the bottleneck. A
    task-level fix cannot show up in the composite score for a case the
    harness never let reach scoring, so every harness-level goal (ranked
    by its own count) is placed ahead of every task-level goal (ranked by
    its own count) rather than letting a merely-more-common task check
    outrank a harness failure that is silently zeroing out a much larger
    share of the score. Falls back to the flat merge above whenever
    ``no_plan_rate`` is missing/non-numeric, or ``harness_checks`` itself
    is empty (nothing to prioritize) -- so a project/run that predates
    ``harness_checks`` behaves exactly as before.

    Returns ``[]`` when ``project_metrics`` is empty/has neither
    ``top_failed_checks`` nor ``harness_checks`` (nothing to build a
    curriculum from) -- callers must treat that as "no curriculum," same
    as an unconfigured/disabled one."""
    no_plan_rate = project_metrics.get("no_plan_rate")
    harness_unreliable = (
        isinstance(no_plan_rate, (int, float))
        and no_plan_rate >= harness_priority_threshold
    )
    if harness_unreliable:
        harness_ranked = _ranked(_source_counts(project_metrics, "harness_checks"))
        if harness_ranked:
            seen = {name for name, _ in harness_ranked}
            task_ranked = [
                entry
                for entry in _ranked(_source_counts(project_metrics, "top_failed_checks"))
                if entry[0] not in seen
            ]
            return (harness_ranked + task_ranked)[:k]
    return _ranked(_combined_check_counts(project_metrics))[:k]


@dataclasses.dataclass
class CurriculumSnapshot:
    """Persisted every EXPAND (the ``adaptive_strategy.json`` precedent --
    see ``hgm.py``'s ``_expand``) as ``curriculum_status.json``, for
    debug/dashboard visibility into what the curriculum is currently
    doing."""

    goals: list[str]
    seed_counts: dict[str, int]
    current_index: int
    current_goal: Optional[str]
    rounds_on_current: int
    patience: int
    resolution_threshold: float
    current_failure_rate: Optional[float]
    resolved_goals: list[str]
    # "resolved" | "patience_exhausted" | None -- set only on the call
    # that just advanced past a goal, i.e. describes what just happened,
    # not a running state.
    advance_reason: Optional[str]
    done: bool


class Curriculum:
    """Ordered sub-goal list over failing checks, one EXPAND-scoped focus
    at a time.

    Advancement is 100% code, never LLM judgment (mirrors strategies.md's
    own "prefer code over LLM judgment for anything mechanically
    computable" principle): advance past the current goal when EITHER
    (a) the current best node's failure rate on that check is <=
    ``resolution_threshold``, OR (b) ``patience`` EXPANDs have already
    been spent on it -- whichever comes first. This guarantees the
    curriculum can never get stuck forever on an unresolvable check.
    """

    def __init__(
        self,
        goals: list[tuple[str, Any]],
        *,
        resolution_threshold: float = 0.15,
        patience: int = 5,
        # Optional check-name -> human-readable semantics ("what this
        # check verifies and when it fires"), e.g. loaded by the manager
        # from a project-specific JSON registry (see hgm.py's
        # curriculum_check_descriptions_path). Purely additive: a goal
        # absent from this dict (or the dict itself absent -- most
        # projects won't have one) falls back to directive()'s existing
        # generic phrasing. Keeps this class itself free of any file I/O
        # or project-specific coupling -- the manager owns loading the
        # JSON, this class just renders whatever mapping it's handed.
        check_descriptions: Optional[dict[str, str]] = None,
    ) -> None:
        # `goals` ordering is exactly the seed's top_failed_checks order
        # (already sorted desc by occurrence count by the project's own
        # scorer aggregate()) -- no re-sorting here. Duplicate check names
        # (shouldn't happen from a real top_failed_checks list, but keep
        # this robust) collapse to their first occurrence's position.
        seen: set[str] = set()
        self._goals: list[str] = []
        self._seed_counts: dict[str, int] = {}
        for check, count in goals:
            check = str(check)
            if check in seen:
                continue
            seen.add(check)
            self._goals.append(check)
            try:
                self._seed_counts[check] = int(count)
            except (TypeError, ValueError):
                self._seed_counts[check] = 0
        self.resolution_threshold = resolution_threshold
        self.patience = patience
        self._check_descriptions: dict[str, str] = check_descriptions or {}
        self._index = 0
        self._rounds_on_current = 0
        self._resolved: list[str] = []

    @property
    def done(self) -> bool:
        return self._index >= len(self._goals)

    @property
    def current_goal(self) -> Optional[str]:
        return None if self.done else self._goals[self._index]

    @staticmethod
    def failure_rate_for(
        check: Optional[str], project_metrics: dict, n_evals: int,
    ) -> Optional[float]:
        """A goal's live failure rate, or ``None`` when ``n_evals <= 0`` or
        every eval crashed (nothing evaluated yet for this node -- the
        caller must not treat that as resolved). Never raises: malformed/
        missing data degrades gracefully to "0 occurrences" or ``None``,
        never an exception.

        Searches ``top_failed_checks`` first (task-level constraint
        failures), then ``harness_checks`` (harness-level crash causes) --
        the same two rankings ``infer_curriculum`` merges to build the
        goal list in the first place, so any goal it produced is
        resolvable here by construction. The two use different
        denominators because they're tallied over different populations:

        - A ``top_failed_checks`` hit divides by ``scored_evals`` --
          ``n_evals`` corrected to exclude crashed (no-plan) cases when
          the scorer reports ``no_plan_rate`` in ``project_metrics``
          (absent means no correction, identical to using ``n_evals``
          directly). This matters because ``top_failed_checks``'s own
          count already excludes crashed cases (the scorer's aggregate()
          skips them when tallying failed_checks) -- left uncorrected, a
          heavily-crashed node's rate is falsely deflated toward 0 for
          EVERY check, the same information-ceiling failure mode fixed in
          behavior_summarizer.py's ``_check_fail_case_ids`` (a 100%-crashed
          node must never read as "check resolved").
        - A ``harness_checks`` hit divides by raw ``n_evals`` directly --
          it's already a count of ALL evals attributable to that specific
          crash cause (not scoped to non-crashed cases the way a task
          check is), so no correction applies.
        - A check absent from BOTH counts as 0 occurrences (itself
          evidence of resolution -- it dropped out of both rankings), not
          unknown, using the same scored_evals-corrected denominator as
          top_failed_checks (the more common goal source)."""
        if check is None or n_evals is None or n_evals <= 0:
            return None

        def _scored_evals() -> float:
            no_plan_rate = project_metrics.get("no_plan_rate")
            if isinstance(no_plan_rate, (int, float)) and 0.0 <= no_plan_rate <= 1.0:
                return n_evals * (1.0 - no_plan_rate)
            return float(n_evals)

        for entry in project_metrics.get("top_failed_checks") or []:
            try:
                name, occurrences = entry[0], entry[1]
            except (TypeError, IndexError, KeyError):
                continue
            if name == check:
                scored_evals = _scored_evals()
                if scored_evals <= 0:
                    return None
                try:
                    return float(occurrences) / scored_evals
                except (TypeError, ValueError):
                    return None

        for entry in project_metrics.get("harness_checks") or []:
            try:
                name, occurrences = entry[0], entry[1]
            except (TypeError, IndexError, KeyError):
                continue
            if name == check:
                try:
                    return float(occurrences) / n_evals
                except (TypeError, ValueError):
                    return None

        scored_evals = _scored_evals()
        if scored_evals <= 0:
            return None
        return 0.0

    def record_expand(self) -> None:
        """Call once per EXPAND that consulted the curriculum
        (``hgm.py::_curriculum_directive_for_expand``), after checking
        ``advance_if_ready`` for the round's (possibly just-advanced)
        current goal -- increments the patience counter for whichever
        goal is current at the time of the call. No-op when done."""
        if not self.done:
            self._rounds_on_current += 1

    def advance_if_ready(self, *, failure_rate: Optional[float]) -> Optional[str]:
        """Advance past the current goal if resolved-or-patience-exhausted;
        returns ``"resolved"`` / ``"patience_exhausted"`` / ``None``.
        ``failure_rate=None`` (nothing evaluated yet for the best node)
        never counts as resolved, but patience still accrues regardless of
        whether resolution could even be checked this call -- keeps the
        fallback simple and robust (an unevaluated period still burns
        patience rather than stalling the curriculum indefinitely)."""
        if self.done:
            return None
        resolved = (
            failure_rate is not None and failure_rate <= self.resolution_threshold
        )
        patience_exhausted = self._rounds_on_current >= self.patience
        if not resolved and not patience_exhausted:
            return None
        reason = "resolved" if resolved else "patience_exhausted"
        self._resolved.append(self._goals[self._index])
        self._index += 1
        self._rounds_on_current = 0
        return reason

    def directive(self) -> Optional[str]:
        """Block-agnostic curriculum instruction for this EXPAND, reused
        verbatim at both injection points (the editor's own context and
        the block-suggester's diagnosis prompt). ``None`` once ``done``.
        Contains an explicit escape hatch: a block that genuinely can't
        address the focused check should say so and propose its own best
        genuine fix instead of forcing an irrelevant change.

        As concrete as the manager gave it the means to be: always states
        the seed's own occurrence count for this exact goal (a real
        number, not just a bare identifier), and uses
        ``check_descriptions[goal]`` (see ``__init__``) for a real
        semantic explanation of the failure mode when the manager
        supplied one -- falling back to a generic explanation of the two
        possible KINDS of goal (task check vs. harness crash cause) only
        when no specific description is available."""
        if self.done:
            return None
        goal = self.current_goal
        seed_count = self._seed_counts.get(goal, 0)
        description = self._check_descriptions.get(goal)
        what_this_is = (
            f"`{goal}` -- {description}"
            if description
            else (
                f"`{goal}` -- either a task-based error (a specific check "
                "or constraint the generated output failed) or a "
                "harness-based error (a cause of a case crashing outright, "
                "tallied from the agent's own workflow) -- whichever this "
                "is, treat it as this EXPAND's one target."
            )
        )
        return (
            f"Focus this EXPAND's diagnosis on resolving this specific "
            f"failure mode: {what_this_is}\n\n"
            f"This failed in {seed_count} of the seed's evaluated cases -- "
            "we need this specific failure mode's occurrence rate to go "
            "down, not the composite score in general.\n\n"
            f"This is sub-goal {self._index + 1} of {len(self._goals)} in "
            "the current curriculum (ordered by how often each failure "
            "mode occurred in the seed's own evaluation -- most frequent "
            "first, task checks and harness crash causes ranked together "
            "by raw count). "
            f"Diagnose and propose a fix specifically targeting `{goal}` "
            "within the block you've been assigned, grounded in concrete "
            "evidence from the feedback/failure summary shown to you -- "
            "do not propose a generic or unrelated improvement instead.\n\n"
            "ESCAPE HATCH: if the block you've been assigned genuinely "
            f"cannot address `{goal}` (the real fix belongs in a "
            "different block entirely), say so explicitly and instead "
            "diagnose and propose the best genuine improvement for THIS "
            "block -- never force an irrelevant or superficial change "
            "just to appear to address the curriculum focus. The "
            "curriculum stays on this same goal for the next EXPAND "
            "regardless of which block ends up fixing it."
        )

    def snapshot(
        self,
        *,
        current_failure_rate: Optional[float],
        advance_reason: Optional[str] = None,
    ) -> CurriculumSnapshot:
        return CurriculumSnapshot(
            goals=list(self._goals),
            seed_counts=dict(self._seed_counts),
            current_index=self._index,
            current_goal=self.current_goal,
            rounds_on_current=self._rounds_on_current,
            patience=self.patience,
            resolution_threshold=self.resolution_threshold,
            current_failure_rate=current_failure_rate,
            resolved_goals=list(self._resolved),
            advance_reason=advance_reason,
            done=self.done,
        )
