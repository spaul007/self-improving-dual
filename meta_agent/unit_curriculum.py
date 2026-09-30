"""Unit/dimension-level curriculum -- the additive "unit" granularity for
meta_agent/curriculum.py's opt-in curriculum layer. Groups related failing
CHECKS into named "units" (a project-supplied mapping, e.g. every
hard:train_*/hard:flight_* check under "hard constraints: transport"), and
re-selects which unit to focus on -- via an LLM call reasoning about
root-cause commonality/fix-difficulty/expected-gain -- every time the
current unit is resolved or patience-exhausted, instead of walking one list
fixed at seed time like check-mode's Curriculum does.

Ported/adapted from experiment_harness_staged.py's choose_unit()/
describe_unit()/UNITS. Deliberately NOT "100% code, never LLM judgment"
like curriculum.py's own Curriculum -- unit SELECTION intrinsically needs
LLM reasoning to compare candidates (see choose_unit()'s own docstring for
why raw failure count alone is a bad proxy for which unit is worth
targeting). The mechanical ``max(cands)`` heuristic is kept ONLY as an
automatic fallback -- no unit_selector configured, its call fails or
returns an invalid answer, or there is only one candidate -- exactly
matching experiment_harness_staged.py's own fallback discipline.

Implements the same duck-typed interface as meta_agent/curriculum.py's
Curriculum (done / current_goal / failure_rate_for / advance_if_ready /
record_expand / directive / snapshot), so
HGMManager._curriculum_directive_for_expand needs no per-granularity
branching beyond which class it constructs. A SEPARATE class rather than a
branch inside Curriculum itself: unit-mode must re-decide its current goal
every time it advances (using the live check-count rollup at that moment),
a structurally different control flow from check-mode's list fixed at
construction time.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Optional

from .curriculum import CurriculumSnapshot, scored_evals


def load_units_map(path_str: str) -> dict[str, tuple[str, ...]]:
    """Load a project's units.json (see e.g.
    projects/travel_mas_refactored/adapter/units.json) -- a flat JSON
    object mapping a unit name to a list of check-label PREFIXES, exactly
    experiment_harness_staged.py's UNITS dict shape (now data, not
    framework code, so any project can supply its own without touching
    this module).

    Unlike curriculum.py's decorative description-path knobs (missing ->
    silent fallback), this is the PRIMARY input unit-mode needs to
    function at all -- raises ValueError (never a silent {}) on a missing/
    unreadable/malformed file, an empty/non-dict body, or any entry whose
    value isn't a non-empty list of strings, so a misconfigured
    curriculum_units_path fails loudly at construction time, not with a
    silently-empty curriculum much later."""
    path = Path(path_str)
    if not path.is_absolute():
        path = Path.cwd() / path
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"could not read units file {path_str!r}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"units file {path_str!r} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict) or not data:
        raise ValueError(f"units file {path_str!r} must be a non-empty JSON object")
    units: dict[str, tuple[str, ...]] = {}
    for name, prefixes in data.items():
        if name == "_comment":
            continue
        if (
            not isinstance(name, str)
            or not isinstance(prefixes, list)
            or not prefixes
            or not all(isinstance(p, str) and p for p in prefixes)
        ):
            raise ValueError(
                f"units file {path_str!r}: entry {name!r} must map to a non-empty "
                "list of non-empty prefix strings"
            )
        units[name] = tuple(prefixes)
    if not units:
        raise ValueError(f"units file {path_str!r} has no usable entries (besides _comment)")
    return units


def rollup_unit_counts(
    check_counts: dict[str, int], units: dict[str, tuple[str, ...]],
) -> dict[str, int]:
    """unit -> sum of every check_counts entry whose name starts with any
    of that unit's prefixes. A check matching no unit's prefix is dropped
    (never double-counted into an invented bucket); a check matching more
    than one unit's prefix (shouldn't happen for a well-formed units map,
    but not enforced here) is counted toward every matching unit."""
    out: dict[str, int] = dict.fromkeys(units, 0)
    for check, count in check_counts.items():
        for unit, prefixes in units.items():
            if any(check.startswith(p) for p in prefixes):
                out[unit] += count
    return out


def unit_failure_rate_for(
    unit: Optional[str], project_metrics: dict, n_evals: int,
    units: dict[str, tuple[str, ...]],
) -> Optional[float]:
    """A unit's live failure rate, or ``None`` when ``n_evals <= 0``
    (nothing evaluated yet). Same scored_evals-corrected denominator as
    Curriculum.failure_rate_for (see curriculum.py::scored_evals); the
    numerator sums every top_failed_checks/harness_checks occurrence count
    across all checks matching ``unit``'s prefixes, via rollup_unit_counts.

    NOTE (approximation, intentional and safety-conservative): this
    OVER-COUNTS relative to "fraction of cases where >=1 check in this
    unit failed" whenever a single case fails more than one check in the
    same unit -- HGM's aggregate project_metrics only exposes check-level
    occurrence counts, not per-case check sets, so there is no way to
    de-duplicate at this layer (unlike experiment_harness_staged.py's own
    unit_score(), which has real per-case recs to work from). The
    direction of the bias is safety-conservative: an inflated failure rate
    only makes advance_if_ready SLOWER to declare a unit resolved, never
    falsely resolves it early -- the risk is capped by ``patience`` either
    way."""
    if unit is None or n_evals is None or n_evals <= 0:
        return None
    from .curriculum import _combined_check_counts

    unit_counts = rollup_unit_counts(_combined_check_counts(project_metrics), units)
    se = scored_evals(project_metrics, n_evals)
    if se <= 0:
        return None
    return unit_counts.get(unit, 0) / se


def describe_unit_from_counts(
    unit: str, check_counts: dict[str, int],
    units: dict[str, tuple[str, ...]], check_descriptions: dict[str, str],
) -> str:
    """Per-unit failure summary from AGGREGATE counts (HGM's
    project_metrics has no per-case recs, unlike experiment_harness_staged.
    py's describe_unit(), which can show a failing/applicable ratio per
    check) -- lists each check belonging to this unit with its raw
    occurrence count and description, sorted by count descending."""
    prefixes = units[unit]
    members = [
        (check, count) for check, count in check_counts.items()
        if any(check.startswith(p) for p in prefixes) and count > 0
    ]
    members.sort(key=lambda kv: (-kv[1], kv[0]))
    lines = ["Checks in this unit, with how often each occurred in the CURRENT best node's evaluation:"]
    for check, count in members:
        desc = check_descriptions.get(check, "")
        lines.append(f"- {check}: {count} -- {desc[:220]}")
    return "\n".join(lines)


def derive_unit_descriptions(
    units: dict[str, tuple[str, ...]], check_descriptions: dict[str, str],
) -> dict[str, str]:
    """unit -> one paragraph summarizing every member check's own
    description (joined) -- used by UnitCurriculum's own directive() to
    show something more useful than a bare unit name when the project's
    check-description files are loaded. A unit with no described member
    checks maps to "" (falls back to directive()'s generic phrasing, same
    convention as Curriculum's own check_descriptions)."""
    out: dict[str, str] = {}
    for unit, prefixes in units.items():
        parts = [
            desc for check, desc in sorted(check_descriptions.items())
            if any(check.startswith(p) for p in prefixes) and desc
        ]
        if parts:
            out[unit] = " ".join(parts)
    return out


def choose_unit(
    unit_counts: dict[str, int], attempts: dict[str, int], max_attempts: int,
    *, units: dict[str, tuple[str, ...]], check_descriptions: dict[str, str],
    chooser: Optional[Callable[[dict, dict], Optional[str]]] = None,
) -> Optional[str]:
    """Pick the next unit to target. ``chooser`` (when given -- an injected
    callable wrapping UnitSelector.choose, never an import of
    platform_core.llm_wrapper.call_llm directly, matching this codebase's
    llm_caller-injection convention) is tried first: ANY exception, an
    answer not among the candidates, or ``chooser is None`` falls back to
    the mechanical ``max(cands)`` heuristic (most occurrences) -- exactly
    experiment_harness_staged.py's own fallback discipline (never let a
    selection failure block the curriculum). Returns ``None`` when no
    candidate has count>0 and attempts<max_attempts (nothing left to
    target -- the curriculum is ``done``)."""
    cands = [
        (n, u) for u, n in unit_counts.items()
        if n > 0 and attempts.get(u, 0) < max_attempts
    ]
    if not cands:
        return None
    fallback = max(cands)[1]
    if len(cands) == 1 or chooser is None:
        return fallback
    try:
        chosen = chooser(
            {u: n for n, u in cands},
            {
                u: describe_unit_from_counts(u, unit_counts, units, check_descriptions)
                for _, u in cands
            },
        )
        if chosen in {u for _, u in cands}:
            return chosen
    except Exception:  # noqa: BLE001 -- a selection failure must never block the curriculum
        pass
    return fallback


class UnitCurriculum:
    """Unit-level counterpart to curriculum.py's Curriculum -- same public
    surface (done/current_goal/failure_rate_for/advance_if_ready/
    record_expand/directive/snapshot), re-picking its current unit from
    the live check-count rollup every time it advances, instead of walking
    a list fixed at construction."""

    def __init__(
        self, seed_check_counts: dict[str, int], *,
        units: dict[str, tuple[str, ...]], max_attempts: int,
        resolution_threshold: float = 0.15, patience: int = 5,
        check_descriptions: Optional[dict[str, str]] = None,
        unit_selector: Any = None,
    ) -> None:
        self._units = dict(units)
        self._max_attempts = max_attempts
        self.resolution_threshold = resolution_threshold
        self.patience = patience
        self._check_descriptions: dict[str, str] = check_descriptions or {}
        self._unit_descriptions: dict[str, str] = derive_unit_descriptions(
            self._units, self._check_descriptions,
        )
        self._unit_selector = unit_selector
        self._attempts: dict[str, int] = {}
        self._resolved: list[str] = []
        self._rounds_on_current = 0
        self._current: Optional[str] = None
        self._done = False
        self._advance_to_next(seed_check_counts)

    @classmethod
    def restored(
        cls, snap: dict, *, units: dict[str, tuple[str, ...]], max_attempts: int,
        resolution_threshold: float = 0.15, patience: int = 5,
        check_descriptions: Optional[dict[str, str]] = None,
        unit_selector: Any = None,
    ) -> "UnitCurriculum":
        """Continue from a persisted ``curriculum_status.json`` (a resumed
        run) without re-choosing the current unit -- which may call the
        unit selector's LLM. ``snapshot()`` stores the attempts per unit
        under ``seed_counts``."""
        self = cls.__new__(cls)
        self._units = dict(units)
        self._max_attempts = max_attempts
        self.resolution_threshold = resolution_threshold
        self.patience = patience
        self._check_descriptions = check_descriptions or {}
        self._unit_descriptions = derive_unit_descriptions(self._units, self._check_descriptions)
        self._unit_selector = unit_selector
        self._attempts = {str(k): int(v) for k, v in (snap.get("seed_counts") or {}).items()}
        self._resolved = [str(g) for g in snap.get("resolved_goals", [])]
        self._rounds_on_current = int(snap.get("rounds_on_current", 0))
        self._current = snap.get("current_goal")
        self._done = bool(snap.get("done", self._current is None))
        return self

    def _chooser(self, unit_counts: dict, descriptions: dict) -> Optional[str]:
        # Passes the already-rendered per-candidate descriptions straight through -- UnitSelector.choose never
        # needs to recompute describe_unit_from_counts itself, it just renders a prompt from what it's given.
        return self._unit_selector.choose(unit_counts, descriptions, self._attempts, self._max_attempts)

    def _advance_to_next(self, check_counts: dict[str, int]) -> None:
        unit_counts = rollup_unit_counts(check_counts, self._units)
        chosen = choose_unit(
            unit_counts, self._attempts, self._max_attempts,
            units=self._units, check_descriptions=self._check_descriptions,
            chooser=self._chooser if self._unit_selector is not None else None,
        )
        self._current = chosen
        self._done = chosen is None

    @property
    def done(self) -> bool:
        return self._done

    @property
    def current_goal(self) -> Optional[str]:
        return self._current

    def failure_rate_for(
        self, goal: Optional[str], project_metrics: dict, n_evals: int,
    ) -> Optional[float]:
        return unit_failure_rate_for(goal, project_metrics, n_evals, self._units)

    def record_expand(self) -> None:
        if not self.done:
            self._rounds_on_current += 1

    def advance_if_ready(
        self, *, failure_rate: Optional[float],
        check_counts: Optional[dict[str, int]] = None,
    ) -> Optional[str]:
        if self.done:
            return None
        resolved = failure_rate is not None and failure_rate <= self.resolution_threshold
        patience_exhausted = self._rounds_on_current >= self.patience
        if not resolved and not patience_exhausted:
            return None
        reason = "resolved" if resolved else "patience_exhausted"
        if self._current is not None:
            self._attempts[self._current] = self._attempts.get(self._current, 0) + 1
            self._resolved.append(self._current)
        self._rounds_on_current = 0
        self._advance_to_next(check_counts or {})
        return reason

    def directive(self) -> Optional[str]:
        """Unit-flavored counterpart to Curriculum.directive() -- same
        ESCAPE HATCH pattern, but names the unit's member checks (from
        describe_unit_from_counts-derived text) instead of quoting a bare
        check identifier, and reports attempts-so-far (unit-mode's
        analogue of "sub-goal N of M") since units, unlike checks, can be
        revisited (choose_unit may re-propose a unit that previously
        exhausted patience if it still has failures and attempts remain)."""
        if self.done:
            return None
        unit = self._current
        description = self._unit_descriptions.get(unit)
        what_this_is = (
            f"`{unit}` -- {description}"
            if description
            else (
                f"`{unit}` -- a group of related failing checks/constraints "
                "(see the member-check breakdown you were given for this "
                "unit's selection) -- treat the unit as a whole as this "
                "EXPAND's one target."
            )
        )
        return (
            f"Focus this EXPAND's diagnosis on resolving this unit (a group of "
            f"related failing checks): {what_this_is}\n\n"
            f"This is attempt {self._attempts.get(unit, 0) + 1} at this unit "
            f"(max {self._max_attempts}) -- we need this unit's checks' "
            "combined occurrence rate to go down, not the composite score in "
            "general.\n\n"
            f"Diagnose and propose a fix specifically targeting `{unit}` "
            "within the block you've been assigned, grounded in concrete "
            "evidence from the feedback/failure summary shown to you -- do "
            "not propose a generic or unrelated improvement instead.\n\n"
            "ESCAPE HATCH: if the block you've been assigned genuinely "
            f"cannot address `{unit}` (the real fix belongs in a different "
            "block entirely), say so explicitly and instead diagnose and "
            "propose the best genuine improvement for THIS block -- never "
            "force an irrelevant or superficial change just to appear to "
            "address the curriculum focus. The curriculum stays on this "
            "same unit for the next EXPAND regardless of which block ends "
            "up fixing it."
        )

    def snapshot(
        self, *, current_failure_rate: Optional[float],
        advance_reason: Optional[str] = None,
    ) -> CurriculumSnapshot:
        # Reuses curriculum.CurriculumSnapshot verbatim, with a few fields' semantics remapped for the
        # dashboard/debug consumer (curriculum_status.json) since unit-mode has no fixed goal ORDER the way
        # check-mode does: `goals` is every known unit name (not yet-visited units are simply absent from
        # `resolved_goals`); `current_index` is a monotonic count of units resolved-or-exhausted so far, not a
        # position in `goals`; `seed_counts` here means "attempts spent per unit so far" (unit-mode has no
        # single seed-time occurrence count per unit the way a check does -- units are re-scored every advance).
        return CurriculumSnapshot(
            goals=sorted(self._units),
            seed_counts=dict(self._attempts),
            current_index=len(self._resolved),
            current_goal=self.current_goal,
            rounds_on_current=self._rounds_on_current,
            patience=self.patience,
            resolution_threshold=self.resolution_threshold,
            current_failure_rate=current_failure_rate,
            resolved_goals=list(self._resolved),
            advance_reason=advance_reason,
            done=self.done,
        )
