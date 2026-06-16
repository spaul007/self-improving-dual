"""Travel-specific error categorizer for the HGM dual-optimization manager.

Consumes the per-case ``CaseResult.details`` shapes emitted by
``projects/travel/benchmark/scorer.py`` and groups failures into categories
the manager can use to spawn error-conditioned Stage B variants.

The function is intentionally pure (no I/O, no LLM) and travel-aware:

1. Commonsense grouping reads ``details["dimension_details"]``
   (``{dim_name: {checks: [{name, passed, message}], ...}}``) — see
   ``scorer.py:155-158``.
2. Hard-constraint grouping reads ``details["hard_constraints"]``
   (``{constraint_name: {passed, message}}``) — see ``scorer.py:159-161``.
   Constraints are grouped by the prefix in their name (flight, train,
   hotel, restaurant, attraction, budget).
3. Plan-conversion-failed and runtime/crash cases are folded into
   synthetic ``generic__*`` categories so the editor still gets steering.
4. As a last resort, ``details["failed_checks"]`` (the flat
   ``"commonsense:DIM:CHECK"`` / ``"hard:CONSTRAINT"`` list) is mined.

Categories are returned sorted by ``failure_rate`` descending — highest-
impact categories first, so the manager's ``num_variants`` cap picks them
preferentially.
"""
from __future__ import annotations

import re
from typing import Any, Iterable

from meta_agent.models import CaseResult


_HARD_PREFIXES = (
    "flight",
    "train",
    "hotel",
    "restaurant",
    "attraction",
    "budget",
)

# Bucket priority for HGMDualManager's "balanced_by_type" category selection
# (round-robin across category_type buckets; resolved by convention from this
# module). "generic" (synthetic crash / plan-failed) is last.
category_type_priority = ["commonsense", "hard_constraint", "generic"]


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "unnamed"


def _hard_prefix(constraint_name: str) -> str:
    head = constraint_name.split("_", 1)[0].lower()
    return head if head in _HARD_PREFIXES else "other"


def _failed_check_prefix(check_token: str) -> tuple[str, str] | None:
    """Parse a ``"commonsense:DIM:CHECK"`` / ``"hard:CONSTRAINT"`` token.

    Returns ``(category_type, key)`` where ``category_type`` is
    ``"commonsense"`` or ``"hard_constraint"`` and ``key`` is the
    dimension name (commonsense) or constraint prefix (hard).
    """
    parts = check_token.split(":")
    if len(parts) >= 3 and parts[0] == "commonsense":
        return "commonsense", parts[1]
    if len(parts) >= 2 and parts[0] == "hard":
        return "hard_constraint", _hard_prefix(parts[1])
    return None


def categorize_errors(
    per_case: Iterable[CaseResult],
    *,
    max_representative_errors: int = 5,
) -> list[dict[str, Any]]:
    """Group failures across ``per_case`` into prioritised categories.

    ``max_representative_errors`` only caps the example errors attached to
    each category for the prompt — every case in ``per_case`` is walked
    when counting failures and computing ``failure_rate``.
    """
    cases = list(per_case)
    total = len(cases)
    if total == 0:
        return []

    # category_id -> {category_type, category_name, slug,
    #                 failing_sample_ids: set[str],
    #                 examples: list[{sample_id, checks_failed, messages}]}
    groups: dict[str, dict[str, Any]] = {}

    def _ensure(cat_id: str, cat_type: str, cat_name: str, slug: str) -> dict[str, Any]:
        g = groups.get(cat_id)
        if g is None:
            g = {
                "category_id": cat_id,
                "category_type": cat_type,
                "category_name": cat_name,
                "slug": slug,
                "failing_sample_ids": set(),
                "examples": [],
            }
            groups[cat_id] = g
        return g

    def _add_example(
        group: dict[str, Any],
        sample_id: str,
        checks_failed: list[str],
        messages: list[str],
    ) -> None:
        group["failing_sample_ids"].add(sample_id)
        if len(group["examples"]) < max_representative_errors:
            group["examples"].append(
                {
                    "sample_id": sample_id,
                    "checks_failed": checks_failed,
                    "messages": messages,
                }
            )

    for case in cases:
        sample_id = str(case.case_id)
        details: dict[str, Any] = dict(case.details or {})

        # (4) Runtime/crash synthetic category — CaseResult.error set and
        # details is empty. Take precedence over (3) so a crash with a
        # stub _zero_result doesn't get swallowed by plan-failed.
        if case.error and not details:
            grp = _ensure(
                "generic__runtime_error",
                "generic",
                "Runtime / subprocess crash",
                "runtime_error",
            )
            _add_example(grp, sample_id, ["runtime_error"], [str(case.error)[:300]])
            continue

        # (1) Commonsense grouping.
        had_commonsense_failure = False
        dim_details = details.get("dimension_details") or {}
        if isinstance(dim_details, dict):
            for dim_name, dim in dim_details.items():
                if not isinstance(dim, dict):
                    continue
                checks = dim.get("checks") or []
                failed_in_dim: list[str] = []
                messages_in_dim: list[str] = []
                for check in checks:
                    if not isinstance(check, dict):
                        continue
                    if check.get("passed"):
                        continue
                    failed_in_dim.append(str(check.get("name") or "?"))
                    msg = check.get("message")
                    if msg:
                        messages_in_dim.append(str(msg)[:300])
                if failed_in_dim:
                    had_commonsense_failure = True
                    slug = _slug(str(dim_name))
                    grp = _ensure(
                        f"commonsense__{slug}",
                        "commonsense",
                        str(dim_name),
                        slug,
                    )
                    _add_example(grp, sample_id, failed_in_dim, messages_in_dim)

        # (2) Hard-constraint grouping.
        had_hard_failure = False
        hard = details.get("hard_constraints") or {}
        if isinstance(hard, dict):
            # Aggregate hard failures within this case by prefix so each
            # case contributes one example per prefix, not one per checked
            # constraint.
            by_prefix: dict[str, dict[str, list[str]]] = {}
            for cname, cinfo in hard.items():
                if not isinstance(cinfo, dict) or cinfo.get("passed"):
                    continue
                prefix = _hard_prefix(str(cname))
                slot = by_prefix.setdefault(prefix, {"checks": [], "messages": []})
                slot["checks"].append(str(cname))
                msg = cinfo.get("message")
                if msg:
                    slot["messages"].append(str(msg)[:300])
            for prefix, slot in by_prefix.items():
                had_hard_failure = True
                grp = _ensure(
                    f"hard_constraint__{prefix}",
                    "hard_constraint",
                    f"Hard constraints ({prefix})",
                    prefix,
                )
                _add_example(grp, sample_id, slot["checks"], slot["messages"])

        if had_commonsense_failure or had_hard_failure:
            continue

        # (3) Plan-conversion-failed synthetic category — details has the
        # _zero_result shape (error set, composite_score == 0, no
        # dimension_details / hard_constraints).
        err_text = str(details.get("error") or "")
        if err_text:
            grp = _ensure(
                "generic__plan_conversion_failed",
                "generic",
                "Plan conversion failed",
                "plan_conversion_failed",
            )
            _add_example(grp, sample_id, ["plan_conversion_failed"], [err_text[:300]])
            continue

        # (5) Last-resort fallback — flat failed_checks list.
        failed = details.get("failed_checks") or []
        if isinstance(failed, list) and failed:
            # Re-group the flat list by (type, key) within this case.
            fallback_by_cat: dict[tuple[str, str], list[str]] = {}
            for token in failed:
                if not isinstance(token, str):
                    continue
                parsed = _failed_check_prefix(token)
                if parsed is None:
                    continue
                fallback_by_cat.setdefault(parsed, []).append(token)
            for (cat_type, key), tokens in fallback_by_cat.items():
                slug = _slug(key)
                if cat_type == "commonsense":
                    cat_id = f"commonsense__{slug}"
                    cat_name = key
                else:
                    cat_id = f"hard_constraint__{slug}"
                    cat_name = f"Hard constraints ({slug})"
                grp = _ensure(cat_id, cat_type, cat_name, slug)
                _add_example(grp, sample_id, tokens, [])

    # Materialize, compute failure_rate, sort.
    out: list[dict[str, Any]] = []
    for cat_id, g in groups.items():
        num_failing = len(g["failing_sample_ids"])
        out.append(
            {
                "category_id": cat_id,
                "category_type": g["category_type"],
                "category_name": g["category_name"],
                "slug": g["slug"],
                "num_failing_samples": num_failing,
                "total_samples": total,
                "failure_rate": num_failing / total,
                "representative_errors": g["examples"],
            }
        )
    out.sort(key=lambda c: (-c["failure_rate"], c["category_id"]))
    return out


def per_case_category_checks(case_details: dict, category_id: str) -> list[bool]:
    """Return one bool per individual check/constraint instance in this case
    that belongs to ``category_id``. Empty list ⇒ case not applicable.

    Mirrors the ``category_id`` scheme produced by :func:`categorize_errors`:

    - ``commonsense__<slug>`` → one bool per check in the matching dimension's
      ``checks`` list.
    - ``hard_constraint__<prefix>`` → one bool per hard constraint whose name
      maps to ``<prefix>`` (see :func:`_hard_prefix`).
    - ``generic__*`` → ``[]`` (no per-case mapping).

    Resolved by convention (same module as ``categorize_errors``) and consumed
    by ``HGMDualManager`` when ``select_metric == 'category_significance'``.
    Previously lived in ``meta_agent/managers/hgm_dual.py``; moved here so the
    manager carries no travel-schema knowledge.
    """
    if not case_details or not category_id:
        return []
    if category_id.startswith("commonsense__"):
        wanted = category_id[len("commonsense__"):]
        dims = case_details.get("dimension_details") or {}
        if isinstance(dims, dict):
            for dim_name, dim in dims.items():
                if not isinstance(dim, dict):
                    continue
                if _slug(str(dim_name)) == wanted:
                    return [
                        bool(c.get("passed", False))
                        for c in (dim.get("checks") or [])
                        if isinstance(c, dict) and "passed" in c
                    ]
        return []
    if category_id.startswith("hard_constraint__"):
        wanted_prefix = category_id[len("hard_constraint__"):]
        out: list[bool] = []
        hard = case_details.get("hard_constraints") or {}
        if isinstance(hard, dict):
            for cname, cinfo in hard.items():
                if not isinstance(cinfo, dict):
                    continue
                if _hard_prefix(str(cname)) == wanted_prefix:
                    out.append(bool(cinfo.get("passed", False)))
        return out
    return []   # generic__* not testable per-case
