"""Shopping-specific error categorizer for the HGM dual-optimization manager.

Consumes the per-case ``CaseResult.details`` emitted by
``projects/shopping/benchmark/scorer.py`` and groups failures into prioritised
categories the manager uses to spawn error-conditioned Stage B variants.
Mirrors the contract of ``projects/travel/travel_error_categorizer.py``:

    categorize_errors(per_case, *, max_representative_errors=5) -> list[dict]
    per_case_category_checks(case_details, category_id) -> list[bool]

The function is pure (no I/O, no LLM). The scorer already did the GT-aware
work — it reads ``validation_cases.json`` and emits:

  * ``missing_feature_categories`` — ``{category: {sub_queries, fields,
    predicates, product_ids}}`` for each *missing* gold product, i.e. which
    *kind* of requirement the cart failed (brand / color / price / rating / …).
  * ``gold_feature_categories`` — ``{category: [all gold pids needing it]}``,
    the per-category trial set for ``category_significance`` selection.
  * coupon diffs (``missing_coupons`` / ``extra_coupons`` / ``coupon_details``)
    and ``extra_products`` / ``matched_products``.

This module only reshapes those into categories — it never re-reads the data.
Categories are returned sorted by ``failure_rate`` descending so the manager's
``num_variants`` cap picks the highest-impact categories first.

Category families (``category_type`` feeds ``category_selection:
balanced_by_type``):

  * ``missing_product__<cat>`` (type ``missing_feature``) — a requirement of
    kind <cat> is unmet by the cart.
  * ``missing_coupon`` / ``extra_coupon`` / ``wrong_coupon_quantity``
    (type ``coupon``).
  * ``generic__no_validation`` / ``generic__runtime_error`` (type ``generic``).

Cart precision (``extra_products``) is deliberately NOT categorized: the
``extra_product`` category is suppressed so it never enters the category list
or variant selection. The scorer still emits ``details["extra_products"]`` for
diagnostics; this module just doesn't turn it into a category.

NOTE: the scorer's ``details`` carry GT-derived specifics. That is by design —
this categorizer's output is the *error report* (the only sanctioned channel by
which the meta-agent may see GT). The task agent never sees any of this.
"""
from __future__ import annotations

import re
from typing import Any, Iterable

from meta_agent.models import CaseResult


_MISSING_PREFIX = "missing_product__"
_NO_VALIDATION_MSG = "validation_cases.json not found"

# Bucket priority for HGMDualManager's "balanced_by_type" category selection
# (resolved by convention from this module). Product-requirement misses first,
# then coupon handling, then cart precision (extra products); the shared
# "generic" (crash / missing-data) bucket last.
category_type_priority = ["missing_feature", "coupon", "cart", "generic"]


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", (name or "").lower()).strip("_") or "unnamed"


def _preds_str(preds: list[dict[str, Any]]) -> str:
    """Compact ``field op value`` summary of a category's predicates. Drops the
    field's dotted prefix and skips predicates with no operator."""
    parts: list[str] = []
    for p in preds or []:
        if not isinstance(p, dict):
            continue
        field = str(p.get("field") or "").split(".")[-1]
        if not field:
            continue
        op = p.get("operator")
        parts.append(f"{field} {op} {p.get('operator_value')}" if op else field)
    return ", ".join(parts[:8])


def categorize_errors(
    per_case: Iterable[CaseResult],
    *,
    max_representative_errors: int = 5,
) -> list[dict[str, Any]]:
    """Group failures across ``per_case`` into prioritised categories.

    ``max_representative_errors`` caps the example errors attached to each
    category for the prompt; every case is still walked for failure counts.
    """
    cases = list(per_case)
    total = len(cases)
    if total == 0:
        return []

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
        level = details.get("level")
        lvl_tag = f"[L{level}] " if level is not None else ""

        # (A) Runtime/crash synthetic category — error set, details empty.
        if case.error and not details:
            grp = _ensure(
                "generic__runtime_error",
                "generic",
                "Runtime / subprocess crash",
                "runtime_error",
            )
            _add_example(grp, sample_id, ["runtime_error"], [str(case.error)[:300]])
            continue

        # (B) No ground truth for this case → nothing else is scoreable.
        err_text = str(details.get("error") or "")
        if err_text.startswith(_NO_VALIDATION_MSG):
            grp = _ensure(
                "generic__no_validation",
                "generic",
                "No ground truth (validation missing)",
                "no_validation",
            )
            _add_example(grp, sample_id, ["no_validation"], [err_text[:300]])
            continue

        # (C) Missing-product feature categories (rich view from the scorer).
        mfc = details.get("missing_feature_categories") or {}
        if isinstance(mfc, dict) and mfc:
            for cat, slot in mfc.items():
                if not isinstance(slot, dict):
                    continue
                cat_slug = _slug(str(cat))
                grp = _ensure(
                    f"{_MISSING_PREFIX}{cat_slug}",
                    "missing_feature",
                    f"Missing-product requirement: {cat}",
                    cat_slug,
                )
                fields = [str(f) for f in (slot.get("fields") or [])] or [str(cat)]
                n_pids = len(slot.get("product_ids") or [])
                preds_msg = _preds_str(slot.get("predicates") or [])
                subqs = slot.get("sub_queries") or []
                msg = f"{lvl_tag}{n_pids} gold product(s) unmet for [{cat}]"
                if preds_msg:
                    msg += f"; requires {preds_msg}"
                if subqs:
                    msg += f'; sub-query: "{str(subqs[0])[:140]}"'
                _add_example(grp, sample_id, fields, [msg])
        elif details.get("missing_products"):
            # Fallback when feature categorization is unavailable (legacy
            # details / no meta_info): one coarse bucket.
            grp = _ensure(
                f"{_MISSING_PREFIX}unknown",
                "missing_feature",
                "Missing products (uncategorized)",
                "unknown",
            )
            miss = details.get("missing_products") or []
            _add_example(
                grp,
                sample_id,
                ["missing_product"],
                [f"{lvl_tag}{len(miss)} expected product(s) missing from cart"],
            )

        # (D) Coupon categories.
        # NOTE: extra products in the cart are intentionally NOT emitted as an
        # error category — ``extra_product`` must never appear in the category
        # list (it dominated variant selection without improving scores). The
        # scorer still records ``details["extra_products"]`` for diagnostics.
        missing_cp = details.get("missing_coupons") or []
        if missing_cp:
            grp = _ensure("missing_coupon", "coupon", "Missing coupons", "missing_coupon")
            _add_example(
                grp,
                sample_id,
                ["missing_coupon"],
                [f"{lvl_tag}expected coupons not applied/correct: {list(missing_cp)[:5]}"],
            )
        extra_cp = details.get("extra_coupons") or []
        if extra_cp:
            grp = _ensure("extra_coupon", "coupon", "Extra coupons", "extra_coupon")
            _add_example(
                grp,
                sample_id,
                ["extra_coupon"],
                [f"{lvl_tag}unexpected coupons applied: {list(extra_cp)[:5]}"],
            )
        wrong_qty = [
            c
            for c in (details.get("coupon_details") or [])
            if isinstance(c, dict)
            and not c.get("match")
            and int(c.get("expected_quantity") or 0) > 0
        ]
        if wrong_qty:
            grp = _ensure(
                "wrong_coupon_quantity",
                "coupon",
                "Wrong coupon quantity",
                "wrong_coupon_quantity",
            )
            msgs = [
                f"{c.get('coupon_name')}: applied {c.get('quantity')} "
                f"expected {c.get('expected_quantity')}"
                for c in wrong_qty[:5]
            ]
            _add_example(
                grp, sample_id, ["wrong_coupon_quantity"], [lvl_tag + "; ".join(msgs)]
            )

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
    """Per-category pass/fail trials for one case (used only by
    ``select_metric == 'category_significance'``). Empty list ⇒ not applicable.

    The manager pools these across cases and bootstraps Stage A vs each variant
    on its category, using Stage A's per-case trial count as the shared
    denominator (padding the variant to match). For that comparison to be valid
    the trial count must be fixed by the GROUND TRUTH / the case, NOT by the
    agent's output — otherwise a variant that changes how many items it emits is
    mis-compared. Two shapes:

    - RECALL categories — denominator = number of gold things of that kind:
      * ``missing_product__<cat>``: one bool per gold product whose requirement
        involves <cat> — ``True`` iff matched (from ``gold_feature_categories``
        + ``matched_products``).
      * ``missing_coupon``: one bool per expected coupon — ``True`` for matched,
        ``False`` for each unmatched.
    - PRECISION / absence categories — there is no GT count of things that
      should be *absent*, so each is a SINGLE per-case "is this case clean of
      the error?" trial (denominator = 1 per case, every case applicable). A
      variant that removes the error raises the cleanliness rate, which the
      bootstrap can detect — a plain ``[False] * count`` could not, because the
      count is agent-determined and both sides would show 0 passes:
      * ``extra_coupon``: clean ⇔ no extra coupons applied.
      * ``wrong_coupon_quantity``: clean ⇔ no applied coupon with a wrong qty.

    (``extra_product`` is intentionally not a category here — see the module
    docstring — so it has no per-case trial shape.)
    - ``generic__*``: ``[]`` (no per-case mapping).
    """
    if not case_details or not category_id:
        return []

    if category_id.startswith(_MISSING_PREFIX):
        wanted = category_id[len(_MISSING_PREFIX):]
        gold_cats = case_details.get("gold_feature_categories") or {}
        matched = set(case_details.get("matched_products") or [])
        pids: list[Any] = []
        if isinstance(gold_cats, dict):
            for cat, cat_pids in gold_cats.items():
                if _slug(str(cat)) == wanted:
                    pids = list(cat_pids or [])
                    break
        return [pid in matched for pid in pids]

    if category_id == "extra_coupon":
        return [not bool(case_details.get("extra_coupons"))]

    if category_id == "missing_coupon":
        n_matched = sum(
            1
            for c in (case_details.get("coupon_details") or [])
            if isinstance(c, dict) and c.get("match")
        )
        n_missing = len(case_details.get("missing_coupons") or [])
        return [True] * n_matched + [False] * n_missing

    if category_id == "wrong_coupon_quantity":
        # Precision/absence → one per-case "clean?" trial: clean ⇔ no applied
        # coupon has a wrong quantity (count is agent-determined, so per-item
        # trials would be invisible to the bootstrap — see docstring).
        has_wrong = any(
            isinstance(c, dict)
            and not c.get("match")
            and int(c.get("expected_quantity") or 0) > 0
            for c in (case_details.get("coupon_details") or [])
        )
        return [not has_wrong]

    return []   # generic__* not testable per-case
