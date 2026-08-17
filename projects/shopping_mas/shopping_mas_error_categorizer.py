"""shopping_mas error categorizer for the HGM dual-optimization manager.

Copied verbatim from projects/shopping/shopping_error_categorizer.py (the
sibling single-agent project) -- pure function of the scorer's details
shape, zero logic changes needed since adapter/scorer_impl.py is itself a
verbatim copy of that project's scorer, emitting the byte-identical
failure_causes/budget_check/coupon_ownership/final_price_check shape this
module already knows how to read.

Consumes the per-case ``CaseResult.details`` emitted by
``projects/shopping_mas/adapter/scorer_impl.py`` and groups failures into a small,
fixed set of **clubbed** categories the manager uses to spawn error-conditioned
Stage B variants. Contract (mirrors travel):

    categorize_errors(per_case, *, max_representative_errors=5) -> list[dict]
    per_case_category_checks(case_details, category_id) -> list[bool]

Pure (no I/O, no LLM). The scorer already did the GT-aware work and emits
``details["failure_causes"]`` (each missing gold product classified into exactly
one cause), ``budget_check``, coupon diffs / ``coupon_ownership``, and
``final_price_check``. This module reshapes those into **6 clubbed categories
(+ ``generic``)** — ``category_id == category_type`` (coarse); the granular
detail lives in the one-sentence example MESSAGE, not the id:

  * ``missing_feature``  — a stated query feature is unmet by the cart's closest
    substitute (message names the feature kind).
  * ``user_info``        — the substitute ignores a profile-derived constraint
    (gender / size) the agent should have inferred (message names which).
  * ``not_cheapest``     — the substitute is a pricier valid product than the
    cheaper target.
  * ``missing_product``  — a required product is absent with no substitute tried.
  * ``budget_constraint_mismatch`` — cart total over/under the stated budget.
  * ``suboptimal_coupon`` — any coupon problem (missing / extra / wrong qty /
    not owned / invalid same-brand scope / under-discounted).
  * ``generic``          — runtime crash / missing ground truth.

Deliberately NOT emitted as categories: ``ambiguous`` (the unactionable residual
where a pick met every checkable constraint yet wasn't the gold item — kept in
``details`` only) and ``final_price_gap`` (a symptom that always co-occurs with a
product/coupon cause — kept as a ``details`` number).

NOTE: the scorer's ``details`` carry GT-derived specifics. That is by design —
this categorizer's output is the *error report* (the only sanctioned channel by
which the meta-agent may see GT). The task agent never sees any of this.
"""
from __future__ import annotations

from typing import Any, Iterable

from meta_agent.models import CaseResult


_NO_VALIDATION_MSG = "validation_cases.json not found"

# Bucket priority for HGMDualManager's "balanced_by_type" category selection
# (resolved by convention from this module). category_id == category_type here.
category_type_priority = [
    "missing_feature",
    "user_info",
    "not_cheapest",
    "not_cheapest_cart_level",
    "missing_product",
    "budget_constraint_mismatch",
    "suboptimal_coupon",
    "generic",
]

_CATEGORY_NAMES = {
    "missing_feature": "Unmet product feature",
    "user_info": "Ignored user-profile constraint",
    "not_cheapest": "Not the cheapest valid product (per item)",
    "not_cheapest_cart_level": "Cart not the cheapest valid combination (L2)",
    "missing_product": "Required product missing (no substitute)",
    "budget_constraint_mismatch": "Budget constraint not met",
    "suboptimal_coupon": "Suboptimal coupon usage",
    "generic": "Runtime / missing-data error",
}

# Feature category -> readable noun for the {feature} slot in the template.
_FEATURE_WORD = {
    "delivery_time": "delivery time",
    "review_distribution": "review distribution",
    "rating_score": "rating score",
    "review_count": "review count",
    "sales_volume": "sales volume",
    "demographic": "target demographic",
}


def _feature_word(cat: str) -> str:
    return _FEATURE_WORD.get(cat, str(cat).replace("_", " "))


def _expected_profile_value(attr: str, slot: dict[str, Any], profile: dict[str, Any]):
    """The profile value the agent should have used for ``attr`` (gender/size).
    Prefer the per-miss violation's ``expected``; fall back to the profile."""
    viols = slot.get("violations") or []
    if viols:
        v = viols[0]
        if attr == "size" and v.get("category"):
            return f"{v.get('expected')} ({v.get('category')})"
        return v.get("expected")
    if attr == "gender":
        return profile.get("target_demographic")
    if attr == "size":
        return (profile.get("standard_sizes") or {}).get("tops")
    return None


def categorize_errors(
    per_case: Iterable[CaseResult],
    *,
    max_representative_errors: int = 5,
) -> list[dict[str, Any]]:
    """Group failures across ``per_case`` into the 6 clubbed categories (+ generic).

    ``max_representative_errors`` caps the example messages attached to each
    category; every case is still walked for failure counts.
    """
    cases = list(per_case)
    total = len(cases)
    if total == 0:
        return []

    groups: dict[str, dict[str, Any]] = {}
    sample_level: dict[str, Any] = {}      # sample_id -> level
    level_totals: dict[Any, int] = {}      # level -> #cases at that level

    def _ensure(cat_id: str) -> dict[str, Any]:
        g = groups.get(cat_id)
        if g is None:
            g = {
                "category_id": cat_id,
                "category_type": cat_id,
                "category_name": _CATEGORY_NAMES.get(cat_id, cat_id),
                "slug": cat_id,
                "failing_sample_ids": set(),
                "examples": [],
            }
            groups[cat_id] = g
        return g

    def _add_example(cat_id: str, sample_id: str, checks: list[str], message: str) -> None:
        g = _ensure(cat_id)
        g["failing_sample_ids"].add(sample_id)
        if len(g["examples"]) < max_representative_errors:
            g["examples"].append(
                {"sample_id": sample_id, "checks_failed": checks, "messages": [message]}
            )

    for case in cases:
        sample_id = str(case.case_id)
        details: dict[str, Any] = dict(case.details or {})
        level = details.get("level")
        lvl = f"L{level}" if level is not None else "L?"
        sample_level[sample_id] = level
        level_totals[level] = level_totals.get(level, 0) + 1

        # (A) Runtime/crash — error set, details empty.
        if case.error and not details:
            _add_example("generic", sample_id, ["runtime_error"],
                         f"[{lvl}] The task agent failed before producing a scorable cart.")
            continue
        # (B) No ground truth.
        if str(details.get("error") or "").startswith(_NO_VALIDATION_MSG):
            _add_example("generic", sample_id, ["no_validation"],
                         f"[{lvl}] No ground truth was available to score this case.")
            continue

        causes = details.get("failure_causes") or {}
        bc = details.get("budget_check") or {}

        # missing_feature — name the feature kind AND the exact unmet predicate(s).
        for cat, slot in (causes.get("feature_mismatch") or {}).items():
            if isinstance(slot, dict) and slot.get("product_ids"):
                ps = _preds_str(slot.get("predicates"))
                detail = f" ({ps})" if ps else ""
                _add_example("missing_feature", sample_id, [str(cat)],
                             f"[{lvl}] The cart's closest substitute fails the required "
                             f"{_feature_word(cat)} constraint{detail}.")

        # user_info — name the picked value AND the required profile value.
        profile = details.get("user_profile") or {}
        for attr in ("gender", "size"):
            slot = (causes.get("user_info_mismatch") or {}).get(attr) or {}
            if isinstance(slot, dict) and slot.get("product_ids"):
                v = (slot.get("violations") or [{}])[0]
                exp = _expected_profile_value(attr, slot, profile)
                act = v.get("actual")
                _add_example("user_info", sample_id, [attr],
                             f"[{lvl}] The cart's substitute has {attr} '{act}' but the "
                             f"user profile requires '{exp}'.")

        # not_cheapest — concrete prices and the overspend.
        nc = causes.get("not_cheapest") or {}
        if nc.get("product_ids"):
            g0 = (nc.get("gaps") or [{}])[0]
            if g0.get("gap") is not None:
                msg = (f"[{lvl}] The cart paid ¥{g0.get('picked_price')} for a substitute "
                       f"but a qualifying product costs ¥{g0.get('gold_price')} "
                       f"(¥{g0.get('gap')} more).")
            else:
                msg = f"[{lvl}] The cart chose a qualifying product that is not the cheapest option."
            _add_example("not_cheapest", sample_id, ["not_cheapest"], msg)

        # not_cheapest_cart_level (L2) — distinct category: the whole cart costs more
        # than the cheapest valid combination (cost is a cart-level property at L2).
        if bc.get("frame") == "not_cheapest_cart_level":
            _add_example("not_cheapest_cart_level", sample_id, ["not_cheapest_cart_level"],
                         f"[{lvl}] The cart total ¥{bc.get('cart_total')} exceeds the cheapest "
                         f"valid combination ¥{bc.get('gt_total')} (¥{bc.get('cost_gap')} more).")

        # missing_product — quote the unfulfilled sub-query.
        mp = causes.get("missing_product") or {}
        if mp.get("product_ids"):
            sq = (mp.get("sub_queries") or [""])[0]
            snip = f': "{sq[:120]}"' if sq else ""
            _add_example("missing_product", sample_id, ["missing_product"],
                         f"[{lvl}] No substitute was added for a required product{snip}.")

        # budget_constraint_mismatch — total, bounds, and the overshoot/shortfall.
        bstatus = bc.get("status")
        if bstatus in ("over", "under"):
            amt = bc.get("over_amount") if bstatus == "over" else bc.get("under_amount")
            _add_example("budget_constraint_mismatch", sample_id, [bstatus],
                         f"[{lvl}] Cart total ¥{bc.get('cart_total')} is {bstatus} the budget "
                         f"[¥{bc.get('budget_min')}–¥{bc.get('budget_max')}] by ¥{amt}.")

        # suboptimal_coupon — one detailed line per concrete coupon problem.
        for msg in _coupon_issue_messages(details, lvl):
            _add_example("suboptimal_coupon", sample_id, ["coupon"], msg)

    # Materialize, compute failure_rate, sort by priority then rate.
    prio = {t: i for i, t in enumerate(category_type_priority)}
    out: list[dict[str, Any]] = []
    for cat_id, g in groups.items():
        num = len(g["failing_sample_ids"])
        # Within-level failure rate (for the manager's per-node level focus): for
        # each level, #failing samples of this category at that level / #cases of
        # that level. Keys are the levels that actually appear (None excluded).
        per_level_fail: dict[Any, int] = {}
        for sid in g["failing_sample_ids"]:
            lv = sample_level.get(sid)
            per_level_fail[lv] = per_level_fail.get(lv, 0) + 1
        failure_rate_by_level = {
            lv: per_level_fail.get(lv, 0) / level_totals[lv]
            for lv in level_totals
            if lv is not None and level_totals[lv] > 0 and per_level_fail.get(lv, 0) > 0
        }
        out.append({
            "category_id": cat_id,
            "category_type": g["category_type"],
            "category_name": g["category_name"],
            "slug": g["slug"],
            "num_failing_samples": num,
            "total_samples": total,
            "failure_rate": num / total,
            "failure_rate_by_level": failure_rate_by_level,
            "representative_errors": g["examples"],
        })
    out.sort(key=lambda c: (-c["failure_rate"], prio.get(c["category_type"], 99), c["category_id"]))
    return out


def _preds_str(preds: list[dict[str, Any]] | None) -> str:
    """Compact ``field op value`` summary of a feature category's predicates
    (dotted prefix dropped). e.g. ``transport_time less_than 2, 5_star > 250``."""
    parts: list[str] = []
    for p in preds or []:
        if not isinstance(p, dict):
            continue
        field = str(p.get("field") or "").split(".")[-1]
        if not field:
            continue
        op = p.get("operator")
        ov = p.get("operator_value")
        parts.append(f"{field} {op} {ov}" if op else field)
    return ", ".join(parts[:4])


def _coupon_issue_messages(details: dict[str, Any], lvl: str) -> list[str]:
    """Detailed one-line messages (with coupon names / amounts) for each concrete
    coupon problem on this case — fed to the suboptimal_coupon bucket."""
    out: list[str] = []
    fpc = details.get("final_price_check") or {}
    agent = fpc.get("agent") or {}
    mc = details.get("missing_coupons") or []
    if mc:
        out.append(f"[{lvl}] Missing the optimal coupon(s): {', '.join(map(str, mc[:4]))}.")
    xc = details.get("extra_coupons") or []
    if xc:
        out.append(f"[{lvl}] Applied non-optimal extra coupon(s): {', '.join(map(str, xc[:4]))}.")
    wq = [c for c in (details.get("coupon_details") or [])
          if isinstance(c, dict) and not c.get("match") and int(c.get("expected_quantity") or 0) > 0]
    for c in wq[:3]:
        out.append(f"[{lvl}] Coupon {c.get('coupon_name')} applied {c.get('quantity')}× "
                   f"but expected {c.get('expected_quantity')}×.")
    own = details.get("coupon_ownership") or {}
    if own.get("applied_not_owned"):
        out.append(f"[{lvl}] Applied coupon(s) the user doesn't own: "
                   f"{', '.join(map(str, own['applied_not_owned'][:4]))}.")
    if own.get("over_owned_qty"):
        parts = [f"{o.get('name')} ({o.get('applied')}>{o.get('owned')})" for o in own["over_owned_qty"][:3]]
        out.append(f"[{lvl}] Over-applied owned coupon(s): {', '.join(parts)}.")
    if agent.get("invalid_coupons"):
        out.append(f"[{lvl}] Same-brand coupon(s) applied without a qualifying brand subtotal: "
                   f"{', '.join(map(str, agent['invalid_coupons'][:4]))}.")
    if float(fpc.get("coupon_suboptimality") or 0.0) > 0:
        opt = (fpc.get("optimal_on_agent_cart") or {}).get("final")
        out.append(f"[{lvl}] Left ¥{fpc['coupon_suboptimality']} of coupon savings unused "
                   f"(cart final ¥{agent.get('final')} vs optimal ¥{opt}).")
    return out


def _coupon_issues(details: dict[str, Any]) -> list[tuple[str, bool]]:
    """(issue phrase, present?) pairs for the suboptimal_coupon per-case check."""
    fpc = details.get("final_price_check") or {}
    agent = fpc.get("agent") or {}
    return [
        ("missing coupon", bool(details.get("missing_coupons"))),
        ("extra coupon", bool(details.get("extra_coupons"))),
        ("wrong quantity", any(
            isinstance(c, dict) and not c.get("match") and int(c.get("expected_quantity") or 0) > 0
            for c in (details.get("coupon_details") or [])
        )),
        ("coupon not owned", bool(
            (details.get("coupon_ownership") or {}).get("applied_not_owned")
            or (details.get("coupon_ownership") or {}).get("over_owned_qty")
        )),
        ("invalid same-brand scope", bool(agent.get("invalid_coupons"))),
        ("under-discounted", float(fpc.get("coupon_suboptimality") or 0.0) > 0),
    ]


def per_case_category_checks(case_details: dict, category_id: str) -> list[bool]:
    """Single per-case "clean?" precision trial per clubbed category (one bool:
    is the case free of that category's error?). Used only by
    ``select_metric == 'category_significance'``; returns ``[]`` when the category
    is not applicable to this case shape.
    """
    if not case_details or not category_id:
        return []
    causes = case_details.get("failure_causes") or {}

    if category_id == "missing_feature":
        return [not any(
            isinstance(s, dict) and s.get("product_ids")
            for s in (causes.get("feature_mismatch") or {}).values()
        )]
    if category_id == "user_info":
        ui = causes.get("user_info_mismatch") or {}
        return [not any(
            isinstance(ui.get(a), dict) and ui[a].get("product_ids") for a in ("gender", "size")
        )]
    if category_id == "not_cheapest":
        return [not bool((causes.get("not_cheapest") or {}).get("product_ids"))]
    if category_id == "not_cheapest_cart_level":
        return [(case_details.get("budget_check") or {}).get("frame") != "not_cheapest_cart_level"]
    if category_id == "missing_product":
        return [not bool((causes.get("missing_product") or {}).get("product_ids"))]
    if category_id == "budget_constraint_mismatch":
        return [(case_details.get("budget_check") or {}).get("status") not in ("over", "under")]
    if category_id == "suboptimal_coupon":
        return [not any(present for _, present in _coupon_issues(case_details))]
    return []  # generic / ambiguous / final_price_gap — not testable per-case


def render_case_error_log(details: dict) -> str:
    """The ENTIRE per-case error log rendered to text — every fired failure cause
    plus budget / coupon / final-price diagnostics, one templated line each.

    Reuses the category message templates by running ``categorize_errors`` over
    this single case (so the per-case log and the round report stay in lock-step).
    Returns "" for a clean case."""
    from types import SimpleNamespace
    case = SimpleNamespace(
        case_id="_", error=details.get("error"),
        score=float(details.get("composite_score") or 0.0), details=dict(details),
    )
    lines: list[str] = []
    for c in categorize_errors([case]):
        for ex in c.get("representative_errors") or []:
            lines.extend(ex.get("messages") or [])
    # final-price gap is a symptom (not a category) — surface it for L3 context.
    fpc = details.get("final_price_check") or {}
    gap = fpc.get("final_price_gap")
    if gap:
        lvl = f"L{details.get('level')}" if details.get("level") is not None else "L?"
        lines.append(
            f"[{lvl}] cart final price ¥{(fpc.get('agent') or {}).get('final')} vs optimal "
            f"¥{(fpc.get('gt') or {}).get('final')} (gap ¥{gap}); attribution: {fpc.get('attribution')}."
        )
    return "\n".join(lines)
