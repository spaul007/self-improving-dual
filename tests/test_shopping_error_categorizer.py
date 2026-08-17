"""Tests for the shopping error categorizer (clubbed scheme).

    PYTHONPATH=. python3 -m unittest tests.test_shopping_error_categorizer

The categorizer emits exactly 6 clubbed `category_id`s (== `category_type`) +
`generic`, never the fine-grained ids and never `ambiguous`/`final_price_gap`.
Granularity lives in the one-sentence example message, which carries no digits.
"""
from __future__ import annotations

import unittest
from types import SimpleNamespace

from projects.shopping.shopping_error_categorizer import (
    categorize_errors,
    per_case_category_checks,
    category_type_priority,
)


def _case(case_id, details, *, score=0.0, error=None):
    return SimpleNamespace(case_id=case_id, score=score, error=error, details=details)


def _pc(**kw):
    """Build a failure_causes dict, filling empty defaults."""
    def slot(pids=()):
        return {"sub_queries": [], "fields": [], "predicates": [], "product_ids": list(pids)}
    base = {
        "feature_mismatch": kw.get("feature_mismatch", {}),
        "user_info_mismatch": {
            "gender": {**slot(), "violations": [], **(kw.get("gender") or {})},
            "size": {**slot(), "violations": [], **(kw.get("size") or {})},
        },
        "not_cheapest": {**slot(kw.get("not_cheapest", [])), "gaps": []},
        "ambiguous": slot(kw.get("ambiguous", [])),
        "missing_product": slot(kw.get("missing_product", [])),
    }
    return base


class ClubbedCategoriesTest(unittest.TestCase):
    def test_six_clubbed_ids_and_types(self) -> None:
        cases = [
            _case("L1-1", {"level": 1, "failure_causes": _pc(
                feature_mismatch={"color": {"product_ids": ["g1"], "sub_queries": [],
                                            "fields": ["color"], "predicates": []}})}),
            _case("L1-2", {"level": 1, "failure_causes": _pc(
                gender={"product_ids": ["g2"]})}),
            _case("L1-3", {"level": 1, "failure_causes": _pc(not_cheapest=["g3"])}),
            _case("L1-4", {"level": 1, "failure_causes": _pc(missing_product=["g4"])}),
            _case("L2-5", {"level": 2, "failure_causes": _pc(),
                           "budget_check": {"status": "over"}}),
            _case("L3-6", {"level": 3, "failure_causes": _pc(),
                           "missing_coupons": ["X"]}),
            # ambiguous-only case -> NO category emitted.
            _case("L1-7", {"level": 1, "failure_causes": _pc(ambiguous=["g7"])}),
        ]
        cats = {c["category_id"]: c for c in categorize_errors(cases)}
        self.assertEqual(
            set(cats),
            {"missing_feature", "user_info", "not_cheapest",
             "missing_product", "budget_constraint_mismatch", "suboptimal_coupon"},
        )
        for cid, c in cats.items():
            self.assertEqual(c["category_type"], cid)  # id == type
        self.assertNotIn("ambiguous", cats)
        self.assertNotIn("final_price_gap", cats)
        # ambiguous-only case contributed no category.
        self.assertEqual(cats["missing_feature"]["num_failing_samples"], 1)

    def test_messages_carry_dynamic_detail(self) -> None:
        cases = [
            _case("L1-1", {"level": 1, "failure_causes": _pc(
                feature_mismatch={"delivery_time": {"product_ids": ["g1"], "sub_queries": [],
                    "fields": ["transport_time"],
                    "predicates": [{"field": "transport_time", "operator": "less_than",
                                    "operator_value": 2}]}})}),
            _case("L1-2", {"level": 1, "failure_causes": _pc(
                gender={"product_ids": ["g2"],
                        "violations": [{"attribute": "gender", "expected": "Women", "actual": "Men"}]})},
                ),
            _case("L1-3", {"level": 1, "failure_causes": _pc(
                not_cheapest=["g3"])}),  # gaps filled below
            _case("L2-4", {"level": 2, "failure_causes": _pc(),
                           "budget_check": {"status": "over", "cart_total": 9000,
                                            "budget_min": 100, "budget_max": 200, "over_amount": 8800}}),
            _case("L3-5", {"level": 3, "failure_causes": _pc(),
                           "missing_coupons": ["VIP: ¥200 off every ¥1,000"]}),
        ]
        # fill a real gap on the not_cheapest case
        cases[2].details["failure_causes"]["not_cheapest"]["gaps"] = [
            {"gold_id": "g3", "picked_id": "p3", "gold_price": 100, "picked_price": 150, "gap": 50}]
        cats = {c["category_id"]: c for c in categorize_errors(cases)}

        mf = cats["missing_feature"]["representative_errors"][0]["messages"][0]
        self.assertIn("transport_time less_than 2", mf)        # the exact unmet predicate
        ui = cats["user_info"]["representative_errors"][0]["messages"][0]
        self.assertIn("Men", ui)                               # picked value
        self.assertIn("Women", ui)                             # required profile value
        nc = cats["not_cheapest"]["representative_errors"][0]["messages"][0]
        self.assertIn("150", nc); self.assertIn("100", nc); self.assertIn("50", nc)  # prices + gap
        bd = cats["budget_constraint_mismatch"]["representative_errors"][0]["messages"][0]
        self.assertIn("9000", bd); self.assertIn("8800", bd)   # total + overshoot
        sc = cats["suboptimal_coupon"]["representative_errors"][0]["messages"][0]
        self.assertIn("VIP: ¥200 off every ¥1,000", sc)        # the specific coupon name

    def test_not_cheapest_split_per_item_vs_cart_level(self) -> None:
        cases = [
            # L1 per-item not_cheapest
            _case("L1-1", {"level": 1, "failure_causes": _pc(not_cheapest=["g1"])}),
            # L2 cart-level: frame set, NO per-item
            _case("L2-2", {"level": 2, "failure_causes": _pc(),
                           "budget_check": {"status": "within", "frame": "not_cheapest_cart_level",
                                            "cart_total": 2257.0, "gt_total": 1787.0, "cost_gap": 470.0}}),
        ]
        cats = {c["category_id"]: c for c in categorize_errors(cases)}
        self.assertIn("not_cheapest", cats)
        self.assertIn("not_cheapest_cart_level", cats)
        self.assertEqual(cats["not_cheapest"]["num_failing_samples"], 1)        # L1 only
        self.assertEqual(cats["not_cheapest_cart_level"]["num_failing_samples"], 1)  # L2 only
        msg = cats["not_cheapest_cart_level"]["representative_errors"][0]["messages"][0]
        self.assertIn("2257", msg); self.assertIn("1787", msg); self.assertIn("470", msg)
        # per_case checks are distinct
        self.assertEqual(per_case_category_checks(cases[1].details, "not_cheapest"), [True])
        self.assertEqual(per_case_category_checks(cases[1].details, "not_cheapest_cart_level"), [False])

    def test_failure_rate_by_level(self) -> None:
        cases = [
            _case("L1-1", {"level": 1, "failure_causes": _pc(
                feature_mismatch={"color": {"product_ids": ["g"], "sub_queries": [],
                                            "fields": ["color"], "predicates": []}})}),
            _case("L2-2", {"level": 2, "failure_causes": _pc()}),  # clean L2
            _case("L3-3", {"level": 3, "failure_causes": _pc(), "missing_coupons": ["X"]}),
            _case("L3-4", {"level": 3, "failure_causes": _pc(), "missing_coupons": ["Y"]}),
        ]
        cats = {c["category_id"]: c for c in categorize_errors(cases)}
        # missing_feature: 1 of the 1 L1 case -> L1 rate 1.0
        self.assertEqual(cats["missing_feature"]["failure_rate_by_level"], {1: 1.0})
        # suboptimal_coupon: 2 of the 2 L3 cases -> L3 rate 1.0 (global only 2/4=0.5)
        self.assertEqual(cats["suboptimal_coupon"]["failure_rate_by_level"], {3: 1.0})
        self.assertEqual(cats["suboptimal_coupon"]["failure_rate"], 0.5)

    def test_generic_runtime(self) -> None:
        cats = {c["category_id"]: c for c in categorize_errors(
            [_case("L1-9", {}, error="boom")])}
        self.assertIn("generic", cats)
        self.assertEqual(cats["generic"]["category_type"], "generic")

    def test_priority_list_matches_emitted_types(self) -> None:
        self.assertEqual(category_type_priority, [
            "missing_feature", "user_info", "not_cheapest", "not_cheapest_cart_level",
            "missing_product", "budget_constraint_mismatch", "suboptimal_coupon", "generic"])


class PerCaseChecksTest(unittest.TestCase):
    def test_single_clean_trials(self) -> None:
        dirty = {"failure_causes": _pc(
            feature_mismatch={"color": {"product_ids": ["g"], "sub_queries": [],
                                        "fields": ["color"], "predicates": []}},
            gender={"product_ids": ["g"]}, not_cheapest=["g"], missing_product=["g"]),
            "budget_check": {"status": "over"}, "missing_coupons": ["X"]}
        clean = {"failure_causes": _pc(), "budget_check": {"status": "within"}}
        for cid in ("missing_feature", "user_info", "not_cheapest", "missing_product",
                    "budget_constraint_mismatch", "suboptimal_coupon"):
            self.assertEqual(per_case_category_checks(dirty, cid), [False], cid)
            self.assertEqual(per_case_category_checks(clean, cid), [True], cid)

    def test_not_applicable_returns_empty(self) -> None:
        self.assertEqual(per_case_category_checks({"failure_causes": _pc()}, "generic"), [])
        self.assertEqual(per_case_category_checks({"failure_causes": _pc()}, "ambiguous"), [])
        self.assertEqual(per_case_category_checks({}, "missing_feature"), [])


if __name__ == "__main__":
    unittest.main()
