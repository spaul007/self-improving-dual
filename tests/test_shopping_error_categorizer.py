"""Tests for the shopping error categorizer + its per-case significance trials.

    PYTHONPATH=. python3 -m unittest tests.test_shopping_error_categorizer

Focus: (1) categorize_errors groups the scorer's enriched ``details`` into the
expected category families, and (2) per_case_category_checks returns trial
vectors whose denominator is fixed by ground truth / the case — so the
HGMDualManager's category_significance bootstrap can detect improvements for
BOTH recall categories (missing_*) and precision/absence categories
(extra_coupon, wrong_coupon_quantity). ``extra_product`` is intentionally
suppressed and must never appear as a category.
"""
from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from meta_agent.managers.hgm_dual import _bootstrap_p_greater
from projects.shopping.shopping_error_categorizer import (
    categorize_errors,
    per_case_category_checks,
)


def _case(case_id, details, *, score=0.0, error=None):
    return SimpleNamespace(case_id=case_id, score=score, error=error, details=details)


class CategorizeErrorsTest(unittest.TestCase):
    def test_groups_feature_coupon_cart_and_generic(self) -> None:
        cases = [
            _case(
                "L1-1",
                {
                    "level": 1,
                    "missing_feature_categories": {
                        "brand": {
                            "sub_queries": ["want a Nike item"],
                            "fields": ["brand"],
                            "predicates": [
                                {"field": "brand", "operator": "equals", "operator_value": "Nike"}
                            ],
                            "product_ids": ["p1"],
                        }
                    },
                    "extra_products": ["x1"],
                    "missing_coupons": [],
                    "coupon_details": [],
                },
            ),
            _case(
                "L3-1",
                {
                    "level": 3,
                    "missing_feature_categories": {},
                    "extra_products": [],
                    "missing_coupons": ["C1"],
                    "coupon_details": [
                        {"coupon_name": "C2", "quantity": 2, "expected_quantity": 1, "match": False}
                    ],
                },
            ),
            _case("crash", {}, error="subprocess died"),
        ]
        cats = {c["category_id"]: c for c in categorize_errors(cases)}
        # feature category surfaced with predicates in the message (the report).
        self.assertIn("missing_product__brand", cats)
        self.assertEqual(cats["missing_product__brand"]["category_type"], "missing_feature")
        msg = cats["missing_product__brand"]["representative_errors"][0]["messages"][0]
        self.assertIn("brand equals Nike", msg)
        # extra_product is suppressed — it must never enter the category list,
        # even though the case carried extra_products in its details.
        self.assertNotIn("extra_product", cats)
        # coupon + generic families still present.
        self.assertEqual(cats["missing_coupon"]["category_type"], "coupon")
        self.assertEqual(cats["wrong_coupon_quantity"]["category_type"], "coupon")
        self.assertEqual(cats["generic__runtime_error"]["category_type"], "generic")
        # failure_rate is over ALL cases (3 here).
        self.assertAlmostEqual(cats["missing_product__brand"]["failure_rate"], 1 / 3)

    def test_no_gt_product_ids_in_report_messages(self) -> None:
        # The report carries predicates/sub-queries, not the gold answer ids.
        cases = [
            _case(
                "L1-1",
                {
                    "level": 1,
                    "missing_feature_categories": {
                        "brand": {"sub_queries": ["q"], "fields": ["brand"],
                                  "predicates": [], "product_ids": ["GOLDPID123"]}
                    },
                },
            )
        ]
        cats = categorize_errors(cases)
        blob = repr([c["representative_errors"] for c in cats])
        # checks_failed/messages name fields, not the gold product id.
        self.assertIn("brand", blob)
        self.assertNotIn("GOLDPID123", blob)


class PerCaseChecksRecallTest(unittest.TestCase):
    def test_missing_product_recall_vector(self) -> None:
        d = {
            "gold_feature_categories": {"brand": ["p1", "p2", "p3"]},
            "matched_products": ["p1", "p3"],
        }
        self.assertEqual(
            per_case_category_checks(d, "missing_product__brand"), [True, False, True]
        )

    def test_missing_coupon_recall_vector(self) -> None:
        d = {
            "coupon_details": [{"coupon_name": "C1", "match": True}],
            "missing_coupons": ["C2", "C3"],
        }
        # 1 matched + 2 missing = denominator fixed by GT coupon count.
        self.assertEqual(
            per_case_category_checks(d, "missing_coupon"), [True, False, False]
        )

    def test_not_applicable_is_empty(self) -> None:
        # No brand-gold products in this case ⇒ not applicable.
        self.assertEqual(
            per_case_category_checks({"gold_feature_categories": {}}, "missing_product__brand"),
            [],
        )
        self.assertEqual(per_case_category_checks({}, "extra_product"), [])
        self.assertEqual(per_case_category_checks({"level": 1}, "generic__runtime_error"), [])


class PerCaseChecksPrecisionTest(unittest.TestCase):
    """The fix: precision/absence categories are a single per-case clean trial,
    so the count is GT-independent (1/case) and improvements are detectable."""

    def test_extra_product_is_suppressed(self) -> None:
        # extra_product is no longer a category, so it has no per-case trial
        # shape — it returns [] regardless of how many extra products exist.
        self.assertEqual(per_case_category_checks({"extra_products": ["x1", "x2"]}, "extra_product"), [])
        self.assertEqual(per_case_category_checks({"extra_products": []}, "extra_product"), [])

    def test_extra_coupon_is_single_clean_trial(self) -> None:
        self.assertEqual(per_case_category_checks({"extra_coupons": ["c"]}, "extra_coupon"), [False])
        self.assertEqual(per_case_category_checks({"extra_coupons": []}, "extra_coupon"), [True])

    def test_wrong_coupon_quantity_is_single_clean_trial(self) -> None:
        dirty = {"coupon_details": [{"match": False, "expected_quantity": 2}]}
        clean = {"coupon_details": [{"match": True, "expected_quantity": 2}]}
        self.assertEqual(per_case_category_checks(dirty, "wrong_coupon_quantity"), [False])
        self.assertEqual(per_case_category_checks(clean, "wrong_coupon_quantity"), [True])

    def test_bootstrap_detects_extra_coupon_cleanup(self) -> None:
        """End-to-end on the manager's bootstrap: a variant that removes every
        extra coupon is significant. (extra_product used to be the example here
        but is now suppressed, so this exercises the same path via extra_coupon.)"""
        sa = [{"extra_coupons": ["x1", "x2"]}, {"extra_coupons": ["x3"]}, {"extra_coupons": []}]
        var = [{"extra_coupons": []}, {"extra_coupons": []}, {"extra_coupons": []}]
        sa_pool, v_pool = [], []
        for sd, vd in zip(sa, var):
            s = per_case_category_checks(sd, "extra_coupon")
            if not s:
                continue
            sa_pool += s
            v = per_case_category_checks(vd, "extra_coupon")
            v = (v + [False] * len(s))[: len(s)]
            v_pool += v
        n = len(sa_pool)
        p = _bootstrap_p_greater(sum(v_pool), n, sum(sa_pool), n, np.random.default_rng(0), 2000)
        self.assertEqual((sum(sa_pool), sum(v_pool), n), (1, 3, 3))
        self.assertGreater(p, 0.95)  # significant cleanup


if __name__ == "__main__":
    unittest.main()
