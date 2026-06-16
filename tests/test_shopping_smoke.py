"""Smoke tests for the shopping benchmark + shopping seed.

No API key, no live OpenAI calls. The match-rate scorer is exercised
against hand-built fixtures (passing + failing) to confirm the
composite score and per-level aggregator behave correctly.

Run from the repo root:
    PYTHONPATH=. python -m unittest tests.test_shopping_smoke
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


# --------------------------------------------------------------------------- #
# Tool registration
# --------------------------------------------------------------------------- #


class ShoppingToolRegistrationTests(unittest.TestCase):
    """All 15 shopping tools must register on package import and be marked
    immutable so the seed routes them through call_tool()."""

    def test_all_shopping_tools_registered_and_immutable(self) -> None:
        import platform_core.tools as t

        # Force a fresh discovery via load_project (idempotent).
        t.load_project("shopping")
        schemas = t.all_schemas()
        # Exactly 15 — if this number changes the schema, the tests,
        # and the seed's tools_schema.json all need to move together.
        expected = {
            "search_products",
            "filter_by_brand",
            "filter_by_color",
            "filter_by_size",
            "filter_by_range",
            "filter_by_applicable_coupons",
            "sort_products",
            "get_product_details",
            "calculate_transport_time",
            "get_user_info",
            "add_product_to_cart",
            "delete_product_from_cart",
            "get_cart_info",
            "add_coupon_to_cart",
            "delete_coupon_from_cart",
        }
        registered = set(schemas)
        missing = expected - registered
        extra = (registered - expected) & set(schemas)  # not strict — travel
        # might also be loaded in the same process during the test run.
        self.assertFalse(missing, f"missing shopping tools: {missing}")
        # Ensure each is marked immutable (i.e., callable via the wrapper).
        for name in expected:
            self.assertTrue(t.is_immutable(name), f"{name} not marked immutable")
        # Schema shape sanity check.
        for name in expected:
            schema = schemas[name]
            self.assertIn("input_schema", schema)
            self.assertEqual(schema.get("name"), name)


# --------------------------------------------------------------------------- #
# Seed validator pass — same six the editor runs each round
# --------------------------------------------------------------------------- #


class ShoppingSeedValidatorTests(unittest.TestCase):
    """The shopping seed must pass every default validator."""

    def test_shopping_seed_passes_all_validators(self) -> None:
        import projects.shopping.tools  # noqa: F401 - force tool registration

        from meta_agent.editor_validators import (
            ImmutableFilesValidator,
            ImportValidator,
            LoadTestValidator,
            MutableToolImportValidator,
            SchemaWrapperConsistencyValidator,
            SignatureValidator,
            SyntaxValidator,
        )

        tmp = Path(tempfile.mkdtemp(prefix="shopping_validator_test_"))
        try:
            out = tmp / "round_000"
            (out / "task_agent").mkdir(parents=True)
            shutil.copytree(
                REPO_ROOT / "projects" / "shopping" / "seed",
                out / "task_agent",
                dirs_exist_ok=True,
            )
            base = tmp / "base"
            shutil.copytree(out, base)

            for v in (
                SyntaxValidator(),
                SignatureValidator(),
                ImportValidator(),
                SchemaWrapperConsistencyValidator(),
                MutableToolImportValidator(),
                ImmutableFilesValidator(),
                LoadTestValidator(),
            ):
                issues = v.validate(out, base_dir=base)
                self.assertEqual(
                    issues,
                    [],
                    f"{type(v).__name__} rejected the shopping seed: {issues}",
                )
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


# --------------------------------------------------------------------------- #
# cases.jsonl shape
# --------------------------------------------------------------------------- #


class ShoppingCasesJsonlTests(unittest.TestCase):
    """cases.jsonl must have exactly 120 entries (50 L1 + 50 L2 + 20 L3)
    with the right env, context, and meta_info fields."""

    def test_case_count_and_shape(self) -> None:
        path = REPO_ROOT / "projects" / "shopping" / "benchmark" / "cases.jsonl"
        self.assertTrue(path.exists(), f"cases.jsonl not found at {path}")
        rows: list[dict] = []
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        self.assertEqual(len(rows), 120)

        per_level: dict[int, int] = {}
        for r in rows:
            self.assertIn("id", r)
            self.assertIn("input", r)
            self.assertIn("env", r)
            self.assertIn("context", r)
            self.assertIn("meta_info", r)
            env = r["env"]
            self.assertIn("SHOPPING_SAMPLE_ID", env)
            self.assertIn("SHOPPING_LEVEL", env)
            level = int(env["SHOPPING_LEVEL"])
            self.assertEqual(r["context"].get("level"), level)
            self.assertEqual(r["meta_info"].get("level"), level)
            self.assertRegex(r["id"], r"^L[123]-\d+$")
            per_level[level] = per_level.get(level, 0) + 1

        self.assertEqual(per_level, {1: 50, 2: 50, 3: 20})


# --------------------------------------------------------------------------- #
# Scorer
# --------------------------------------------------------------------------- #


class ShoppingScorerTests(unittest.TestCase):
    """Hand-built validation + cart fixtures exercise the
    set-intersection scoring and the per-level aggregator."""

    def setUp(self) -> None:
        from projects.shopping.benchmark.scorer import ShoppingScorer

        self.scorer_cls = ShoppingScorer
        self.tmp = Path(tempfile.mkdtemp(prefix="shopping_scorer_test_"))
        # Build a tmp data tree at <tmp>/database_level1/case_42/.
        self.case_dir = self.tmp / "database_level1" / "case_42"
        self.case_dir.mkdir(parents=True)
        self._set_validation(
            {
                "ground_truth_products": [
                    {"product_id": "A"},
                    {"product_id": "B"},
                    {"product_id": "C"},
                ],
                "ground_truth_coupons": {
                    "Cross-store: ¥30 off every ¥300": 1,
                },
            }
        )
        # Repoint SHOPPING_DATABASE_ROOT at the tmp tree for the test.
        import os

        self._orig_env = os.environ.get("SHOPPING_DATABASE_ROOT")
        os.environ["SHOPPING_DATABASE_ROOT"] = str(self.tmp)

    def tearDown(self) -> None:
        import os

        if self._orig_env is None:
            os.environ.pop("SHOPPING_DATABASE_ROOT", None)
        else:
            os.environ["SHOPPING_DATABASE_ROOT"] = self._orig_env
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _set_validation(self, data: dict) -> None:
        (self.case_dir / "validation_cases.json").write_text(
            json.dumps(data), encoding="utf-8"
        )

    def _case(self) -> dict:
        return {
            "id": "L1-42",
            "input": "test query",
            "env": {"SHOPPING_SAMPLE_ID": "42", "SHOPPING_LEVEL": "1"},
            "context": {"level": 1},
            "meta_info": {"level": 1},
        }

    def _cart(self, pids, used_coupons=None) -> str:
        return json.dumps(
            {
                "items": [{"product_id": p} for p in pids],
                "used_coupons": used_coupons or [],
                "summary": {},
            }
        )

    def test_perfect_match_scores_1(self) -> None:
        scorer = self.scorer_cls()
        out = scorer.score(
            self._case(),
            self._cart(
                ["A", "B", "C"],
                used_coupons=[
                    {"coupon_name": "Cross-store: ¥30 off every ¥300", "quantity": 1}
                ],
            ),
        )
        self.assertTrue(out["passed"])
        self.assertEqual(out["score"], 1.0)
        details = out["details"]
        self.assertEqual(details["matched_count"], 4)
        self.assertEqual(details["expected_count"], 4)
        self.assertEqual(details["level"], 1)

    def test_partial_match(self) -> None:
        scorer = self.scorer_cls()
        out = scorer.score(self._case(), self._cart(["A", "Z"]))
        # 1 matched (A) out of 4 expected (3 products + 1 coupon).
        self.assertFalse(out["passed"])
        self.assertAlmostEqual(out["score"], 1 / 4)
        details = out["details"]
        self.assertIn("Z", details["extra_products"])
        self.assertIn("B", details["missing_products"])
        self.assertIn("C", details["missing_products"])

    def test_empty_cart_scores_zero(self) -> None:
        scorer = self.scorer_cls()
        out = scorer.score(self._case(), '{"items":[]}')
        self.assertFalse(out["passed"])
        self.assertEqual(out["score"], 0.0)

    def test_no_validation_file_handled_gracefully(self) -> None:
        # Point the scorer at a case without a validation file.
        scorer = self.scorer_cls()
        case = self._case()
        case["env"]["SHOPPING_SAMPLE_ID"] = "9999"  # no such case dir
        out = scorer.score(case, self._cart(["A"]))
        self.assertFalse(out["passed"])
        self.assertEqual(out["score"], 0.0)
        self.assertIn("error", out["details"])

    def test_aggregate_per_level(self) -> None:
        # Use real CaseResult objects — the scorer's emitted dict lands on
        # ``.details`` (CaseResult has no ``.metrics``). A fake namedtuple
        # with a ``.metrics`` field masked the bug where aggregate read the
        # non-existent ``.metrics`` and left per_level silently empty.
        from meta_agent.models import CaseResult

        scorer = self.scorer_cls()

        def _case(cid: str, score: float, level: int) -> CaseResult:
            return CaseResult(
                case_id=cid, passed=score >= 1.0, score=score,
                details={"level": level, "expected_count": 4},
            )

        per_case = [
            _case("L1-1", 1.0, 1),
            _case("L1-2", 0.5, 1),
            _case("L2-1", 0.0, 2),
            _case("L3-1", 0.8, 3),
        ]
        agg = scorer.aggregate(per_case, trace_events=[])
        self.assertAlmostEqual(agg["score_overall"], 2.3 / 4)
        self.assertAlmostEqual(agg["per_level"][1], 0.75)
        self.assertEqual(agg["per_level"][2], 0.0)
        self.assertAlmostEqual(agg["per_level"][3], 0.8)
        self.assertEqual(agg["level_n"], {1: 2, 2: 1, 3: 1})
        self.assertEqual(agg["cases_with_ground_truth"], 4)


# --------------------------------------------------------------------------- #
# _db reset_cart
# --------------------------------------------------------------------------- #


class ShoppingDbResetCartTests(unittest.TestCase):
    """`reset_cart` must restore the per-case cart.json to its empty
    initial shape so multi-round meta-agent loops start clean."""

    def setUp(self) -> None:
        import os
        from projects.shopping.tools import _db

        self._db = _db
        self.tmp = Path(tempfile.mkdtemp(prefix="shopping_db_test_"))
        self.case_dir = self.tmp / "database_level1" / "case_7"
        self.case_dir.mkdir(parents=True)
        (self.case_dir / "user_info.json").write_text(
            json.dumps({"user_id": "u-7", "username": "tester"}),
            encoding="utf-8",
        )
        # Write a non-empty cart to simulate end-of-previous-round state.
        (self.case_dir / "cart.json").write_text(
            json.dumps(
                {
                    "items": [
                        {"product_id": "X", "name": "stale", "quantity": 2, "price": 9.9}
                    ],
                    "used_coupons": [
                        {"coupon_name": "Cross-store: ¥30 off every ¥300", "quantity": 1}
                    ],
                    "summary": {"total_items_count": 2, "total_price": 19.8},
                }
            ),
            encoding="utf-8",
        )
        self._orig_env = {
            k: os.environ.get(k)
            for k in ("SHOPPING_DATABASE_ROOT", "SHOPPING_LEVEL", "SHOPPING_SAMPLE_ID")
        }
        os.environ["SHOPPING_DATABASE_ROOT"] = str(self.tmp)
        os.environ["SHOPPING_LEVEL"] = "1"
        os.environ["SHOPPING_SAMPLE_ID"] = "7"
        # Caches are stale across tests because tmpdir changes.
        self._db._PRODUCTS_CACHE.clear()
        self._db._USER_CACHE.clear()
        self._db._VALIDATION_CACHE.clear()

    def tearDown(self) -> None:
        import os

        for k, v in self._orig_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        self._db._PRODUCTS_CACHE.clear()
        self._db._USER_CACHE.clear()
        self._db._VALIDATION_CACHE.clear()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_reset_cart_clears_state(self) -> None:
        self._db.reset_cart()
        cart = self._db.load_cart()
        self.assertEqual(cart.get("items"), [])
        self.assertEqual(cart.get("used_coupons"), [])
        self.assertEqual(cart.get("summary", {}).get("total_items_count"), 0)
        self.assertEqual(cart.get("summary", {}).get("total_price"), 0.0)
        # user_info should still be reachable.
        self.assertEqual(cart.get("user_id"), "u-7")


class ShoppingCartIsolationTests(unittest.TestCase):
    """The mutable cart must live under the evaluator's per-run scratch dir
    (``META_AGENT_SCRATCH_DIR``) when set, so concurrent evaluations of the
    same case never collide on a shared cart.json. Read-only data stays in
    the shared data root. When the scratch var is unset, behavior falls back
    to ``case_dir()/cart.json`` (standalone / load-test path)."""

    def setUp(self) -> None:
        import os
        from projects.shopping.tools import _db

        self._db = _db
        self.data_root = Path(tempfile.mkdtemp(prefix="shopping_cart_data_"))
        # Read-only data for (level=1, sample=7): user_info only.
        case_dir = self.data_root / "database_level1" / "case_7"
        case_dir.mkdir(parents=True)
        (case_dir / "user_info.json").write_text(
            json.dumps({"user_id": "u-7", "username": "tester"}), encoding="utf-8"
        )
        self.scratch_a = Path(tempfile.mkdtemp(prefix="shopping_cart_scratchA_"))
        self.scratch_b = Path(tempfile.mkdtemp(prefix="shopping_cart_scratchB_"))

        self._orig_env = {
            k: os.environ.get(k)
            for k in (
                "SHOPPING_DATABASE_ROOT",
                "SHOPPING_LEVEL",
                "SHOPPING_SAMPLE_ID",
                "META_AGENT_SCRATCH_DIR",
            )
        }
        os.environ["SHOPPING_DATABASE_ROOT"] = str(self.data_root)
        os.environ["SHOPPING_LEVEL"] = "1"
        os.environ["SHOPPING_SAMPLE_ID"] = "7"
        os.environ.pop("META_AGENT_SCRATCH_DIR", None)
        self._db._PRODUCTS_CACHE.clear()
        self._db._USER_CACHE.clear()
        self._db._VALIDATION_CACHE.clear()

    def tearDown(self) -> None:
        import os

        for k, v in self._orig_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        self._db._PRODUCTS_CACHE.clear()
        self._db._USER_CACHE.clear()
        self._db._VALIDATION_CACHE.clear()
        shutil.rmtree(self.data_root, ignore_errors=True)
        shutil.rmtree(self.scratch_a, ignore_errors=True)
        shutil.rmtree(self.scratch_b, ignore_errors=True)

    def test_cart_written_under_scratch_not_data_root(self) -> None:
        import os

        os.environ["META_AGENT_SCRATCH_DIR"] = str(self.scratch_a)
        self._db.write_cart(
            {"items": [{"product_id": "X", "quantity": 1}], "used_coupons": []}
        )
        expected = self.scratch_a / "database_level1" / "case_7" / "cart.json"
        self.assertTrue(expected.exists(), "cart must land under the scratch dir")
        # The shared read-only data root must NOT receive a cart.json.
        self.assertFalse(
            (self.data_root / "database_level1" / "case_7" / "cart.json").exists(),
            "cart must not be written into the shared data tree",
        )

    def test_two_scratch_dirs_are_isolated(self) -> None:
        import os

        os.environ["META_AGENT_SCRATCH_DIR"] = str(self.scratch_a)
        self._db.write_cart(
            {"items": [{"product_id": "X", "quantity": 1}], "used_coupons": []}
        )
        # A second concurrent "run" with its own scratch sees an empty cart.
        os.environ["META_AGENT_SCRATCH_DIR"] = str(self.scratch_b)
        self.assertEqual(self._db.load_cart().get("items"), [])
        # The first run's cart is untouched by the second.
        os.environ["META_AGENT_SCRATCH_DIR"] = str(self.scratch_a)
        self.assertEqual(
            self._db.load_cart().get("items"), [{"product_id": "X", "quantity": 1}]
        )

    def test_within_run_cases_do_not_collide(self) -> None:
        import os

        # One scratch dir shared by all cases of a run; distinct (level,sample)
        # must map to distinct files.
        os.environ["META_AGENT_SCRATCH_DIR"] = str(self.scratch_a)
        os.environ["SHOPPING_LEVEL"] = "1"
        os.environ["SHOPPING_SAMPLE_ID"] = "7"
        self._db.write_cart({"items": [{"product_id": "A"}], "used_coupons": []})
        os.environ["SHOPPING_LEVEL"] = "2"
        os.environ["SHOPPING_SAMPLE_ID"] = "3"
        self._db.write_cart({"items": [{"product_id": "B"}], "used_coupons": []})

        p1 = self.scratch_a / "database_level1" / "case_7" / "cart.json"
        p2 = self.scratch_a / "database_level2" / "case_3" / "cart.json"
        self.assertTrue(p1.exists() and p2.exists())
        self.assertEqual(json.loads(p1.read_text())["items"], [{"product_id": "A"}])
        self.assertEqual(json.loads(p2.read_text())["items"], [{"product_id": "B"}])

    def test_fallback_to_case_dir_when_scratch_unset(self) -> None:
        import os

        os.environ.pop("META_AGENT_SCRATCH_DIR", None)
        expected = self.data_root / "database_level1" / "case_7" / "cart.json"
        self.assertEqual(self._db.cart_path(), expected)

    def test_cart_path_none_without_level_or_sample(self) -> None:
        import os

        os.environ.pop("META_AGENT_SCRATCH_DIR", None)
        os.environ.pop("SHOPPING_LEVEL", None)
        os.environ.pop("SHOPPING_SAMPLE_ID", None)
        self.assertIsNone(self._db.cart_path())


if __name__ == "__main__":
    unittest.main()
