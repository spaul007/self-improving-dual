"""Smoke tests for the travel benchmark + travel_baseline seed.

No API key, no live OpenAI calls. The constraint-evaluation path is
exercised against a hand-built passing fixture and a fixture with one
deliberately failing hard constraint, to confirm the scorer surfaces a
composite score that drops below 1.0.

Run from the repo root:
    PYTHONPATH=. python -m unittest tests.test_travel_smoke
"""
from __future__ import annotations

import importlib.util
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


class TravelToolRegistrationTests(unittest.TestCase):
    """All nine travel tools must register on package import and be marked
    as immutable so the seed routes them through call_tool()."""

    def test_all_travel_tools_registered_and_immutable(self) -> None:
        import platform_core.tools as t
        import projects.travel.tools  # noqa: F401  - triggers registration

        names = [
            "query_flight_info",
            "query_train_info",
            "query_hotel_info",
            "query_attraction_details",
            "recommend_attractions",
            "query_restaurant_details",
            "recommend_restaurants",
            "search_location",
            "query_road_route_info",
        ]
        schemas = t.all_schemas()
        for n in names:
            with self.subTest(tool=n):
                self.assertIn(n, schemas, f"{n} not registered")
                self.assertTrue(t.is_immutable(n), f"{n} should be immutable")
                schema = schemas[n]
                self.assertEqual(schema["name"], n)
                self.assertIn("input_schema", schema)
                self.assertEqual(schema["input_schema"]["type"], "object")


# --------------------------------------------------------------------------- #
# Travel seed validation
# --------------------------------------------------------------------------- #


class TravelSeedValidatorTests(unittest.TestCase):
    """travel_baseline must pass every validator that ships with the framework
    (the same six the editor will run against the agent on every round)."""

    def test_travel_baseline_passes_all_validators(self) -> None:
        # Force tool registration before SchemaWrapperConsistencyValidator
        # checks tools_schema.json against the immutable registry.
        import projects.travel.tools  # noqa: F401

        from meta_agent.editor_validators import (
            ImmutableFilesValidator,
            ImportValidator,
            MutableToolImportValidator,
            SchemaWrapperConsistencyValidator,
            SignatureValidator,
            SyntaxValidator,
        )

        tmp = Path(tempfile.mkdtemp(prefix="travel_validator_test_"))
        try:
            out = tmp / "round_000"
            (out / "task_agent").mkdir(parents=True)
            shutil.copytree(
                REPO_ROOT / "projects" / "travel" / "seed",
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
            ):
                issues = v.validate(out, base_dir=base)
                self.assertEqual(
                    issues,
                    [],
                    f"{type(v).__name__} rejected the travel_baseline seed: {issues}",
                )
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


# --------------------------------------------------------------------------- #
# cases.jsonl shape
# --------------------------------------------------------------------------- #


class TravelCasesJsonlTests(unittest.TestCase):
    """Every line in cases.jsonl must be parseable and have the per-case
    env-override structure the SubprocessEvaluator expects."""

    def test_cases_jsonl_well_formed(self) -> None:
        path = REPO_ROOT / "projects" / "travel" / "benchmark" / "cases.jsonl"
        self.assertTrue(path.exists(), f"missing {path}")
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.assertGreater(len(rows), 0, "cases.jsonl is empty")
        for r in rows[:5]:  # spot-check the first five so the suite stays fast
            with self.subTest(case_id=r.get("id")):
                self.assertIn("id", r)
                self.assertIn("input", r)
                self.assertIn("env", r)
                self.assertIn("meta_info", r)
                self.assertEqual(
                    r["env"].get("TRAVEL_SAMPLE_ID"),
                    str(r["id"]),
                    "env.TRAVEL_SAMPLE_ID must match the case id",
                )


# --------------------------------------------------------------------------- #
# Scorer constraint helpers (no LLM)
# --------------------------------------------------------------------------- #


def _load_scorer_module():
    """Load projects/travel/benchmark/scorer.py via the same importlib path
    the evaluator uses, so the test exercises the same code path."""
    spec = importlib.util.spec_from_file_location(
        "_test_scorer",
        REPO_ROOT / "projects" / "travel" / "benchmark" / "scorer.py",
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class ScorerConstraintTests(unittest.TestCase):
    """Run the constraint helpers against tiny hand-built fixtures so the
    scorer's _evaluate path is exercised without needing an OpenAI call."""

    def test_registry_lookup_returns_scorer_class(self) -> None:
        _load_scorer_module()  # runs @register("scorer", "travel_default")
        from meta_agent.registry import get
        cls = get("scorer", "travel_default")
        scorer = cls()
        self.assertTrue(callable(scorer.score))

    def test_scorer_returns_zero_on_empty_output(self) -> None:
        mod = _load_scorer_module()
        result = mod.score({"id": "0", "meta_info": {"hard_constraints": {}}}, "")
        self.assertEqual(result["score"], 0.0)
        self.assertFalse(result["passed"])
        self.assertIn("error", result["details"])

    def test_scorer_returns_zero_on_unparseable_plan(self) -> None:
        # Without an OPENAI_API_KEY the conversion step fails fast. We use
        # that to verify the scorer returns a graceful zero instead of crashing.
        import os
        prev_key = os.environ.pop("OPENAI_API_KEY", None)
        try:
            mod = _load_scorer_module()
            result = mod.score(
                {"id": "0", "meta_info": {"hard_constraints": {}}},
                "<plan>arbitrary text</plan>",
            )
            self.assertEqual(result["score"], 0.0)
            self.assertFalse(result["passed"])
            self.assertIn("error", result["details"])
            self.assertIn("OPENAI_API_KEY", result["details"]["error"])
        finally:
            if prev_key is not None:
                os.environ["OPENAI_API_KEY"] = prev_key


class TravelScorerAggregateTests(unittest.TestCase):
    """``TravelCompositeScorer`` owns both per-case ``score()`` and
    round-level ``aggregate(per_case, trace_events)``. The framework's
    ``DefaultFeedbackGatherer`` calls ``aggregate()`` and lands the
    result on ``AgentFeedback.project_metrics``."""

    def _make_cases(self, per_case_dicts: list[dict]):
        from meta_agent.models import CaseResult

        return [CaseResult(**c) for c in per_case_dicts]

    def test_aggregate_populates_expected_keys(self) -> None:
        mod = _load_scorer_module()
        scorer = mod.TravelCompositeScorer()
        per_case = self._make_cases([
            {
                "case_id": "0", "passed": False, "score": 0.0,
                "details": {"error": "plan conversion failed: agent produced no plan"},
            },
            {
                "case_id": "1", "passed": False, "score": 0.0,
                "details": {"error": "plan conversion failed: empty"},
            },
            {
                "case_id": "2", "passed": False, "score": 0.2,
                "details": {
                    "failed_checks": [
                        "commonsense:Cost Calculation Accuracy:cost_calculation_correctness",
                        "commonsense:Itinerary Structure:ends_with_accommodation",
                    ],
                    "dimension_scores": {
                        "Cost Calculation Accuracy": 0.0,
                        "Itinerary Structure": 0.25,
                    },
                },
            },
            {
                "case_id": "3", "passed": False, "score": 0.4,
                "details": {
                    "failed_checks": [
                        "commonsense:Cost Calculation Accuracy:cost_calculation_correctness",
                    ],
                    "dimension_scores": {
                        "Cost Calculation Accuracy": 0.5,
                        "Itinerary Structure": 0.75,
                    },
                },
            },
        ])
        m = scorer.aggregate(per_case, trace_events=[])
        self.assertAlmostEqual(m["no_plan_rate"], 0.5)  # 2 of 4 cases
        names = [n for n, _ in m["top_failed_checks"]]
        self.assertEqual(
            names[0],
            "commonsense:Cost Calculation Accuracy:cost_calculation_correctness",
        )
        self.assertEqual(dict(m["top_failed_checks"]).get(names[0]), 2)
        # Dimension means averaged only over the 2 cases that produced a plan.
        self.assertAlmostEqual(m["dimension_means"]["Cost Calculation Accuracy"], 0.25)
        self.assertAlmostEqual(m["dimension_means"]["Itinerary Structure"], 0.5)

    def test_default_gatherer_dispatches_to_scorer_aggregate(self) -> None:
        """End-to-end: when DefaultFeedbackGatherer holds a
        TravelCompositeScorer, its ``compile()`` populates
        ``project_metrics`` via the scorer's ``aggregate``."""
        import shutil
        import tempfile

        from meta_agent.feedback_gatherer import DefaultFeedbackGatherer
        from meta_agent.models import EvaluationResult, EvolutionStrategy

        mod = _load_scorer_module()
        scorer = mod.TravelCompositeScorer()
        per_case = self._make_cases([
            {
                "case_id": "0", "passed": False, "score": 0.4,
                "details": {
                    "failed_checks": ["commonsense:X:y"],
                    "dimension_scores": {"X": 0.5},
                },
            },
        ])
        eval_result = EvaluationResult(score=0.4, passed=0, failed=1, per_case=per_case)

        tmp = Path(tempfile.mkdtemp(prefix="travel_aggregate_test_"))
        self.addCleanup(shutil.rmtree, tmp, True)
        (tmp / "logs").mkdir()
        (tmp / "logs" / "trace.jsonl").write_text("", encoding="utf-8")

        strategy = EvolutionStrategy(
            target_files=["workflow.py"], optimization_goal="x", proposed_changes="y"
        )
        gatherer = DefaultFeedbackGatherer(scorer=scorer)
        fb = gatherer.compile(
            round_number=1,
            base_round=0,
            strategy=strategy,
            eval_result=eval_result,
            round_dir=tmp,
        )
        m = fb.project_metrics
        self.assertAlmostEqual(m["no_plan_rate"], 0.0)
        self.assertEqual(dict(m["top_failed_checks"]).get("commonsense:X:y"), 1)
        self.assertAlmostEqual(m["dimension_means"]["X"], 0.5)


if __name__ == "__main__":
    unittest.main()
