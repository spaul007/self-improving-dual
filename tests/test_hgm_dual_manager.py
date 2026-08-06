"""Smoke tests for HGMDualManager — drives the full dual-optimization
expansion with stubbed AgentEditor + Evaluator (no LLM, no subprocess).

    PYTHONPATH=. python3 -m unittest tests.test_hgm_dual_manager
"""
from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from meta_agent.feedback_gatherer import DefaultFeedbackGatherer
from meta_agent.managers.hgm_dual import HGMDualManager, _VariantSpec
from meta_agent.models import CaseResult, EditResult, EvaluationResult, EvolutionStrategy


# --------------------------------------------------------------------------- #
# Spec factory for category-significance selection tests
# --------------------------------------------------------------------------- #


def _make_spec_with_dim_checks(
    *, index: int, category_id, case_check_passes: list[list[bool]],
    dim_name: str = "Time Feasibility",
) -> _VariantSpec:
    """Build a _VariantSpec where each case has dimension_details populated
    with a list of per-check pass/fail bools for the given dimension name.

    ``case_check_passes`` is a list-of-lists: outer dimension = cases,
    inner = per-check pass/fail for that case.
    """
    per_case = []
    for i, checks in enumerate(case_check_passes):
        n_checks = len(checks)
        n_pass = sum(checks)
        dim_score = n_pass / n_checks if n_checks else 0.0
        details = {
            "dimension_details": {dim_name: {
                "score": dim_score,
                "checks": [
                    {"name": f"check_{j}", "passed": bool(p), "message": None}
                    for j, p in enumerate(checks)
                ],
            }}
        }
        per_case.append(CaseResult(
            case_id=str(i), passed=(dim_score == 1.0),
            score=dim_score, details=details,
        ))
    mean = (sum(c.score for c in per_case) / len(per_case)) if per_case else 0.0
    return _VariantSpec(
        index=index,
        category=(
            {"category_id": category_id, "category_name": dim_name}
            if category_id else None
        ),
        agent_dir=Path("/tmp"),
        eval_result=EvaluationResult(
            score=mean,
            passed=sum(1 for c in per_case if c.passed),
            failed=sum(1 for c in per_case if not c.passed),
            per_case=per_case,
        ),
        strategy=EvolutionStrategy(
            target_files=["workflow.py"],
            optimization_goal="stub", proposed_changes="", rationale="",
        ),
    )


def _make_spec_with_hard_constraint(
    *, index: int, category_id,
    case_constraints: list[list[tuple[str, bool]]],
) -> _VariantSpec:
    """Build a _VariantSpec where each case has hard_constraints populated.
    ``case_constraints`` is a list-of-lists: outer = cases, inner =
    (constraint_name, passed) tuples for that case."""
    per_case = []
    for i, cs in enumerate(case_constraints):
        details = {
            "hard_constraints": {
                name: {"passed": bool(p), "message": None}
                for name, p in cs
            }
        }
        all_passed = all(p for _, p in cs) if cs else True
        per_case.append(CaseResult(
            case_id=str(i), passed=all_passed,
            score=1.0 if all_passed else 0.0, details=details,
        ))
    mean = (sum(c.score for c in per_case) / len(per_case)) if per_case else 0.0
    return _VariantSpec(
        index=index,
        category=(
            {"category_id": category_id, "category_name": category_id}
            if category_id else None
        ),
        agent_dir=Path("/tmp"),
        eval_result=EvaluationResult(
            score=mean,
            passed=sum(1 for c in per_case if c.passed),
            failed=sum(1 for c in per_case if not c.passed),
            per_case=per_case,
        ),
        strategy=EvolutionStrategy(
            target_files=["workflow.py"],
            optimization_goal="stub", proposed_changes="", rationale="",
        ),
    )


# --------------------------------------------------------------------------- #
# Stub editor — copies workspace, marks every variant with a per-variant
# optimization_goal so the manager's selection can pick a distinct winner.
# --------------------------------------------------------------------------- #


class _StubEditor:
    """Mimics AgentEditor.apply: copies base_dir/task_agent -> out_dir, no
    LLM. ``fail_predicate`` (out_dir -> bool) forces the edit-failed path
    for matching variants."""

    def __init__(self, *, fail_predicate=None) -> None:
        self.calls = 0
        self.fail_predicate = fail_predicate or (lambda out_dir: False)

    def apply(self, feedback, base_dir, out_dir, *, context=None):
        self.calls += 1
        out = Path(out_dir)
        src = Path(base_dir) / "task_agent"
        dst = out / "task_agent"
        if dst.exists():
            shutil.rmtree(dst)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dst)
        # Record which call produced this directory so the test can tell
        # winner from loser by reading the file.
        (dst / "EDIT_TAG.txt").write_text(out.name, encoding="utf-8")
        strategy = EvolutionStrategy(
            target_files=["workflow.py"],
            optimization_goal=f"stub edit for {out.name}",
            proposed_changes="stub",
            rationale="",
        )
        if self.fail_predicate(out):
            return EditResult(
                success=False, errors=["stub forced fail"], strategy=strategy,
            )
        return EditResult(success=True, edited_files=["workflow.py"], strategy=strategy)


# --------------------------------------------------------------------------- #
# Stub evaluator — emits travel-shaped CaseResult.details so the real
# travel_error_categorizer produces deterministic categories, and the per-
# case score depends on which directory is being evaluated so variant
# selection has a unique winner.
# --------------------------------------------------------------------------- #


def _failing_dim_case(case_id: str, dim: str, score: float = 0.3) -> CaseResult:
    return CaseResult(
        case_id=case_id,
        passed=False,
        score=score,
        details={
            "commonsense_score": score,
            "hard_score": score,
            "composite_score": score,
            "dimension_details": {
                dim: {
                    "score": 0.0,
                    "checks": [
                        {"name": f"{dim}_check_1", "passed": False, "message": "oops"},
                    ],
                }
            },
            "hard_constraints": {},
            "failed_checks": [],
        },
    )


def _passing_case(case_id: str, *, score: float = 1.0) -> CaseResult:
    return CaseResult(
        case_id=case_id,
        passed=True,
        score=score,
        details={
            "commonsense_score": score,
            "hard_score": score,
            "composite_score": score,
            "dimension_details": {},
            "hard_constraints": {},
            "failed_checks": [],
        },
    )


class _StubEvaluator:
    """Per-directory score model:

    - The seed (round_000) is evaluated on the full train set with a mix
      of failures across two dimensions, so when expanded it produces
      categories the manager can spawn variants for.
    - intermediate / var_0 / var_1 get directory-specific mean scores so
      _select_winner has a clear winner. By default ``var_1`` wins.
    """

    def __init__(self, *, winner_dir_name: str = "var_1") -> None:
        self.winner_dir_name = winner_dir_name
        self.run_calls = 0
        self.seen_dirs: list[str] = []

    def run(self, round_dir, benchmark_dir, *, case_ids=None):
        self.run_calls += 1
        name = Path(round_dir).name
        self.seen_dirs.append(name)
        case_ids = list(case_ids or [])

        if name == "round_000":
            # Seed: mix of failures so categorize_errors produces ≥2 cats.
            per_case = []
            for i, cid in enumerate(case_ids):
                if i % 3 == 0:
                    per_case.append(_failing_dim_case(cid, "Time Feasibility"))
                elif i % 3 == 1:
                    per_case.append(_failing_dim_case(cid, "Route Consistency"))
                else:
                    per_case.append(_passing_case(cid))
        elif name == "intermediate":
            # Stage A intermediate: keep two distinct categories failing so
            # the categorizer produces ≥2 categories and the manager can
            # spawn num_variants=2 variants.
            per_case = []
            for i, cid in enumerate(case_ids):
                if i % 3 == 0:
                    per_case.append(_failing_dim_case(cid, "Time Feasibility", score=0.4))
                elif i % 3 == 1:
                    per_case.append(_failing_dim_case(cid, "Route Consistency", score=0.4))
                else:
                    per_case.append(_passing_case(cid, score=1.0))
        elif name == self.winner_dir_name:
            # Winner variant: all pass.
            per_case = [_passing_case(cid, score=1.0) for cid in case_ids]
        elif name.startswith("var_"):
            # Loser variants: half pass.
            per_case = [
                _passing_case(cid, score=1.0) if i % 2 else _failing_dim_case(cid, "Time Feasibility")
                for i, cid in enumerate(case_ids)
            ]
        else:
            # Finalize / unknown — score 1.0.
            per_case = [_passing_case(cid, score=1.0) for cid in case_ids]

        passed = sum(1 for c in per_case if c.passed)
        score = sum(c.score for c in per_case) / max(len(per_case), 1)
        return EvaluationResult(
            score=score,
            passed=passed,
            failed=len(per_case) - passed,
            per_case=per_case,
        )


# --------------------------------------------------------------------------- #
# Test harness
# --------------------------------------------------------------------------- #


class HGMDualExpandTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="hgm_dual_"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.seed = self.tmp / "seed"
        (self.seed).mkdir()
        (self.seed / "workflow.py").write_text(
            "def run_task(task):\n    return None\n", encoding="utf-8"
        )
        self.experiment = self.tmp / "exp"
        self.experiment.mkdir()

    def _make_manager(self, *, variant_parallelism=2, **overrides):
        defaults = dict(
            eval_budget=80,
            init_expansions=1,
            eval_batch_size=2,
            alpha=0.6,
            seed=7,
            num_variants=2,
            intra_expand_eval_size=4,
            variant_parallelism=variant_parallelism,
            categorizer_max_representative_errors=3,
            min_remaining_budget_for_dual=0,
            error_categorizer="projects.travel.travel_error_categorizer:categorize_errors",
        )
        defaults.update(overrides)
        return HGMDualManager(**defaults)

    def _evolve(self, *, fail_predicate=None, **overrides):
        manager = self._make_manager(**overrides)
        editor = _StubEditor(fail_predicate=fail_predicate)
        evaluator = _StubEvaluator()
        outcome = manager.evolve(
            editor=editor,
            evaluator=evaluator,
            gatherer=DefaultFeedbackGatherer(),
            seed_dir=self.seed,
            benchmark_dir=self.tmp / "bench",
            experiment_dir=self.experiment,
            max_rounds=30,
            score_target=None,
            train_case_ids=[f"c{i}" for i in range(12)],
            eval_case_ids=None,
        )
        return manager, editor, evaluator, outcome

    # ---- Layout & winner promotion ----
    def test_expansion_writes_variants_layout_and_promotes_winner(self) -> None:
        manager, editor, evaluator, _ = self._evolve()
        # round_001 should be the first dual expansion.
        round1 = self.experiment / "round_001"
        self.assertTrue(round1.is_dir())
        variants = round1 / "variants"
        self.assertTrue((variants / "intermediate" / "task_agent").is_dir())
        self.assertTrue((variants / "var_0" / "task_agent").is_dir())
        self.assertTrue((variants / "var_1" / "task_agent").is_dir())
        self.assertTrue((variants / "selection.json").is_file())
        # Winner copied into round_001/task_agent — EDIT_TAG identifies it.
        tag = (round1 / "task_agent" / "EDIT_TAG.txt").read_text().strip()
        self.assertEqual(tag, "var_1")

    # ---- Each variant directory holds its own steering context + eval ----
    def test_variant_directory_has_context_and_eval_artifacts(self) -> None:
        self._evolve()
        var0 = self.experiment / "round_001" / "variants" / "var_0"
        ctx = (var0 / "variant_context.txt").read_text(encoding="utf-8")
        # The focused context names the variant's assigned failure category.
        self.assertIn("Assigned failure category", ctx)
        # The category text references one of the two failing dimensions.
        self.assertTrue(
            "Time Feasibility" in ctx or "Route Consistency" in ctx,
            ctx[:300],
        )
        # Each variant gets its own eval json.
        self.assertTrue((var0 / "variant_eval.json").is_file())

    # ---- Budget accounting ----
    def test_budget_debits_match_intra_expand_evals(self) -> None:
        manager, _, evaluator, _ = self._evolve(init_expansions=1, eval_budget=200)
        # First expansion: 1 Stage A eval + up to 2 variant evals × 4 cases
        # = up to 12 cases charged to the budget before any plain _evaluate
        # batches run.
        # We can't assert exact spend at this point (the bandit loop also
        # spends), but we can assert the stub evaluator was called for the
        # intermediate and both variants, on top of the seed pre-eval.
        names = evaluator.seen_dirs
        self.assertIn("intermediate", names)
        self.assertIn("var_0", names)
        self.assertIn("var_1", names)
        # Stage A intermediate sample size matches intra_expand_eval_size.
        # We can't grep the call args, but the stub records each run so the
        # first three eval calls (after the seed) are the intra-expand ones.
        self.assertEqual(names[0], "round_000")  # seed pre-eval

    # ---- Tight budget skips Stage B ----
    def test_tight_budget_skips_stage_b(self) -> None:
        manager, _, evaluator, _ = self._evolve(
            eval_budget=4,
            init_expansions=1,
            intra_expand_eval_size=4,
            min_remaining_budget_for_dual=100,  # always too tight
        )
        # With Stage B skipped, no variants directory should be evaluated:
        # the stub evaluator never sees "intermediate" or "var_*". A node
        # is still added off the root (the intermediate gets promoted as
        # the child since Stage B is skipped).
        names = evaluator.seen_dirs
        self.assertNotIn("intermediate", names)
        self.assertNotIn("var_0", names)
        self.assertNotIn("var_1", names)
        round1 = self.experiment / "round_001"
        self.assertTrue((round1 / "task_agent").is_dir())

    # ---- Stage A edit failure short-circuits to edit_failed node ----
    def test_stage_a_failure_marks_node_edit_failed(self) -> None:
        # Fail Stage A for the FIRST expansion only — subsequent
        # expansions must still succeed so the evolve loop terminates.
        first_intermediate = {"hit": False}

        def fp(out_dir: Path) -> bool:
            if out_dir.name == "intermediate" and not first_intermediate["hit"]:
                first_intermediate["hit"] = True
                return True
            return False

        manager, _, _, _ = self._evolve(fail_predicate=fp)
        # round_001 was the first expansion attempted.
        n1 = manager._tree.nodes[1]
        self.assertTrue(n1.edit_failed)
        # No variants directory because Stage A failed.
        self.assertFalse((self.experiment / "round_001" / "variants" / "var_0").exists())

    # ---- All variants fail: Stage A intermediate stays as winner ----
    def test_all_variants_fail_keeps_stage_a_intermediate(self) -> None:
        def fp(out_dir: Path) -> bool:
            return out_dir.name.startswith("var_")
        manager, _, _, _ = self._evolve(fail_predicate=fp)
        round1 = self.experiment / "round_001"
        tag = (round1 / "task_agent" / "EDIT_TAG.txt").read_text().strip()
        self.assertEqual(tag, "intermediate")
        # selection.json's pool includes the Stage A intermediate as winner.
        import json as _json
        sel = _json.loads((round1 / "variants" / "selection.json").read_text())
        self.assertEqual(sel["winner"]["index"], -1)
        # Failed variants are not in the pool (edit failure excludes them).
        self.assertEqual([s["index"] for s in sel["pool"] if s["index"] >= 0], [])

    # ---- Stage B win records composite "Stage A: … / Stage B (…): …" ----
    def test_stage_b_winner_records_composite_optimization_goal(self) -> None:
        manager, _, _, _ = self._evolve()
        # var_1 wins round_001 by the stub's canned scoring; node 1 = that
        # promoted child. The composite is stored on its AgentFeedback, and
        # also reflected in selection.json.
        goal = manager._feedback[1].strategy.optimization_goal
        self.assertIn("Stage A: ", goal)
        self.assertIn("Stage B (", goal)
        self.assertIn("specialist): ", goal)
        self.assertGreaterEqual(len(goal.split("\n")), 2)
        # The Stage A line carries the intermediate's stub goal verbatim.
        self.assertIn("stub edit for intermediate", goal)
        self.assertIn("stub edit for var_1", goal)

        import json as _json
        sel = _json.loads(
            (self.experiment / "round_001" / "variants" / "selection.json").read_text()
        )
        self.assertIn("Stage A: ", sel["winner"]["optimization_goal"])
        self.assertIn("Stage B (", sel["winner"]["optimization_goal"])
        # Losing variants in the pool keep their original specialist-only goal.
        losers = [s for s in sel["pool"] if s["index"] >= 0 and s["index"] != sel["winner"]["index"]]
        for loser in losers:
            self.assertNotIn("Stage A: ", loser["optimization_goal"])

    # ---- Stage A win leaves optimization_goal as a single-line goal ----
    def test_stage_a_winner_keeps_single_line_optimization_goal(self) -> None:
        # Forcing every variant to fail makes the Stage A intermediate the
        # only viable candidate — composite assembly must NOT fire.
        def fp(out_dir: Path) -> bool:
            return out_dir.name.startswith("var_")
        manager, _, _, _ = self._evolve(fail_predicate=fp)
        goal = manager._feedback[1].strategy.optimization_goal
        self.assertNotIn("Stage A: ", goal)
        self.assertNotIn("Stage B (", goal)
        self.assertNotIn("\n", goal)

    # ---- Multi-line goals render as indented continuations in the lineage ----
    def test_multiline_goal_renders_with_indent_continuation(self) -> None:
        manager, _, _, _ = self._evolve()
        # Render the lineage that an EXPAND off node 1 would see. The stub's
        # composite for node 1 has two lines, so the rendered output should
        # include the indented continuation form.
        ctx = manager._render_expand_context(manager._tree[1])
        self.assertIn("[depth 1] Stage A: stub edit for intermediate", ctx)
        self.assertIn("             Stage B (", ctx)

    # ---- _run_top_k_full_eval dedups against per-node existing evals ----
    def test_full_eval_audit_dedups_against_existing_node_evals(self) -> None:
        import json as _json
        from meta_agent.managers.hgm_tree import HGMNode

        manager = self._make_manager(full_eval_top_k=2)
        manager._benchmark_dir = self.tmp / "bench"
        manager._experiment_dir = self.experiment
        manager._train_case_ids = ["c1", "c2", "c3", "c4"]
        manager._eval_case_ids = ["c5", "c6"]

        # Two nodes with pre-existing evidence on different case subsets.
        # Node A: fully evaluated on all 4 train cases (no held-out).
        # Node B: evaluated on just 2 train cases.
        def _pp(cid, score, passed):
            return CaseResult(case_id=cid, passed=passed, score=score, details={})

        node_a_dir = self.experiment / "round_001"
        node_a_dir.mkdir(parents=True, exist_ok=True)
        node_a = HGMNode(node_id=1, parent_id=0, round_dir=node_a_dir)
        for cid, s, p in [("c1", 1.0, True), ("c2", 1.0, True),
                          ("c3", 0.0, False), ("c4", 1.0, True)]:
            node_a.record(_pp(cid, s, p))

        node_b_dir = self.experiment / "round_002"
        node_b_dir.mkdir(parents=True, exist_ok=True)
        node_b = HGMNode(node_id=2, parent_id=0, round_dir=node_b_dir)
        for cid, s, p in [("c1", 0.5, False), ("c2", 0.8, True)]:
            node_b.record(_pp(cid, s, p))

        # Seed (kept so candidates filter doesn't pick it as a finalist,
        # since mean=0 with n_evals=0). Top-2 finalists should be A then B
        # by mean_utility.
        seed_dir = self.experiment / "round_000"
        seed_dir.mkdir(parents=True, exist_ok=True)
        seed = HGMNode(node_id=0, parent_id=None, round_dir=seed_dir)

        manager._tree.add(seed)
        manager._tree.add(node_a)
        manager._tree.add(node_b)

        # Stub evaluator records what case_ids each call was given.
        class _StubEval:
            def __init__(self): self.calls = []
            def run(self, round_dir, benchmark_dir, *, case_ids=None):
                cids = list(case_ids or [])
                self.calls.append((Path(round_dir).name, cids))
                # New scores for the audit's missing cases — held-out and any
                # missing train cases for node B.
                per_case = []
                for cid in cids:
                    s, p = (1.0, True) if cid != "c6" else (0.0, False)
                    per_case.append(CaseResult(
                        case_id=cid, passed=p, score=s, details={}))
                passed = sum(1 for c in per_case if c.passed)
                return EvaluationResult(
                    score=sum(c.score for c in per_case) / max(len(per_case), 1),
                    passed=passed, failed=len(per_case) - passed,
                    per_case=per_case,
                )

        ev = _StubEval()
        manager._run_top_k_full_eval(ev)

        # Two evaluator calls (one per top-2 finalist).
        self.assertEqual(len(ev.calls), 2)
        # Node A audit call: should only ask for held-out cases (c5, c6) —
        # it has all 4 train cases already.
        name_to_cids = dict(ev.calls)
        self.assertEqual(set(name_to_cids["round_001"]), {"c5", "c6"})
        # Node B audit call: should ask for c3, c4 (missing train) + c5, c6
        # (held-out) — c1 and c2 are already cached on the node.
        self.assertEqual(set(name_to_cids["round_002"]), {"c3", "c4", "c5", "c6"})

        # Node A's full_eval_score.json: 4 cached + 2 new = 6 cases.
        a_audit = _json.loads((node_a_dir / "full_eval_score.json").read_text())
        self.assertEqual(a_audit["n_cases"], 6)
        self.assertEqual(a_audit["audit_pre_existing_evals"], 4)
        self.assertEqual(a_audit["audit_new_evals"], 2)
        # Composite = (1+1+0+1+1+0)/6 = 4/6 = 0.6667
        self.assertAlmostEqual(a_audit["composite_score"], 4.0 / 6.0, places=4)
        self.assertEqual(a_audit["passed"], 4)
        self.assertEqual(a_audit["failed"], 2)

        # Node B's full_eval_score.json: 2 cached + 4 new = 6 cases.
        b_audit = _json.loads((node_b_dir / "full_eval_score.json").read_text())
        self.assertEqual(b_audit["n_cases"], 6)
        self.assertEqual(b_audit["audit_pre_existing_evals"], 2)
        self.assertEqual(b_audit["audit_new_evals"], 4)

    # ---- Balanced-by-type category selection ----
    def test_balanced_by_type_picks_one_of_each_when_available(self) -> None:
        cats = [
            {"category_id": "commonsense__a", "category_type": "commonsense", "failure_rate": 0.9},
            {"category_id": "commonsense__b", "category_type": "commonsense", "failure_rate": 0.5},
            {"category_id": "hard_constraint__x", "category_type": "hard_constraint", "failure_rate": 0.3},
        ]
        m = self._make_manager(category_selection="balanced_by_type")
        picked = m._select_active_categories(cats, k_max=2)
        self.assertEqual(len(picked), 2)
        self.assertEqual({p["category_type"] for p in picked}, {"commonsense", "hard_constraint"})
        # commonsense slot took the highest-failure-rate commonsense category
        self.assertEqual(picked[0]["category_id"], "commonsense__a")
        self.assertEqual(picked[1]["category_id"], "hard_constraint__x")

    def test_balanced_by_type_falls_back_to_available_bucket(self) -> None:
        cats = [
            {"category_id": "commonsense__a", "category_type": "commonsense", "failure_rate": 0.9},
            {"category_id": "commonsense__b", "category_type": "commonsense", "failure_rate": 0.5},
        ]  # no hard_constraint at all
        m = self._make_manager(category_selection="balanced_by_type")
        picked = m._select_active_categories(cats, k_max=2)
        self.assertEqual(len(picked), 2)
        self.assertTrue(all(p["category_type"] == "commonsense" for p in picked))
        self.assertEqual([p["category_id"] for p in picked], ["commonsense__a", "commonsense__b"])

    def test_top_failure_rate_is_default(self) -> None:
        cats = [
            {"category_id": "commonsense__a", "category_type": "commonsense", "failure_rate": 0.9},
            {"category_id": "hard_constraint__x", "category_type": "hard_constraint", "failure_rate": 0.8},
            {"category_id": "commonsense__b", "category_type": "commonsense", "failure_rate": 0.7},
        ]
        m = self._make_manager()  # default policy
        picked = m._select_active_categories(cats, k_max=2)
        self.assertEqual([p["category_id"] for p in picked], ["commonsense__a", "hard_constraint__x"])

    def test_category_selection_unknown_value_raises(self) -> None:
        with self.assertRaises(ValueError):
            self._make_manager(category_selection="nonsense")

    # ---- Per-node level-focused selection (sampling) ----
    _PNL_CATS = [
        {"category_id": "missing_feature", "category_type": "missing_feature",
         "failure_rate": 0.19, "failure_rate_by_level": {1: 0.18, 2: 0.18, 3: 0.25}},
        {"category_id": "user_info", "category_type": "user_info",
         "failure_rate": 0.13, "failure_rate_by_level": {1: 0.12, 2: 0.18, 3: 0.10}},
        # globally tiny (0.10) but dominant within L3 (0.60)
        {"category_id": "suboptimal_coupon", "category_type": "suboptimal_coupon",
         "failure_rate": 0.10, "failure_rate_by_level": {3: 0.60}},
    ]

    def test_focus_level_selects_that_levels_top_categories(self) -> None:
        m = self._make_manager(category_selection="per_node_level")
        # focus L3 -> suboptimal_coupon is the #1 category (invisible to global top-rate)
        picked = m._select_active_categories(
            [dict(c) for c in self._PNL_CATS], k_max=2, focus_level=3)
        self.assertEqual(picked[0]["category_id"], "suboptimal_coupon")
        self.assertTrue(all(p["_focus_level"] == 3 for p in picked))
        # focus L2 -> the L2-rate top (missing_feature/user_info tie -> stable order)
        picked2 = m._select_active_categories(
            [dict(c) for c in self._PNL_CATS], k_max=1, focus_level=2)
        self.assertEqual(picked2[0]["_focus_level"], 2)
        self.assertIn(picked2[0]["category_id"], {"missing_feature", "user_info"})

    def test_focus_level_none_falls_back_to_top_failure_rate(self) -> None:
        m = self._make_manager(category_selection="per_node_level")
        picked = m._select_active_categories(
            [dict(c) for c in self._PNL_CATS], k_max=1, focus_level=None)
        self.assertEqual(picked[0]["category_id"], "missing_feature")  # highest global rate

    def test_sample_focus_level_follows_data_distribution(self) -> None:
        m = self._make_manager(category_selection="per_node_level")
        # 25 L1, 25 L2, 10 L3 ids -> 50/50/20 (i.e. 5:5:2)
        m._train_case_ids = ([f"L1-{i}" for i in range(25)]
                             + [f"L2-{i}" for i in range(25)]
                             + [f"L3-{i}" for i in range(10)])
        from collections import Counter
        draws = Counter(m._sample_focus_level(nid) for nid in range(3000))
        # L3 should be sampled ~10/60 = 17% (well above 0), L1≈L2≈42%
        self.assertGreater(draws[3], 250)          # ~500 expected
        self.assertGreater(draws[1], 900)
        self.assertGreater(draws[2], 900)
        self.assertEqual(set(draws), {1, 2, 3})

    def test_intra_batch_reserves_focus_level_cases(self) -> None:
        m = self._make_manager(category_selection="per_node_level",
                               intra_expand_eval_size=16)
        m._train_case_ids = ([f"L1-{i}" for i in range(25)]
                             + [f"L2-{i}" for i in range(25)]
                             + [f"L3-{i}" for i in range(10)])
        batch = m._intra_sample_cases(node_id=1, focus_level=3)
        self.assertEqual(len(batch), 16)
        n_l3 = sum(1 for c in batch if c.startswith("L3-"))
        self.assertGreaterEqual(n_l3, max(2, 16 // 3))   # guaranteed quota present
        # without a focus level the batch is just a uniform sample (no guarantee)
        plain = m._intra_sample_cases(node_id=1)
        self.assertEqual(len(plain), 16)

    # ----------------------------------------------------------------- #
    # Category-significance selection (select_metric="category_significance")
    # ----------------------------------------------------------------- #

    def test_category_significance_picks_significant_variant(self) -> None:
        """v0 (Time Feasibility specialist) clearly beats SA on TF checks
        -> significant -> v0 wins."""
        # SA: 6/16 cases pass TF dim (one check per case, pass=True for 6 cases)
        sa  = _make_spec_with_dim_checks(
            index=-1, category_id=None,
            case_check_passes=[[True]] * 6 + [[False]] * 10,
        )
        # v0: 14/16 cases pass TF (clear improvement on the assigned category)
        v0  = _make_spec_with_dim_checks(
            index=0, category_id="commonsense__time_feasibility",
            case_check_passes=[[True]] * 14 + [[False]] * 2,
        )
        # v1: 7/16 cases pass — marginal, not significant
        v1  = _make_spec_with_dim_checks(
            index=1, category_id="commonsense__time_feasibility",
            case_check_passes=[[True]] * 7 + [[False]] * 9,
        )
        m = self._make_manager(
            select_metric="category_significance",
            significance_alpha=0.05, bootstrap_n_samples=2000,
        )
        winner = m._select_winner([sa, v0, v1], node_id=1)
        self.assertEqual(winner.index, 0)

    def test_category_significance_falls_back_to_stage_a_marginal(self) -> None:
        """Both variants have only a 1-case lead on their category vs SA ->
        not significant at p<0.05 -> Stage A wins."""
        sa  = _make_spec_with_dim_checks(
            index=-1, category_id=None,
            case_check_passes=[[True]] * 8 + [[False]] * 8,
        )
        v0  = _make_spec_with_dim_checks(
            index=0, category_id="commonsense__time_feasibility",
            case_check_passes=[[True]] * 9 + [[False]] * 7,
        )
        v1  = _make_spec_with_dim_checks(
            index=1, category_id="commonsense__time_feasibility",
            case_check_passes=[[True]] * 9 + [[False]] * 7,
        )
        m = self._make_manager(
            select_metric="category_significance",
            significance_alpha=0.05, bootstrap_n_samples=2000,
        )
        winner = m._select_winner([sa, v0, v1], node_id=1)
        self.assertEqual(winner.index, -1)

    def test_category_significance_picks_higher_p_among_multiple_significant(self) -> None:
        """Two variants both significant; max_p_greater tiebreak picks the
        one with the largest P(V > SA)."""
        sa  = _make_spec_with_dim_checks(
            index=-1, category_id=None,
            case_check_passes=[[True]] * 6 + [[False]] * 10,
        )
        # v0: strong improvement (13/16)
        v0  = _make_spec_with_dim_checks(
            index=0, category_id="commonsense__time_feasibility",
            case_check_passes=[[True]] * 13 + [[False]] * 3,
        )
        # v1: stronger improvement (16/16)
        v1  = _make_spec_with_dim_checks(
            index=1, category_id="commonsense__time_feasibility",
            case_check_passes=[[True]] * 16,
        )
        m = self._make_manager(
            select_metric="category_significance",
            significance_alpha=0.05, bootstrap_n_samples=4000,
            category_tiebreak="max_p_greater",
        )
        winner = m._select_winner([sa, v0, v1], node_id=1)
        self.assertEqual(winner.index, 1)

    def test_category_significance_generic_category_falls_back(self) -> None:
        """Generic category has no per-case check mapping -> variant excluded
        -> Stage A wins."""
        sa  = _make_spec_with_dim_checks(
            index=-1, category_id=None,
            case_check_passes=[[True]] * 8 + [[False]] * 8,
        )
        v0  = _make_spec_with_dim_checks(
            index=0, category_id="generic__runtime_error",
            case_check_passes=[[True]] * 16,
        )
        m = self._make_manager(
            select_metric="category_significance",
            significance_alpha=0.05, bootstrap_n_samples=2000,
        )
        winner = m._select_winner([sa, v0], node_id=1)
        self.assertEqual(winner.index, -1)

    def test_category_significance_no_applicable_cases_falls_back(self) -> None:
        """A hard_constraint variant where no case has that constraint in
        Stage A's details -> 0 applicable trials -> variant excluded ->
        Stage A wins."""
        sa  = _make_spec_with_dim_checks(
            index=-1, category_id=None,
            case_check_passes=[[True]] * 8 + [[False]] * 8,
            dim_name="Time Feasibility",
        )
        # Variant assigned to hard_constraint__restaurant but SA's per_case
        # only has dimension_details (no hard_constraints) -> 0 applicable.
        v0  = _make_spec_with_dim_checks(
            index=0, category_id="hard_constraint__restaurant",
            case_check_passes=[[True]] * 16,
        )
        m = self._make_manager(
            select_metric="category_significance",
            significance_alpha=0.05, bootstrap_n_samples=2000,
        )
        winner = m._select_winner([sa, v0], node_id=1)
        self.assertEqual(winner.index, -1)

    def test_category_significance_crashed_variant_treated_as_failure(self) -> None:
        """Variant with empty details (crashed) is treated as having failed
        every applicable check -> P(V > SA) is very low -> Stage A wins."""
        sa  = _make_spec_with_dim_checks(
            index=-1, category_id=None,
            case_check_passes=[[True]] * 12 + [[False]] * 4,
        )
        # Build v0 with same case_ids but EMPTY details (simulating crash)
        crashed_cases = [
            CaseResult(case_id=str(i), passed=False, score=0.0, details={})
            for i in range(16)
        ]
        v0 = _VariantSpec(
            index=0,
            category={"category_id": "commonsense__time_feasibility",
                      "category_name": "Time Feasibility"},
            agent_dir=Path("/tmp"),
            eval_result=EvaluationResult(
                score=0.0, passed=0, failed=16, per_case=crashed_cases,
            ),
            strategy=EvolutionStrategy(
                target_files=["workflow.py"],
                optimization_goal="stub", proposed_changes="", rationale="",
            ),
        )
        m = self._make_manager(
            select_metric="category_significance",
            significance_alpha=0.05, bootstrap_n_samples=2000,
        )
        winner = m._select_winner([sa, v0], node_id=1)
        self.assertEqual(winner.index, -1)

    def test_category_significance_hard_constraint_per_check_counting(self) -> None:
        """Hard-constraint category counts EACH constraint instance as a
        trial, not case-level all-pass. Variant fixes 4 of 5 failing
        restaurant constraints -> significant -> v0 wins."""
        # SA: 5 cases with 2 restaurant constraints each, 5 failing in total
        sa_constraints = [
            [("restaurant_must_eat_named", True), ("restaurant_tag_nearby", False)],
            [("restaurant_must_eat_named", False), ("restaurant_tag_nearby", True)],
            [("restaurant_must_eat_named", False), ("restaurant_tag_nearby", False)],
            [("restaurant_must_eat_named", False), ("restaurant_tag_nearby", True)],
            [("restaurant_must_eat_named", True), ("restaurant_tag_nearby", True)],
        ]
        sa = _make_spec_with_hard_constraint(
            index=-1, category_id=None, case_constraints=sa_constraints,
        )
        # v0: same cases but fixes 4 of 5 failing constraints
        v0_constraints = [
            [("restaurant_must_eat_named", True), ("restaurant_tag_nearby", True)],
            [("restaurant_must_eat_named", True), ("restaurant_tag_nearby", True)],
            [("restaurant_must_eat_named", True), ("restaurant_tag_nearby", True)],
            [("restaurant_must_eat_named", True), ("restaurant_tag_nearby", True)],
            [("restaurant_must_eat_named", True), ("restaurant_tag_nearby", True)],
        ]
        v0 = _make_spec_with_hard_constraint(
            index=0, category_id="hard_constraint__restaurant",
            case_constraints=v0_constraints,
        )
        m = self._make_manager(
            select_metric="category_significance",
            significance_alpha=0.05, bootstrap_n_samples=4000,
        )
        winner = m._select_winner([sa, v0], node_id=1)
        # v0 has 10/10 vs SA's 5/10 — clearly significant
        self.assertEqual(winner.index, 0)

    def test_select_metric_mean_default_unchanged(self) -> None:
        """Sanity: default select_metric='mean' picks the higher-mean
        candidate regardless of significance."""
        sa = _make_spec_with_dim_checks(
            index=-1, category_id=None,
            case_check_passes=[[True]] * 5 + [[False]] * 11,
        )
        v0 = _make_spec_with_dim_checks(
            index=0, category_id="commonsense__time_feasibility",
            case_check_passes=[[True]] * 6 + [[False]] * 10,
        )
        m = self._make_manager()  # default select_metric=mean
        winner = m._select_winner([sa, v0])
        self.assertEqual(winner.index, 0)

    def test_select_metric_validation(self) -> None:
        with self.assertRaises(ValueError):
            self._make_manager(select_metric="nonsense")
        with self.assertRaises(ValueError):
            self._make_manager(select_metric="category_significance",
                                category_tiebreak="bad")
        with self.assertRaises(ValueError):
            self._make_manager(select_metric="category_significance",
                                significance_alpha=1.5)
        with self.assertRaises(ValueError):
            self._make_manager(select_metric="category_significance",
                                bootstrap_n_samples=10)

    # ---- Parallel and sequential paths agree on the winner ----
    def test_parallel_and_sequential_produce_identical_winner(self) -> None:
        # Sequential.
        manager_seq, _, _, _ = self._evolve(variant_parallelism=1)
        tag_seq = (
            self.experiment / "round_001" / "task_agent" / "EDIT_TAG.txt"
        ).read_text().strip()
        # Fresh experiment dir for the parallel run.
        for child in list(self.experiment.iterdir()):
            shutil.rmtree(child) if child.is_dir() else child.unlink()
        manager_par, _, _, _ = self._evolve(variant_parallelism=3)
        tag_par = (
            self.experiment / "round_001" / "task_agent" / "EDIT_TAG.txt"
        ).read_text().strip()
        self.assertEqual(tag_seq, tag_par)
        self.assertEqual(tag_par, "var_1")  # the canned winner


if __name__ == "__main__":
    unittest.main()
