"""HGM with Dual Optimization.

Subclass of ``HGMManager`` that overrides ``_expand`` only. Each expansion
runs two stages:

* **Stage A** — identical to today's HGM expansion: parent → editor.apply
  → intermediate agent (``variants/intermediate/``).
* **Stage B** — evaluate the intermediate, group its failures into error
  categories via a project-specific categorizer, spawn one error-conditioned
  variant per category (each = another editor.apply on top of the
  intermediate), evaluate each variant, and pick the highest-mean candidate
  (Stage A intermediate is in the pool). The winner is copied to the
  round's ``task_agent/`` and registered as the actual HGM child node.

Per-stage evaluation is delegated to ``_evaluate_candidate``, a separate
helper that mirrors ``HGMManager._evaluate`` (one eval call → budget debit).
``_expand`` itself only orchestrates.

Variant generation runs in parallel by default (one thread per variant).
``variant_parallelism: 1`` in the config forces sequential execution.
"""
from __future__ import annotations

import importlib
import json
import random
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from statistics import pvariance
from typing import Any, Optional

import numpy as np

from ..agent_editor import AgentEditor, fallback_strategy
from ..evaluator import Evaluator
from ..failure_report import (
    FailureReportConfig,
    build_failure_report,
    render_failure_report,
)
from ..feedback_gatherer import FeedbackGatherer
from ..models import AgentFeedback, CaseResult, EvaluationResult, EvolutionStrategy
from ..registry import register
from .hgm import HGMManager
from .hgm_tree import HGMNode


def _bootstrap_p_greater(
    s_L: int, n_L: int, s_O: int, n_O: int, rng: np.random.Generator, K: int
) -> float:
    """Parametric Binomial bootstrap. Returns Mann-Whitney-style
    ``P(leader > other)`` with 0.5 tie-split (the standard handling of
    discrete-outcome ties)."""
    if n_L == 0 or n_O == 0:
        return 0.5
    p_L = s_L / n_L
    p_O = s_O / n_O
    L_sim = rng.binomial(n_L, p_L, size=K) / n_L
    O_sim = rng.binomial(n_O, p_O, size=K) / n_O
    return float((L_sim > O_sim).mean() + 0.5 * (L_sim == O_sim).mean())


@dataclass
class _VariantSpec:
    """One candidate produced inside an expansion. ``index == -1`` is
    reserved for the Stage A intermediate."""

    index: int
    category: Optional[dict[str, Any]]
    agent_dir: Path
    eval_result: Optional[EvaluationResult]
    strategy: EvolutionStrategy


@register("manager", "hgm_dual")
class HGMDualManager(HGMManager):
    def __init__(
        self,
        *,
        num_variants: int = 3,
        intra_expand_eval_size: int = 8,
        variant_parallelism: Optional[int] = None,
        select_metric: str = "mean",
        categorizer_max_representative_errors: int = 5,
        min_remaining_budget_for_dual: int = 12,
        error_categorizer: str,
        category_selection: str = "top_failure_rate",
        category_types: Optional[list[str]] = None,
        significance_alpha: float = 0.05,
        bootstrap_n_samples: int = 1000,
        category_tiebreak: str = "max_p_greater",
        **base_kwargs: Any,
    ) -> None:
        super().__init__(**base_kwargs)
        self.num_variants = max(0, num_variants)
        self.intra_expand_eval_size = max(1, intra_expand_eval_size)
        # Default to full parallel over the configured variant count.
        self.variant_parallelism = (
            self.num_variants if variant_parallelism is None
            else max(1, int(variant_parallelism))
        )
        # Per-expansion selection metric. "mean" (default) picks argmax
        # composite mean with variance tiebreak. "category_significance"
        # uses a Binomial bootstrap on the variant's assigned category's
        # individual check pass/fail trials; promotes a variant only when
        # P(variant > Stage A) > 1 - significance_alpha on its category.
        allowed_select_metrics = {"mean", "category_significance"}
        if select_metric not in allowed_select_metrics:
            raise ValueError(
                f"select_metric must be one of {sorted(allowed_select_metrics)}, "
                f"got {select_metric!r}"
            )
        self.select_metric = select_metric
        self.categorizer_max_representative_errors = max(
            1, categorizer_max_representative_errors
        )
        self.min_remaining_budget_for_dual = max(0, min_remaining_budget_for_dual)
        # How to pick `num_variants` categories from the categorizer's
        # (sorted) list. "top_failure_rate" preserves the original behavior;
        # "balanced_by_type" round-robins across category_type buckets to
        # enforce specialist diversity (bucket order from the categorizer's
        # optional `category_type_priority`, else derived — see
        # `_select_active_categories`).
        allowed_selections = {"top_failure_rate", "balanced_by_type"}
        if category_selection not in allowed_selections:
            raise ValueError(
                f"category_selection must be one of {sorted(allowed_selections)}, "
                f"got {category_selection!r}"
            )
        self.category_selection = category_selection
        # Optional allowlist of category_type values eligible for Stage B. When
        # set (e.g. ["commonsense", "hard_constraint"]), categories of any other
        # type (e.g. the synthetic "generic" crash/plan-failed bucket) are
        # excluded before selection — so balancing/top-k only considers these
        # types. ``None`` (default) considers all types.
        self.category_types = (
            [str(t) for t in category_types] if category_types else None
        )

        # category_significance-specific knobs (no-op when select_metric=="mean")
        if not 0.0 < significance_alpha < 1.0:
            raise ValueError(
                f"significance_alpha must be in (0, 1), got {significance_alpha!r}"
            )
        if bootstrap_n_samples < 100:
            raise ValueError(
                f"bootstrap_n_samples must be >= 100, got {bootstrap_n_samples!r}"
            )
        allowed_tiebreaks = {"max_p_greater", "max_pass_diff", "max_composite_mean"}
        if category_tiebreak not in allowed_tiebreaks:
            raise ValueError(
                f"category_tiebreak must be one of {sorted(allowed_tiebreaks)}, "
                f"got {category_tiebreak!r}"
            )
        self.significance_alpha = float(significance_alpha)
        self.bootstrap_n_samples = int(bootstrap_n_samples)
        self.category_tiebreak = category_tiebreak

        mod_path, _, func_name = error_categorizer.partition(":")
        if not mod_path or not func_name:
            raise ValueError(
                "error_categorizer must be 'module.path:function_name', "
                f"got {error_categorizer!r}"
            )
        _cat_mod = importlib.import_module(mod_path)
        self._categorize_errors = getattr(_cat_mod, func_name)
        # Per-case category check function — only used by
        # select_metric=="category_significance". Resolved BY CONVENTION from
        # the same module as the categorizer: the module may export
        # ``per_case_category_checks(case_details, category_id) -> list[bool]``.
        # Keeps this manager domain-agnostic (no hardcoded schema knowledge);
        # ``None`` is fine when select_metric=="mean".
        self._category_checks_fn = getattr(
            _cat_mod, "per_case_category_checks", None
        )
        if (
            self.select_metric == "category_significance"
            and self._category_checks_fn is None
        ):
            raise ValueError(
                "select_metric='category_significance' requires the categorizer "
                f"module {mod_path!r} to export a 'per_case_category_checks' "
                "function; none found."
            )
        # Optional project-declared bucket priority for "balanced_by_type"
        # selection — a list of category_type strings, resolved by convention
        # from the categorizer module. When absent, the manager derives an
        # order from the categories themselves (see _select_active_categories).
        self._category_type_priority = getattr(
            _cat_mod, "category_type_priority", None
        )

        # Thread-safety for the parallel Stage B path.
        self._budget_lock = threading.Lock()

    def _min_budget_to_expand(self) -> int:
        """A dual expansion runs an intra-evaluation (Stage A, then variants),
        so it needs at least ``intra_expand_eval_size`` of remaining budget to
        be funded. Below that the main loop early-stops rather than committing
        un-evaluated nodes (which would still cost a Stage A editor call)."""
        return self.intra_expand_eval_size

    # ------------------------------------------------------------------ #
    # Overridden EXPAND — orchestrates Stage A + Stage B + selection.
    # ------------------------------------------------------------------ #

    def _expand(
        self,
        parent_id: int,
        editor: AgentEditor,
        gatherer: FeedbackGatherer,
    ) -> int:
        # NOTE: ``evaluator`` is captured off ``self._evaluator`` set in the
        # subclass override of ``evolve`` (see below).
        evaluator = self._evaluator  # type: ignore[attr-defined]

        parent = self._tree[parent_id]
        node_id = self._next_id
        self._next_id += 1
        out_dir = self._experiment_dir / f"round_{node_id:03d}"
        variants_root = out_dir / "variants"
        (out_dir / "logs").mkdir(parents=True, exist_ok=True)
        variants_root.mkdir(parents=True, exist_ok=True)

        # ---------- Stage A: today's HGM edit ---------- #
        intermediate_dir = variants_root / "intermediate"
        (intermediate_dir / "logs").mkdir(parents=True, exist_ok=True)
        context_a = self._render_expand_context(parent)
        res_a = editor.apply(
            self._feedback.get(parent_id),
            parent.round_dir,
            intermediate_dir,
            context=context_a,
        )
        strategy_a = res_a.strategy or fallback_strategy()

        if not res_a.success:
            node = HGMNode(node_id=node_id, parent_id=parent_id, round_dir=out_dir)
            node.edit_failed = True
            self._tree.add(node)
            self._feedback[node_id] = self._synth_failed_edit_feedback(
                node_id, parent_id, strategy_a, res_a.errors, out_dir
            )
            print(
                f"node {node_id}: EXPAND (dual) from {parent_id} — Stage A "
                f"edit FAILED ({res_a.errors[0][:80] if res_a.errors else '?'})",
                flush=True,
            )
            return node_id

        # ---------- Decide whether to do Stage B ---------- #
        remaining = self.eval_budget - self._budget_spent
        do_dual = (
            self.num_variants > 0
            and remaining >= self.min_remaining_budget_for_dual
        )

        sample_cids = self._intra_sample_cases(node_id)
        ev_a: Optional[EvaluationResult] = None
        categories: list[dict[str, Any]] = []
        if do_dual and sample_cids:
            ev_a = self._evaluate_candidate(intermediate_dir, evaluator, sample_cids)
            (intermediate_dir / "intermediate_eval.json").write_text(
                ev_a.model_dump_json(indent=2), encoding="utf-8"
            )
            categories = list(
                self._categorize_errors(
                    ev_a.per_case,
                    max_representative_errors=self.categorizer_max_representative_errors,
                )
            )

        # ---------- Stage B: error-conditioned variants ---------- #
        k_max = min(
            self.num_variants,
            len(categories),
            max(
                0,
                (self.eval_budget - self._budget_spent)
                // self.intra_expand_eval_size,
            ),
        )
        active_categories = self._select_active_categories(categories, k_max)
        # Stage B specialists edit ON TOP of the Stage A agent, so they must be
        # steered by the STAGE A agent's own feedback (its eval ``ev_a``), not
        # the parent's. Compile it once here and hand it to every variant.
        feedback_a = (
            gatherer.compile(node_id, parent_id, strategy_a, ev_a, intermediate_dir)
            if ev_a is not None
            else None
        )
        stage_a_cases = list(ev_a.per_case) if ev_a is not None else []
        # Summarize the Stage A intermediate now (before Stage B) so its
        # behavior memory exists for the variant steering context — the Stage B
        # specialist edits ON TOP of this intermediate, so it is the variant's
        # true parent. (+1 summarizer call per dual expansion when enabled.)
        # Guard: only when Stage B will actually run variants (active categories
        # + sample cases) — otherwise the intermediate is never edited, so its
        # summary would be wasted.
        will_run_stage_b = bool(active_categories) and bool(sample_cids)
        if self._summarizer is not None and ev_a is not None and will_run_stage_b:
            try:
                self._summarizer.summarize(
                    round_dir=intermediate_dir,
                    parent_round_dir=parent.round_dir,
                    eval_dir=intermediate_dir,
                    eval_result=ev_a,
                    node_id=node_id,
                    parent_id=parent_id,
                )
            except Exception as exc:  # noqa: BLE001
                print(
                    f"[summarizer] intermediate summary failed for node {node_id}: {exc!r}",
                    flush=True,
                )
        candidates = self._run_variants(
            parent=parent,
            parent_id=parent_id,
            intermediate_dir=intermediate_dir,
            variants_root=variants_root,
            categories=active_categories,
            editor=editor,
            evaluator=evaluator,
            sample_cids=sample_cids,
            feedback_a=feedback_a,
            stage_a_cases=stage_a_cases,
        )

        # ---------- Selection ---------- #
        pool: list[_VariantSpec] = list(candidates)
        if ev_a is not None:
            pool.append(
                _VariantSpec(
                    index=-1,
                    category=None,
                    agent_dir=intermediate_dir,
                    eval_result=ev_a,
                    strategy=strategy_a,
                )
            )
        if pool:
            winner = self._select_winner(pool, node_id=node_id)
        else:
            # No Stage B (no budget or no categories) and no Stage A eval.
            winner = _VariantSpec(
                index=-1,
                category=None,
                agent_dir=intermediate_dir,
                eval_result=None,
                strategy=strategy_a,
            )

        # When a Stage B variant wins, compose a two-line optimization_goal
        # so descendants' lineage block shows BOTH the Stage A edit and the
        # specialist layered on top (with category attribution). When Stage A
        # wins, the intermediate's goal is already the only contribution.
        if winner.index != -1 and winner.category is not None:
            cat_name = winner.category.get("category_name", "?")
            stage_a_goal = (strategy_a.optimization_goal or "(empty)").strip()
            stage_b_goal = (winner.strategy.optimization_goal or "(empty)").strip()
            winner.strategy.optimization_goal = (
                f"Stage A: {stage_a_goal}\n"
                f"Stage B ({cat_name} specialist): {stage_b_goal}"
            )

        self._promote_winner(winner, out_dir, pool)

        # ---------- Register child in the HGM tree ---------- #
        node = HGMNode(node_id=node_id, parent_id=parent_id, round_dir=out_dir)
        if winner.eval_result is not None:
            for case in winner.eval_result.per_case:
                node.record(case)
            # The winner's intra batch is reused as this node's evaluation, so
            # it counts toward the widening schedule (like a top-up). The
            # losing variants' trials charge _budget_spent only, not this.
            self._node_evals_spent += len(winner.eval_result.per_case)
        self._tree.add(node)
        self._feedback[node_id] = gatherer.compile(
            node_id,
            parent_id,
            winner.strategy,
            self._build_eval_result(node),
            out_dir,
        )
        self._write_node_sidecar(node)

        # Fire the behavior summarizer for the winning agent's eval. The
        # winner's traces live in its variant subdir (variants/var_k/ or
        # variants/intermediate/), not the canonical round_dir — pass
        # ``eval_dir=winner.agent_dir`` so the summarizer reads them from
        # the right place. Only fires when the winner actually has an eval
        # result; otherwise the base ``_evaluate`` will summarize when the
        # bandit later evaluates this node.
        if (
            self._summarizer is not None
            and winner.eval_result is not None
            and node.parent_id is not None
        ):
            try:
                self._summarizer.summarize(
                    round_dir=out_dir,
                    parent_round_dir=parent.round_dir,
                    eval_dir=winner.agent_dir,
                    eval_result=winner.eval_result,
                    node_id=node.node_id,
                    parent_id=node.parent_id,
                )
            except Exception as exc:  # noqa: BLE001
                print(
                    f"[summarizer] unexpected error on dual node {node.node_id}: {exc!r}",
                    flush=True,
                )

        cat_id = (winner.category or {}).get("category_id") if winner.category else "stage_a"
        winner_mean = (
            winner.eval_result.score if winner.eval_result is not None else float("nan")
        )
        print(
            f"node {node_id}: EXPAND (dual) from {parent_id} — winner={cat_id} "
            f"mean={winner_mean:.3f} ({len(candidates)} variant(s), pool={len(pool)})",
            flush=True,
        )
        return node_id

    # ------------------------------------------------------------------ #
    # Variant generation — parallel by default, sequential when pool == 1.
    # ------------------------------------------------------------------ #

    def _run_variants(
        self,
        *,
        parent: HGMNode,
        parent_id: int,
        intermediate_dir: Path,
        variants_root: Path,
        categories: list[dict[str, Any]],
        editor: AgentEditor,
        evaluator: Evaluator,
        sample_cids: list[str],
        feedback_a: Optional[AgentFeedback] = None,
        stage_a_cases: Optional[list[CaseResult]] = None,
    ) -> list[_VariantSpec]:
        if not categories or not sample_cids:
            return []

        sibling_names = [c.get("category_name", "?") for c in categories]

        # De-duplicate Stage B context: each variant is a single-category
        # specialist and already gets that category's examples (query → plan →
        # what failed) in its steering context (_render_variant_context). So
        # strip the BROAD failure_report from the feedback digest here — keeping
        # both would repeat ~all the failure analysis in every variant prompt.
        # All other Stage A feedback (scores, tool errors, exceptions) is kept.
        feedback_focused = (
            feedback_a.model_copy(update={"failure_report": {}})
            if feedback_a is not None
            else None
        )

        def _build_and_eval(args: tuple[int, dict[str, Any]]) -> Optional[_VariantSpec]:
            k, category = args
            var_dir = variants_root / f"var_{k}"
            (var_dir / "logs").mkdir(parents=True, exist_ok=True)
            ctx = self._render_variant_context(
                parent=parent,
                category=category,
                sibling_category_names=[
                    n for i, n in enumerate(sibling_names) if i != k
                ],
                stage_a_cases=stage_a_cases or [],
                intermediate_dir=intermediate_dir,
            )
            (var_dir / "variant_context.txt").write_text(ctx, encoding="utf-8")
            # Steer with the Stage A agent's feedback (the agent being edited),
            # not the parent's — variants are built off ``intermediate_dir``.
            # failure_report stripped (see feedback_focused above); the focused
            # category's examples are already in ``ctx``.
            res = editor.apply(
                feedback_focused,
                intermediate_dir,
                var_dir,
                context=ctx,
            )
            if not res.success:
                (var_dir / "edit_errors.txt").write_text(
                    "\n".join(res.errors), encoding="utf-8"
                )
                return None
            ev = self._evaluate_candidate(var_dir, evaluator, sample_cids)
            (var_dir / "variant_eval.json").write_text(
                ev.model_dump_json(indent=2), encoding="utf-8"
            )
            return _VariantSpec(
                index=k,
                category=category,
                agent_dir=var_dir,
                eval_result=ev,
                strategy=res.strategy or fallback_strategy(),
            )

        items = list(enumerate(categories))
        pool_size = max(1, min(self.variant_parallelism, len(items)))
        candidates: list[_VariantSpec] = []
        if pool_size == 1:
            for item in items:
                spec = _build_and_eval(item)
                if spec is not None:
                    candidates.append(spec)
        else:
            with ThreadPoolExecutor(max_workers=pool_size) as ex:
                for spec in ex.map(_build_and_eval, items):
                    if spec is not None:
                        candidates.append(spec)
        return candidates

    # ------------------------------------------------------------------ #
    # Evaluation helper — the SEPARATE eval method (no HGM-tree side
    # effects). Mirrors HGMManager._evaluate's eval + debit step.
    # ------------------------------------------------------------------ #

    def _evaluate_candidate(
        self,
        agent_dir: Path,
        evaluator: Evaluator,
        case_ids: list[str],
    ) -> EvaluationResult:
        result = evaluator.run(agent_dir, self._benchmark_dir, case_ids=case_ids)
        with self._budget_lock:
            self._budget_spent += len(result.per_case)
        return result

    # ------------------------------------------------------------------ #
    # Pure helpers — sampling, selection, prompt assembly, promotion.
    # ------------------------------------------------------------------ #

    def _select_active_categories(
        self, categories: list[dict[str, Any]], k_max: int
    ) -> list[dict[str, Any]]:
        """Pick up to ``k_max`` categories for Stage B variants.

        Two policies, controlled by ``self.category_selection``:

        - ``"top_failure_rate"`` (default) — take the k highest-failure-rate
          categories. Categories arrive already sorted by the categorizer.
        - ``"balanced_by_type"`` — round-robin across category_type buckets,
          within each bucket take by failure_rate descending. Ensures
          specialists address distinct failure flavors. Bucket order comes
          from the categorizer's optional ``category_type_priority`` (a list
          of category_type strings); any types it doesn't name — and the
          whole order when it's absent — are sorted by each bucket's strongest
          failure_rate, with the shared ``"generic"`` (synthetic crash /
          missing-data) bucket last. Falls back to filling from non-empty
          buckets if some types have no failures, so ``num_variants`` slots
          are always filled when categories exist. Domain-agnostic: no
          project-specific type names are baked in here.
        """
        if k_max <= 0:
            return []
        # Restrict to the allowed category types (e.g. exclude "generic") before
        # any selection policy runs.
        if self.category_types is not None:
            categories = [
                c for c in categories if c.get("category_type") in self.category_types
            ]
            if not categories:
                return []
        if self.category_selection == "balanced_by_type":
            from collections import defaultdict
            buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for c in categories:
                buckets[c["category_type"]].append(c)
            # Round-robin bucket order: honor the categorizer's declared
            # ``category_type_priority`` first (those present, in order), then
            # append any remaining types sorted by their strongest failure_rate
            # — with the shared "generic" bucket last. No project-specific type
            # names are hardcoded here.
            explicit = [
                t for t in (self._category_type_priority or []) if t in buckets
            ]

            def _fallback_key(type_key: str) -> tuple:
                best = max(
                    (c.get("failure_rate", 0.0) for c in buckets[type_key]),
                    default=0.0,
                )
                return (type_key == "generic", -best, type_key)

            extra = sorted(
                (t for t in buckets if t not in explicit), key=_fallback_key
            )
            order = explicit + extra
            out: list[dict[str, Any]] = []
            while len(out) < k_max:
                took = False
                for type_key in order:
                    if buckets[type_key]:
                        out.append(buckets[type_key].pop(0))
                        took = True
                        if len(out) >= k_max:
                            break
                if not took:
                    break
            return out
        # default: top_failure_rate
        return categories[:k_max]

    def _intra_sample_cases(self, node_id: int) -> list[str]:
        """A deterministic sample of train cases shared across Stage A and
        every Stage B variant for this expansion. Seeded per expansion so
        successive expansions use different samples."""
        rng = random.Random(self.seed ^ node_id ^ 0x5BD1E995)
        pool = list(self._train_case_ids)
        if not pool:
            return []
        n = min(self.intra_expand_eval_size, len(pool))
        return rng.sample(pool, n)

    def _select_winner(
        self, pool: list[_VariantSpec], *, node_id: int = 0
    ) -> _VariantSpec:
        """Pick the winning candidate from the expansion pool.

        Dispatches on ``self.select_metric``:
          - ``"mean"`` (default): argmax composite mean with variance
            tiebreak. Original behavior.
          - ``"category_significance"``: parametric Binomial bootstrap on
            each variant's assigned-category individual checks vs Stage A
            on the same cases. Promotes a variant only when
            ``P(V > Stage A) > 1 - significance_alpha``; otherwise falls
            back to Stage A. See ``_select_winner_category_significance``.

        ``node_id`` is used to seed the bootstrap RNG deterministically
        (only relevant for ``category_significance``). Defaults to 0 so
        unit tests can call this method without simulating an expansion.
        """
        scored = [s for s in pool if s.eval_result is not None]
        if not scored:
            return pool[0]
        if len(scored) == 1:
            return scored[0]
        if self.select_metric == "mean":
            return self._select_winner_mean(scored)
        if self.select_metric == "category_significance":
            return self._select_winner_category_significance(
                scored, node_id=node_id
            )
        raise ValueError(f"Unknown select_metric: {self.select_metric!r}")

    @staticmethod
    def _select_winner_mean(scored: list[_VariantSpec]) -> _VariantSpec:
        """Highest mean composite, tiebreak by lower variance (more
        consistent). The original (and default) selection rule."""
        def _key(s: _VariantSpec) -> tuple[float, float]:
            ev = s.eval_result
            assert ev is not None
            scores = [c.score for c in ev.per_case] or [ev.score]
            mean = ev.score
            var = pvariance(scores) if len(scores) > 1 else 0.0
            # max mean, min variance.
            return (mean, -var)
        return max(scored, key=_key)

    def _select_winner_category_significance(
        self, scored: list[_VariantSpec], *, node_id: int
    ) -> _VariantSpec:
        """Per-check Binomial bootstrap on each variant's assigned category.

        Algorithm:
          1. Identify Stage A intermediate (``index == -1``). If absent, fall
             back to mean selection.
          2. For each variant V with assigned category C:
             a. Walk Stage A's per_case as the applicability + trial-count
                reference. Extract Stage A's per-check bools for C.
             b. For the same case, extract V's per-check bools — pad with
                ``False`` if V has fewer checks or empty details (e.g. when
                the variant crashed, attributing failures to the variant).
             c. Bootstrap ``P(V > SA)`` over the pooled check counts. V is
                "significant" iff this exceeds ``1 - significance_alpha``.
          3. If no variant is significant, Stage A wins. Otherwise pick from
             the significant set by ``category_tiebreak``.
        """
        stage_a = next((s for s in scored if s.index == -1), None)
        if stage_a is None:
            return self._select_winner_mean(scored)
        variants = [s for s in scored if s.index >= 0]
        if not variants:
            return stage_a

        sa_by_id = {c.case_id: c for c in stage_a.eval_result.per_case}
        seed = (self.seed ^ int(node_id) ^ 0xB007511A) & 0xFFFFFFFF
        rng = np.random.default_rng(seed)
        significant: list[tuple[_VariantSpec, float, int]] = []

        for v in variants:
            cat_id = (v.category or {}).get("category_id") if v.category else None
            if not cat_id:
                continue
            v_by_id = {c.case_id: c for c in v.eval_result.per_case}
            sa_pool: list[bool] = []
            v_pool: list[bool] = []
            for cid, sa_case in sa_by_id.items():
                sa_checks = self._category_checks_fn(
                    sa_case.details or {}, cat_id
                )
                if not sa_checks:
                    continue   # case not applicable to this category
                sa_pool.extend(sa_checks)
                v_case = v_by_id.get(cid)
                if v_case is None or not v_case.details:
                    # Variant missing / crashed → treat as failed every check
                    v_pool.extend([False] * len(sa_checks))
                    continue
                v_checks = self._category_checks_fn(v_case.details, cat_id)
                # Align variant's bool list to Stage A's trial count.
                if len(v_checks) < len(sa_checks):
                    v_checks = v_checks + [False] * (
                        len(sa_checks) - len(v_checks)
                    )
                v_pool.extend(v_checks[: len(sa_checks)])

            n = len(sa_pool)
            if n == 0:
                continue   # no applicable trials for this category
            s_v = sum(1 for p in v_pool if p)
            s_sa = sum(1 for p in sa_pool if p)
            p_greater = _bootstrap_p_greater(
                s_v, n, s_sa, n, rng, self.bootstrap_n_samples
            )
            if p_greater > 1.0 - self.significance_alpha:
                significant.append((v, p_greater, s_v - s_sa))

        if not significant:
            return stage_a
        if len(significant) == 1:
            return significant[0][0]

        if self.category_tiebreak == "max_p_greater":
            return max(significant, key=lambda t: t[1])[0]
        if self.category_tiebreak == "max_pass_diff":
            return max(significant, key=lambda t: t[2])[0]
        # "max_composite_mean"
        return max(significant, key=lambda t: t[0].eval_result.score)[0]

    def _promote_winner(
        self,
        winner: _VariantSpec,
        out_dir: Path,
        pool: list[_VariantSpec],
    ) -> None:
        src = winner.agent_dir / "task_agent"
        dst = out_dir / "task_agent"
        if dst.exists():
            shutil.rmtree(dst)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dst)

        # Promote the winner's eval logs (case_*.json + trace.jsonl + scratch/)
        # into the canonical round folder. In dual mode the evaluator only ever
        # writes into the variant subdir (variants/var_k/logs), so out_dir/logs
        # is otherwise left empty — and downstream readers that key off the
        # round dir read nothing. Notably FeedbackGatherer.compile() (called
        # right after promotion with out_dir) reads round_dir/logs/trace.jsonl;
        # without this copy its trace-derived feedback (tool_usage, llm_calls,
        # tool_error_rate, log_excerpt) is silently empty for every dual round.
        # This runs before that compile() call, so the gatherer sees the trace.
        src_logs = winner.agent_dir / "logs"
        if src_logs.exists():
            dst_logs = out_dir / "logs"
            shutil.copytree(src_logs, dst_logs, dirs_exist_ok=True)

        def _summary(spec: _VariantSpec) -> dict[str, Any]:
            ev = spec.eval_result
            return {
                "index": spec.index,
                "category_id": (spec.category or {}).get("category_id"),
                "category_name": (spec.category or {}).get("category_name"),
                "agent_dir": str(spec.agent_dir),
                "mean_score": (ev.score if ev is not None else None),
                "n_cases": (len(ev.per_case) if ev is not None else 0),
                "optimization_goal": spec.strategy.optimization_goal,
            }

        (out_dir / "variants" / "selection.json").write_text(
            json.dumps(
                {
                    "winner": _summary(winner),
                    "pool": [_summary(s) for s in pool],
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    # ------------------------------------------------------------------ #
    # Stage B steering context — category-focused, no parent source.
    # ------------------------------------------------------------------ #

    def _render_variant_context(
        self,
        *,
        parent: HGMNode,
        category: dict[str, Any],
        sibling_category_names: list[str],
        stage_a_cases: Optional[list[CaseResult]] = None,
        intermediate_dir: Optional[Path] = None,
    ) -> str:
        parts: list[str] = []

        lineage = self._ancestor_goals(parent.node_id)
        if len(lineage) > 1:
            parts.append("## Edits already applied along this lineage (root → parent):")
            for depth, goal in lineage:
                lines = goal.split("\n")
                parts.append(f"  [depth {depth}] {lines[0][:200]}")
                for cont in lines[1:]:
                    parts.append(f"             {cont[:200]}")

        # Lineage behavior memories, newest-first within the token budget. The
        # Stage A intermediate (the agent this variant edits ON TOP of) is the
        # true parent, so it is injected as the most-recent memory, then parent
        # → grandparent → ... up the chain.
        extra_recent = (
            ("Stage A intermediate (the agent you are editing)", intermediate_dir)
            if intermediate_dir is not None
            else None
        )
        memory_block = self._render_lineage_memory(parent, extra_recent=extra_recent)
        if memory_block:
            parts.append(memory_block)

        parts.append(
            "\nAn intermediate edit (Stage A) has already been applied to this "
            "parent. You are editing on TOP of that intermediate."
        )

        cat_name = category.get("category_name", "?")
        # Focused, example-driven view of THIS category only (a diverse pair of
        # the Stage A agent's failing cases shown as query → plan → what failed).
        # Keeps the specialist's context tight — no broad top-N list — which
        # avoids context bloat and dilution.
        focused = build_failure_report(
            list(stage_a_cases or []),
            [category],
            cfg=FailureReportConfig(),
            focus_category_id=category.get("category_id"),
        )
        focused_block = render_failure_report(focused)
        if focused_block:
            parts.append("\n" + focused_block.rstrip())
        else:
            cat_type = category.get("category_type", "?")
            n_fail = category.get("num_failing_samples", 0)
            n_total = category.get("total_samples", 0)
            rate = category.get("failure_rate", 0.0)
            parts.append(f"\n## Assigned error category: {cat_name} ({cat_type})")
            parts.append(
                f"failing on {n_fail}/{n_total} sample(s) — failure rate {rate:.2f}."
            )

        parts.append(
            f"\nYou are a specialist editor for **{cat_name}** failures. "
            "Other specialists are addressing other categories in parallel — "
            "you must fix ONLY this category. Do not address other "
            "dimensions. Keep the change scope minimal."
        )

        if sibling_category_names:
            parts.append(
                "\n## Sibling specialists are handling these other categories:"
            )
            for name in sibling_category_names[:8]:
                parts.append(f"  - {name}")

        return "\n".join(parts)

    # ------------------------------------------------------------------ #
    # evolve() override — capture the evaluator so _expand can see it
    # without changing HGMManager's public signature.
    # ------------------------------------------------------------------ #

    def evolve(
        self,
        editor: AgentEditor,
        evaluator: Evaluator,
        gatherer: FeedbackGatherer,
        seed_dir: Path,
        benchmark_dir: Path,
        experiment_dir: Path,
        max_rounds: int,
        score_target: float | None,
        train_case_ids: Optional[list[str]] = None,
        eval_case_ids: Optional[list[str]] = None,
        summarizer: Any = None,
        # Accepted so main_loop.py's unconditional kwarg doesn't crash this
        # manager (see meta_agent/failure_summarizer.py); forwarded to
        # HGMManager.evolve() below so self._failure_summarizer gets set the
        # same way self._summarizer already does. Dual's own Stage A/B
        # EXPAND/EVALUATE paths aren't specifically wired to call it though
        # (out of scope for now — this manager's variant logic differs
        # enough from vanilla HGM's _evaluate that it wasn't verified here).
        failure_summarizer: Any = None,
    ) -> Any:
        # Stash the evaluator so _expand can call _evaluate_candidate
        # without changing the HGMManager._expand signature.
        self._evaluator = evaluator  # type: ignore[attr-defined]
        # Subprocess oversubscription heuristic (warning only).
        inner = getattr(evaluator, "parallelism", None) or getattr(
            getattr(evaluator, "config", None), "parallelism", None
        )
        if isinstance(inner, int) and self.variant_parallelism * inner > 32:
            print(
                f"warning: variant_parallelism={self.variant_parallelism} × "
                f"evaluator.parallelism={inner} = "
                f"{self.variant_parallelism * inner} concurrent subprocesses; "
                "consider lowering one if the host gets overloaded.",
                flush=True,
            )
        return super().evolve(
            editor=editor,
            evaluator=evaluator,
            gatherer=gatherer,
            seed_dir=seed_dir,
            benchmark_dir=benchmark_dir,
            experiment_dir=experiment_dir,
            max_rounds=max_rounds,
            score_target=score_target,
            train_case_ids=train_case_ids,
            eval_case_ids=eval_case_ids,
            summarizer=summarizer,
            failure_summarizer=failure_summarizer,
        )
