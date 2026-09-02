"""Huxley-Gödel Machine (HGM) evolution manager.

A tree search over agent self-modifications, ported from arXiv 2510.21614
(github.com/metauto-ai/HGM). Contrast with ``HillClimbingManager``'s linear
loop: HGM keeps a *tree* of agents and *decouples* expansion from
evaluation under an adaptive schedule, with a budget counted in agent-task
evaluations rather than rounds.

Algorithm (the paper's Algorithm 1, adapted to continuous [0,1] scores):

    seed the root (round_000) and PRE-EVALUATE it on the full train set
      (free — not charged to the budget; makes the root expandable);
    do `init_expansions` unconditional EXPANDs off the root;
    while loop evaluations spent < eval_budget:
      if budget_spent**alpha >= (real node count - 1) and a node is expandable:
        EXPAND  — pick parent by Thompson sampling over CLADE tallies
                  (Clade Metaproductivity). Only evaluated, positive-mean
                  nodes are expandable.
      else:
        EVALUATE — pick a node by Thompson sampling over its OWN tallies;
                   drip a random `eval_batch_size` batch of un-run cases.
    finalize: re-evaluate the top-k finalists on the FULL train split so
      selection compares them on equal evidence; return the lcb-best node.

Tree ↔ framework mapping: node id == ``round_number`` (a monotonic
counter), parent == ``base_round``. Round dirs are therefore *not*
contiguous-by-depth — every consumer keys off ``base_round``.

A self-modification is ONE editor call: ``editor.apply`` self-diagnoses and
edits in a single step and returns the ``EvolutionStrategy`` summary on
``EditResult.strategy``. The manager only selects the node and builds a
cheap (non-LLM) steering context string.
"""
from __future__ import annotations

import dataclasses
import json
import random
import shutil
from pathlib import Path
from typing import Optional

from ..agent_editor import AgentEditor, fallback_strategy
from ..block_bandit import AdaptiveStrategy, BlockBandit
from ..evaluator import Evaluator, load_cases
from ..feedback_gatherer import FeedbackGatherer, persist_round_artifacts, render_metrics
from ..models import (
    AgentFeedback,
    CaseResult,
    EvaluationResult,
    EvolutionOutcome,
    EvolutionStrategy,
)
from ..registry import register
from ..tree_snapshot import NodeSnapshot, TreeSnapshotWriter
from .hgm_tree import HGMNode, HGMTree

# Rough char<->token proxy for the lineage-memory budget (avoids a hard
# tiktoken dependency in the eval path; the budget is approximate by design).
_CHARS_PER_TOKEN = 4


@register("manager", "hgm")
class HGMManager:
    def __init__(
        self,
        *,
        eval_budget: int = 400,
        init_expansions: int = 5,
        alpha: float = 0.6,
        epsilon: float = 0.25,
        beta_prior: float = 1.0,
        clade_pseudo_count: int = 10000,
        cool_down: bool = False,
        beta: float = 1.0,
        eval_batch_size: int = 16,
        # How many times a single node may be evaluated on the SAME case
        # before it's excluded from that node's candidate pool. 1 (default)
        # is today's exact "at most once, ever" behavior -- every existing
        # config is implicitly this. >1 opts into repeat-eligible dripping:
        # each repeat is folded as its own fresh observation into the
        # node's Beta tallies (HGMNode.record()), not averaged first, so
        # the posterior actually tightens with more repeats. Exists for
        # small (down to 1-case) demos, where breadth across many distinct
        # cases -- the thing that makes the Beta-Bernoulli bandit meaningful
        # at n=20+ -- isn't available to fall back on.
        eval_repeats: int = 1,
        finalize_top_k: int = 5,
        full_eval_top_k: int = 0,
        snapshot_tree: bool = False,
        lineage_memory_token_budget: int = 15000,
        seed: int = 42,
        # Which block _select_block targets for every EXPAND (see
        # meta_agent/block_suggester.py for what each block means).
        # "collaboration" (default -- preserves the stage-1 hardcoded
        # behavior with zero config changes for every existing config)
        # always returns "collaboration_workflow"; "single_agent" always
        # returns "individual_subagent"; "verifiers" always returns
        # "verifiers"; "foundations" always returns "foundation_capability".
        # "non_adaptive" samples uniformly at random among all four
        # (seeded via self._block_rng, so the sequence of choices is
        # reproducible for a given seed) -- "non_adaptive" because it
        # still uses no feedback/performance signal to pick, unlike
        # "adaptive" below. "adaptive" Thompson-samples a block from a
        # BlockBandit (see meta_agent/block_bandit.py) built from the
        # accumulated success/failure mass of every node evaluated so far,
        # grouped by which block its creating edit targeted -- the same
        # Beta-Bernoulli convention hgm_tree.py already uses for node
        # selection, applied to a new axis. See tier_based_hgm.md's
        # "Block-level Thompson sampling" section for the design.
        block_selection_strategy: str = "collaboration",
        # Reward signal the "adaptive" strategy's BlockBandit uses to score
        # each block. "fractional_score" (default -- zero change for every
        # existing config) sums each qualifying node's own accumulated
        # continuous score mass (HGMNode.n_success/n_failure) into its
        # block's tally, so nodes with more evals contribute more.
        # "boolean_increase" instead gives each qualifying node exactly one
        # Bernoulli trial: success iff its mean_utility >= its direct
        # parent's, regardless of how many evals backed that mean -- see
        # meta_agent/block_bandit.py::BlockBandit for the full contrast.
        block_reward_metric: str = "fractional_score",
    ) -> None:
        self.eval_budget = eval_budget
        self.init_expansions = init_expansions
        self.alpha = alpha
        self.epsilon = epsilon
        self.beta_prior = beta_prior
        self.clade_pseudo_count = clade_pseudo_count
        allowed_block_selection_strategies = {
            "collaboration", "single_agent", "verifiers", "foundations",
            "non_adaptive", "adaptive",
        }
        if block_selection_strategy not in allowed_block_selection_strategies:
            raise ValueError(
                "block_selection_strategy must be one of "
                f"{sorted(allowed_block_selection_strategies)}, got "
                f"{block_selection_strategy!r}"
            )
        self.block_selection_strategy = block_selection_strategy
        self.block_reward_metric = block_reward_metric
        # τ scheduler: off by default, matching the reference's committed
        # config.yaml (`cool_down: false`). When on, τ = (B/b)**beta.
        self.cool_down = cool_down
        self.beta = beta
        self.eval_batch_size = max(1, eval_batch_size)
        self.eval_repeats = max(1, eval_repeats)
        # How many top finalists are re-evaluated on the full train split
        # before the final selection (the small-sample-overfit fix).
        self.finalize_top_k = finalize_top_k
        # Sidecar audit: 0 disables; >0 runs that many top finalists on the
        # FULL benchmark (train + held-out eval split) after LCB selection.
        # Does not influence selection — purely for head-to-head comparison.
        self.full_eval_top_k = full_eval_top_k
        # Approx token budget for the lineage behavior-memory block injected into
        # editor steering. Ancestors' full memories are included newest-first up
        # the chain until this budget is reached (~4 chars/token). Applies to
        # vanilla HGM expand + dual Stage A; dual Stage B also prepends the Stage
        # A intermediate's memory as the most-recent.
        self.lineage_memory_token_budget = lineage_memory_token_budget
        # Opt-in time-series snapshots of the whole tree, written after every
        # EXPAND/EVALUATE so the best node at any budget level can be recovered
        # and re-evaluated later (analysis/debug). See meta_agent/tree_snapshot.py
        # and snapshot_eval.py. Off by default — zero behavior change.
        self.snapshot_tree = snapshot_tree
        self.seed = seed

        # Per-run state (reset at the top of evolve()).
        self._tree: HGMTree = HGMTree()
        self._feedback: dict[int, AgentFeedback] = {}
        self._train_case_ids: list[str] = []
        self._eval_case_ids: Optional[list[str]] = None
        self._benchmark_dir: Path = Path()
        self._experiment_dir: Path = Path()
        self._next_id: int = 0
        # Loop evaluations spent — excludes the root's free pre-evaluation
        # and the finalization re-evaluations.
        self._budget_spent: int = 0
        self._task_rng: random.Random = random.Random(seed)
        # Dedicated RNG for the "non_adaptive" block_selection_strategy
        # (uniform-random block choice per EXPAND) -- kept separate from
        # _task_rng so block selection doesn't perturb the task-sampling
        # draw sequence (comparing block_selection_strategy values with
        # everything else held fixed would otherwise also change which
        # tasks get sampled, confounding the comparison).
        self._block_rng: random.Random = random.Random(seed)
        # Blackbox for the "adaptive" block_selection_strategy -- swappable
        # for a different adaptive algorithm later without touching
        # _select_block/_expand beyond this construction line, since both
        # only ever call BlockBandit.select(tree, feedback). Shares
        # self._block_rng with "non_adaptive" (both are the block-selection
        # RNG, kept isolated from _task_rng -- see above).
        self._block_bandit: BlockBandit = BlockBandit(
            beta_prior=beta_prior, rng=self._block_rng,
            reward_metric=block_reward_metric,
        )
        # Snapshot of the most recent adaptive selection (None for every
        # other strategy, and reset at the top of every _select_block call)
        # so _expand can persist it to adaptive_strategy.json without
        # _select_block's return type changing from plain str.
        self._last_block_selection: Optional[AdaptiveStrategy] = None
        # Optional per-round behavior summarizer. When set, ``_evaluate``
        # fires it after a node's feedback is final, and steering contexts
        # inject the lineage's behavior_memory.md files. ``None`` keeps
        # the legacy behavior (no memory written, no memory in prompts).
        self._summarizer: Any = None
        # Optional per-round failure summarizer (see
        # meta_agent/failure_summarizer.py) -- unlike ``_summarizer``, fires
        # even for the root/seed's evaluation (no parent/diff needed) since
        # its job is "what's failing right now", not "what did this edit do".
        self._failure_summarizer: Any = None
        # Optional per-EXPAND block-level improvement suggestion (see
        # meta_agent/block_suggester.py) -- injects a block-scoped, grounded
        # diagnosis into the editor's steering context alongside the
        # (currently hardcoded, see _select_block) selected block. ``None``
        # keeps legacy behavior: no block section beyond the plain "Selected
        # block" line, ``EvolutionStrategy.block`` stays set but unsupported
        # by any suggestion text.
        self._block_suggester: Any = None
        # Time-series tree snapshotter (a no-op unless snapshot_tree is on);
        # (re)created at the top of evolve() once experiment_dir is known.
        self._snapshotter: Optional[TreeSnapshotWriter] = None

    # ------------------------------------------------------------------ #
    # Public API (EvolutionManager protocol)
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
        failure_summarizer: Any = None,
        block_suggester: Any = None,
    ) -> EvolutionOutcome:
        self._benchmark_dir = benchmark_dir
        self._experiment_dir = experiment_dir
        self._eval_case_ids = eval_case_ids
        self._summarizer = summarizer
        self._failure_summarizer = failure_summarizer
        self._block_suggester = block_suggester
        self._tree = HGMTree(
            beta_prior=self.beta_prior,
            clade_pseudo_count=self.clade_pseudo_count,
            rng=random.Random(self.seed),
        )
        self._feedback = {}
        self._next_id = 0
        self._budget_spent = 0
        # Evaluations attributed to committed nodes (main-loop top-ups plus,
        # in the dual manager, the winner's reused intra batch). This drives
        # the widening schedule; ``_budget_spent`` (which also includes the
        # dual manager's throw-away variant trials) only caps total spend.
        self._node_evals_spent = 0
        self._task_rng = random.Random(self.seed)
        self._block_rng = random.Random(self.seed)
        # Rebuild against the freshly-seeded RNG above -- BlockBandit holds
        # its rng by reference, so without this it would keep drawing from
        # the (stale) RNG object created in __init__.
        self._block_bandit = BlockBandit(
            beta_prior=self.beta_prior, rng=self._block_rng,
            reward_metric=self.block_reward_metric,
        )
        self._last_block_selection = None
        self._snapshotter = TreeSnapshotWriter(
            experiment_dir, enabled=self.snapshot_tree
        )

        # max_rounds directly caps tree size (n_real_nodes() < max_rounds
        # gates the scheduled loop below, same as init_expansions' own
        # guard) -- NOT just a soft hint, and separate from eval_budget.
        # root (1) + init_expansions unconditional EXPANDs already fills
        # max_rounds <= 1 + init_expansions, leaving zero room for the
        # alpha-driven schedule to ever expand again regardless of
        # eval_budget -- a real misconfiguration caught live in this
        # project's own first sanity run (max_rounds=3, init_expansions=1
        # silently capped the search at 2 EXPANDs total, no matter how
        # large eval_budget was). Fail fast, before spending the seed's
        # free pre-evaluation, rather than silently producing a
        # prematurely-frozen tree.
        if max_rounds <= 1 + self.init_expansions:
            raise ValueError(
                f"max_rounds={max_rounds} leaves no room for schedule-driven "
                f"expansion beyond init_expansions={self.init_expansions} "
                f"(root + init_expansions = {1 + self.init_expansions} "
                "node(s) already fills the cap) -- increase max_rounds or "
                "reduce init_expansions."
            )
        # Second, independent check: schedule_favors_expand's own natural
        # ceiling. Its condition (budget_spent**alpha >= n_real_nodes - 1)
        # means that, given the FULL configured eval_budget eventually
        # gets spent, the alpha-schedule itself is designed to grow the
        # tree up to roughly eval_budget**alpha + 1 nodes (this is exactly
        # what size_eval_budget.py::solve_alpha_from_xy solves alpha FOR --
        # alpha = ln(target_agents)/ln(eval_budget) makes
        # eval_budget**alpha == target_agents by construction). If
        # max_rounds is smaller than that natural ceiling, max_rounds --
        # not the schedule -- becomes the actual thing deciding when
        # growth stops, silently capping expansion below what alpha was
        # tuned to reach and defeating the point of tuning it at all.
        schedule_ceiling = self.eval_budget**self.alpha
        if max_rounds <= schedule_ceiling:
            raise ValueError(
                f"max_rounds={max_rounds} is at or below the widening "
                f"schedule's own natural ceiling at this eval_budget/alpha "
                f"(eval_budget**alpha = {self.eval_budget}**{self.alpha} = "
                f"{schedule_ceiling:.2f}) -- max_rounds would cap growth "
                "before the schedule itself would, defeating whatever "
                "eval_budget/alpha was tuned to reach (see "
                "size_eval_budget.py). Increase max_rounds above "
                f"{schedule_ceiling:.2f}, or lower alpha/eval_budget to "
                "match the intended max_rounds."
            )

        # Train cases drive the budget; fall back to every case when no
        # split is configured (the eval split, if any, stays a sidecar).
        if train_case_ids is not None:
            self._train_case_ids = list(train_case_ids)
        else:
            self._train_case_ids = [
                str(c.get("id") or c.get("case_id"))
                for c in load_cases(benchmark_dir)
            ]

        # Root: copy the seed and PRE-EVALUATE it on the full train set
        # (free — not charged to eval_budget), so it qualifies as an
        # expansion parent. Then `init_expansions` unconditional EXPANDs;
        # only evaluated, positive-mean nodes are expandable, so these all
        # branch off the freshly pre-evaluated root.
        self._run_seed(seed_dir, evaluator, gatherer)
        self._snapshot("seed")
        for _ in range(self.init_expansions):
            expandable = self._expandable()
            if not expandable or self._tree.n_real_nodes() > max_rounds:
                break
            # Same affordability guard as the main loop (matters when
            # eval_budget is tiny relative to the dual expansion cost).
            if self.eval_budget - self._budget_spent < self._min_budget_to_expand():
                break
            self._expand(self._tree.argmax_expand(1.0, expandable), editor, gatherer)
            self._snapshot("expand")

        # Scheduled EXPAND/EVALUATE loop. The while-stop keys off total spend
        # (``_budget_spent``), but the widening schedule keys off
        # ``_node_evals_spent`` — evaluations attributed to committed nodes —
        # so the dual manager's throw-away variant trials don't distort it.
        # The root's pre-evaluation is excluded from both.
        while self._budget_spent < self.eval_budget:
            remaining = self.eval_budget - self._budget_spent
            # Early stop: if the remaining budget can't fund an expansion's
            # intra-evaluation, stop instead of spawning un-evaluated nodes that
            # still cost an editor call. ``_min_budget_to_expand`` is 0 for
            # vanilla HGM (expand is free at expand-time; the bandit evaluates
            # the node later), so vanilla's decoupled behavior is unchanged; the
            # dual manager returns ``intra_expand_eval_size``.
            if remaining < self._min_budget_to_expand():
                break
            tau = self._tree.tau(
                remaining, self.eval_budget,
                cool_down=self.cool_down, beta=self.beta,
            )
            expandable = self._expandable()
            evaluable = self._evaluable()
            can_grow = bool(expandable) and self._tree.n_real_nodes() < max_rounds
            if (
                self._tree.schedule_favors_expand(self.alpha, self._node_evals_spent)
                and can_grow
            ):
                self._expand(self._tree.argmax_expand(tau, expandable), editor, gatherer)
                self._snapshot("expand")
            elif evaluable:
                node_id = self._tree.argmax_evaluate(tau, evaluable)
                spent = self._evaluate(node_id, evaluator, gatherer)
                self._budget_spent += spent
                self._node_evals_spent += spent
                self._snapshot("evaluate")
                if (
                    score_target is not None
                    and self._tree[node_id].mean_utility >= score_target
                ):
                    break
            elif can_grow:
                # Nothing left to evaluate, but the tree can still widen.
                self._expand(self._tree.argmax_expand(tau, expandable), editor, gatherer)
                self._snapshot("expand")
            else:
                break

        return self._finalize(evaluator, gatherer)

    # ------------------------------------------------------------------ #
    # EXPAND / EVALUATE
    # ------------------------------------------------------------------ #

    def _expand(
        self, parent_id: int, editor: AgentEditor, gatherer: FeedbackGatherer
    ) -> int:
        """Self-modify ``parent_id`` into a fresh child node via one editor
        call. The editor emits its strategy summary on ``EditResult``."""
        parent = self._tree[parent_id]
        node_id = self._next_id
        self._next_id += 1
        out_dir = self._experiment_dir / f"round_{node_id:03d}"
        (out_dir / "logs").mkdir(parents=True, exist_ok=True)

        block = self._select_block(parent)
        if self._last_block_selection is not None:
            (out_dir / "adaptive_strategy.json").write_text(
                json.dumps(
                    dataclasses.asdict(self._last_block_selection), indent=2
                )
            )
        context = self._render_expand_context(parent, block, out_dir, node_id)
        edit_result = editor.apply(
            self._feedback.get(parent_id), parent.round_dir, out_dir,
            context=context,
        )
        strategy = edit_result.strategy or fallback_strategy()
        strategy.block = block
        node = HGMNode(node_id=node_id, parent_id=parent_id, round_dir=out_dir)

        if not edit_result.success:
            node.edit_failed = True
            self._tree.add(node)
            self._feedback[node_id] = self._synth_failed_edit_feedback(
                node_id, parent_id, strategy, edit_result.errors, out_dir
            )
            print(
                f"node {node_id}: EXPAND from {parent_id} — edit FAILED "
                f"({edit_result.errors[0][:80] if edit_result.errors else '?'})",
                flush=True,
            )
            return node_id

        self._tree.add(node)
        # A fresh child starts unevaluated; compile a zero-eval feedback so
        # the round folder is complete. _evaluate() rewrites it later.
        self._feedback[node_id] = gatherer.compile(
            node_id, parent_id, strategy, self._empty_eval(), out_dir
        )
        self._write_node_sidecar(node)
        print(f"node {node_id}: EXPAND from {parent_id}", flush=True)
        return node_id

    def _evaluate(
        self, node_id: int, evaluator: Evaluator, gatherer: FeedbackGatherer
    ) -> int:
        """Drip a random batch of un-run train cases to a node. Returns the
        number of evaluations actually spent."""
        node = self._tree[node_id]
        if self.eval_repeats > 1:
            # Repeat-eligible: every case is always a valid candidate,
            # regardless of how many times this node has already seen it.
            # eval_repeats is an on/off switch here, not a per-case cap --
            # how often a given case actually gets re-picked is left to
            # self._task_rng's random sampling, bounded only by the total
            # eval_budget (same as any other eval).
            unevaluated = list(self._train_case_ids)
        else:
            unevaluated = [
                cid
                for cid in self._train_case_ids
                if cid not in node.evaluated_case_ids
            ]
        # Cap the batch at the remaining budget so the run lands on
        # eval_budget exactly. Tasks are sampled at RANDOM — the reference
        # runs with eval_random_level=1.0 (fully random task selection).
        remaining = self.eval_budget - self._budget_spent
        n_take = min(self.eval_batch_size, len(unevaluated), max(remaining, 0))
        if n_take <= 0:
            return 0
        batch = self._task_rng.sample(unevaluated, n_take)

        result = evaluator.run(node.round_dir, self._benchmark_dir, case_ids=batch)
        for case in result.per_case:
            node.record(case)

        self._refresh_node_feedback(node, gatherer)
        print(
            f"node {node_id}: EVALUATE +{len(batch)} "
            f"-> mean={node.mean_utility:.3f} n={node.n_evals} "
            f"cmp={self._tree.cmp(node_id):.3f}",
            flush=True,
        )
        # Run the behavior summarizer after every batch. The first batch creates
        # behavior_memory.md; later top-ups UPDATE it cumulatively (the
        # summarizer reads the existing memo and merges this batch in). Pass the
        # BATCH result (``result``) — not the cumulative one — so the aggregate's
        # mutable_log/tool_calls cross-tab matches the per-batch trace.jsonl the
        # summarizer reads (the evaluator truncates the trace each batch).
        if self._summarizer is not None and node.parent_id is not None:
            parent_round_dir = self._tree[node.parent_id].round_dir
            try:
                self._summarizer.summarize(
                    round_dir=node.round_dir,
                    parent_round_dir=parent_round_dir,
                    eval_dir=node.round_dir,
                    eval_result=result,
                    node_id=node.node_id,
                    parent_id=node.parent_id,
                )
            except Exception as exc:  # noqa: BLE001
                print(
                    f"[summarizer] unexpected error on node {node.node_id}: {exc!r}",
                    flush=True,
                )
        self._run_failure_summarizer(node)
        return len(batch)

    # ------------------------------------------------------------------ #
    # Block selection — swappable seam
    # ------------------------------------------------------------------ #

    def _select_block(self, parent: HGMNode) -> str:
        """Which block (see meta_agent/block_suggester.py) this EXPAND
        should target, per ``self.block_selection_strategy``.

        Stage 1 of tier_based_hgm.md's roadmap: four FIXED strategies
        ("collaboration" -> "collaboration_workflow", "single_agent" ->
        "individual_subagent", "verifiers" -> "verifiers", "foundations"
        -> "foundation_capability") that always return the same block
        regardless of ``parent``, plus "non_adaptive", which samples
        UNIFORMLY AT RANDOM among all four every call (via
        self._block_rng, seeded -- reproducible for a given seed, but not
        fixed to one block; still "non_adaptive" because the choice uses
        no feedback/performance signal, unlike "adaptive"). "adaptive"
        Thompson-samples a block from ``self._block_bandit`` (see
        meta_agent/block_bandit.py), built from the accumulated
        success/failure mass of every node evaluated so far grouped by
        block -- real feedback/performance signal, unlike the other five.
        All six are validated against this exact set in __init__, so an
        unknown value fails fast at construction, not here. Called exactly
        once per _expand -- callers must reuse the same value for both
        steering the editor and stamping EvolutionStrategy.block, since a
        stochastic strategy must not be sampled twice for one EXPAND (this
        is exactly why "non_adaptive"/"adaptive" must live in this single
        method rather than being sampled at each call site separately)."""
        # Reset here (not just in evolve()) so a stale AdaptiveStrategy from
        # a previous "adaptive" call can never leak into this round's
        # persisted artifact if block_selection_strategy isn't "adaptive".
        self._last_block_selection = None
        if self.block_selection_strategy == "collaboration":
            return "collaboration_workflow"
        if self.block_selection_strategy == "single_agent":
            return "individual_subagent"
        if self.block_selection_strategy == "verifiers":
            return "verifiers"
        if self.block_selection_strategy == "foundations":
            return "foundation_capability"
        if self.block_selection_strategy == "non_adaptive":
            # Import here (not module-level) matching the existing lazy
            # -import pattern for other meta_agent submodules used within
            # a single method (see render_failure_summary_for_steering
            # below). Reads the canonical block-name set directly from
            # block_suggester.py's own _BLOCK_BODIES rather than
            # duplicating the 4 literal names again here, so this stays
            # correct automatically if a block is ever added/renamed
            # there. sorted() makes the choice order deterministic before
            # sampling, independent of dict insertion order.
            from ..block_suggester import _BLOCK_BODIES

            return self._block_rng.choice(sorted(_BLOCK_BODIES))
        if self.block_selection_strategy == "adaptive":
            adaptive = self._block_bandit.select(self._tree, self._feedback)
            self._last_block_selection = adaptive
            return adaptive.block
        # Unreachable: __init__ already validates block_selection_strategy
        # against the exact same set. Kept as a loud failure (not a
        # silent fallback to one block) in case that invariant is ever
        # broken by a future edit to either place.
        raise ValueError(
            f"unhandled block_selection_strategy={self.block_selection_strategy!r}"
        )

    # ------------------------------------------------------------------ #
    # Steering context for the editor's self-improvement call
    # ------------------------------------------------------------------ #

    def _render_expand_context(
        self, parent: HGMNode, block: str, out_dir: Path, node_id: int
    ) -> str:
        """Build the manager's steering context for an EXPAND: the parent's
        edit lineage, performance + clade metaproductivity, the best node
        so far, and the parent's feedback digest. Pure string assembly —
        the editor reads the parent's actual code itself."""
        parts: list[str] = []

        # Edit lineage — the chain of optimization goals already applied
        # from the root down to this parent, so the editor does not
        # re-propose changes its ancestors already made.
        lineage = self._ancestor_goals(parent.node_id)
        if len(lineage) > 1:
            parts.append("## Edits already applied along this lineage (root → parent):")
            for depth, goal in lineage:
                lines = goal.split("\n")
                parts.append(f"  [depth {depth}] {lines[0][:200]}")
                for cont in lines[1:]:
                    parts.append(f"             {cont[:200]}")

        parts.append(f"\n## Parent performance — node {parent.node_id}")
        if parent.n_evals > 0:
            parts.append(
                f"mean score = {parent.mean_utility:.3f} over "
                f"{parent.n_evals} task(s) evaluated so far; clade "
                f"metaproductivity (CMP) = {self._tree.cmp(parent.node_id):.3f}"
            )
        else:
            parts.append(
                "not yet evaluated — propose a promising exploratory edit."
            )
        best = self._best_evaluated()
        if best is not None:
            parts.append(
                f"best evaluated node in the tree: node {best[0]} at mean "
                f"{best[1]:.3f} — aim to beat it."
            )

        pf = self._feedback.get(parent.node_id)
        if pf is not None:
            if pf.tool_error_rate:
                ranked = sorted(
                    ((n, r) for n, r in pf.tool_error_rate.items() if r > 0),
                    key=lambda kv: -kv[1],
                )
                if ranked:
                    parts.append(
                        "parent tool error rates: "
                        + ", ".join(f"{n}={r:.2f}" for n, r in ranked[:5])
                    )
            if pf.project_metrics:
                parts.append(
                    "parent project metrics (from the cases evaluated so far):"
                )
                parts.extend(render_metrics(pf.project_metrics, cap=10, indent="  "))
            for exc in pf.runtime_exceptions[:3]:
                parts.append(f"  parent error: {exc[:200]}")

        # LLM-synthesized cross-case failure summary (main patterns + hardest
        # cases), when configured -- grounded in ALL failing cases, not just
        # the small char-capped sample failure_report.py renders elsewhere.
        # Skipped silently when the failure summarizer isn't configured or
        # the parent had no failing cases (failure_summary.md wasn't written).
        from ..failure_summarizer import render_failure_summary_for_steering

        failure_summary = render_failure_summary_for_steering(parent.round_dir)
        if failure_summary:
            parts.append(
                f"\n## Failure summary — parent (node {parent.node_id}):\n{failure_summary}"
            )

        # Sibling edits already branched off this parent. Computed once,
        # before the block suggester call, so it can be handed the same
        # data -- see below for why the editor no longer sees this as a
        # separate, potentially-conflicting instruction when a suggester
        # is configured.
        siblings = [
            self._feedback[c] for c in parent.children if c in self._feedback
        ]

        # Block-scoped improvement suggestion (see block_suggester.py), when
        # configured. Selected block always shown (even if the suggester is
        # off/unconfigured or its call failed) so the editor still gets a
        # lightweight steer, degrading gracefully to "no suggestion
        # available" rather than silently vanishing.
        parts.append(f"\n## Selected block for this EXPAND: {block}")
        # Tracks whether a real suggestion was actually produced -- stays
        # None both when no suggester is configured AND when a configured
        # suggester's call fails/errors, so the sibling-directive fallback
        # below (which used to check `self._block_suggester is None` only)
        # correctly fires in BOTH cases. Previously, a suggester configured
        # but erroring on one EXPAND would silently lose ALL sibling-
        # differentiation guidance -- neither the suggestion (which owns
        # that responsibility when it succeeds) nor the fallback directive
        # (gated on the wrong condition) would appear.
        suggestion = None
        if self._block_suggester is not None:
            try:
                suggestion = self._block_suggester.suggest(
                    block=block,
                    agent_dir=parent.round_dir / "task_agent",
                    out_dir=out_dir,
                    node_id=node_id,
                    feedback=self._feedback.get(parent.node_id),
                    failure_summary=failure_summary,
                    siblings=[
                        (sib.strategy.block, sib.strategy.optimization_goal)
                        for sib in siblings
                    ],
                )
            except Exception as exc:  # noqa: BLE001
                print(
                    f"[block_suggester] unexpected error on node {node_id}: {exc!r}",
                    flush=True,
                )
                suggestion = None
            if suggestion:
                parts.append(
                    f"\n## Block-scoped suggestion ({block}) — grounded "
                    "diagnosis + proposal for THIS block specifically. "
                    "This already accounts for any sibling edits already "
                    "tried off this parent (see its own reasoning above) "
                    "-- implement it largely as given; you do not need to "
                    "separately differentiate from siblings yourself. "
                    "Deviate only if your own reading of the current "
                    "source clearly contradicts this suggestion's "
                    "diagnosis. Either way, say so explicitly in your "
                    "rationale (per the editor's own hard rules).\n\n"
                    + suggestion
                )

        # Fallback: fires whenever no ACTUAL suggestion made it into the
        # context above -- either no suggester is configured, or one is
        # but failed/errored for this EXPAND. When a suggestion WAS
        # produced, sibling differentiation is entirely its responsibility
        # (see the `siblings=` kwarg above and BlockSuggester's own
        # _SYSTEM_CLOSING) -- showing this too would hand the editor two
        # instructions that can pull in different directions (see
        # tier_based_hgm.md's stage-1 findings).
        if siblings and not suggestion:
            parts.append(
                f"\n## {len(siblings)} sibling edit(s) already branch off this "
                "parent — make a DIFFERENT change from these:"
            )
            for sib in siblings[:8]:
                parts.append(f"  - {sib.strategy.optimization_goal[:160]}")

        # Inject the lineage's behavior memories, newest-first up the chain
        # within a token budget. Skipped silently when the summarizer isn't
        # configured or no ancestor has a memory file.
        memory_block = self._render_lineage_memory(parent)
        if memory_block:
            parts.append(memory_block)

        parts.append(
            "\nMake targeted improvement to this parent agent. Keep the "
            "scope small enough to apply correctly in one pass."
        )
        return "\n".join(parts)

    def _lineage_memory_label(self, node_id: int) -> str:
        """Label shown before an ancestor's behavior-memory block in
        ``_render_lineage_memory``'s rendered output. Default: just the
        round number -- carries no indication of which block (if any)
        that round's creating edit targeted. Factored out as its own
        overridable hook (rather than inlined in the walk loop below) so
        a subclass can add more context -- e.g. the block name -- without
        duplicating the walk/budget logic. See
        ``meta_agent/managers/hgm_block_tagged.py`` for that override."""
        return f"round {node_id}"

    def _render_lineage_memory(
        self,
        parent: HGMNode,
        *,
        extra_recent: Optional[tuple[str, Path]] = None,
    ) -> str:
        """Render lineage behavior memories newest-first up the chain, within
        ``lineage_memory_token_budget`` (approx; ~4 chars/token).

        Walks ``parent -> grandparent -> ... -> root``, including each
        ancestor's FULL ``behavior_memory.md`` greedily until the budget is
        reached; the first memory that doesn't fully fit is truncated to the
        remaining budget and the walk stops. ``extra_recent`` (a
        ``(label, round_dir)`` pair — dual Stage B passes the Stage A
        intermediate) is treated as the most-recent memory. Rendered oldest →
        newest so the editor reads the evolution over time.

        Returns ``""`` when no memory files exist along the lineage (e.g.
        summarizer disabled, seed round, or write failed).
        """
        from ..behavior_summarizer import render_memory_for_steering

        char_budget = max(0, self.lineage_memory_token_budget) * _CHARS_PER_TOKEN
        if char_budget <= 0:
            return ""

        # Ordered most-recent-first: [extra_recent?, parent, grandparent, ..., root]
        chain: list[tuple[str, Path]] = []
        if extra_recent is not None:
            chain.append(extra_recent)
        nid: Optional[int] = parent.node_id
        while nid is not None and nid in self._tree.nodes:
            node = self._tree.nodes[nid]
            chain.append((self._lineage_memory_label(nid), node.round_dir))
            nid = node.parent_id

        included: list[tuple[str, str]] = []  # (label, text), most-recent-first
        used = 0
        for label, round_dir in chain:
            mem = render_memory_for_steering(round_dir)
            if not mem:
                continue
            if used + len(mem) <= char_budget:
                included.append((label, mem))
                used += len(mem)
            else:
                remaining = char_budget - used
                if remaining >= 400:  # only include a tail if a useful chunk fits
                    included.append(
                        (f"{label} (excerpt)", mem[:remaining].rstrip() + "\n<... truncated ...>")
                    )
                break  # budget exhausted — stop walking older ancestors

        if not included:
            return ""
        parts: list[str] = []
        for label, mem in reversed(included):  # oldest -> newest
            parts.append(f"\n## Observed behavior memory — {label}:")
            parts.append(mem)
        return "\n".join(parts)

    def _ancestor_goals(self, node_id: int) -> list[tuple[int, str]]:
        """(depth, optimization_goal) for every node from the root down to
        ``node_id`` inclusive — the edit lineage of this branch."""
        path: list[int] = []
        nid: Optional[int] = node_id
        while nid is not None:
            path.append(nid)
            nid = self._tree[nid].parent_id
        path.reverse()
        out: list[tuple[int, str]] = []
        for depth, n in enumerate(path):
            fb = self._feedback.get(n)
            if fb is not None and fb.strategy.optimization_goal:
                out.append((depth, fb.strategy.optimization_goal))
        return out

    def _best_evaluated(self) -> Optional[tuple[int, float]]:
        """(node_id, mean_utility) of the best-scoring evaluated node."""
        scored = [
            (nid, n.mean_utility)
            for nid, n in self._tree.nodes.items()
            if not n.edit_failed and n.n_evals > 0
        ]
        return max(scored, key=lambda kv: kv[1]) if scored else None

    def _snapshot(self, event: str) -> None:
        """Append a full-tree snapshot keyed by the current eval budget. A
        no-op unless ``snapshot_tree`` is enabled. Records every node (incl.
        the seed root and edit-failed placeholders) plus a pointer to the
        current best-by-mean node, so an analyst can recover and re-evaluate
        the best agent at any budget level (see snapshot_eval.py)."""
        if self._snapshotter is None or not self._snapshotter.enabled:
            return
        nodes = [
            NodeSnapshot(
                node_id=n.node_id,
                parent_id=n.parent_id,
                round_dir=n.round_dir.name,
                edit_failed=n.edit_failed,
                n_evals=n.n_evals,
                mean_utility=n.mean_utility,
                n_success=n.n_success,
                n_failure=n.n_failure,
                cmp=self._tree.cmp(nid),
            )
            for nid, n in sorted(self._tree.nodes.items())
        ]
        best = self._best_evaluated()
        best_id = best[0] if best is not None else None
        best_round_dir = (
            self._tree[best_id].round_dir.name if best_id is not None else None
        )
        self._snapshotter.record(
            event=event,
            manager=type(self).__name__,
            budget_spent=self._budget_spent,
            node_evals_spent=self._node_evals_spent,
            nodes=nodes,
            best_node_id=best_id,
            best_mean_utility=best[1] if best is not None else None,
            best_round_dir=best_round_dir,
        )

    # ------------------------------------------------------------------ #
    # Selectable-node sets
    # ------------------------------------------------------------------ #

    def _expandable(self) -> list[int]:
        """A node may be expanded only once it has been evaluated and has a
        positive mean score — faithful to hgm.py::expand()'s filter
        ``np.isfinite(mean_utility) and mean_utility > 0``. Edit-failed
        placeholders are excluded too."""
        return [
            nid
            for nid, n in self._tree.nodes.items()
            if not n.edit_failed and n.n_evals > 0 and n.mean_utility > 0
        ]

    def _evaluable(self) -> list[int]:
        """Real nodes that still have un-evaluated train cases."""
        if self.eval_repeats > 1:
            # Repeats never exhaust the pool -- any real node is always
            # evaluable as long as there's at least one train case to drip.
            if not self._train_case_ids:
                return []
            return [nid for nid, n in self._tree.nodes.items() if not n.edit_failed]
        n_train = len(self._train_case_ids)
        return [
            nid
            for nid, n in self._tree.nodes.items()
            if not n.edit_failed and len(n.evaluated_case_ids) < n_train
        ]

    def _min_budget_to_expand(self) -> int:
        """Minimum remaining eval budget required to make an EXPAND worthwhile.

        Vanilla HGM evaluates lazily — ``_expand`` charges nothing at
        expand-time (the bandit evaluates the new node later) — so there is no
        minimum here (returns 0; behavior unchanged). The dual manager, whose
        ``_expand`` runs an intra-evaluation, overrides this to
        ``intra_expand_eval_size`` so the main loop stops instead of spawning
        un-evaluated nodes once the budget can no longer fund one."""
        return 0

    # ------------------------------------------------------------------ #
    # Finalization
    # ------------------------------------------------------------------ #

    def _finalize(
        self, evaluator: Evaluator, gatherer: FeedbackGatherer
    ) -> EvolutionOutcome:
        # Bring the top-k finalists to a full-train estimate, THEN select —
        # so a small-sample fluke can no longer win the LCB comparison.
        self._finalize_top_k(evaluator, gatherer)
        # Final snapshot: finalists are now fully train-evaluated, so the
        # best-by-mean recorded here reflects the post-finalization tree.
        self._snapshot("finalize")

        # Rewrite every sidecar so clade stats are final and consistent.
        for node in self._tree.nodes.values():
            self._write_node_sidecar(node)

        # Select only among fully-train-evaluated nodes (the root + the
        # finalists `_finalize_top_k` just topped up). A thinly-evaluated
        # non-finalist's optimistic partial estimate must not win.
        n_train = len(self._train_case_ids)
        fully_evaluated = {
            nid
            for nid, n in self._tree.nodes.items()
            if not n.edit_failed and n.n_evals >= n_train
        }
        best_id = self._tree.lcb_select(
            self.epsilon, restrict_to=fully_evaluated
        )
        if self._eval_case_ids:
            self._run_eval_split(best_id, evaluator)
        self._run_top_k_full_eval(evaluator)

        rounds = [self._feedback[nid] for nid in sorted(self._feedback)]
        final_score = self._tree[best_id].mean_utility
        print(
            f"HGM done: {self._tree.n_real_nodes()} nodes, "
            f"{self._budget_spent} budget evals "
            f"({self._tree.total_evals} incl. free pre-eval + finalize); "
            f"best = node {best_id} (train mean {final_score:.3f}, "
            f"n={self._tree[best_id].n_evals})",
            flush=True,
        )
        return EvolutionOutcome(
            rounds=rounds, best_round=best_id, final_score=final_score
        )

    def _finalize_top_k(
        self, evaluator: Evaluator, gatherer: FeedbackGatherer
    ) -> None:
        """Re-evaluate the top-k finalists (by current train mean) on the
        train cases they have not yet seen, so every finalist has a full
        train-split estimate before ``lcb_select``. These evaluations are a
        separate finalization budget — not charged to ``eval_budget``."""
        candidates = [
            n for n in self._tree.nodes.values()
            if not n.edit_failed and n.n_evals > 0
        ]
        if not candidates:
            return
        candidates.sort(key=lambda n: n.mean_utility, reverse=True)
        finalists = candidates[: max(1, self.finalize_top_k)]

        n_train = len(self._train_case_ids)
        spent = 0
        for node in finalists:
            if self.eval_repeats > 1:
                # Same on/off switch as _evaluate: one additional full pass
                # over every train case, on top of whatever repeats this
                # finalist already accumulated during the search loop.
                missing = list(self._train_case_ids)
            else:
                missing = [
                    cid for cid in self._train_case_ids
                    if cid not in node.evaluated_case_ids
                ]
            if not missing:
                continue
            result = evaluator.run(
                node.round_dir, self._benchmark_dir, case_ids=missing
            )
            for case in result.per_case:
                node.record(case)
            spent += len(missing)
            self._refresh_node_feedback(node, gatherer)
            self._run_failure_summarizer(node)
            print(
                f"finalize: node {node.node_id} +{len(missing)} "
                f"-> mean={node.mean_utility:.3f} n={node.n_evals}/{n_train}",
                flush=True,
            )
        print(
            f"finalize: re-evaluated {len(finalists)} finalist(s), {spent} "
            f"extra evals (not charged to eval_budget)",
            flush=True,
        )

    def _run_eval_split(self, best_id: int, evaluator: Evaluator) -> None:
        """Held-out eval of the chosen best node — a sidecar metric, not
        fed back into the search (mirrors HillClimbingManager)."""
        node = self._tree[best_id]
        result = evaluator.run(
            node.round_dir, self._benchmark_dir, case_ids=self._eval_case_ids
        )
        (node.round_dir / "eval_score.json").write_text(
            json.dumps(
                {
                    "node_id": best_id,
                    "composite_score": result.score,
                    "passed": result.passed,
                    "failed": result.failed,
                    "wall_time_s": result.wall_time_s,
                    "crashed": result.crashed,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(
            f"held-out eval on best node {best_id}: {result.score:.3f}",
            flush=True,
        )

    def _run_top_k_full_eval(self, evaluator: Evaluator) -> None:
        """Evaluate top-k finalists on the full benchmark (train + eval split).

        Sidecar audit metric — does NOT influence LCB selection. Gives every
        finalist a head-to-head score on identical, comprehensive ground so
        the user can compare generalization independent of the LCB winner.

        Skips cases the finalist already has scores for (from intra-expand,
        bandit top-up, or ``_finalize_top_k``) — only newly-needed cases hit
        the evaluator. The audit's composite score combines the pre-existing
        per-case results (from ``node.case_results``) with the newly-run
        ones, so the reported number is over the full 120-case set without
        the cost of re-evaluating cases this node has already been scored on.

        Writes one ``full_eval_score.json`` per finalist into its round_dir.
        Disabled when ``full_eval_top_k == 0`` (the default).
        """
        if self.full_eval_top_k <= 0:
            return
        if not self._train_case_ids and not self._eval_case_ids:
            return

        all_cases = list(self._train_case_ids or []) + list(self._eval_case_ids or [])
        all_set = set(all_cases)

        candidates = [
            n for n in self._tree.nodes.values()
            if not n.edit_failed and n.n_evals > 0
        ]
        candidates.sort(key=lambda n: n.mean_utility, reverse=True)
        finalists = candidates[: self.full_eval_top_k]

        print(
            f"full-benchmark audit: running top-{len(finalists)} finalist(s) "
            f"on {len(all_cases)} cases — dedup against per-node existing evals "
            f"(sidecar, not used for selection)",
            flush=True,
        )
        for node in finalists:
            # Cases this node already has per-case results for, restricted to
            # the audit's full-benchmark set (defensive: node could have
            # results for case_ids that aren't in the current split, e.g. if
            # train_case_ids changed between runs).
            existing_by_id = {
                c.case_id: c
                for c in node.case_results
                if c.case_id in all_set
            }
            missing = [cid for cid in all_cases if cid not in existing_by_id]

            new_per_case = []
            new_wall_time = 0.0
            new_crashed = False
            if missing:
                new_result = evaluator.run(
                    node.round_dir, self._benchmark_dir, case_ids=missing
                )
                new_per_case = list(new_result.per_case)
                new_wall_time = new_result.wall_time_s
                new_crashed = new_result.crashed
            new_by_id = {c.case_id: c for c in new_per_case}

            # Combine in `all_cases` order. Falls through if a case is in
            # neither bucket (shouldn't happen unless the evaluator dropped
            # something).
            combined = []
            for cid in all_cases:
                if cid in existing_by_id:
                    combined.append(existing_by_id[cid])
                elif cid in new_by_id:
                    combined.append(new_by_id[cid])
            n_cases = len(combined)
            composite = sum(c.score for c in combined) / n_cases if n_cases else 0.0
            passed = sum(1 for c in combined if c.passed)
            failed = n_cases - passed
            audit_pre_existing = n_cases - len(new_per_case)

            (node.round_dir / "full_eval_score.json").write_text(
                json.dumps(
                    {
                        "node_id": node.node_id,
                        "composite_score": composite,
                        "passed": passed,
                        "failed": failed,
                        "n_cases": n_cases,
                        "wall_time_s": new_wall_time,
                        "crashed": new_crashed,
                        "train_mean": node.mean_utility,
                        "train_n": node.n_evals,
                        # Audit cost transparency:
                        "audit_pre_existing_evals": audit_pre_existing,
                        "audit_new_evals": len(new_per_case),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(
                f"  node {node.node_id}: full={composite:.3f} "
                f"(passed={passed}/{n_cases})  "
                f"train_mean={node.mean_utility:.3f} n={node.n_evals}  "
                f"[audit: {audit_pre_existing} cached + {len(new_per_case)} new]",
                flush=True,
            )

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _run_seed(
        self, seed_dir: Path, evaluator: Evaluator, gatherer: FeedbackGatherer
    ) -> None:
        out_dir = self._experiment_dir / "round_000"
        agent_dst = out_dir / "task_agent"
        if agent_dst.exists():
            shutil.rmtree(agent_dst)
        agent_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(seed_dir, agent_dst)
        (out_dir / "logs").mkdir(exist_ok=True)

        node = HGMNode(node_id=0, parent_id=None, round_dir=out_dir)
        self._tree.add(node)
        self._next_id = 1

        # Pre-evaluate the seed on the FULL train set. Like the reference's
        # initial-agent evaluation this is free — not charged to
        # eval_budget — and gives the root a real score so it is an
        # eligible expansion parent for the init expansions.
        result = evaluator.run(
            out_dir, self._benchmark_dir, case_ids=self._train_case_ids
        )
        for case in result.per_case:
            node.record(case)

        zero_strategy = EvolutionStrategy(
            target_files=[],
            optimization_goal="Seed agent (HGM tree root).",
            proposed_changes="(none — seed used as-is)",
            rationale="HGM tree root.",
        )
        self._feedback[0] = gatherer.compile(
            0, 0, zero_strategy, self._build_eval_result(node), out_dir
        )
        self._write_node_sidecar(node)
        self._run_failure_summarizer(node)
        print(
            f"node 0: SEED pre-eval -> mean={node.mean_utility:.3f} "
            f"n={node.n_evals} (free, not charged to budget)",
            flush=True,
        )

    def _run_failure_summarizer(self, node: HGMNode) -> None:
        """Fire the failure summarizer (if configured) on a node's current
        CUMULATIVE evaluation result -- unlike the behavior summarizer, this
        runs unconditionally (no ``parent_id is not None`` guard): there's no
        diff involved, so the root/seed round benefits from it just as much
        as any other. Overwrites ``failure_summary.md`` each call; see
        ``FailureSummarizer.summarize``'s own docstring for why that's safe."""
        if self._failure_summarizer is None:
            return
        try:
            self._failure_summarizer.summarize(
                eval_result=self._feedback[node.node_id].eval_result,
                round_dir=node.round_dir,
                node_id=node.node_id,
            )
        except Exception as exc:  # noqa: BLE001
            print(
                f"[failure_summarizer] unexpected error on node {node.node_id}: {exc!r}",
                flush=True,
            )

    def _refresh_node_feedback(
        self, node: HGMNode, gatherer: FeedbackGatherer
    ) -> None:
        """Rebuild a node's cumulative EvaluationResult and feedback after
        an eval batch, and rewrite its sidecar. The trace-derived feedback
        stats only cover the last batch — a documented limitation; the
        authoritative tallies live on the node and in hgm_node.json."""
        strategy = self._feedback[node.node_id].strategy
        self._feedback[node.node_id] = gatherer.compile(
            node.node_id, node.parent_id or 0, strategy,
            self._build_eval_result(node), node.round_dir,
        )
        self._write_node_sidecar(node)

    @staticmethod
    def _empty_eval() -> EvaluationResult:
        return EvaluationResult(
            score=0.0, metrics={}, passed=0, failed=0, per_case=[],
            wall_time_s=0.0, crashed=False,
        )

    def _build_eval_result(self, node: HGMNode) -> EvaluationResult:
        """A cumulative EvaluationResult over every batch a node has seen."""
        cases: list[CaseResult] = list(node.case_results)
        passed = sum(1 for c in cases if c.passed)
        return EvaluationResult(
            score=node.mean_utility,
            metrics={},
            passed=passed,
            failed=len(cases) - passed,
            per_case=cases,
            wall_time_s=0.0,
            crashed=False,
        )

    def _synth_failed_edit_feedback(
        self,
        node_id: int,
        base_round: int,
        strategy: EvolutionStrategy,
        errors: list[str],
        out_dir: Path,
    ) -> AgentFeedback:
        feedback = AgentFeedback(
            round_number=node_id,
            base_round=base_round,
            strategy=strategy,
            eval_result=self._empty_eval(),
            tool_usage={},
            llm_calls=0,
            runtime_exceptions=[],
            log_excerpt="",
            edit_errors=errors,
        )
        persist_round_artifacts(out_dir, feedback)
        return feedback

    def _write_node_sidecar(self, node: HGMNode) -> None:
        """Authoritative per-node HGM state — survives evaluator.run
        overwriting eval_result.json, and is what a future --resume reads."""
        (node.round_dir / "hgm_node.json").write_text(
            json.dumps(
                {
                    "node_id": node.node_id,
                    "parent_id": node.parent_id,
                    "children": node.children,
                    "edit_failed": node.edit_failed,
                    "n_evals": node.n_evals,
                    "mean_utility": node.mean_utility,
                    "n_success": node.n_success,
                    "n_failure": node.n_failure,
                    "clade_success": self._tree.clade_success(node.node_id),
                    "clade_failure": self._tree.clade_failure(node.node_id),
                    "cmp": self._tree.cmp(node.node_id),
                    "utility_measures": node.utility_measures,
                    "evaluated_case_ids": sorted(node.evaluated_case_ids),
                    # Repeat counts per case -- always 0/1 unless
                    # eval_repeats > 1, in which case this is the only place
                    # that shows how many times each case was re-run.
                    "case_eval_counts": dict(node.evaluated_case_ids),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
