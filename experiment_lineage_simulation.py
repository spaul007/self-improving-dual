#!/usr/bin/env python3
"""Standalone experiment: walk the REAL framework components (AgentEditor,
BlockSuggester, the subprocess evaluator, the default gatherer -- the exact
same classes, model (DeepSeek v4 Pro), agentic tool sets, agentic_max_turns,
and verbose round-transcript logging a live HGM run uses) through one
SCRIPTED, FORCED sequence of EXPANDs, instead of letting HGMManager's
Thompson-sampling bandits choose block/implementation_strategy/target.

Why: a prior analysis of Claude's own winning two-pass harness redesign for
this project (harness_claude_hard/REPORT.md, composite 0.501 -> 0.824 held
out) found its recipe decomposes into steps across the block_suggester
taxonomy (per-role individual_subagent+harness_heavy rewrites, a shared
foundation_capability normalization layer early enough for later roles to
reuse rather than duplicate, then more foundation_capability substrate,
more individual_subagent integration, then one genuinely cross-cutting
mixed fix) -- but reaching it requires all of them landing on ONE
unbroken, dependency-ordered lineage. Live HGM runs with
Thompson sampling spread EXPANDs across many parents rather than
compounding depth on the best branch, and separately, a single real EXPAND
measured in an actual run (round 26->32 of
runs/20260925_235724_travel_mas_refactored_deepseek27b_two_tier_no_curriculum_no_backbone_X100Y180)
showed even a narrowly-targeted, correctly-diagnosed harness_heavy fix on
the single most prevalent failing check moves raw per-case composite by
less than the ~0.017 noise floor (240 evals). This script tests the
cleaner question directly: if this exact lineage IS walked end to end
(target forced per step, not discovered by the model), does the composite
gain compound to something comparable to Claude's recipe?

Each step is a real EXPAND: BlockSuggester.suggest(block=<forced>,
curriculum_directive=<forced target hint>) for a grounded diagnosis scoped
to that step's target, then AgentEditor.apply(context=<lineage + block +
implementation_strategy + suggestion>) to implement it, then a FULL 60-case
train evaluation, then DefaultFeedbackGatherer.compile(...) -- which is the
sole writer of strategy.json/eval_result.json/feedback.json, so every round
directory this script produces is byte-shape-identical to a live HGM
round's. No tree, no bandits, no branching: round_000 (seed) -> round_001 ->
... -> round_012, one linear chain, each step's out_dir becoming the next
step's base_dir.

Resumable: a step whose round_NNN/eval_result.json already exists is
skipped (its persisted feedback is loaded and used as the next step's
parent) rather than re-run, so a killed/restarted run picks up where it
left off without repaying already-spent LLM/evaluator cost.

Usage (source .env FIRST -- this process needs OPENAI_API_KEY in its own
environment for the OpenRouter-backed editor/block_suggester calls; nothing
in this script or the framework loads .env on its own, matching this repo's
existing launch-script convention, e.g. launch_gemma_no_backbone.sh):
  source /groups/AIC-MV/v.kulkarni1/.env
  PYTHONPATH=. python3 experiment_lineage_simulation.py --out-dir runs_lineage_sim/staged_lineage_v1
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from meta_agent import config as cfg_mod  # noqa: E402
from meta_agent import runtime_env  # noqa: E402
from meta_agent.implementation_strategy import _IMPLEMENTATION_STRATEGY_BODIES  # noqa: E402
from meta_agent.models import AgentFeedback, EvaluationResult, EvolutionStrategy  # noqa: E402

DEFAULT_CONFIG = REPO_ROOT / "configs" / "experiment_lineage_simulation_X100Y300.yaml"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


@dataclass(frozen=True)
class Step:
    label: str
    block: str
    implementation_strategy: str
    directive: str


# The scripted lineage. implementation_strategy is "harness_heavy" throughout,
# matching Claude's own recipe (near-total replacement of LLM reasoning with
# deterministic code, LLM retained only for narrow extraction). Each
# directive is handed to BlockSuggester.suggest(curriculum_directive=...) --
# a real, public steering hook (see block_suggester.py's
# _render_curriculum_focus) -- so the diagnosis is still genuinely produced
# by the model reading real failure data, just scoped to a named target
# instead of self-selected.
STEPS: list[Step] = [
    Step(
        "01_flight_harness_heavy",
        "individual_subagent",
        "harness_heavy",
        "This is step 1 of a scripted multi-stage experiment that works "
        "through each pipeline role's own harness-heavy conversion in turn, "
        "one role per EXPAND. For THIS EXPAND, scope your diagnosis "
        "specifically to the Flight stage/role (the function that selects "
        "a flight from tool results) -- even if another role currently "
        "shows more failures; later steps in this same scripted sequence "
        "cover the other roles. Diagnose why this stage's LLM call doesn't "
        "reliably apply the traveler's stated selection rule "
        "(cheapest/fastest/direct), and propose replacing the selection "
        "logic with deterministic code driven by an upstream LLM "
        "extraction of the stated preference -- keep the LLM call narrow "
        "(structured extraction only), move the actual selection to code.",
    ),
    Step(
        "02_shared_extraction_layer",
        "foundation_capability",
        "harness_heavy",
        "Step 2 of the same scripted sequence. The Flight stage was just "
        "converted to a two-step design: a narrow LLM call extracts the "
        "traveler's selection rule, then deterministic code parses the "
        "flight tool's raw results and applies that rule. For THIS "
        "EXPAND, diagnose whether the code that turns the tool's raw JSON "
        "into usable fields (duration, price, departure/arrival time, "
        "segment/direct-vs-connecting count) is currently written inline "
        "inside Flight's own selection function, specific to Flight's own "
        "code path. If so, extract it into ONE shared module (e.g. a new "
        "`tdata.py`-style helper) that produces flat, typed records from "
        "the raw tool JSON -- not scoped to Flight, written so the Train "
        "stage (converted in the NEXT step) can reuse it directly instead "
        "of re-deriving its own copy of the same parsing logic. This is a "
        "foundation_capability diagnosis specifically: ground it in why "
        "this normalization logic is shared substrate every stage that "
        "queries transport tools will need, not a Flight-specific fix.",
    ),
    Step(
        "03_train_harness_heavy",
        "individual_subagent",
        "harness_heavy",
        "Step 3 of the same scripted sequence. Flight was converted to a "
        "two-step design, and a shared raw-tool-result normalization "
        "module now exists (from the prior EXPAND) -- do NOT re-diagnose "
        "Flight, and do NOT write Train's own copy of that normalization "
        "logic. For THIS EXPAND, scope your diagnosis specifically to the "
        "Train stage/role: apply the identical two-step treatment (narrow "
        "LLM extraction, then deterministic selection), reusing the "
        "shared normalization module the prior EXPAND created rather than "
        "duplicating its parsing logic inside train.py.",
    ),
    Step(
        "04_accounting_harness_heavy",
        "individual_subagent",
        "harness_heavy",
        "Step 4 of the same scripted sequence. Flight and Train have "
        "already been converted this way -- do not re-diagnose either. "
        "For THIS EXPAND, scope your diagnosis specifically to the "
        "Accounting/budget-summary stage/role. Diagnose whether its "
        "budget computation is arithmetic over line items already present "
        "in upstream stage messages (flight/train/hotel/attraction "
        "prices) that an LLM call is unreliable at computing exactly, and "
        "propose computing it in deterministic code directly from those "
        "upstream fields, dropping the LLM call for this stage if the "
        "diagnosis bears this out.",
    ),
    Step(
        "05_sightseeing_harness_heavy",
        "individual_subagent",
        "harness_heavy",
        "Step 5 of the same scripted sequence. Flight, Train, and "
        "Accounting have already been converted this way -- do not "
        "re-diagnose any of them. For THIS EXPAND, scope your diagnosis "
        "specifically to the Sightseeing stage/role (attraction/restaurant "
        "selection and day-by-day scheduling). Propose moving selection "
        "and scheduling logic to deterministic code, keeping the LLM call "
        "narrow (requirement extraction from the free-text request only).",
    ),
    Step(
        "06_tool_call_substrate",
        "foundation_capability",
        "harness_heavy",
        "Step 6 of the same scripted sequence. Flight/Train already share "
        "a raw-tool-result normalization module (added two EXPANDs ago) -- "
        "do not redo that part. The four prior EXPANDs converted "
        "Flight/Train/Accounting/Sightseeing to deterministic, code-driven "
        "selection logic, and each likely independently reimplements the "
        "surrounding tool-invocation-and-error-handling pattern (call a "
        "tool, handle an empty or malformed result, retry/log on failure) "
        "separately from the data-normalization module. For THIS EXPAND, "
        "diagnose whether THAT pattern is duplicated verbatim-or-near-"
        "verbatim across those stages' own code, and propose extracting "
        "the shared tool-call/empty-result-handling logic into ONE module "
        "imported by all of them. This is a foundation_capability "
        "diagnosis specifically -- it must be about logic shared across "
        "roles, not any one role's own remaining behavior.",
    ),
    Step(
        "07_entity_resolution_substrate",
        "foundation_capability",
        "harness_heavy",
        "Step 7 of the same scripted sequence. The role rewrites made "
        "entity selection (flights/trains/hotels/restaurants/attractions) "
        "deterministic and code-driven, but likely still compare entity "
        "names/identities with exact-string matching against tool "
        "results. Diagnose whether this now causes hard-constraint "
        "failures when the correct, well-formed entity a stage picks "
        "doesn't match the grader's expected name exactly (e.g. a "
        "transliteration/partial-name/quoted-name variant). If so, "
        "propose a SHARED fuzzy/approximate name-resolution utility "
        "usable by every stage's selection code -- not scoped to one "
        "stage -- since the same resolution gap would recur across "
        "flight/train/hotel/restaurant/attraction matching alike. This is "
        "a foundation_capability diagnosis: ground it in which stages "
        "would all need this, not just one.",
    ),
    Step(
        "08_seat_count_integration",
        "individual_subagent",
        "harness_heavy",
        "Step 8 of the same scripted sequence. A shared entity-resolution "
        "utility was just added in the prior EXPAND. For THIS EXPAND, "
        "scope your diagnosis specifically to the Train/Flight "
        "role's selection logic: check whether it verifies that a chosen "
        "seat/ticket option actually has enough remaining capacity for "
        "the traveler's whole party before selecting it, and propose "
        "adding that as an explicit filter condition in its deterministic "
        "selection code.",
    ),
    Step(
        "09_hotel_tie_break",
        "individual_subagent",
        "harness_heavy",
        "Step 9 of the same scripted sequence. For THIS EXPAND, scope "
        "your diagnosis specifically to the Hotel/lodging selection "
        "logic: when multiple hotel results satisfy the stated filters "
        "equally, check how ties are currently broken, and propose a "
        "concrete, deterministic tie-break rule (e.g. prefer the "
        "highest-rated option among ties).",
    ),
    Step(
        "10_restaurant_filter_relax",
        "individual_subagent",
        "harness_heavy",
        "Step 10 of the same scripted sequence. For THIS EXPAND, scope "
        "your diagnosis specifically to the Restaurant selection logic: "
        "check whether an overly strict filter (e.g. an exact cuisine-"
        "class match) is dropping otherwise-valid matches, and whether "
        "restaurant opening hours are checked against the scheduled meal "
        "time at all. Propose relaxing the over-strict filter and adding "
        "an opening-hours check in its place.",
    ),
    Step(
        "11_sightseeing_scheduling_density",
        "individual_subagent",
        "harness_heavy",
        "Step 11 of the same scripted sequence. For THIS EXPAND, scope "
        "your diagnosis specifically to the Sightseeing stage's "
        "scheduling logic (not its selection logic, already addressed in "
        "an earlier step): check whether a request with many required "
        "attractions is scheduled densely enough to fit them all, and "
        "whether arrival/departure days correctly reserve slots for "
        "required attractions rather than being left mostly empty. "
        "Propose concrete scheduling-density fixes.",
    ),
    Step(
        "12_pseudo_place_mixed",
        "mixed",
        "harness_heavy",
        "Step 12, the final step of this scripted sequence. Earlier "
        "EXPANDs added a shared entity-resolution utility "
        "(foundation_capability) and converted Sightseeing's own "
        "selection/scheduling logic to code (individual_subagent). For "
        "THIS EXPAND, diagnose whether an ambiguous 'located pseudo-place' "
        "reference (e.g. a district/area name standing in for a specific "
        "venue in a tool result or request) can defeat BOTH the shared "
        "resolver AND Sightseeing's own call site at once -- i.e. a fix "
        "confined to only one of those two layers would be incomplete. "
        "Use the mixed block only if this is genuinely true; if your "
        "diagnosis actually fits cleanly in one layer, say so and use "
        "that block's territory in your proposed change instead.",
    ),
]


def load_feedback(round_dir: Path) -> AgentFeedback:
    return AgentFeedback.model_validate_json(
        (round_dir / "feedback.json").read_text(encoding="utf-8")
    )


def composite_of(feedback: AgentFeedback) -> Optional[float]:
    scores = [c.score for c in feedback.eval_result.per_case]
    return statistics.fmean(scores) if scores else None


def no_plan_rate_of(feedback: AgentFeedback) -> float:
    """Prefer the project scorer's own ``no_plan_rate`` (computed by
    ``scorer_impl.py``'s ``aggregate()`` and surfaced via
    ``AgentFeedback.project_metrics`` -- the same number a live HGM round's
    feedback digest shows). Falls back to 0.0 if the gatherer has no scorer
    wired (project_metrics empty) rather than guessing from case details."""
    rate = (feedback.project_metrics or {}).get("no_plan_rate")
    return float(rate) if rate is not None else 0.0


def render_lineage_context(history: list[dict], step: Step) -> list[str]:
    parts: list[str] = []
    if history:
        parts.append(
            "## Edits already applied along this scripted lineage (earliest first):"
        )
        for rec in history:
            parts.append(f"  [{rec['label']}] {rec['optimization_goal'][:200]}")
        last = history[-1]
        parts.append(
            f"\n## Current checkpoint performance -- {last['label']}: "
            f"composite={last['composite']:.4f}, no_plan_rate={last['no_plan_rate']:.3f} "
            f"(seed composite was {history[0]['composite']:.4f})"
        )
    else:
        parts.append("## This is the first scripted EXPAND off the bare seed agent.")
    parts.append(f"\n## Selected block for this EXPAND: {step.block}")
    parts.append("\n" + _IMPLEMENTATION_STRATEGY_BODIES[step.implementation_strategy])
    return parts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument(
        "--n-steps",
        type=int,
        default=len(STEPS),
        help="run only the first N scripted steps (for a cheap smoke test)",
    )
    args = ap.parse_args()
    assert 1 <= args.n_steps <= len(STEPS)
    steps = STEPS[: args.n_steps]

    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit(
            "OPENAI_API_KEY is not set in this process's environment -- the "
            "editor/block_suggester calls (deepseek/deepseek-v4-pro via "
            "OpenRouter) will fail. Source .env first, in THIS shell, before "
            "launching this script:\n"
            "  source /groups/AIC-MV/v.kulkarni1/.env\n"
            "(nothing in this script or the framework loads .env on its "
            "own -- matching this repo's existing launch scripts, e.g. "
            "launch_gemma_no_backbone.sh)"
        )

    out = args.out_dir.resolve()
    fresh = not out.exists()
    out.mkdir(parents=True, exist_ok=True)
    snapshot = out / "config.snapshot.yaml"
    if fresh or not snapshot.exists():
        snapshot.write_text(args.config.read_text(encoding="utf-8"), encoding="utf-8")

    cfg = cfg_mod.load(args.config)
    runtime_env.apply_all(cfg)
    fw = cfg_mod.build_components(cfg)
    train_ids = [str(x) for x in fw.train_case_ids]
    log(
        f"output dir: {out}; project={cfg.project}; seed_dir={fw.seed_dir}; "
        f"{len(train_ids)} train cases; {len(steps)} scripted step(s)"
    )

    manifest_path = out / "lineage_manifest.json"
    history: list[dict] = (
        json.loads(manifest_path.read_text()) if manifest_path.exists() else []
    )

    # ---- round_000: the bare seed, pre-evaluated on the full train set ----
    seed_dir_path = out / "round_000"
    if not history:
        agent_dst = seed_dir_path / "task_agent"
        if agent_dst.exists():
            shutil.rmtree(agent_dst)
        agent_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(fw.seed_dir, agent_dst)
        (seed_dir_path / "logs").mkdir(exist_ok=True)
        log("round_000: evaluating the bare seed on all train cases ...")
        result = fw.evaluator.run(seed_dir_path, fw.benchmark_dir, case_ids=train_ids)
        zero_strategy = EvolutionStrategy(
            target_files=[],
            optimization_goal="Seed agent (scripted-lineage experiment root).",
            proposed_changes="(none -- seed used as-is)",
            rationale="Root of the scripted lineage simulation.",
        )
        feedback0 = fw.gatherer.compile(0, 0, zero_strategy, result, seed_dir_path)
        comp0 = composite_of(feedback0) or 0.0
        history.append(
            {
                "step_index": 0,
                "label": "00_seed",
                "round_dir": str(seed_dir_path),
                "block": None,
                "implementation_strategy": None,
                "optimization_goal": zero_strategy.optimization_goal,
                "composite": comp0,
                "no_plan_rate": no_plan_rate_of(feedback0),
                "edit_success": True,
            }
        )
        manifest_path.write_text(json.dumps(history, indent=2))
        log(f"round_000: seed composite={comp0:.4f}")
    else:
        log(f"resuming: {len(history) - 1} scripted step(s) already completed")

    parent_dir = Path(history[-1]["round_dir"])
    parent_feedback = load_feedback(parent_dir)

    for i, step in enumerate(steps, start=1):
        if i < len(history):
            rec = history[i]
            log(
                f"round_{i:03d} [{rec['label']}]: already completed "
                f"(composite={rec['composite']:.4f}) -- skipping"
            )
            parent_dir = Path(rec["round_dir"])
            parent_feedback = load_feedback(parent_dir)
            continue

        out_dir = out / f"round_{i:03d}"
        (out_dir / "logs").mkdir(parents=True, exist_ok=True)
        log(f"round_{i:03d} [{step.label}]: block={step.block} "
            f"implementation_strategy={step.implementation_strategy}")

        suggestion = fw.block_suggester.suggest(
            block=step.block,
            agent_dir=parent_dir / "task_agent",
            out_dir=out_dir,
            node_id=i,
            feedback=parent_feedback,
            failure_summary=None,
            siblings=[],
            curriculum_directive=step.directive,
        ) if fw.block_suggester is not None else None

        context_parts = render_lineage_context(history, step)
        if suggestion:
            context_parts.append(
                f"\n## Block-scoped suggestion ({step.block}) -- grounded "
                "diagnosis + proposal for THIS scripted step specifically. "
                "Implement it largely as given; deviate only if your own "
                "reading of the current source clearly contradicts its "
                "diagnosis, and say so explicitly in your rationale.\n\n"
                + suggestion
            )
        else:
            context_parts.append(
                f"\n## Scripted target for this EXPAND (no block_suggester "
                f"diagnosis available):\n{step.directive}"
            )
        context = "\n".join(context_parts)

        t0 = time.time()
        edit_result = fw.editor.apply(
            parent_feedback,
            parent_dir,
            out_dir,
            context=context,
            has_suggestion=bool(suggestion),
        )
        log(f"round_{i:03d}: editor.apply done in {time.time() - t0:.0f}s "
            f"success={edit_result.success}")

        strategy = edit_result.strategy
        strategy.block = step.block
        strategy.implementation_strategy = step.implementation_strategy

        if not edit_result.success:
            # Matches hgm.py's own _synth_failed_edit_feedback convention: an
            # EMPTY eval_result, not the parent's -- a failed edit was never
            # actually evaluated, so copying the parent's scores in here
            # would misleadingly imply otherwise.
            feedback = AgentFeedback(
                round_number=i,
                base_round=i - 1,
                strategy=strategy,
                eval_result=EvaluationResult(score=0.0, per_case=[]),
                edit_errors=edit_result.errors or ["editor returned no file edits"],
            )
            from meta_agent.feedback_gatherer import persist_round_artifacts

            persist_round_artifacts(out_dir, feedback)
            history.append(
                {
                    "step_index": i,
                    "label": step.label,
                    "round_dir": str(out_dir),
                    "block": step.block,
                    "implementation_strategy": step.implementation_strategy,
                    "optimization_goal": strategy.optimization_goal,
                    "composite": history[-1]["composite"],
                    "no_plan_rate": history[-1]["no_plan_rate"],
                    "edit_success": False,
                    "edit_errors": edit_result.errors,
                }
            )
            manifest_path.write_text(json.dumps(history, indent=2))
            log(
                f"round_{i:03d} [{step.label}]: EDIT FAILED "
                f"({(edit_result.errors or ['?'])[0][:160]}) -- stopping the "
                "lineage here; fix and re-run with the same --out-dir to resume."
            )
            sys.exit(1)

        log(f"round_{i:03d}: evaluating on all {len(train_ids)} train cases ...")
        t0 = time.time()
        eval_result = fw.evaluator.run(out_dir, fw.benchmark_dir, case_ids=train_ids)
        log(f"round_{i:03d}: evaluation done in {time.time() - t0:.0f}s")

        feedback = fw.gatherer.compile(i, i - 1, strategy, eval_result, out_dir)
        comp = composite_of(feedback) or 0.0
        no_plan = no_plan_rate_of(feedback)
        history.append(
            {
                "step_index": i,
                "label": step.label,
                "round_dir": str(out_dir),
                "block": step.block,
                "implementation_strategy": step.implementation_strategy,
                "optimization_goal": strategy.optimization_goal,
                "composite": comp,
                "no_plan_rate": no_plan,
                "edit_success": True,
            }
        )
        manifest_path.write_text(json.dumps(history, indent=2))
        prev_comp = history[-2]["composite"]
        log(
            f"round_{i:03d} [{step.label}]: composite {prev_comp:.4f} -> "
            f"{comp:.4f} (delta {comp - prev_comp:+.4f}), no_plan_rate={no_plan:.3f}"
        )
        parent_dir, parent_feedback = out_dir, feedback

    log("scripted lineage complete.")
    table = [
        "# Scripted lineage simulation",
        "",
        "Forced (block, implementation_strategy, target) sequence driven through the "
        "real BlockSuggester/AgentEditor (DeepSeek v4 Pro, full agentic tool sets, "
        "agentic_max_turns=100) -- no Thompson sampling, one linear chain.",
        "",
        "| Step | Block | Impl. strategy | Composite | No-plan rate | Delta | Edit OK |",
        "|---|---|---|---|---|---|---|",
    ]
    for idx, rec in enumerate(history):
        prev = history[idx - 1]["composite"] if idx > 0 else rec["composite"]
        table.append(
            f"| {rec['label']} | {rec.get('block') or '-'} | "
            f"{rec.get('implementation_strategy') or '-'} | {rec['composite']:.4f} | "
            f"{rec['no_plan_rate']:.3f} | {rec['composite'] - prev:+.4f} | "
            f"{rec['edit_success']} |"
        )
    (out / "REPORT.md").write_text("\n".join(table) + "\n")
    print("\n" + (out / "REPORT.md").read_text())
    print(f"\noutputs: {out}")


if __name__ == "__main__":
    main()
