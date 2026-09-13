"""Standalone validation: does ITERATIVE refinement of a single EXPAND --
one lineage, K iterations, always keeping the best -- reduce a specific
curriculum sub-goal's failure rate more than the current production
behavior (one-shot EXPAND, then the tree moves on)?

As close to the real HGM EXPAND as this standalone harness can get:
STAGE 1 (improvement proposer) is the real ``BlockSuggester`` with
``agentic_access=True`` (read_file/grep over the harness source AND the
current best's own real eval_result.json/logs) -- it diagnoses and
proposes a fix, but writes no code. STAGE 2 (implementer) is the real
``AgentEditor`` with ``agentic_editing=True``, given the proposer's
suggestion as ``context`` (exactly ``hgm.py::_expand``'s own wiring:
suggestion text -> ``context``, ``has_suggestion=True``) -- it reads/
writes files itself and implements the proposed fix. Both come straight
from ``fw.block_suggester`` / ``fw.editor``, built by the SAME production
config already running live, so they're the exact same components/models
the production run uses -- this script only supplies the manual K-loop +
keep-the-best logic that HGM's own EXPAND does implicitly once per node.

Motivation: manual forensic analysis this session found that the live
curriculum+agentic production run's essential_meal_coverage window (5
one-shot EXPANDs off 5 DIFFERENT parents) never converged -- the failure
rate went from 38.7% to 44.9% instead of down, largely because attempts
never built on each other.

Starts from the CLEAN seed (node 0) -- isolates this experiment from
whatever confounds already exist in the live run's messy tree.

IMPORTANT lesson learned live while building this (kept here so it isn't
silently reintroduced): ``FeedbackGatherer.compile()`` has a WRITE side
effect -- it calls ``persist_round_artifacts(round_dir, feedback)``,
overwriting ``round_dir/{feedback,eval_result,strategy}.json``. An
earlier version of this script passed the LIVE production run's own real
round_000 directly as ``round_dir`` to reuse its already-real seed eval,
which silently corrupted that run's on-disk round_000 artifacts (repaired
afterward from its still-intact per-case ``logs/case_*.json`` files,
which ``compile()`` never touches). This version copies what it needs
from the live run's round_000 into ITS OWN sandboxed ``iter_000_baseline``
dir first, and every ``compile()``/``suggest()``/``apply()`` call from
then on only ever touches paths under ``OUT_ROOT`` -- never the live run.

Never reads the benchmark's scorer source -- only the harness's own
mutable source (already fair game) and real per-case evaluation output
(``CaseResult.details``, produced by the same scorer the production run
uses -- OUTPUT, not source).

Usage: python3 validate_iterative_refinement.py
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

REPO_ROOT = Path("/groups/AIC-MV/v.kulkarni1/unified_framework/self-improving-dual")
sys.path.insert(0, str(REPO_ROOT))

from meta_agent.config import load, build_components  # noqa: E402
from meta_agent import runtime_env  # noqa: E402
from meta_agent.curriculum import Curriculum  # noqa: E402
from meta_agent.models import CaseResult, EvaluationResult, EvolutionStrategy  # noqa: E402

# Already-evaluated seed (node 0) from the live local curriculum+agentic
# production run -- same seed code (byte-identical to fw.seed_dir, a
# straight shutil.copytree with zero edits), already evaluated on the
# FULL 60-case train split. Read from ONLY -- see the module docstring's
# "IMPORTANT lesson learned" for why nothing ever calls compile()/
# persist_round_artifacts with this path as round_dir.
SEED_EVAL_ROUND_DIR = (
    REPO_ROOT / "runs"
    / "20260908_043243_travel_mas_refactored_full_scale_block_tagged_curriculum_agentic_suggester_X100Y180"
    / "round_000"
)

CHECK_FULL = "commonsense:Itinerary Structure:essential_meal_coverage"
CHECK_BARE = CHECK_FULL.split(":")[-1]
CHECK_DESCRIPTION = (
    "Fires when a day doesn't have the right number of meals scheduled "
    "for its type -- a full sightseeing day needs both lunch and dinner "
    "with a real gap between them; a travel/transfer day's meal "
    "requirement depends on when the traveler arrives or departs."
)
# From the live local curriculum+agentic run's curriculum_status.json
# seed_counts (round_026's snapshot) -- the real seed occurrence count for
# this exact goal, reused here purely for an accurate directive() render.
SEED_COUNT = 19

# Which block_suggester.py block to sample for the proposer step. "mixed"
# deliberately doesn't constrain the diagnosis to one layer (prompt vs.
# verifier code) -- real production attempts at this exact goal used both
# (rounds 26/27/28/30: verifier code; round 29: prompt-only), so a
# generic 2-stage validation shouldn't artificially exclude either.
BLOCK = "mixed"

K = 5
N_CASES = 32
# Guardrail added after a real failure mode observed live: a candidate can
# make the TARGETED check's failure rate look better purely by making the
# task_agent fail to produce a plan at all on most cases (essential_meal_
# coverage can only fire on cases that actually produced a plan, so
# collapsing the denominator from 32 to 6 cases made a 0.5 rate look like
# progress over the baseline's 0.625 while overall score cratered from
# 0.318 to 0.084 -- 26/32 cases stopped completing at all). A candidate is
# only "kept" if it both improves (or ties) the targeted check AND retains
# at least this fraction of the current best's overall score.
SCORE_REGRESSION_TOLERANCE = 0.9
# Known real cases (from this session's forensic dig into round_030) whose
# final day has a late (>15:00) departure and a meal-coverage violation --
# deliberately over-represented in the fixed eval set so the signal isn't
# diluted by a mostly-passing random sample; the rest of the 32 slots are
# filled with other train cases so genuine regressions are still visible.
KNOWN_RELEVANT_CASES = [
    "105", "111", "40", "52", "53", "68", "103", "109", "101", "114",
    "12", "14", "54", "72",
]

CONFIG_PATH = REPO_ROOT / "configs/hgm_travel_full_scale_block_tagged_curriculum_agentic_suggester_X100Y180.yaml"
OUT_ROOT = REPO_ROOT / "iterative_validation_runs" / "essential_meal_coverage_v2"


def _load_seed_eval(case_ids: list[str]) -> EvaluationResult:
    """Filter the live run's real round_000 eval_result.json down to
    ``case_ids`` -- READ-ONLY (json.loads on a Path.read_text; never a
    write)."""
    raw = json.loads((SEED_EVAL_ROUND_DIR / "eval_result.json").read_text(encoding="utf-8"))
    by_id = {c["case_id"]: c for c in raw["per_case"]}
    missing = [cid for cid in case_ids if cid not in by_id]
    if missing:
        raise KeyError(f"case_ids not found in seed eval: {missing}")
    filtered = [CaseResult(**by_id[cid]) for cid in case_ids]
    passed = sum(1 for c in filtered if c.passed)
    return EvaluationResult(
        score=sum(c.score for c in filtered) / len(filtered),
        metrics={}, passed=passed, failed=len(filtered) - passed,
        per_case=filtered, wall_time_s=0.0, crashed=False,
    )


def _count_no_plan(eval_result) -> int:
    """How many cases produced no scoreable plan at all (task_failure /
    output_truncated -- the task_agent never emitted <itinerary>). These
    cases can't fire any per-check violation, so a metric scoped to one
    check silently excludes them from its denominator -- see
    SCORE_REGRESSION_TOLERANCE's comment for why that's dangerous to
    optimize against blindly."""
    return sum(1 for c in eval_result.per_case if "no plan" in ((c.details or {}).get("error") or ""))


def _build_directive() -> str:
    curriculum = Curriculum(
        goals=[(CHECK_FULL, SEED_COUNT)],
        resolution_threshold=0.15,
        patience=999,  # unused -- this script drives iterations manually
        check_descriptions={CHECK_FULL: CHECK_DESCRIPTION},
    )
    base = curriculum.directive()
    debug_instructions = (
        "\n\nThis EXPAND's job is specifically to DEBUG AND FIX this one "
        "recurring failure -- not to redesign anything else. Before "
        "proposing a fix:\n"
        "1. Look at BOTH passing and failing real cases for this check "
        "(use your read_file/grep tools over 'eval_result.json' and "
        "'logs/case_<id>.json') -- compare what a PASSING case's day "
        "looks like against a FAILING case's day of the same TYPE "
        "(transfer/departure day vs. full sightseeing day), not just "
        "failing cases in isolation. The contrast between a case that "
        "gets it right and one that doesn't is usually the fastest way "
        "to see exactly what's different.\n"
        "2. Ground your diagnosis in the harness's own source "
        "('harness/...') for whichever stage actually produces or "
        "checks this -- read the real code, don't guess at what it "
        "does.\n"
        "3. Then tell me explicitly: (a) exactly what the bug/gap is, "
        "citing the specific evidence (a case id, a line of code, an "
        "exact violation message) you found it from, and (b) exactly "
        "what fix you propose -- concrete enough that an editor could "
        "implement it without having to re-diagnose from scratch."
    )
    return base + debug_instructions


def main() -> None:
    print(f"Loading config {CONFIG_PATH} ...", flush=True)
    cfg = load(CONFIG_PATH)
    # Must run BEFORE build_components -- pushes LLM_MODEL/LLM_BASE_URL/
    # META_AGENT_PROJECT/the YAML's env: block into os.environ so the
    # task_agent subprocess (which never sees the YAML) actually calls the
    # real vLLM endpoint instead of crashing immediately.
    runtime_env.apply_all(cfg)
    fw = build_components(cfg)
    print(
        f"block_suggester.agentic_access={getattr(fw.block_suggester, 'agentic_access', None)} "
        f"editor.agentic_editing={getattr(fw.editor, 'agentic_editing', None)} "
        f"model={fw.editor.model} base_url={fw.editor.base_url}",
        flush=True,
    )
    if fw.block_suggester is None:
        raise RuntimeError("config has no block_suggester configured")

    train_ids = list(fw.train_case_ids or [])
    known = [c for c in KNOWN_RELEVANT_CASES if c in train_ids]
    rest = [c for c in sorted(train_ids, key=lambda x: (len(x), x)) if c not in known]
    case_ids = (known + rest)[:N_CASES]
    print(
        f"Fixed eval set: {len(case_ids)} cases "
        f"({len(known)} known meal-coverage-relevant): {case_ids}",
        flush=True,
    )

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    directive = _build_directive()

    def eval_dir(d: Path, round_number: int, base_round: int, strategy: EvolutionStrategy):
        eval_result = fw.evaluator.run(d, fw.benchmark_dir, case_ids=case_ids)
        feedback = fw.gatherer.compile(round_number, base_round, strategy, eval_result, d)
        fr = Curriculum.failure_rate_for(CHECK_FULL, feedback.project_metrics, len(case_ids))
        return feedback, fr

    # --- iteration 0: clean seed baseline. Copy what's needed from the
    # live run's round_000 into OUR OWN iter0 dir first, then compile()
    # (which writes feedback/eval_result/strategy.json) only ever touches
    # iter0 -- never the live run again.
    iter0 = OUT_ROOT / "iter_000_baseline"
    if not (iter0 / "task_agent").exists():
        iter0.mkdir(parents=True, exist_ok=True)
        shutil.copytree(fw.seed_dir, iter0 / "task_agent")
    (iter0 / "logs").mkdir(parents=True, exist_ok=True)
    for cid in case_ids:
        src = SEED_EVAL_ROUND_DIR / "logs" / f"case_{cid}.json"
        dst = iter0 / "logs" / f"case_{cid}.json"
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
    trace_src = SEED_EVAL_ROUND_DIR / "logs" / "trace.jsonl"
    trace_dst = iter0 / "logs" / "trace.jsonl"
    if trace_src.exists() and not trace_dst.exists():
        shutil.copy2(trace_src, trace_dst)

    baseline_strategy = EvolutionStrategy(
        target_files=[], optimization_goal="baseline seed (no edit)",
        proposed_changes="", rationale="",
    )
    t0 = time.time()
    seed_eval_result = _load_seed_eval(case_ids)
    best_feedback = fw.gatherer.compile(0, 0, baseline_strategy, seed_eval_result, iter0)
    best_fr = Curriculum.failure_rate_for(CHECK_FULL, best_feedback.project_metrics, len(case_ids))
    best_dir = iter0
    best_score = best_feedback.eval_result.score
    best_no_plan = _count_no_plan(best_feedback.eval_result)
    print(
        f"[iter 0 baseline] score={best_score:.3f} {CHECK_BARE}_failure_rate={best_fr} "
        f"no_plan={best_no_plan}/{len(case_ids)} ({time.time() - t0:.1f}s)",
        flush=True,
    )
    history = [{
        "iter": 0, "dir": str(iter0), "score": best_score,
        "failure_rate": best_fr, "no_plan": best_no_plan, "kept": True, "note": "baseline",
    }]
    (OUT_ROOT / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    for i in range(1, K + 1):
        cand_dir = OUT_ROOT / f"iter_{i:03d}_candidate"
        (cand_dir / "logs").mkdir(parents=True, exist_ok=True)
        print(f"\n=== iteration {i}: proposing + editing off {best_dir.name} ===", flush=True)

        # --- STAGE 1: improvement proposer (real BlockSuggester, agentic) ---
        t0 = time.time()
        suggestion = fw.block_suggester.suggest(
            block=BLOCK,
            agent_dir=best_dir / "task_agent",
            out_dir=cand_dir,
            node_id=i,
            feedback=best_feedback,
            failure_summary=None,
            siblings=None,
            curriculum_directive=directive,
        )
        print(
            f"  propose: {'got ' + str(len(suggestion)) + ' chars' if suggestion else 'EMPTY/None'} "
            f"({time.time() - t0:.0f}s)",
            flush=True,
        )
        if suggestion:
            print("  --- suggestion ---")
            print("  " + suggestion.replace("\n", "\n  "))
            print("  --- end suggestion ---", flush=True)

        # --- STAGE 2: implementer (real AgentEditor, agentic), given the
        # proposer's suggestion as context -- exact same wiring as
        # hgm.py::_expand (suggestion -> context, has_suggestion=True). ---
        t0 = time.time()
        edit_result = fw.editor.apply(
            best_feedback, best_dir, cand_dir,
            context=suggestion, has_suggestion=bool(suggestion),
        )
        print(
            f"  implement: success={edit_result.success} "
            f"edited_files={edit_result.edited_files} errors={edit_result.errors} "
            f"({time.time() - t0:.0f}s)",
            flush=True,
        )
        if not edit_result.success or not edit_result.edited_files:
            history.append({
                "iter": i, "dir": str(cand_dir), "score": None, "failure_rate": None,
                "kept": False, "note": f"edit failed/empty: {edit_result.errors}",
                "suggestion": suggestion,
            })
            (OUT_ROOT / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
            continue

        t0 = time.time()
        cand_strategy = edit_result.strategy or EvolutionStrategy(
            target_files=edit_result.edited_files, optimization_goal="(no strategy emitted)",
            proposed_changes="", rationale="",
        )
        cand_feedback, cand_fr = eval_dir(cand_dir, i, i - 1, cand_strategy)
        cand_score = cand_feedback.eval_result.score
        cand_no_plan = _count_no_plan(cand_feedback.eval_result)
        print(
            f"  eval: score={cand_score:.3f} {CHECK_BARE}_failure_rate={cand_fr} "
            f"no_plan={cand_no_plan}/{len(case_ids)} ({time.time() - t0:.0f}s)",
            flush=True,
        )

        improves_check = cand_fr is not None and (best_fr is None or cand_fr <= best_fr)
        retains_score = cand_score >= best_score * SCORE_REGRESSION_TOLERANCE
        kept = improves_check and retains_score
        if kept:
            best_dir, best_feedback = cand_dir, cand_feedback
            best_fr, best_score, best_no_plan = cand_fr, cand_score, cand_no_plan
        elif improves_check and not retains_score:
            print(
                f"  -> check improved but score regressed too much "
                f"({cand_score:.3f} < {best_score:.3f} * {SCORE_REGRESSION_TOLERANCE}) "
                "-- discarding despite the check-level improvement",
                flush=True,
            )
        history.append({
            "iter": i, "dir": str(cand_dir),
            "score": cand_score, "failure_rate": cand_fr, "no_plan": cand_no_plan,
            "kept": kept, "optimization_goal": cand_strategy.optimization_goal,
            "suggestion": suggestion,
        })
        (OUT_ROOT / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(f"  -> {'KEPT as new best' if kept else 'discarded (best unchanged)'}", flush=True)

    print("\n=== FINAL HISTORY ===", flush=True)
    for h in history:
        print({k: v for k, v in h.items() if k != "suggestion"}, flush=True)
    print(
        f"\nBaseline: score={history[0]['score']:.3f} failure_rate={history[0]['failure_rate']}",
        flush=True,
    )
    print(
        f"Best after {K} iterations: score={best_score:.3f} failure_rate={best_fr} "
        f"(dir={best_dir})",
        flush=True,
    )


if __name__ == "__main__":
    main()
