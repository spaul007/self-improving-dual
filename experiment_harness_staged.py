#!/usr/bin/env python3
"""Standalone experiment: build the redesigned harness in STAGES with harness-enforced gates.

Starts from an existing partial redesign (default: Qwen run 1's workspace, a working code-owned planner that scores the commonsense scheduling
dimensions well but has no hard-constraint / intercity machinery). It then runs a sequence of short, focused meta-agent sessions (local Qwen3.5-122B,
node-5), each with a FRESH context and ONE unit to improve (a grader dimension or a hard-constraint family), chosen from the current pipeline's own
train-split failures. After each session the harness -- not the model -- decides:

  * the candidate workspace is re-evaluated on ALL 60 TRAIN cases;
  * it is ACCEPTED (becomes the new checkpoint and the source of the next stage's corpus) only if the TARGETED unit's own score improves on the
    checkpoint's (by --min-gain) and the no-plan rate does not get worse; otherwise it is ROLLED BACK. Overall composite is reported but is not itself
    the gate (see unit_score()).

Information rule (same categories as HGM's editor): each session sees the seed-derived workspace it will edit, the train-split evaluation of the CURRENT
checkpoint (per case: request, plan, failed-check labels and grader messages, dimension scores), a train-only digest, the trace of that evaluation and the
project's check descriptions. Never the gold scorer's source, benchmark data or any eval-split log. Tools are those of experiment_harness_redesign.py
(list_cases / show_case / read_file / grep / write_file / str_replace_file / check_workspace / run_python / evaluate_variant / submit_variant).

The final checkpoint is measured on all held-out EVAL cases (fresh repeats) against the stored baseline passes: composite, paired bootstrap CI,
no-plan rate, dimension pass rates, per-check deltas. Local vLLM only (task agent Qwen3.5-35B on node-6).

A sibling script: it reuses the components of experiment_harness_redesign.py (import) and the framework; nothing else.
Usage: python3 experiment_harness_staged.py --out-dir staged_run --start-workspace harness_redesign_run_1/work/workspace
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

import experiment_harness_redesign as h  # noqa: E402
from meta_agent import config as cfg_mod  # noqa: E402
from meta_agent import runtime_env  # noqa: E402

log = h.log

UNITS: dict[str, tuple[str, ...]] = {
    "Route Consistency": ("commonsense:Route Consistency:",),
    "Itinerary Structure": ("commonsense:Itinerary Structure:",),
    "Business Hours": ("commonsense:Business Hours:",),
    "Time Feasibility": ("commonsense:Time Feasibility:",),
    "Duration Rationality": ("commonsense:Duration Rationality:",),
    "Activity Diversity": ("commonsense:Activity Diversity:",),
    "Sandbox Compliance": ("commonsense:Sandbox Compliance:",),
    "Cost Calculation Accuracy": ("commonsense:Cost Calculation Accuracy:",),
    "hard constraints: transport (train/flight)": ("hard:train_", "hard:flight_"),
    "hard constraints: restaurant": ("hard:restaurant_",),
    "hard constraints: hotel": ("hard:hotel_",),
    "hard constraints: attraction": ("hard:attraction_",),
    "hard constraints: budget": ("hard:budget_",),
}

STAGE_SYSTEM = """You are a meta-agent continuing the redesign of a travel-planning harness around a WEAK runtime model, one focused stage at a time.

BACKGROUND. The task agent turns a traveler's request into a full day-by-day itinerary. A hidden grader scores every itinerary against many named checks;
you never see its source, only real graded runs. The workspace ALREADY contains a working code-owned pipeline (written in earlier sessions): code makes
all tool calls (through ToolWrapper), schedules the days and renders the required line format.
The runtime model ({task_model}) is a small local backbone. Small local backbones in this project have repeatedly shown unreliable behavior on
tasks like multi-step tool comparisons, exact value transcription, and composing a long itinerary in one generation -- do not assume this run's backbone
is immune to that just because its own baseline score looks healthy. Based on the reliability and instruction-following of this backbone, be
strategic about what you want to use the task agent LLM for and what you want to do deterministically in code. The task agent's LLM can be used
not just for full-stage generation but also for smaller helper subtasks you judge it's well-suited to (e.g. extracting key values from free text).

GRADING STRUCTURE (visible in the logs' dimension_scores / hard_score): composite = (commonsense + hard) / 2. Each commonsense dimension counts (1/8) only
if EVERY check in it passes; the hard score counts only if EVERY hard constraint of the case passes (they depend on what the traveler asked for). A case
with no plan scores 0. Fixing one check rarely moves the score; whole dimensions must pass consistently, and a fix for one check must not break others.

THIS STAGE'S UNIT: {unit}
{unit_block}

HOW YOUR WORK IS JUDGED. After your session the harness re-evaluates your workspace on ALL 60 train cases and ACCEPTS it only if THIS UNIT's own score
(currently {unit_score:.4f} on train) improves by at least {min_gain} AND the no-plan rate does not get worse; otherwise your whole stage is rolled back.
(Overall composite is currently {ckpt_composite:.4f} and is reported alongside, but it is NOT itself the accept/reject gate -- it blends 8 other units
this stage does not touch, so it is a much noisier signal for a single-unit fix than the unit's own pass rate.) So improve this unit WITHOUT regressing
what already works (the pipeline already satisfies most scheduling checks -- do not rewrite the scheduler; extend it). A crash or an empty result on any
case is a regression, and will worsen the no-plan guard even if it happens not to touch this unit's own checks.

RULES
- Keep the workflow.py -> mas_workflow.run_task(task) -> AgentOutput contract; the final result is the complete itinerary text. Extend the existing
  code in small, testable pieces (add functions/modules; do not restart from scratch). No hard-coding of case ids, cities or answers; general behavior only.
- No file/network/process/env access in the code you write. The model and endpoint are fixed (local {task_model}): do not try to change them.
  workflow.py, tool_wrapper.py, tools_schema.json and agents/immutable/ are frozen.
- run_python is free and makes no LLM calls. check_workspace's static/policy checks are free too, but it also makes ONE real local-LLM run on a
  fixed train case to confirm a plan is actually produced (does not consume an evaluate_variant call). evaluate_variant evaluates ALL 60 train
  cases by default (takes a while) -- pass case_ids (train ids only) to check just a specific subset faster while iterating. You have
  {eval_rounds} evaluate_variant calls total. Turn budget: {max_turns}.
- Test on the ACTUAL cases where this unit currently fails, ALL of them if you can, not just one or two -- and be aware that a small, fixed
  sample looking clean is not the same as being fixed. It is easy to overfit to whatever small set you keep re-testing against without ever
  re-checking the wider set the fix is meant to generalize over. run_python against train_data.failing(...) is free -- use it to check every
  failing case offline before spending a limited evaluate_variant call, and spend at least one evaluate_variant call with NO case_ids (the
  full 60) before you submit, however good a small sample looks.
- In addition to the real stored cases, feel free to write your OWN small synthetic test inputs for any parsing/extraction logic you write --
  short constructed strings covering variations you'd expect in real requests (different phrasings, punctuation, quoting styles, name formats
  with apostrophes or multi-word names) -- and check your function against them directly with run_python, the same way you'd test any parsing
  code. The real stored cases tell you whether it works on what's already in front of you; synthetic edge cases tell you whether the
  underlying logic actually generalizes, rather than happening to match only the specific inputs you've seen so far.

WORKFLOW
1. Read corpus/digest.md and the train cases where this unit fails (list_cases with failed_check=..., show_case) -- these are the CURRENT pipeline's own
   plans and the grader's messages. Read the relevant workspace code (start with mas_workflow.py and the planner).
2. Use the provided tools to analyze and understand the root cause, and plan a fix.
3. Use run_python heavily: print REAL data (plans, tool outputs) first and parse those; do not assume formats from prompts (square brackets in the
   format spec are placeholders, not literal). Test offline on the stored train requests (tool lookups work offline through ToolWrapper after
   train_data.use_case(cid); LLM calls do not) before spending an evaluate_variant call.
4. After editing, call check_workspace (full policy/syntax scan, HGM's real validator suite, and a check that a real train case still produces a
   plan) to catch cross-file mistakes and no-plan regressions BEFORE spending a limited evaluate_variant call on something that would just crash,
   get rejected, or silently produce no plan.
5. When implementing a fix, evaluate your changes by calling evaluate_variant -- pass case_ids=[the train ids where this unit currently fails] for a fast targeted check while you're
   still iterating, and call it with no case_ids (all 60) once you believe it's fixed, since that full-60 result is what decides accept/rollback.
   Read the per-case results, refine. Only submit_variant once the unit improves and nothing else got worse.
6. Before calling submit_variant, re-read your OWN changed code and check every claim you are about to write in the summary against what the
   code actually does line by line -- not what you intended it to do. A summary that overstates what a change does (e.g. "the LLM no longer
   decides X" when the LLM's tool schema still lets it decide X) is worse than an accurate but modest one: it hides exactly the failure mode
   most worth catching before this gets built on further.
7. submit_variant with a short summary of what you changed."""


def eval_train(fw: Any, ws: Path, run_dir: Path, train_ids: list[str]) -> Any:
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    shutil.copytree(ws, run_dir / "task_agent", ignore=shutil.ignore_patterns("__pycache__"))
    res = fw.evaluator.run(run_dir, fw.benchmark_dir, case_ids=train_ids)
    bad = h.route_problems(run_dir)
    if bad:
        log(f"WARNING: non-local LLM calls in {run_dir.name}: {bad}")
    return res


class _LoadedCase:
    __slots__ = ("case_id", "score", "details", "error")

    def __init__(self, case_id: str, score: float, details: dict, error: Optional[str]):
        self.case_id, self.score, self.details, self.error = case_id, score, details, error


class _LoadedResult:
    def __init__(self, per_case: list[_LoadedCase]):
        self.per_case = per_case


def load_eval_dir(round_dir: Path, train_ids: list[str]) -> Any:
    """Reconstruct a res-like object (just the .per_case surface eval_train's callers use) from an EARLIER run's already-written
    logs/case_<id>.json files -- reused when the starting workspace is byte-identical to a prior run's, to skip re-paying the
    ~4 min full 60-case evaluation. Raises if any expected case is missing (caller falls back to a fresh eval_train)."""
    logs = round_dir / "logs"
    per_case = []
    for cid in train_ids:
        p = logs / f"case_{cid}.json"
        if not p.exists():
            # evaluator._safe_case_filename may sanitize ids differently; fall back to a scan.
            matches = [f for f in logs.glob("case_*.json") if json.loads(f.read_text()).get("case_id") == cid]
            if not matches:
                raise FileNotFoundError(f"no cached case result for {cid!r} in {logs}")
            p = matches[0]
        d = json.loads(p.read_text())
        err = d.get("error")
        per_case.append(_LoadedCase(str(d["case_id"]), float(d.get("score") or 0.0), d.get("details") or {}, None if err in (None, "None") else err))
    return _LoadedResult(per_case)


def summarize_eval(res: Any) -> dict:
    recs = {str(c.case_id): h.compact(c.details, c.score) for c in res.per_case}
    return {"composite": statistics.fmean(r["score"] for r in recs.values()), "no_plan_rate": statistics.fmean(r["no_plan"] for r in recs.values()),
            "recs": recs}


def unit_failures(recs: dict[str, dict]) -> dict[str, int]:
    """Failing (case, check) instances per unit in an evaluation (no-plan cases excluded)."""
    out = {u: 0 for u in UNITS}
    for r in recs.values():
        if r["no_plan"]:
            continue
        for lab in r["failed"]:
            for u, pref in UNITS.items():
                if lab.startswith(pref):
                    out[u] += 1
    return out


def unit_score(recs: dict[str, dict], unit: str) -> Optional[float]:
    """The TARGETED unit's own score: fraction of cases -- among those the unit applies to (no-plan cases excluded from
    the numerator, counted as failing the unit if it would otherwise apply) -- where EVERY applicable check/constraint
    in this one unit passes. None if the unit never applies to any case. This is what a stage's gate judges, per the
    same logic as a grader dimension: a stage that fixes its OWN target unit should be judged on that unit's own
    progress, not on overall composite (composite blends 8 other units the stage never touched and is much noisier
    as a signal for a single-unit edit)."""
    pref = UNITS[unit]
    n = k = 0
    for r in recs.values():
        applicable = [lab for lab in r["applicable"] if lab.startswith(pref)]
        if not applicable:
            continue
        n += 1
        if not r["no_plan"] and all(lab not in r["failed"] for lab in applicable):
            k += 1
    return (k / n) if n else None


def build_stage_corpus(stage_dir: Path, res: Any, baseline_dir: Path, run_dir: Path, seed: int, inloop_n: int) -> dict:
    """Corpus for a stage = the current checkpoint's TRAIN evaluation (per case: request, plan, graded failures), like a parent's eval in HGM."""
    corpus = stage_dir / "corpus"
    if corpus.exists():
        shutil.rmtree(corpus)
    (corpus / "cases").mkdir(parents=True)
    (corpus / "semantics").mkdir()
    index, comp, dim_scores, fail_count, n_noplan = [], [], {}, {}, 0
    for c in res.per_case:
        d = c.details or {}
        msgs = h.failed_messages(d)
        _, _, np_ = h.gold_label_sets(d)
        n_noplan += np_
        comp.append(float(c.score or 0.0))
        for dim, v in (d.get("dimension_scores") or {}).items():
            dim_scores.setdefault(dim, []).append(float(v))
        for lab in msgs:
            fail_count[lab] = fail_count.get(lab, 0) + 1
        row = {"case_id": str(c.case_id), "composite": c.score, "query": d.get("query"),
               "raw_plan_text": d.get("raw_plan_text") or d.get("raw_result") or "", "no_plan": bool(np_),
               "error": str(d.get("error") or c.error or "")[:600], "commonsense_score": d.get("commonsense_score"),
               "hard_score": d.get("hard_score"), "dimension_scores": d.get("dimension_scores"), "dimension_details": d.get("dimension_details"),
               "hard_constraints": d.get("hard_constraints"), "failed_checks": msgs}
        (corpus / "cases" / f"{c.case_id}.json").write_text(json.dumps(row, indent=1))
        index.append({"case_id": str(c.case_id), "composite": c.score, "n_failed_checks": len(msgs)})
    (corpus / "index.json").write_text(json.dumps(index, indent=1))
    digest = [f"# Train-split digest of the CURRENT pipeline's evaluation ({len(index)} train cases)", "",
              f"- mean composite: {statistics.fmean(comp):.4f}", f"- no_plan_rate: {n_noplan / max(len(index), 1):.3f}",
              "- dimension means: " + ", ".join(f"{k}={statistics.fmean(v):.3f}" for k, v in sorted(dim_scores.items())), "",
              "## Failed checks (number of train cases failing the check)", ""]
    digest += [f"- {lab}: {n}" for lab, n in sorted(fail_count.items(), key=lambda kv: -kv[1])]
    (corpus / "digest.md").write_text("\n".join(digest))
    tp = run_dir / "logs" / "trace.jsonl"
    if tp.exists():
        shutil.copy(tp, corpus / "trace.jsonl")
    else:
        (corpus / "trace.jsonl").write_text("")
    for name in h.SEMANTICS_FILES:
        src = h.PROJECT_DIR / "adapter" / name
        if src.exists():
            shutil.copy(src, corpus / "semantics" / name)
    rng = random.Random(seed)
    order = sorted(index, key=lambda r: (float(r["composite"] or 0.0), rng.random()))
    n_low = max(1, round(inloop_n * 2 / 3))
    low = [r["case_id"] for r in order[:n_low]]
    rest = [r["case_id"] for r in order[n_low:]]
    rng.shuffle(rest)
    inloop = low + rest[: max(0, inloop_n - n_low)]
    comps = {r["case_id"]: float(r["composite"] or 0.0) for r in index}
    info = {"inloop_train_cases": inloop, "inloop_pass1": comps, "final_eval_cases": [], "stored_control": {}}
    (stage_dir / "targets.json").write_text(json.dumps(info))
    return info


def describe_unit(unit: str, recs: dict[str, dict], semantics: dict, fail_examples: int = 3) -> str:
    pref = UNITS[unit]
    labs: dict[str, int] = {}
    applied: dict[str, int] = {}
    for r in recs.values():
        if r["no_plan"]:
            continue
        for l in r["applicable"]:
            if l.startswith(pref):
                applied[l] = applied.get(l, 0) + 1
        for l in r["failed"]:
            if l.startswith(pref):
                labs[l] = labs.get(l, 0) + 1
    lines = ["Checks in this unit, with how often each fails in the CURRENT pipeline's train evaluation (failing cases / cases where it applies):"]
    for l in sorted(applied, key=lambda x: -labs.get(x, 0)):
        desc = semantics.get(l) or semantics.get(l.replace("hard:", "", 1)) or ""
        lines.append(f"- {l}: {labs.get(l, 0)}/{applied[l]} -- {desc[:220]}")
    if unit.startswith("hard"):
        lines.append("Note: the hard score is all-or-nothing per case (every hard constraint of the case must pass), and which constraints apply depends "
                     "on what the traveler asked for in the request; read the request text and satisfy exactly what it states.")
    return "\n".join(lines)


def _fmt_opt(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{x:.4f}"


def choose_unit(fails: dict[str, int], attempts: dict[str, int], max_attempts: int) -> Optional[str]:
    cands = [(n, u) for u, n in fails.items() if n > 0 and attempts.get(u, 0) < max_attempts]
    return max(cands)[1] if cands else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--start-workspace", type=Path, default=REPO_ROOT / "harness_redesign_run_1/work/workspace")
    ap.add_argument("--baseline-dir", type=Path, default=h.DEFAULT_BASELINE)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-stages", type=int, default=6)
    ap.add_argument("--max-attempts-per-unit", type=int, default=2)
    ap.add_argument("--stage-max-turns", type=int, default=100)
    ap.add_argument("--stage-eval-rounds", type=int, default=4)
    ap.add_argument("--inloop-cases", type=int, default=8)
    ap.add_argument("--min-gain", type=float, default=0.005, help="composite gain on the 60 train cases required to accept a stage")
    ap.add_argument("--final-repeats", type=int, default=3)
    ap.add_argument("--parallelism", type=int, default=15)
    ap.add_argument("--case-timeout", type=float, default=3000.0)
    ap.add_argument("--reuse-eval-dir", type=Path, default=None,
                    help="an earlier stage-0 eval dir (e.g. harness_staged_run/eval_ckpt_0) to reuse instead of re-evaluating the "
                         "starting workspace, IF that dir's copied task_agent is byte-identical to --start-workspace")
    ap.add_argument("--force-unit", default=None, help="force stage 1's unit instead of auto-selecting by failure count (e.g. "
                    "for a quick single-stage check); later stages still auto-select. One of: " + ", ".join(UNITS))
    args = ap.parse_args()
    if args.force_unit is not None:
        assert args.force_unit in UNITS, f"--force-unit must be one of {list(UNITS)}, got {args.force_unit!r}"
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    log(f"output dir: {out}; starting from {args.start_workspace}")

    cfg = cfg_mod.load(str(h.CONFIG))
    runtime_env.apply_all(cfg)
    live = h.assert_local_only(args.baseline_dir)
    fw = cfg_mod.build_components(cfg)
    fw.evaluator.parallelism = args.parallelism
    fw.evaluator.wall_time_s = args.case_timeout
    train_ids = [str(x) for x in fw.train_case_ids]
    semantics = json.loads((h.PROJECT_DIR / "adapter" / "error_semantics.json").read_text())
    seed_dir = Path(fw.seed_dir)

    # ---- checkpoint 0: the starting workspace (or resume from the last accepted checkpoint) ----
    hist_path = out / "staged_history.json"
    history: list[dict] = json.loads(hist_path.read_text()) if hist_path.exists() else []
    ckpt_ws = out / "checkpoint"
    if not history:
        start = args.start_workspace.resolve()
        probs = h.workspace_problems(start, seed_dir)
        assert not probs, f"start workspace violates the edit policy: {probs[:3]}"
        smoke = h.run_smoke(start)
        assert smoke is None, f"start workspace does not import: {smoke}"
        if ckpt_ws.exists():
            shutil.rmtree(ckpt_ws)
        shutil.copytree(start, ckpt_ws, ignore=shutil.ignore_patterns("__pycache__"))
        cur_eval_dir = out / "eval_ckpt_0"
        res = None
        if args.reuse_eval_dir is not None:
            reuse_dir = args.reuse_eval_dir.resolve()
            try:
                # cheap correctness check: the reused dir's own copied starting code must be byte-identical to ours.
                import filecmp
                cmp = filecmp.dircmp(reuse_dir / "task_agent", ckpt_ws)
                identical = not cmp.diff_files and not cmp.left_only and not cmp.right_only and \
                    all(not filecmp.dircmp(reuse_dir / "task_agent" / sd, ckpt_ws / sd).diff_files for sd in cmp.common_dirs)
                if identical:
                    log(f"stage 0: reusing precomputed train evaluation from {reuse_dir} (starting workspace verified byte-identical)")
                    res = load_eval_dir(reuse_dir, train_ids)
                    cur_eval_dir = reuse_dir
                else:
                    log(f"stage 0: --reuse-eval-dir workspace differs from the current starting workspace -- ignoring, evaluating fresh")
            except Exception as exc:  # noqa: BLE001
                log(f"stage 0: could not reuse {args.reuse_eval_dir} ({exc!r}) -- evaluating fresh")
        if res is None:
            log("stage 0: evaluating the starting workspace on all 60 train cases ...")
            res = eval_train(fw, ckpt_ws, cur_eval_dir, train_ids)
        cur = summarize_eval(res)
        history.append({"stage": 0, "unit": "(start)", "accepted": True, "composite": cur["composite"], "no_plan_rate": cur["no_plan_rate"]})
        hist_path.write_text(json.dumps(history, indent=1))
        log(f"stage 0: composite {cur['composite']:.4f}, no-plan {cur['no_plan_rate']:.3f}")
    else:
        log("resuming: re-evaluating the last accepted checkpoint on all 60 train cases ...")
        cur_eval_dir = out / "eval_ckpt_current"
        res = eval_train(fw, ckpt_ws, cur_eval_dir, train_ids)
        cur = summarize_eval(res)
    attempts: dict[str, int] = {}
    for r in history:
        if r["stage"] > 0:
            attempts[r["unit"]] = attempts.get(r["unit"], 0) + 1

    for k in range(len(history), args.n_stages + 1):
        fails = unit_failures(cur["recs"])
        if k == 1 and args.force_unit is not None:
            unit = args.force_unit
            log(f"stage {k}: unit FORCED to '{unit}' (--force-unit)")
        else:
            unit = choose_unit(fails, attempts, args.max_attempts_per_unit)
        if unit is None:
            log("no unit left to improve; stopping")
            break
        attempts[unit] = attempts.get(unit, 0) + 1
        log(f"stage {k}: unit '{unit}' ({fails[unit]} failing check instances on train); checkpoint composite {cur['composite']:.4f}")
        sdir = out / f"stage_{k}"
        info = build_stage_corpus(sdir, res, args.baseline_dir, cur_eval_dir, args.seed + k, args.inloop_cases)
        sargs = argparse.Namespace(baseline_dir=args.baseline_dir, seed=args.seed + k, max_turns=args.stage_max_turns,
                                   eval_rounds=args.stage_eval_rounds, inloop_cases=args.inloop_cases, final_repeats=args.final_repeats,
                                   parallelism=args.parallelism, case_timeout=args.case_timeout, redo=False)
        sess = h.Session(sdir, fw, sargs, info)
        sess.setup(start_workspace=ckpt_ws)
        esc = lambda t: t.replace("{", "{{").replace("}", "}}")  # noqa: E731 -- these strings pass through str.format later
        start_unit_score = unit_score(cur["recs"], unit)
        task_model = os.environ.get("LLM_MODEL", "Qwen3.5-35B-A3B")
        system = STAGE_SYSTEM.replace("{unit}", esc(unit)).replace("{unit_block}", esc(describe_unit(unit, cur["recs"], semantics))) \
            .replace("{ckpt_composite:.4f}", f"{cur['composite']:.4f}") \
            .replace("{unit_score:.4f}", _fmt_opt(start_unit_score)).replace("{min_gain}", f"{args.min_gain:g}") \
            .replace("{task_model}", esc(task_model))
        user = (f"Improve the unit '{unit}' as described. Start with corpus/digest.md and the train cases where this unit fails, read the relevant "
                "workspace code, and use run_python on real data before you change anything.")
        t0 = time.time()
        dev = sess.develop(system_prompt=system, user_message=user)
        log(f"stage {k} session done in {time.time() - t0:.0f}s: submitted={dev['submitted']} turns={dev['turns_used']} eval_calls={dev['eval_calls_used']}")
        rec = {"stage": k, "unit": unit, "turns": dev["turns_used"], "eval_calls": dev["eval_calls_used"], "submitted": dev["submitted"],
               "summary": dev.get("summary", "")[:800], "before_composite": cur["composite"], "before_no_plan": cur["no_plan_rate"]}
        cand = sess.ws
        probs = h.workspace_problems(cand, sess.orig)
        smoke = h.run_smoke(cand) if not probs else None
        if probs or smoke:
            rec.update(accepted=False, reason=f"invalid: {(probs or [smoke])[0][:200]}")
        else:
            cres = eval_train(fw, cand, out / f"eval_stage_{k}", train_ids)
            cs = summarize_eval(cres)
            before_unit, after_unit = unit_score(cur["recs"], unit), unit_score(cs["recs"], unit)
            # Gate on the TARGETED unit's own score, not overall composite: composite blends 8 other units this
            # stage never touched, and is a much noisier signal for a single-unit edit than the unit's own progress.
            # The no-plan guard stays composite-independent -- a fix that starts breaking plans is rejected regardless.
            ok = (before_unit is not None and after_unit is not None and after_unit >= before_unit + args.min_gain
                  and cs["no_plan_rate"] <= cur["no_plan_rate"] + 0.02)
            rec.update(accepted=ok, composite=cs["composite"], no_plan_rate=cs["no_plan_rate"],
                       unit_score_before=before_unit, unit_score_after=after_unit,
                       reason=("accepted" if ok else
                               f"rolled back (unit score {_fmt_opt(after_unit)} vs {_fmt_opt(before_unit)}, "
                               f"composite {cs['composite']:.4f} vs {cur['composite']:.4f}, no-plan {cs['no_plan_rate']:.3f})"))
            if ok:
                shutil.rmtree(ckpt_ws)
                shutil.copytree(cand, ckpt_ws, ignore=shutil.ignore_patterns("__pycache__"))
                res, cur, cur_eval_dir = cres, cs, out / f"eval_stage_{k}"
        history.append(rec)
        hist_path.write_text(json.dumps(history, indent=1))
        log(f"stage {k}: {rec.get('reason')} | '{unit}' score now {_fmt_opt(rec.get('unit_score_after', rec.get('unit_score_before')))} "
            f"| checkpoint composite now {cur['composite']:.4f}")

    # ---- final measurement of the last accepted checkpoint on all held-out eval cases ----
    log("final measurement of the last accepted checkpoint on all held-out eval cases ...")
    fdir = out / "final"
    fargs = argparse.Namespace(baseline_dir=args.baseline_dir, seed=args.seed, max_turns=1, eval_rounds=0, inloop_cases=args.inloop_cases,
                               final_repeats=args.final_repeats, parallelism=args.parallelism, case_timeout=args.case_timeout, redo=False)
    fdir.mkdir(exist_ok=True)
    finfo = h.prepare(fargs, fw, fdir)
    fsess = h.Session(fdir, fw, fargs, finfo)
    fsess.setup(start_workspace=ckpt_ws)
    m = fsess.measure()
    tot = {"n_llm_responses": 0, "n_status_failed_retries": 0, "n_exception_retries": 0, "n_terminal_failed_responses": 0}
    for tr in fsess.dir.glob("final_*/logs/trace.jsonl"):
        fh = h.analyze_trace_file(tr)
        for kk in tot:
            tot[kk] += fh.get(kk, 0)
    tot["incidence_rate_pct"] = h.incidence_rate_pct(tot) if tot["n_llm_responses"] else 0.0
    dev = {"submitted": True, "summary": "; ".join(f"stage {r['stage']} [{r['unit']}]: {r.get('reason', 'start')}" for r in history),
           "turns_used": sum(r.get("turns", 0) for r in history), "eval_calls_used": sum(r.get("eval_calls", 0) for r in history), "inloop_rounds": []}
    rep = h.build_report(out, m, dev, fargs, live, tot)
    table = ["", "## Stage history (gate = the TARGETED unit's own score on all 60 TRAIN cases, not overall composite)", "",
             "| Stage | Unit | Turns | Eval calls | Unit score before -> after | Composite before -> after | No-plan after | Outcome |",
             "|---|---|---|---|---|---|---|---|"]
    for r in history:
        table.append(f"| {r['stage']} | {r['unit']} | {r.get('turns', '-')} | {r.get('eval_calls', '-')} | "
                     f"{_fmt_opt(r.get('unit_score_before'))} -> {_fmt_opt(r.get('unit_score_after'))} | "
                     f"{r.get('before_composite', float('nan')):.4f} -> {r.get('composite', float('nan')):.4f} | {r.get('no_plan_rate', float('nan')):.3f} | "
                     f"{r.get('reason', 'start')} |")
    (out / "REPORT.md").write_text(rep.replace("# Harness redesign experiment", "# Staged harness redesign (continuation of an earlier partial redesign)", 1) + "\n".join(table))
    (out / "summary.json").write_text(json.dumps({"history": history, **{k2: v for k2, v in m.items() if k2 != "records"}}, indent=1))
    print("\n" + (out / "REPORT.md").read_text())
    print(f"\noutputs: {out}")


if __name__ == "__main__":
    main()
