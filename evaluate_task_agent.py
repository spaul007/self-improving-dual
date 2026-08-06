"""Evaluate any given task-agent directory against a benchmark, using the
exact same evaluator/scorer/gatherer pipeline the real HGM loop uses -- no
``snapshot_tree`` dependency, no budget-selection machinery (see
``snapshot_eval.py`` for that use case instead: picking the best agent at a
given optimization budget from a run's history).

Prints the full project-specific metrics (e.g. for math_mas: ``accuracy``,
``predictor_accuracy``, ``fixed_by_reflector``, ``broken_by_reflector``,
timing), not just the plain composite score -- by calling
``fw.gatherer.compile(...)`` for real (the same call HGM itself makes),
which dispatches to the project scorer's ``aggregate()`` method.

    # Full benchmark, the pristine seed (the "baseline" agent):
    python3 evaluate_task_agent.py --config configs/hgm_math_mas_sanity.yaml \\
        --agent-dir projects/math_mas/math_mas

    # A specific HGM round's edited code, from a completed run:
    python3 evaluate_task_agent.py --config configs/hgm_math_mas_sanity.yaml \\
        --agent-dir runs/<experiment>/round_003/task_agent

    # A specific set of case ids:
    python3 evaluate_task_agent.py --config configs/hgm_math_mas_sanity.yaml \\
        --agent-dir projects/math_mas/math_mas \\
        --case-ids test/algebra/101.json,test/precalculus/697.json

    # Quick smoke check on the first 10 benchmark cases:
    python3 evaluate_task_agent.py --config configs/hgm_math_mas_sanity.yaml \\
        --agent-dir projects/math_mas/math_mas --limit 10

    # 3 repeats averaged (LLM stochasticity), higher concurrency:
    python3 evaluate_task_agent.py --config configs/hgm_math_mas_sanity.yaml \\
        --agent-dir projects/math_mas/math_mas --repeats 3 --parallelism 20

Writes ``<out-dir>/<label>_<case_set>/summary.json`` (mean/std score across
repeats, per-run breakdown, full project_metrics) plus one real
``feedback.json``/``eval_result.json``/``strategy.json`` per run (same shape
as any HGM round's, viewable in ``hgm_dashboard.py``) under
``<out-dir>/<label>_<case_set>/run_<i>/``.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Optional

from meta_agent import config as cfg_mod
from meta_agent import runtime_env
from meta_agent.evaluator import load_cases
from meta_agent.feedback_gatherer import render_metrics
from meta_agent.models import EvolutionStrategy


def _isolated_round_dir(base: Path, agent_dir: Path, label: str) -> Path:
    """A throwaway round-dir whose ``task_agent`` symlinks the real agent
    code, so the evaluator's own log writes (``case_*.json``,
    ``trace.jsonl``, ``scratch/``) never clobber the original round's logs
    -- or, for a pristine seed dir, never write into the seed itself. Same
    pattern as ``snapshot_eval.py``'s own helper; any prior dir at the same
    label is removed first so re-invocations don't mix with stale artifacts.
    """
    run_dir = base / label
    if run_dir.exists() or run_dir.is_symlink():
        shutil.rmtree(run_dir, ignore_errors=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "task_agent").symlink_to(agent_dir.resolve(), target_is_directory=True)
    return run_dir


def _resolve_case_ids(
    args: argparse.Namespace, fw: Any
) -> tuple[Optional[list[str]], str]:
    if args.case_ids:
        return [c.strip() for c in args.case_ids.split(",") if c.strip()], "custom"
    if args.eval_split:
        if not fw.eval_case_ids:
            raise SystemExit("--eval-split given but the config has no eval split")
        return fw.eval_case_ids, "eval_split"
    if args.limit is not None:
        all_cases = load_cases(fw.benchmark_dir)
        ids = [str(c.get("id") or c.get("case_id")) for c in all_cases[: args.limit]]
        return ids, f"first_{args.limit}"
    return None, "full_benchmark"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate a given task-agent directory against a benchmark, "
        "printing the full project-specific metrics."
    )
    parser.add_argument(
        "--config", type=Path, required=True,
        help="YAML config (supplies evaluator/scorer/gatherer wiring, benchmark dir)",
    )
    parser.add_argument(
        "--agent-dir", type=Path, required=True,
        help="Directory holding the runnable agent code (a workflow.py sibling) -- "
        "the pristine seed, or any round's task_agent/",
    )
    parser.add_argument("--case-ids", type=str, default=None, help="Comma-separated case ids")
    parser.add_argument(
        "--eval-split", action="store_true", help="Use the config's held-out eval split"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Evaluate only the first N cases of the full benchmark "
        "(ignored if --case-ids/--eval-split given)",
    )
    parser.add_argument(
        "--parallelism", type=int, default=None,
        help="Override evaluator.config.parallelism for this run only",
    )
    parser.add_argument(
        "--repeats", type=int, default=1,
        help="Number of full evaluation runs, averaged (mean +/- std reported). Default: 1.",
    )
    parser.add_argument(
        "--label", type=str, default=None,
        help="Name for this run's output dir (default: derived from --agent-dir)",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path("runs/adhoc_eval"),
        help="Where isolated per-run logs + the summary JSON are written. Default: runs/adhoc_eval",
    )
    args = parser.parse_args()

    agent_dir = args.agent_dir.resolve()
    # Resolve to absolute: the evaluator subprocess runs with cwd set to the
    # isolated round dir, so a relative out_dir would make trace_path (passed
    # via META_AGENT_TRACE_PATH) resolve against the wrong directory and
    # crash every case on the first call_llm (FileNotFoundError opening
    # trace.jsonl).
    args.out_dir = args.out_dir.resolve()
    if not agent_dir.is_dir():
        raise SystemExit(f"--agent-dir not found: {agent_dir}")
    if not (agent_dir / "workflow.py").is_file():
        print(
            f"warning: {agent_dir}/workflow.py not found -- is this really a "
            "task-agent dir?"
        )

    cfg = cfg_mod.load(args.config)
    runtime_env.apply_all(cfg)
    fw = cfg_mod.build_components(cfg)

    if args.parallelism is not None:
        fw.evaluator.parallelism = max(1, int(args.parallelism))
        print(f"# evaluator parallelism overridden to {fw.evaluator.parallelism}")

    case_ids, case_set = _resolve_case_ids(args, fw)
    label = args.label or agent_dir.name
    eval_base = args.out_dir / f"{label}_{case_set}"
    repeats = max(1, int(args.repeats))

    print(f"# evaluating {agent_dir}")
    print(
        f"# case_set={case_set}  n_cases={'(full benchmark)' if case_ids is None else len(case_ids)}  "
        f"repeats={repeats}"
    )

    per_run: list[dict[str, Any]] = []
    for run_idx in range(1, repeats + 1):
        iso_round_dir = _isolated_round_dir(eval_base, agent_dir, f"run_{run_idx}")
        result = fw.evaluator.run(iso_round_dir, fw.benchmark_dir, case_ids=case_ids)

        strategy = EvolutionStrategy(
            target_files=[],
            optimization_goal=f"Standalone evaluation of {agent_dir} (not an HGM round)",
            proposed_changes="",
            rationale="",
        )
        # The real gatherer call HGM itself makes -- persists feedback.json/
        # eval_result.json/strategy.json and dispatches to the project
        # scorer's aggregate(), so project_metrics come out identical in
        # shape to any real HGM round's.
        feedback = fw.gatherer.compile(0, 0, strategy, result, iso_round_dir)

        total = result.passed + result.failed
        print(
            f"\nrun {run_idx}/{repeats}: score={result.score:.4f} "
            f"passed={result.passed}/{total} crashed={result.crashed} "
            f"wall_time_s={result.wall_time_s:.1f}"
        )
        if feedback.project_metrics:
            for line in render_metrics(feedback.project_metrics, cap=20, indent="    "):
                print(line)

        per_run.append(
            {
                "run": run_idx,
                "score": result.score,
                "passed": result.passed,
                "failed": result.failed,
                "n_cases": total,
                "wall_time_s": result.wall_time_s,
                "crashed": result.crashed,
                "project_metrics": feedback.project_metrics,
                "logs_dir": str(iso_round_dir / "logs"),
            }
        )

    scores = [r["score"] for r in per_run]
    n = len(scores)
    mean_score = sum(scores) / n
    std_score = (sum((s - mean_score) ** 2 for s in scores) / n) ** 0.5

    summary = {
        "agent_dir": str(agent_dir),
        "config": str(args.config),
        "case_set": case_set,
        "n_cases": per_run[0]["n_cases"],
        "repeats": repeats,
        "mean_score": mean_score,
        "std_score": std_score,
        "per_run": per_run,
    }
    out_path = eval_base / "summary.json"
    out_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    print()
    if repeats > 1:
        print(f"=== MEAN score over {repeats} runs: {mean_score:.4f} (+/- {std_score:.4f}) ===")
    else:
        print(f"=== score: {mean_score:.4f} ===")
    print(f"summary written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
