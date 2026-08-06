"""
- Use the config from the run directory
- Use full path to config and run directory

Pick the best agent at a given budget from a run's tree snapshots and
re-evaluate it — for head-to-head, equal-budget method comparison.

Requires the run to have been produced with ``snapshot_tree: true`` (see
meta_agent/tree_snapshot.py), which writes
``<experiment_dir>/snapshots/tree_snapshots.jsonl``.

For each requested budget B, this picks the LATEST snapshot whose
``budget_spent <= B`` (the tree state once B evals had been spent), reads its
``best_node_id``/``best_round_dir``, resolves ``<experiment_dir>/<round_dir>/
task_agent``, and re-evaluates it on the chosen case set. Results are written to
``<experiment_dir>/snapshots/eval_at_budget_<B>.json``.

    # Just list the best agent at each budget (no evaluation, no API key):
    python3 snapshot_eval.py --experiment-dir runs/<exp> --all --list

    # Budget-vs-performance curve in one command: select the best agent by the
    # framework's LCB rule at each 1k budget, evaluate on the full benchmark,
    # and emit a plot-ready CSV (snapshots/budget_curve_lcb_full_benchmark.csv):
    python3 snapshot_eval.py --config configs/hgm_travel.yaml \\
        --experiment-dir runs/<exp> --budgets 1000,2000,3000,4000 --select lcb

    # Same, but dial evaluator concurrency up for a quick run without editing the
    # YAML (--parallelism overrides evaluator.config.parallelism for this run):
    python3 snapshot_eval.py --config configs/hgm_travel.yaml \\
        --experiment-dir runs/<exp> --budgets 1000,2000,3000,4000 --select lcb \\
        --parallelism 40

Selection (--select): 'mean' (recorded argmax-mean best node) or 'lcb' (the
manager's final-selection rule recomputed offline from each node's Beta tallies
stored in the snapshot). Outputs per-budget eval_at_budget_<B>.json plus a
budget_curve_<select>_<case_set>.csv.

Concurrency (--parallelism): override the evaluator's subprocess concurrency
(how many cases run at once) for this run only, leaving the shared YAML
untouched. Each case still runs in its own subprocess with its own RLIMIT_AS, so
this only changes how many run concurrently — watch host RAM and the per-case
API request rate when raising it. Omit to use the config's
evaluator.config.parallelism.

Repeats (--repeats N): evaluate each budget's best agent N times and report the
MEAN composite_score (plus std and per-run breakdown) as the final value — for
averaging out scorer/LLM stochasticity. Each run is given its own isolated
round-dir (snapshots/eval_runs/<select>_<case_set>/budget_<B>/run_<i>/, with a
task_agent symlink to the real agent), so the evaluator's per-run logs
(case_<id>.json, trace.jsonl, scratch/, stderr) never overwrite each other or
the original round's optimization logs.

    # 3 evaluation rounds, averaged, 40-wide, isolated logs:
    python3 snapshot_eval.py --config configs/hgm_dual_travel_8000.yaml \\
        --experiment-dir runs/<exp> --at-budget 99999999 --select lcb \\
        --parallelism 40 --repeats 3
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Optional

from meta_agent import config as cfg_mod
from meta_agent import runtime_env


def _isolated_round_dir(base: Path, agent_dir: Path, label: str) -> Path:
    """Build a throwaway round-dir whose ``task_agent`` symlinks the real agent.

    The evaluator always writes its artifacts under ``<round_dir>/logs``
    (per-case ``case_<id>.json``, ``trace.jsonl`` — which it truncates —,
    ``scratch/``, ``case_<id>.stderr``). Pointing it at the agent's real round
    dir would clobber that round's original optimization logs, and running
    several repeats over the same dir would have them overwrite each other.

    Instead we hand the evaluator an isolated dir per run: ``<base>/<label>``
    with a ``task_agent`` symlink to the real code (so the agent still imports
    its own ``workflow.py`` / ``mutable_tools`` / ``tools_schema.json``), and
    its own empty ``logs/`` for this run only. Any prior dir at the same label
    is removed first so re-invocations don't mix with stale artifacts.
    """
    run_dir = base / label
    if run_dir.exists() or run_dir.is_symlink():
        shutil.rmtree(run_dir, ignore_errors=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "task_agent").symlink_to(agent_dir.resolve(), target_is_directory=True)
    return run_dir


def load_snapshots(experiment_dir: Path) -> list[dict[str, Any]]:
    """Read the snapshot JSONL into a list of records, ordered as written."""
    path = experiment_dir / "snapshots" / "tree_snapshots.jsonl"
    if not path.is_file():
        raise FileNotFoundError(
            f"no snapshots at {path} — was the run produced with "
            "snapshot_tree: true?"
        )
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def snapshot_at_budget(
    records: list[dict[str, Any]], budget: int
) -> Optional[dict[str, Any]]:
    """The latest snapshot whose ``budget_spent <= budget`` — the tree state
    once ``budget`` evals had been spent. ``None`` if none qualifies."""
    eligible = [r for r in records if r.get("budget_spent", 0) <= budget]
    if not eligible:
        return None
    # Records are append-ordered by non-decreasing budget; take the last that
    # qualifies (also robust if order is ever perturbed).
    return max(eligible, key=lambda r: (r.get("budget_spent", 0), r.get("snapshot_idx", 0)))


def _select_best(
    snap: dict[str, Any], method: str, epsilon: float
) -> Optional[dict[str, Any]]:
    """Pick the best node in a snapshot by the chosen selection method.

    - ``mean``: the snapshot's recorded ``best_node_id`` (argmax mean utility).
    - ``lcb``: the framework's actual final-selection rule recomputed from each
      node's Beta tallies (``n_success``/``n_failure``) — the highest
      ε-quantile of ``Beta(1+n_success, 1+n_failure)``, over evaluated,
      non-edit-failed nodes. Reuses ``HGMTree.lcb_select`` so it matches the
      manager exactly. (Restricted to evaluated nodes — i.e. the LCB of the
      evidence available at that budget; mid-run nodes are only partially
      evaluated, which is the honest "what would it pick if stopped here".)
    """
    nodes = snap.get("nodes") or []
    if method == "mean":
        bid = snap.get("best_node_id")
        if bid is None:
            return None
        return {
            "node_id": bid,
            "round_dir": snap.get("best_round_dir"),
            "mean_utility": snap.get("best_mean_utility"),
            "snapshot_budget": snap.get("budget_spent"),
            "select": "mean",
        }
    # lcb
    from meta_agent.managers.hgm_tree import HGMNode, HGMTree

    tree = HGMTree()
    restrict: set[int] = set()
    by_id: dict[int, dict[str, Any]] = {}
    for nd in sorted(nodes, key=lambda n: n["node_id"]):
        node = HGMNode(
            node_id=nd["node_id"],
            parent_id=nd.get("parent_id"),
            round_dir=Path(str(nd.get("round_dir"))),
        )
        node.edit_failed = bool(nd.get("edit_failed"))
        node.n_success = float(nd.get("n_success", 0.0))
        node.n_failure = float(nd.get("n_failure", 0.0))
        tree.add(node)
        by_id[nd["node_id"]] = nd
        if not node.edit_failed and int(nd.get("n_evals", 0)) > 0:
            restrict.add(nd["node_id"])
    if not restrict:
        return None
    bid = tree.lcb_select(epsilon, restrict_to=restrict)
    nd = by_id.get(bid)
    if nd is None:
        return None
    return {
        "node_id": bid,
        "round_dir": nd.get("round_dir"),
        "mean_utility": nd.get("mean_utility"),
        "snapshot_budget": snap.get("budget_spent"),
        "select": "lcb",
    }


def _resolve_budgets(args: argparse.Namespace, records: list[dict[str, Any]]) -> list[int]:
    if args.all:
        # One entry per distinct budget level actually recorded.
        return sorted({int(r.get("budget_spent", 0)) for r in records})
    budgets: list[int] = []
    for b in args.at_budget or []:
        budgets.append(int(b))
    if args.budgets:
        budgets.extend(int(x.strip()) for x in args.budgets.split(",") if x.strip())
    return sorted(set(budgets))


def run(args: argparse.Namespace) -> None:
    experiment_dir: Path = args.experiment_dir
    records = load_snapshots(experiment_dir)
    budgets = _resolve_budgets(args, records)
    if not budgets:
        raise SystemExit("specify --at-budget, --budgets, or --all")

    # Resolve the (budget -> best node) plan up front so --list needs no
    # component assembly (no API key required just to inspect).
    plan: list[dict[str, Any]] = []
    for b in budgets:
        snap = snapshot_at_budget(records, b)
        best = _select_best(snap, args.select, args.epsilon) if snap else None
        plan.append({"budget": b, "snapshot": snap, "best": best})

    print(f"# best agent at budget (select={args.select}) from {experiment_dir}")
    print(f"{'req_budget':>10}  {'snap_budget':>11}  {'node':>5}  {'round_dir':<12}  {'train_mean':>10}")
    for item in plan:
        best = item["best"]
        if best is None:
            print(f"{item['budget']:>10}  {'-':>11}  {'-':>5}  {'(no evaluated node yet)':<12}")
            continue
        mu = best["mean_utility"]
        mu_s = "n/a" if mu is None else f"{mu:.4f}"
        print(
            f"{item['budget']:>10}  {best['snapshot_budget']:>11}  "
            f"{best['node_id']:>5}  {str(best['round_dir']):<12}  {mu_s:>10}"
        )

    if args.list:
        return

    # ---- Re-evaluation path: assemble components once, evaluate each pick. ---- #
    cfg = cfg_mod.load(args.config)
    runtime_env.apply_all(cfg)
    fw = cfg_mod.build_components(cfg)

    # Optional per-run override of the evaluator's subprocess concurrency, so
    # quick experiments can dial parallelism up/down without editing the YAML
    # (which is shared with the optimization loop). Each case still runs in its
    # own subprocess with its own RLIMIT_AS; this only changes how many run at
    # once. ``run()`` reads ``self.parallelism`` at call time.
    if args.parallelism is not None:
        fw.evaluator.parallelism = max(1, int(args.parallelism))
        print(f"# evaluator parallelism overridden to {fw.evaluator.parallelism}")

    if args.case_ids:
        case_ids: Optional[list[str]] = [c.strip() for c in args.case_ids.split(",")]
    elif args.eval_split:
        case_ids = fw.eval_case_ids
        if not case_ids:
            raise SystemExit("--eval-split given but the config has no eval split")
    else:
        case_ids = None  # full benchmark

    case_set = (
        "custom" if args.case_ids
        else "eval_split" if args.eval_split
        else "full_benchmark"
    )
    out_dir = experiment_dir / "snapshots"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_rows: list[list[Any]] = []

    repeats = max(1, int(args.repeats))
    # Each eval run gets its own isolated logs dir under here (see
    # _isolated_round_dir) so repeats never overlap and the original round's
    # optimization logs are left untouched.
    eval_logs_base = out_dir / "eval_runs" / f"{args.select}_{case_set}"

    print(
        f"\n# re-evaluating best agent at each budget "
        f"(select={args.select}, case_set={case_set}, repeats={repeats})"
    )
    if repeats > 1:
        print(f"# final composite_score per budget = mean over {repeats} runs")
    print(f"# per-run logs under: {eval_logs_base}")
    for item in plan:
        best = item["best"]
        b = item["budget"]
        if best is None:
            print(f"budget {b}: no evaluated node — skipped")
            continue
        round_dir = experiment_dir / str(best["round_dir"])
        agent_dir = round_dir / "task_agent"
        if not agent_dir.is_dir():
            print(f"budget {b}: agent dir missing ({agent_dir}) — skipped")
            continue

        per_run: list[dict[str, Any]] = []
        for run_idx in range(1, repeats + 1):
            iso_round_dir = _isolated_round_dir(
                eval_logs_base, agent_dir, f"budget_{b}/run_{run_idx}"
            )
            result = fw.evaluator.run(iso_round_dir, fw.benchmark_dir, case_ids=case_ids)
            total = result.passed + result.failed
            # TRUE per-plan timing/llm: one entry per case, measured by the
            # evaluator (subprocess wall time + llm_call trace count). Filtered
            # to non-None so legacy/partial results never poison the means.
            plan_wall_times = [
                c.wall_time_s for c in result.per_case if c.wall_time_s is not None
            ]
            plan_llm_calls = [
                c.llm_calls for c in result.per_case if c.llm_calls is not None
            ]
            plan_wt_mean = (
                sum(plan_wall_times) / len(plan_wall_times) if plan_wall_times else 0.0
            )
            llm_mean = (
                sum(plan_llm_calls) / len(plan_llm_calls) if plan_llm_calls else 0.0
            )
            per_run.append({
                "run": run_idx,
                "composite_score": result.score,
                "passed": result.passed,
                "failed": result.failed,
                "n_cases": total,
                # Whole-batch wall clock under parallelism (throughput proxy),
                # NOT per-plan latency. Kept for backward compatibility.
                "wall_time_s": result.wall_time_s,
                # TRUE mean per-plan wall time / llm calls for this run.
                "plan_wall_time_s_mean": plan_wt_mean,
                "llm_calls_per_plan_mean": llm_mean,
                # Full per-case lists so percentiles (p90, etc.) can be computed
                # later by the consumer; we store the raw data only.
                "plan_wall_times": plan_wall_times,
                "plan_llm_calls": plan_llm_calls,
                "crashed": result.crashed,
                "logs_dir": str(iso_round_dir / "logs"),
            })
            print(
                f"budget {b}: run {run_idx}/{repeats} node {best['node_id']} "
                f"({best['round_dir']}) -> composite={result.score:.4f} "
                f"passed={result.passed}/{total}"
            )

        scores = [r["composite_score"] for r in per_run]
        n = len(scores)
        mean_score = sum(scores) / n
        std_score = (sum((s - mean_score) ** 2 for s in scores) / n) ** 0.5
        mean_passed = sum(r["passed"] for r in per_run) / n
        mean_failed = sum(r["failed"] for r in per_run) / n
        n_cases = per_run[0]["n_cases"]
        any_crashed = any(r["crashed"] for r in per_run)
        # Wall-clock timing. ``wall_time_s`` is the total time for a whole eval
        # run (all cases); divide by case count for the per-sample figure. Note
        # this is wall clock under the configured parallelism, so it reflects
        # observed THROUGHPUT, not per-plan latency. For the accurate per-plan
        # figures use ``plan_wall_time_s_mean`` / ``llm_calls_per_plan_mean``
        # (and the raw ``plan_wall_times_all`` list for percentiles) below.
        mean_wall_time = sum(r["wall_time_s"] for r in per_run) / n
        wall_time_per_sample = mean_wall_time / n_cases if n_cases else 0.0

        # Accurate per-plan aggregates pooled over ALL cases across ALL repeats
        # (case-weighted — not a mean-of-run-means). The flattened lists are
        # retained verbatim so the consumer can compute percentiles (p90, ...).
        plan_wall_times_all = [wt for r in per_run for wt in r["plan_wall_times"]]
        plan_llm_calls_all = [lc for r in per_run for lc in r["plan_llm_calls"]]
        m = len(plan_wall_times_all)
        plan_wall_time_s_mean = sum(plan_wall_times_all) / m if m else 0.0
        plan_wall_time_s_std = (
            (sum((x - plan_wall_time_s_mean) ** 2 for x in plan_wall_times_all) / m) ** 0.5
            if m else 0.0
        )
        k = len(plan_llm_calls_all)
        llm_calls_per_plan_mean = sum(plan_llm_calls_all) / k if k else 0.0

        out_path = out_dir / f"eval_at_budget_{b}.json"
        out_path.write_text(
            json.dumps(
                {
                    "requested_budget": b,
                    "snapshot_budget": best["snapshot_budget"],
                    "select": best.get("select", args.select),
                    "node_id": best["node_id"],
                    "round_dir": best["round_dir"],
                    "train_mean_at_snapshot": best["mean_utility"],
                    "case_set": case_set,
                    "repeats": repeats,
                    # Headline value: mean composite over all repeats.
                    "composite_score": mean_score,
                    "composite_score_std": std_score,
                    "composite_score_runs": scores,
                    "passed_mean": mean_passed,
                    "failed_mean": mean_failed,
                    "n_cases": n_cases,
                    # Throughput approximations (whole-batch wall / parallelism).
                    # Kept for backward compat — see the accurate keys below.
                    "wall_time_s_mean": mean_wall_time,
                    "wall_time_per_sample_s": wall_time_per_sample,
                    # Accurate per-plan metrics (mean over all cases x repeats).
                    "plan_wall_time_s_mean": plan_wall_time_s_mean,
                    "plan_wall_time_s_std": plan_wall_time_s_std,
                    "llm_calls_per_plan_mean": llm_calls_per_plan_mean,
                    # Raw flattened lists for percentile computation downstream.
                    "plan_wall_times_all": plan_wall_times_all,
                    "plan_llm_calls_all": plan_llm_calls_all,
                    "crashed_any": any_crashed,
                    "per_run": per_run,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        csv_rows.append([
            b, best["snapshot_budget"], best.get("select", args.select),
            best["node_id"], best["round_dir"], best["mean_utility"], repeats,
            round(mean_score, 6), round(std_score, 6),
            round(mean_passed, 3), round(mean_failed, 3), n_cases,
            round(mean_wall_time, 3), round(wall_time_per_sample, 3),
            round(plan_wall_time_s_mean, 3), round(llm_calls_per_plan_mean, 3),
        ])
        if repeats > 1:
            print(
                f"budget {b}: MEAN composite={mean_score:.4f} (±{std_score:.4f}) "
                f"over {repeats} runs, passed={mean_passed:.1f}/{n_cases}, "
                f"plan_time={plan_wall_time_s_mean:.2f}s "
                f"llm/plan={llm_calls_per_plan_mean:.1f}  "
                f"[{out_path.name}]"
            )
        else:
            print(
                f"budget {b}: plan_time={plan_wall_time_s_mean:.2f}s "
                f"llm/plan={llm_calls_per_plan_mean:.1f} "
                f"-> [{out_path.name}]"
            )

    # Budget-vs-performance curve, ready to plot.
    if csv_rows:
        import csv as _csv
        csv_path = out_dir / f"budget_curve_{args.select}_{case_set}.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            w = _csv.writer(fh)
            w.writerow([
                "requested_budget", "snapshot_budget", "select", "node_id",
                "round_dir", "train_mean_at_snapshot", "repeats",
                "composite_score", "composite_score_std",
                "passed_mean", "failed_mean", "n_cases",
                "wall_time_s_mean", "wall_time_per_sample_s",
                "plan_wall_time_s_mean", "llm_calls_per_plan_mean",
            ])
            w.writerows(csv_rows)
        print(f"\ncurve CSV: {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Re-evaluate the best agent at a given budget from tree snapshots."
    )
    parser.add_argument("--config", type=Path, help="Path to YAML config (needed unless --list)")
    parser.add_argument(
        "--experiment-dir", type=Path, required=True,
        help="A run directory containing snapshots/tree_snapshots.jsonl",
    )
    parser.add_argument(
        "--at-budget", type=int, action="append",
        help="A budget level to evaluate at (repeatable).",
    )
    parser.add_argument(
        "--budgets", type=str, default=None,
        help="Comma-separated budget levels, e.g. '100,200,300'.",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Evaluate at every distinct budget level recorded.",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="Only print the best-at-budget table; do not re-evaluate.",
    )
    parser.add_argument(
        "--case-ids", type=str, default=None,
        help="Comma-separated case ids to evaluate on (overrides --eval-split).",
    )
    parser.add_argument(
        "--eval-split", action="store_true",
        help="Evaluate on the config's held-out eval split (default: full benchmark).",
    )
    parser.add_argument(
        "--parallelism", type=int, default=None,
        help="Override the evaluator's subprocess concurrency (how many cases run "
        "at once) for this run only, without editing the YAML. Default: use the "
        "config's evaluator.config.parallelism.",
    )
    parser.add_argument(
        "--repeats", type=int, default=1,
        help="Number of full evaluation runs (rounds) per budget; the final "
        "composite_score is their mean (std also reported). Each run writes to "
        "its own isolated logs dir, so repeats never overwrite each other or the "
        "original round's logs. Default: 1.",
    )
    parser.add_argument(
        "--select", choices=("mean", "lcb"), default="mean",
        help="Best-agent selection at each budget: 'mean' (recorded argmax mean) "
        "or 'lcb' (the framework's final-selection rule recomputed from the "
        "snapshot's Beta tallies). Default: mean.",
    )
    parser.add_argument(
        "--epsilon", type=float, default=0.25,
        help="LCB quantile for --select lcb (matches manager.config.epsilon; default 0.25).",
    )
    args = parser.parse_args()
    if not args.list and args.config is None:
        parser.error("--config is required unless --list is given")
    run(args)
