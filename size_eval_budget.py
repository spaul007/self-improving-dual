"""Size an HGM ``eval_budget`` to target a wall-clock time budget, from
measured per-case throughput -- the same calculation used to size
``configs/hgm_math_mas_full.yaml``'s ``eval_budget: 11000`` (target ~12h).

Two costs are modeled, not just raw case-evaluation time:

1. **Case evaluation** itself: ``eval_budget`` cases at the observed
   aggregate throughput (s/case, at whatever concurrency you measured it at
   -- get this from a real run, e.g. by timing the vendor's own bulk
   inference script, or an ``evaluate_task_agent.py`` run; don't guess it).
2. **Meta-agent LLM call overhead**, easy to under-count: with
   ``behavior_summarizer``/``failure_summarizer`` enabled, EVERY EVALUATE
   step fires extra meta-agent LLM calls -- at a small ``eval_batch_size``
   this can dominate the whole time budget. Modeled as
   ``(eval_budget / eval_batch_size) * t_summarizer_pair``, plus
   ``max_rounds * t_editor`` for EXPAND calls (upper bound -- not every
   allowed round necessarily gets used, so real time is usually a bit less
   than projected).

Solves the closed-form ``eval_budget`` for a target number of hours, then
prints a projection table around it so you can sanity-check the trade-off
before committing to a number.

Usage (defaults match the math_mas full-run calculation):

    python3 size_eval_budget.py --t-case 2.528 --hours 12

    # override anything:
    python3 size_eval_budget.py --t-case 2.528 --hours 8 \\
        --eval-batch-size 10 --max-rounds 50 --safety-frac 0.8

    # just project a few candidate eval_budgets instead of solving:
    python3 size_eval_budget.py --t-case 2.528 --try 5000,8000,11000,15000

    # X/Y mode: "I want ~X good-enough agents, each evaluated on ~Y examples
    # on average" -- derives eval_budget=X*Y and the branching-factor alpha
    # such that eval_budget**alpha == X (so the widening schedule -- see
    # hgm_tree.py::schedule_favors_expand -- naturally arrives at ~X nodes by
    # the time the budget is spent). Also prints the time projection for the
    # resulting eval_budget, same as the other modes, when --t-case is given.
    python3 size_eval_budget.py --t-case 2.528 --target-agents 20 --evals-per-agent 100
"""
from __future__ import annotations

import argparse
import math
from typing import Optional


def solve_alpha_from_xy(target_agents: int, evals_per_agent: int) -> tuple[int, float]:
    """B = X*Y; alpha solved from B**alpha == X, i.e. alpha = ln(X)/ln(B).

    Matches ``hgm_tree.py``'s actual schedule
    (``budget_spent**alpha >= n_real_nodes - 1``) exactly: at
    ``budget_spent == B``, ``B**alpha == X``, so the schedule has allowed
    ~X nodes to exist by the time the full budget is spent, each having
    received ~Y evals on average (B evals total / X nodes)."""
    eval_budget = target_agents * evals_per_agent
    alpha = math.log(target_agents) / math.log(eval_budget)
    return eval_budget, alpha


def solve_eval_budget(
    *,
    t_case: float,
    hours: float,
    eval_batch_size: int,
    t_summarizer_pair: float,
    max_rounds: int,
    t_editor: float,
    fixed_unbudgeted_evals: int,
    safety_frac: float,
) -> float:
    """Closed-form solve for eval_budget E from:

        safety_frac * hours*3600
            = E*t_case + (E/eval_batch_size)*t_summarizer_pair
              + max_rounds*t_editor + fixed_unbudgeted_evals*t_case
    """
    total_s = hours * 3600 * safety_frac
    rhs_const = max_rounds * t_editor + fixed_unbudgeted_evals * t_case
    denom = t_case + t_summarizer_pair / eval_batch_size
    return (total_s - rhs_const) / denom


def project_hours(
    eval_budget: float,
    *,
    t_case: float,
    eval_batch_size: int,
    t_summarizer_pair: float,
    max_rounds: int,
    t_editor: float,
    fixed_unbudgeted_evals: int,
) -> dict[str, float]:
    n_eval_steps = eval_budget / eval_batch_size
    case_eval_s = eval_budget * t_case
    summarizer_s = n_eval_steps * t_summarizer_pair
    editor_s = max_rounds * t_editor
    fixed_s = fixed_unbudgeted_evals * t_case
    total_s = case_eval_s + summarizer_s + editor_s + fixed_s
    return {
        "eval_steps": n_eval_steps,
        "case_eval_h": case_eval_s / 3600,
        "summarizer_overhead_h": summarizer_s / 3600,
        "editor_overhead_h": editor_s / 3600,
        "fixed_unbudgeted_h": fixed_s / 3600,
        "total_h": total_s / 3600,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--t-case", type=float, required=True,
        help="Measured aggregate s/case throughput at your chosen concurrency "
        "(NOT per-case latency -- e.g. wall_time_for_N_cases / N). Get this from "
        "a real timed run, don't guess it.",
    )
    p.add_argument("--hours", type=float, default=12.0, help="Target wall-clock budget. Default: 12.")
    p.add_argument(
        "--eval-batch-size", type=int, default=25,
        help="manager.config.eval_batch_size -- also controls how many EVALUATE "
        "steps (and therefore summarizer calls) eval_budget implies. Default: 25.",
    )
    p.add_argument(
        "--t-summarizer-pair", type=float, default=15.0,
        help="Seconds per EVALUATE step for behavior_summarizer + failure_summarizer "
        "combined (0 if neither is enabled). Rough estimate -- both are single "
        "meta-agent LLM calls, not code-writing calls, so usually cheaper than the "
        "editor's. Default: 15.",
    )
    p.add_argument(
        "--max-rounds", type=int, default=100,
        help="loop.max_rounds -- upper bound used for EXPAND/editor-call overhead "
        "(worst case; real time is usually less if fewer rounds actually get "
        "created). Default: 100.",
    )
    p.add_argument(
        "--t-editor", type=float, default=30.0,
        help="Seconds per EXPAND (editor LLM call -- writes real code, so usually "
        "costs more than a summarizer call). Default: 30.",
    )
    p.add_argument(
        "--fixed-unbudgeted-evals", type=int, default=150,
        help="Unbudgeted eval cost outside the main loop: seed pre-eval "
        "(train_size) + finalize_top_k top-up (worst case, up to train_size) + "
        "the held-out full_eval_top_k pass (eval_size), if enabled. Default: 150 "
        "(matches train_size=eval_size=50, finalize_top_k=full_eval_top_k=1).",
    )
    p.add_argument(
        "--safety-frac", type=float, default=0.85,
        help="Fraction of --hours to actually target, reserving the rest as "
        "margin for host-contention variance / estimate error. Default: 0.85.",
    )
    p.add_argument(
        "--try", dest="try_values", type=str, default=None,
        help="Comma-separated eval_budget values to project instead of solving "
        "(e.g. '5000,8000,11000,15000'). If omitted, solves for --hours and "
        "prints a table around the solved value.",
    )
    p.add_argument(
        "--target-agents", type=int, default=None,
        help="X/Y mode: target number of good-enough agents (X). Requires "
        "--evals-per-agent too. Derives eval_budget=X*Y and alpha=ln(X)/ln(X*Y); "
        "overrides --hours/--try.",
    )
    p.add_argument(
        "--evals-per-agent", type=int, default=None,
        help="X/Y mode: average evals per agent (Y). See --target-agents.",
    )
    args = p.parse_args()

    common = dict(
        t_case=args.t_case,
        eval_batch_size=args.eval_batch_size,
        t_summarizer_pair=args.t_summarizer_pair,
        max_rounds=args.max_rounds,
        t_editor=args.t_editor,
        fixed_unbudgeted_evals=args.fixed_unbudgeted_evals,
    )

    if args.target_agents is not None or args.evals_per_agent is not None:
        if args.target_agents is None or args.evals_per_agent is None:
            p.error("--target-agents and --evals-per-agent must be given together")
        eval_budget, alpha = solve_alpha_from_xy(args.target_agents, args.evals_per_agent)
        print(
            f"# X={args.target_agents} agents, Y={args.evals_per_agent} evals/agent "
            f"-> eval_budget=X*Y={eval_budget}, alpha=ln(X)/ln(eval_budget)={alpha:.4f}"
        )
        candidates = [eval_budget]
    elif args.try_values:
        candidates = [int(v.strip()) for v in args.try_values.split(",") if v.strip()]
    else:
        solved = solve_eval_budget(hours=args.hours, safety_frac=args.safety_frac, **common)
        print(f"# solved eval_budget for {args.hours}h (safety_frac={args.safety_frac}): ~{solved:.0f}")
        step = max(1, round(solved * 0.1 / 500) * 500)
        base = max(step, round(solved / step) * step)
        candidates = sorted({max(1, base + k * step) for k in (-2, -1, 0, 1, 2)})

    print(
        f"\n{'eval_budget':>11}  {'eval_steps':>10}  {'case_eval':>10}  "
        f"{'summarizer':>10}  {'editor':>8}  {'fixed':>7}  {'total_h':>8}"
    )
    for e in candidates:
        proj = project_hours(e, **common)
        print(
            f"{e:>11}  {proj['eval_steps']:>10.0f}  {proj['case_eval_h']:>9.2f}h  "
            f"{proj['summarizer_overhead_h']:>9.2f}h  {proj['editor_overhead_h']:>7.2f}h  "
            f"{proj['fixed_unbudgeted_h']:>6.2f}h  {proj['total_h']:>7.2f}h"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
