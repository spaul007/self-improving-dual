"""Standalone diagnostic tool: quantifies the prevalence of each error
bucket in strategies.md's failure taxonomy (tool_omission, wrong_tool_or_arg,
constraint_misreading, wrong_tool_calling_order, long_horizon_state_loss,
apply_info_incorrectly, tool_calling_budget_exceeded) across one or more
completed evals' logs/ directories (trace.jsonl + case_*.json).

This is a read of an existing eval's own trace.jsonl/case_*.json, using an
LLM only to classify WHY each already-failing case failed, into the fixed
taxonomy above -- it never re-runs the agent or the scorer, and (like
analyze_llm_call_failures.py) it is never imported by the HGM/main_loop
itself, so it cannot influence block selection or scoring.

The classifying LLM defaults to whatever platform_core.llm_wrapper.call_llm
falls back to on its own (the LLM_MODEL/LLM_BASE_URL/LLM_REASONING_EFFORT
env vars) -- i.e. it is meant to be the SAME backbone this project already
uses for its other meta-agent roles (block_suggester/failure_summarizer/
editor), not a separate model choice. Override with --model/--base-url/
--reasoning-effort if you want a different one for this analysis specifically.

Usage:
    python3 analyze_error_buckets.py <logs_dir_or_run_dir> [<...> ...] \\
        [--model MODEL] [--base-url URL] [--reasoning-effort EFFORT] \\
        [--batch-size N] [--max-cases N] [--out-dir DIR]

<logs_dir_or_run_dir> is any of:
  - a logs/ dir directly (contains case_*.json + trace.jsonl)
  - a dir one level up (contains logs/case_*.json + logs/trace.jsonl)
  - a run/experiment root (every logs/ dir found anywhere under it via
    a recursive search is analyzed and reported separately, then combined)

Each analyzed logs/ dir gets its own artifacts written under --out-dir
(default: <that logs_dir>/error_bucket_analysis/): the raw per-batch LLM
prompts/responses, and error_bucket_analysis.json (the full structured
result). A combined summary across every logs/ dir found is printed last.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

from meta_agent.error_bucket_analyzer import BUCKETS, aggregate, analyze, render_report
from platform_core.llm_wrapper import call_llm


def _find_logs_dirs(path: Path) -> list[Path]:
    """A single logs/ dir, its direct parent, or a root to search
    recursively for every logs/ dir containing at least one case_*.json."""
    if (path / "case_0.json").exists() or any(path.glob("case_*.json")):
        return [path]
    if (path / "logs").is_dir():
        return [path / "logs"]
    found = sorted({p.parent for p in path.rglob("case_*.json")})
    return found


def _merge_aggregates(aggs: list[dict]) -> dict:
    """Recombine several logs/ dirs' aggregate() outputs into one,
    without re-deriving anything from raw case/label data -- just sums
    the counts each aggregate() already computed."""
    n_total = sum(a["n_total_cases"] for a in aggs)
    n_failing = sum(a["n_failing_cases"] for a in aggs)
    n_checked = sum(a.get("n_checked_cases", a["n_failing_cases"]) for a in aggs)
    n_det = sum(a["n_deterministic_budget_exceeded"] for a in aggs)
    n_llm = sum(a["n_llm_classified"] for a in aggs)

    per_bucket = {k: {"instances": 0, "distinct_cases": 0, "examples": []} for k in BUCKETS}
    for a in aggs:
        for row in a["buckets"]:
            b = per_bucket[row["bucket"]]
            b["instances"] += row["instances"]
            b["distinct_cases"] += row["distinct_cases"]
            b["examples"].extend(row["examples"])

    buckets = []
    for key, b in per_bucket.items():
        buckets.append({
            "bucket": key,
            "description": BUCKETS[key],
            "instances": b["instances"],
            "distinct_cases": b["distinct_cases"],
            "pct_of_checked_cases": round(100.0 * b["distinct_cases"] / n_checked, 1) if n_checked else 0.0,
            "examples": b["examples"][:3],
        })
    buckets.sort(key=lambda r: r["distinct_cases"], reverse=True)

    return {
        "n_total_cases": n_total,
        "n_failing_cases": n_failing,
        "n_checked_cases": n_checked,
        "n_deterministic_budget_exceeded": n_det,
        "n_llm_classified": n_llm,
        "buckets": buckets,
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--model", default=None, help="Overrides LLM_MODEL for the classification calls only.")
    parser.add_argument("--base-url", default=None, help="Overrides LLM_BASE_URL for the classification calls only.")
    parser.add_argument("--reasoning-effort", default=None, help="Overrides LLM_REASONING_EFFORT for the classification calls only.")
    parser.add_argument("--batch-size", type=int, default=4, help="Failing cases classified per LLM call (default: 4).")
    parser.add_argument("--max-cases", type=int, default=None, help="Cap on failing cases sent to the LLM per logs/ dir, worst-scoring first (default: no cap).")
    parser.add_argument("--out-dir", type=Path, default=None, help="Where to write artifacts (default: <logs_dir>/error_bucket_analysis/, one per logs/ dir found).")
    args = parser.parse_args(argv)

    logs_dirs: list[Path] = []
    for p in args.paths:
        if not p.exists():
            print(f"SKIP (not found): {p}")
            continue
        logs_dirs.extend(_find_logs_dirs(p))

    if not logs_dirs:
        print("No logs/ directories with case_*.json found.")
        return 1

    per_dir_aggs = []
    for logs_dir in logs_dirs:
        out_dir = args.out_dir or (logs_dir / "error_bucket_analysis")
        print(f"\n=== {logs_dir} ===", flush=True)
        agg = analyze(
            logs_dir,
            llm_caller=call_llm,
            model=args.model,
            base_url=args.base_url,
            reasoning_effort=args.reasoning_effort,
            batch_size=args.batch_size,
            max_cases=args.max_cases,
            artifacts_dir=out_dir,
        )
        print(render_report(agg))
        print(f"(artifacts written to {out_dir})")
        per_dir_aggs.append(agg)

    if len(per_dir_aggs) > 1:
        print("\n\n=== COMBINED ACROSS ALL logs/ DIRS ===")
        print(render_report(_merge_aggregates(per_dir_aggs)))

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
