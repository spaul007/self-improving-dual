"""Standalone debug/monitoring tool: reports how often
platform_core.llm_wrapper.call_llm hits the OpenRouter Responses-API
"status=failed" empty-response condition (see platform_core/llm_wrapper.py
-- the response comes back without raising an exception, but its own
status/stop_reason field says the generation failed).

This is purely observational -- it parses each round's trace.jsonl after
the fact and prints a summary. It is NEVER read by the HGM/main_loop
itself (no manager, block_suggester, or evaluator code imports or calls
this file), so it cannot influence block selection, scoring, or any
other HGM decision. Its only purpose is to let a human see, independent
of the in-run scores, how much this specific infra issue is affecting a
given run -- especially useful now that call_llm retries this condition
internally (see llm_wrapper.py's fix, 2026-09-11): after the fix, most
occurrences should show up as "retried and recovered" rather than
"reached a terminal case/plan as a failed, empty response."

Two things are counted, per round and in aggregate:
  1. llm_call_retry events caused specifically by status=failed (as
     opposed to a genuine thrown exception -- distinguished by this
     tool's own error-message text set in the fix).
  2. llm_response events whose own stop_reason ended up "failed" --
     i.e. every retry for that call was exhausted and the failure
     reached the caller anyway (should be rare/zero after the fix,
     common before it, since before the fix EVERY status=failed response
     reached the caller immediately with zero retries).

Usage:
    python3 analyze_llm_call_failures.py <run_dir> [<run_dir> ...]
    python3 analyze_llm_call_failures.py <run_dir>/round_005/logs/trace.jsonl

<run_dir> is a directory like runs/20260911_.../ (every round_*/logs/
trace.jsonl under it is scanned) or a path directly to one trace.jsonl.

The actual parsing logic lives in meta_agent/llm_failure_health.py
(iter_trace_files/analyze_trace_file) -- shared with HGMManager, which
uses the same signal to write a per-round monitoring artifact and
(opt-in) exclude terminal-failure-corrupted cases from a node's reward.
This file is just the CLI wrapper.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

from meta_agent.llm_failure_health import analyze_trace_file as _analyze_trace_file
from meta_agent.llm_failure_health import iter_trace_files as _iter_trace_files


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 1

    all_results: list[dict] = []
    for arg in argv:
        p = Path(arg)
        if not p.exists():
            print(f"SKIP (not found): {p}")
            continue
        for trace_path in _iter_trace_files(p):
            all_results.append(_analyze_trace_file(trace_path))

    if not all_results:
        print("No trace.jsonl files found.")
        return 1

    print(
        f"{'round':<50} {'calls':>7} {'responses':>10} "
        f"{'status=failed retried':>22} {'exc retried':>12} {'terminal failed':>16}"
    )
    totals = Counter()
    all_error_codes: Counter = Counter()
    for r in all_results:
        # Best-effort round label: parent-of-parent dir name (round_NNN).
        round_label = Path(r["path"]).parent.parent.name
        run_label = Path(r["path"]).parent.parent.parent.name
        label = f"{run_label}/{round_label}"
        print(
            f"{label:<50} {r['n_llm_calls']:>7} {r['n_llm_responses']:>10} "
            f"{r['n_status_failed_retries']:>22} {r['n_exception_retries']:>12} "
            f"{r['n_terminal_failed_responses']:>16}"
        )
        totals["calls"] += r["n_llm_calls"]
        totals["responses"] += r["n_llm_responses"]
        totals["status_failed_retries"] += r["n_status_failed_retries"]
        totals["exception_retries"] += r["n_exception_retries"]
        totals["terminal_failed"] += r["n_terminal_failed_responses"]
        all_error_codes.update(r["error_codes"])
        if r["n_terminal_failed_responses"]:
            print(
                f"    -> terminal failures by case: {r['cases_with_terminal_failure']}"
            )
        if r["n_status_failed_retries"]:
            print(
                f"    -> status=failed retries by case: {r['cases_with_status_failed_retry']}"
            )

    print()
    print("=== TOTALS ===")
    print(f"llm_call events:                {totals['calls']}")
    print(f"llm_response events:            {totals['responses']}")
    print(f"status=failed retries (fixed):  {totals['status_failed_retries']}")
    print(f"exception retries (pre-existing): {totals['exception_retries']}")
    print(f"terminal failed responses:      {totals['terminal_failed']}")
    if totals["responses"]:
        rate = 100.0 * (totals["status_failed_retries"] + totals["terminal_failed"]) / totals["responses"]
        print(f"status=failed incidence rate:   {rate:.2f}% of all responses")
    if all_error_codes:
        print(f"error codes given by the API:    {dict(all_error_codes)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
