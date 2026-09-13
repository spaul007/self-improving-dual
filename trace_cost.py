"""Token + dollar accounting for one evaluation run's trace.

Sums the ``llm_response`` usage recorded by ``platform_core.llm_wrapper``
(see ``trace.emit`` at llm_wrapper.py:466) over a run's ``trace.jsonl`` and
prices it. Per-case breakdown comes from the ambient ``case_id`` that
``platform_core.trace.case_scope`` merges into every event.

Only task-agent calls land in this trace: the scorer's plan->JSON conversion
runs in the parent process, so when it points at a local vLLM node the totals
here ARE the OpenRouter bill.

    PYTHONPATH=. python3 trace_cost.py \\
        runs/eval_<stamp>_seed/round_eval/logs/trace.jsonl \\
        --in-price 0.66 --out-price 1.98

Prices are USD per 1M tokens (OpenRouter list, deepseek-v4-pro-0813 by
default). ``output_tokens`` already includes reasoning tokens.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("trace", type=Path, help="path to round_eval/logs/trace.jsonl")
    ap.add_argument("--in-price", type=float, default=0.66, help="USD per 1M input tokens")
    ap.add_argument("--out-price", type=float, default=1.98, help="USD per 1M output tokens")
    ap.add_argument("--per-case", action="store_true", help="print a per-case table")
    args = ap.parse_args()

    tin: dict[str, int] = defaultdict(int)
    tout: dict[str, int] = defaultdict(int)
    treason: dict[str, int] = defaultdict(int)
    calls: dict[str, int] = defaultdict(int)
    retries = 0

    with args.trace.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                # A killed/timed-out case can leave a half-written final line.
                continue
            kind = ev.get("kind")
            if kind == "llm_call_retry":
                retries += 1
                continue
            if kind != "llm_response":
                continue
            p = ev.get("payload", {})
            cid = str(p.get("case_id", "?"))
            calls[cid] += 1
            tin[cid] += p.get("input_tokens") or 0
            tout[cid] += p.get("output_tokens") or 0
            treason[cid] += p.get("reasoning_tokens") or 0

    cases = sorted(calls, key=lambda c: (len(c), c))
    if not cases:
        print(f"no llm_response events in {args.trace}")
        return

    def cost(c: str) -> float:
        return tin[c] * args.in_price / 1e6 + tout[c] * args.out_price / 1e6

    if args.per_case:
        print(f"{'case':>6} {'calls':>6} {'in':>10} {'out':>10} {'reason':>9} {'usd':>8}")
        for c in cases:
            print(
                f"{c:>6} {calls[c]:>6} {tin[c]:>10,} {tout[c]:>10,} "
                f"{treason[c]:>9,} {cost(c):>8.4f}"
            )
        print()

    n = len(cases)
    total_in, total_out = sum(tin.values()), sum(tout.values())
    total = sum(cost(c) for c in cases)
    per = sorted(cost(c) for c in cases)
    print(f"cases            : {n}")
    print(f"llm calls        : {sum(calls.values())}  (retries: {retries})")
    print(f"input tokens     : {total_in:,}")
    print(f"output tokens    : {total_out:,}  (reasoning: {sum(treason.values()):,})")
    print(f"prices           : ${args.in_price}/M in, ${args.out_price}/M out")
    print(f"TOTAL            : ${total:.4f}")
    print(f"per case  mean   : ${total / n:.4f}")
    print(f"          min/med/max: ${per[0]:.4f} / ${per[n // 2]:.4f} / ${per[-1]:.4f}")
    print(f"extrapolated 120 : ${total / n * 120:.2f}")


if __name__ == "__main__":
    main()
