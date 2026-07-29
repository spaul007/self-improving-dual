"""CLI: score raw inference results with eval/metrics and report recall.

    python evaluate.py --run-name smoke

Reads results/raw/<run-name>.json, writes results/scored/<run-name>.json.

`score_run` is the reusable entry point — run_inference.py calls it directly for
its --evaluate flag, so inference+scoring can be one command. Scoring is fully
deterministic (no judge LLM): the FINAL: line parser in
tools/immutable/label_extraction.py decides the predicted labels.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import config
from eval import metrics


def score_run(
    run_name: str,
    raw_path: Path | None = None,
    show_errors: int = 0,
) -> dict[str, Any] | None:
    """Score one raw results file, write results/scored/<run_name>.json, print a report.

    Returns the summary dict, or None if the raw file is missing.
    """
    raw_path = raw_path or config.RAW_DIR / f"{run_name}.json"
    if not raw_path.exists():
        print(f"No raw results at {raw_path}. Run run_inference.py first.", file=sys.stderr)
        return None

    with open(raw_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    scored = [metrics.score_record(r) for r in payload.get("records", [])]
    summary = metrics.summarize(scored)

    out_path = config.SCORED_DIR / f"{run_name}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "run_name": payload.get("run_name", run_name),
                "model": payload.get("model"),
                "dataset": payload.get("dataset"),
                "tools_enabled": payload.get("tools_enabled"),
                "use_compressed_context": payload.get("use_compressed_context"),
                "db_coverage": payload.get("db_coverage"),
                "summary": summary,
                "records": scored,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"=== {payload.get('run_name', run_name)} ({payload.get('model')}) ===")
    print(f"  task score (recall) : {summary['recall']:.4f}")
    print(f"  precision           : {summary['precision']:.4f}")
    print(f"  f1                  : {summary['f1']:.4f}")
    print(f"  exact match         : {summary['exact_match']}/{summary['total']} "
          f"({summary['exact_match_rate']:.4f})")
    print(f"  extraction failed   : {summary['extraction_failed']}")
    print(f"  over-named verdicts : {summary['over_named']}")
    print(f"  errored tasks       : {summary['errors']}")
    print(f"  avg elapsed (s)     : {summary['avg_elapsed_s']}")
    tu = summary["tool_usage"]
    print(f"  query_db calls      : {tu['query_db_calls']} "
          f"(avg {tu['avg_calls_per_task']}/task, by replay: {tu['by_replay']})")
    print("  per-label recall    :")
    for label, st in summary["per_label"].items():
        rec = "   n/a" if st["recall"] is None else f"{st['recall']:6.3f}"
        print(f"    {label:<18}: {rec}  ({st['recovered']}/{st['gold_tasks']} gold tasks)")

    if show_errors:
        misses = [r for r in scored if r["recall"] < 1.0][:show_errors]
        for r in misses:
            print(f"\n--- task {r['unique_id']} (recall={r['recall']}) ---")
            print(f"  gold      : {r.get('root_causes')}")
            print(f"  predicted : {r.get('predicted')}  [extraction={r.get('extraction')}]")
            if r.get("error"):
                print(f"  error     : {r['error']}")

    print(f"\nWrote {out_path}")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-name", default="database", help="basename under results/raw/")
    parser.add_argument("--raw", default=None, help="explicit raw results path (overrides --run-name)")
    parser.add_argument("--show-errors", type=int, default=0, help="print N imperfect samples")
    args = parser.parse_args()

    summary = score_run(
        args.run_name,
        raw_path=Path(args.raw) if args.raw else None,
        show_errors=args.show_errors,
    )
    return 0 if summary is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
