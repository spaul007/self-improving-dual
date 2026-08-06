"""CLI: score raw inference results with eval/metrics and report accuracy.

    python evaluate.py --run-name math500

Reads results/raw/<run-name>.json, writes results/scored/<run-name>.json.

`score_run` is the reusable entry point — run_inference.py calls it directly for
its --evaluate flag, so inference+scoring can be one command.
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
                "summary": summary,
                "records": scored,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"=== {payload.get('run_name', run_name)} ({payload.get('model')}) ===")
    print(f"  accuracy           : {summary['accuracy']:.4f}  "
          f"({summary['correct']}/{summary['total']})")
    print(f"  predictor accuracy : {summary['predictor_accuracy']:.4f}")
    print(f"  fixed by reflector : {summary['fixed_by_reflector']}")
    print(f"  broken by reflector: {summary['broken_by_reflector']}")
    print(f"  fixed by verifier  : {summary['fixed_by_verifier']}")
    print(f"  broken by verifier : {summary['broken_by_verifier']}")
    print(f"  errored tasks      : {summary['errors']}")
    print(f"  avg elapsed (s)    : {summary['avg_elapsed_s']}")

    if show_errors:
        for r in [x for x in scored if not x["correct"]][:show_errors]:
            print(f"\n--- {r['unique_id']} ---")
            print(f"  gold: {r['normalized_gold']}")
            print(f"  pred: {r['normalized_prediction']}")
            if r.get("error"):
                print(f"  error: {r['error']}")

    print(f"\nWrote {out_path}")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-name", default="math500", help="basename under results/raw/")
    parser.add_argument("--raw", default=None, help="explicit raw results path (overrides --run-name)")
    parser.add_argument("--show-errors", type=int, default=0, help="print N incorrect samples")
    args = parser.parse_args()

    summary = score_run(
        args.run_name,
        raw_path=Path(args.raw) if args.raw else None,
        show_errors=args.show_errors,
    )
    return 0 if summary is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
