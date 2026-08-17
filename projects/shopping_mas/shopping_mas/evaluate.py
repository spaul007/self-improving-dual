"""CLI: score results/raw/<run> using eval/metrics.py -> results/scored/.

Usage:
    python evaluate.py --run level1_n50
    python evaluate.py --run level1_n50 --cases case_1 case_2
"""

import argparse
import json
from pathlib import Path

from config import RESULTS_RAW, RESULTS_SCORED
from eval import metrics


def main():
    ap = argparse.ArgumentParser(description="Evaluate a Shopping MAS run")
    ap.add_argument("--run", required=True, help="run name under results/raw/")
    ap.add_argument("--cases", nargs="+", default=None, help="only these case dirs")
    args = ap.parse_args()

    run_dir = RESULTS_RAW / args.run
    db_root = run_dir / "database"
    if not db_root.exists():
        raise SystemExit(f"no database dir in {run_dir}")

    case_dirs = sorted([d for d in db_root.iterdir()
                        if d.is_dir() and d.name.startswith("case_")],
                       key=lambda d: int(d.name.split("_")[1]))
    if args.cases:
        wanted = set(args.cases)
        case_dirs = [d for d in case_dirs if d.name in wanted]
    if not case_dirs:
        raise SystemExit("no case dirs found")

    case_results = [metrics.score_case(d) for d in case_dirs]
    summary = metrics.summarize(case_results)

    manifest = {}
    mf = run_dir / "manifest.json"
    if mf.exists():
        manifest = json.loads(mf.read_text(encoding="utf-8"))

    out = {
        "run": args.run,
        "manifest": manifest,
        "summary": summary,
        "case_results": case_results,
    }
    RESULTS_SCORED.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_SCORED / f"{args.run}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"{'case':<10} {'score':>6} {'case_score':>10} {'matched':>9} {'extra':>6}  notes")
    for r in case_results:
        print(f"{r['case_name']:<10} {r.get('score', 0):>6.2f} "
              f"{r.get('case_score', 0):>10.0f} "
              f"{r.get('matched_count', 0):>4}/{r.get('expected_count', 0):<4} "
              f"{r.get('extra_products_count', 0):>6}  "
              f"{'' if r.get('is_completed') else 'INCOMPLETE'}")
    print("\nsummary:")
    print(json.dumps(summary, indent=2))
    print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    main()
