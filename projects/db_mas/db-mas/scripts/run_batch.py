"""Step 3 verification / general batch runner: run N tasks sequentially, then score them all."""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import subprocess

import mas_workflow

# One task_id per real anomaly type (INSERT_LARGE_DATA, REDUNDANT_INDEX,
# LOCK_CONTENTION, VACUUM, FETCH_LARGE_DATA), plus two 2-anomaly/3-label tasks,
# picked directly from database_main.jsonl for the Step 3 verification batch.
DEFAULT_SAMPLE_TASK_IDS = [1, 3, 5, 6, 7, 51, 52]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-ids", type=int, nargs="*", default=None)
    parser.add_argument("--all", action="store_true", help="Run all 100 benchmark tasks")
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Number of tasks to run concurrently (each gets its own Docker container "
        "and host port). Set to 1 for the original strictly-sequential behavior.",
    )
    parser.add_argument(
        "--coordination-eval",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Passed through to score.py: also score coordination/communication "
        "quality. Use --no-coordination-eval to skip it.",
    )
    args = parser.parse_args()

    if args.all:
        task_ids = None
    else:
        task_ids = args.task_ids or DEFAULT_SAMPLE_TASK_IDS

    results = mas_workflow.run_many(task_ids=task_ids, max_workers=args.max_workers)
    print(f"\nCompleted {len(results)} tasks.")

    print("\nRunning score.py over results/raw ...")
    score_cmd = [sys.executable, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "score.py")]
    score_cmd.append("--coordination-eval" if args.coordination_eval else "--no-coordination-eval")
    subprocess.run(score_cmd, check=True)


if __name__ == "__main__":
    main()
