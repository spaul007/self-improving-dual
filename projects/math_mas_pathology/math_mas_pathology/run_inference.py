"""CLI: load MATH-500, run the predictor -> reflector MAS, write raw results.

    python run_inference.py --limit 20 --run-name smoke

Writes results/raw/<run-name>.json. Score it with:

    python evaluate.py --run-name smoke

Or do both in one command:

    python run_inference.py --run-name math500_full --evaluate
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import config
import evaluate
import mas_workflow


def load_math500(path: Path, limit: int | None = None, start: int = 0) -> list[dict[str, Any]]:
    """Read the MATH-500 jsonl directly — no separate benchmark module."""
    items: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            items.append(
                {
                    "unique_id": str(row.get("unique_id", idx)),
                    config.PROBLEM_KEY: row[config.PROBLEM_KEY],
                    config.ANSWER_KEY: row.get(config.ANSWER_KEY, ""),
                }
            )

    items = items[start:]
    if limit is not None:
        items = items[:limit]
    return items


async def _main(args: argparse.Namespace) -> int:
    items = load_math500(Path(args.data), limit=args.limit, start=args.start)
    if not items:
        print("No items to run.", file=sys.stderr)
        return 1

    print(
        f"Running {len(items)} problem(s) | model={config.MAIN_LLM.model} "
        f"| concurrency={args.max_concurrent or config.MAX_CONCURRENT_TASKS}"
    )

    done = 0

    def progress(_record: dict[str, Any]) -> None:
        nonlocal done
        done += 1
        print(f"  [{done}/{len(items)}] done", end="\r", flush=True)

    results = await mas_workflow.run_many(
        items, max_concurrent=args.max_concurrent, on_done=progress
    )
    print()

    out_path = config.RAW_DIR / f"{args.run_name}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_name": args.run_name,
        "model": config.MAIN_LLM.model,
        "dataset": str(args.data),
        "n": len(results),
        "use_compressed_context": config.USE_COMPRESSED_CONTEXT,
        "records": results,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    failed = sum(1 for r in results if r.get("error"))
    print(f"Wrote {out_path}  ({len(results)} records, {failed} errored)")

    if args.evaluate:
        print()
        summary = evaluate.score_run(args.run_name, show_errors=args.show_errors)
        return 0 if summary is not None else 1

    print(f"Next: python evaluate.py --run-name {args.run_name}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", default=str(config.MATH500_PATH), help="MATH-500 jsonl path")
    parser.add_argument("--limit", type=int, default=None, help="run only the first N problems")
    parser.add_argument("--start", type=int, default=0, help="skip the first N problems")
    parser.add_argument("--run-name", default="math500", help="output basename under results/raw/")
    parser.add_argument(
        "--max-concurrent", type=int, default=None, help="tasks in flight (default from config)"
    )
    parser.add_argument(
        "--evaluate", action="store_true", help="score the run immediately after inference"
    )
    parser.add_argument(
        "--show-errors", type=int, default=0, help="with --evaluate: print N incorrect samples"
    )
    return asyncio.run(_main(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
