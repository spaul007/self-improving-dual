"""CLI: load the database tasks, run the 5-investigators + lead-DBA MAS, write raw results.

    python run_inference.py --limit 3 --run-name smoke

Writes results/raw/<run-name>.json. Score it with:

    python evaluate.py --run-name smoke

Or do both in one command:

    python run_inference.py --run-name database_full --evaluate

No live database is needed: every query_db call replays from the per-task
snapshots in data/marble-db/db_cache/ (see snapshot/record_db_cache.py for how
they are produced).
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import config
import evaluate
import llm_client
import mas_workflow
from tools.immutable.query_db import format_db_stats, get_db_stats, reset_db_stats


def load_database_tasks(
    path: Path,
    limit: int | None = None,
    start: int = 0,
    task_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Read the flattened MARBLE database jsonl directly."""
    items: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            item = {
                config.ID_KEY: str(row.get(config.ID_KEY, idx)),
                config.PROBLEM_KEY: row[config.PROBLEM_KEY],
                config.GOLD_KEY: row.get(config.GOLD_KEY, []),
                config.LABELS_KEY: row.get(config.LABELS_KEY, config.LABELS),
                "number_of_labels_pred": row.get("number_of_labels_pred"),
            }
            items.append(item)

    if task_ids is not None:
        items = [it for it in items if it[config.ID_KEY] in task_ids]
    items = items[start:]
    if limit is not None:
        items = items[:limit]
    return items


def _check_snapshots(items: list[dict[str, Any]]) -> None:
    """Warn (loudly) about tasks whose recorded snapshot is missing."""
    missing = [
        it[config.ID_KEY]
        for it in items
        if not (config.DB_CACHE_DIR / f"{it[config.ID_KEY]}.json").exists()
    ]
    if missing:
        preview = ", ".join(missing[:10]) + (" ..." if len(missing) > 10 else "")
        print(
            f"WARNING: {len(missing)}/{len(items)} task snapshot(s) missing under "
            f"{config.DB_CACHE_DIR} (ids: {preview}).\n"
            "         Those tasks will run with no query_db evidence. Generate "
            "snapshots with: python snapshot/record_db_cache.py",
            file=sys.stderr,
        )


async def _main(args: argparse.Namespace) -> int:
    task_ids = (
        {t.strip() for t in args.task_ids.split(",") if t.strip()}
        if args.task_ids else None
    )
    items = load_database_tasks(
        Path(args.data), limit=args.limit, start=args.start, task_ids=task_ids
    )
    if not items:
        print("No items to run.", file=sys.stderr)
        return 1

    if args.no_tools:
        config.TOOLS_ENABLED = False

    if config.TOOLS_ENABLED:
        # Preflight: confirm the endpoint accepts OpenAI tool-calling before
        # spending a full run.
        ok, msg = await llm_client.get_client().acheck_tool_support()
        print(f"[preflight] {msg}")
        if not ok:
            print("[preflight] Aborting. Re-run with --no-tools to run without tools.",
                  file=sys.stderr)
            return 1
        _check_snapshots(items)
    else:
        print("Tools DISABLED: investigators answer from the task text alone (ablation mode).")

    print(
        f"Running {len(items)} task(s) | model={config.MAIN_LLM.model} "
        f"| concurrency={args.max_concurrent or config.MAX_CONCURRENT_TASKS} "
        f"| tool rounds<={config.TOOL_MAX_ROUNDS}"
    )

    reset_db_stats()
    done = 0

    def progress(_record: dict[str, Any]) -> None:
        nonlocal done
        done += 1
        print(f"  [{done}/{len(items)}] done", end="\r", flush=True)

    results = await mas_workflow.run_many(
        items, max_concurrent=args.max_concurrent, on_done=progress
    )
    print()
    print(format_db_stats(args.run_name))

    out_path = config.RAW_DIR / f"{args.run_name}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_name": args.run_name,
        "model": config.MAIN_LLM.model,
        "dataset": str(args.data),
        "n": len(results),
        "use_compressed_context": config.USE_COMPRESSED_CONTEXT,
        "tools_enabled": config.TOOLS_ENABLED,
        "tool_max_rounds": config.TOOL_MAX_ROUNDS,
        # Persisted (not just printed): how often query_db was answerable from
        # the recorded snapshots during THIS inference run.
        "db_coverage": get_db_stats(),
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
    parser.add_argument("--data", default=str(config.DATASET_PATH), help="database tasks jsonl path")
    parser.add_argument("--limit", type=int, default=None, help="run only the first N tasks")
    parser.add_argument("--start", type=int, default=0, help="skip the first N tasks")
    parser.add_argument("--task-ids", default=None,
                        help='run only these unique_ids, comma-separated (e.g. "1,42,87")')
    parser.add_argument("--run-name", default="database", help="output basename under results/raw/")
    parser.add_argument(
        "--max-concurrent", type=int, default=None, help="tasks in flight (default from config)"
    )
    parser.add_argument(
        "--no-tools", action="store_true",
        help="disable the query_db tool loop (ablation: no evidence gathering)"
    )
    parser.add_argument(
        "--evaluate", action="store_true", help="score the run immediately after inference"
    )
    parser.add_argument(
        "--show-errors", type=int, default=0, help="with --evaluate: print N imperfect samples"
    )
    return asyncio.run(_main(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
