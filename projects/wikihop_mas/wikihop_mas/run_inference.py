"""CLI: load 2WikiMultihopQA (jsonl), run the wikihop MAS, write raw results.

    python run_inference.py --limit 20 --run-name smoke

Writes results/raw/<run-name>.json. Score it with:

    python evaluate.py --run-name smoke

Or do both in one command:

    python run_inference.py --run-name dev_full --evaluate

Reads jsonl only -- see scripts/convert_parquet_to_jsonl.py for the one-off
parquet->jsonl conversion step (README.md has the download+convert steps).
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import config
import evaluate
import mas_workflow
from mas_state import Paragraph


def normalize_context(context: list[list]) -> list[Paragraph]:
    """`context` is a list of [title, [sentence, ...]] pairs (upstream
    2WikiMultihopQA / HotpotQA format), verified against the actual downloaded
    parquet -- not the {title, content} dict shape a dataset-card summary
    initially suggested."""
    out = []
    for title, sentences in context:
        for sent_id, text in enumerate(sentences):
            out.append(Paragraph(title=title, sent_id=sent_id, text=text))
    return out


def load_wikihop(path: Path, limit: int | None = None, start: int = 0) -> list[dict[str, Any]]:
    """Read the 2WikiMultihopQA jsonl directly -- no separate benchmark module."""
    items: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            items.append({
                "unique_id": str(row.get(config.ID_KEY, idx)),
                config.QUESTION_KEY: row[config.QUESTION_KEY],
                config.ANSWER_KEY: row.get(config.ANSWER_KEY, ""),
                config.TYPE_KEY: row.get(config.TYPE_KEY, ""),
                "context_paragraphs": normalize_context(row.get(config.CONTEXT_KEY, [])),
                # supporting_facts is already [[title, sent_id], ...] -- upstream list-pair format.
                "supporting_facts": [list(sf) for sf in row.get(config.SUPPORTING_FACTS_KEY, [])],
                "evidences": row.get(config.EVIDENCES_KEY, []),
            })

    items = items[start:]
    if limit is not None:
        items = items[:limit]
    return items


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", default=str(config.WIKIHOP_DEV_PATH), help="2WikiMultihopQA jsonl path")
    parser.add_argument("--limit", type=int, default=None, help="run only the first N questions")
    parser.add_argument("--start", type=int, default=0, help="skip the first N questions")
    parser.add_argument("--run-name", default="wikihop_dev", help="output basename under results/raw/")
    parser.add_argument("--max-workers", type=int, default=None, help="questions in flight (default from config)")
    parser.add_argument("--evaluate", action="store_true", help="score the run immediately after inference")
    parser.add_argument("--show-errors", type=int, default=0, help="with --evaluate: print N incorrect samples")
    args = parser.parse_args()

    items = load_wikihop(Path(args.data), limit=args.limit, start=args.start)
    if not items:
        print("No items to run.", file=sys.stderr)
        return 1

    print(
        f"Running {len(items)} question(s) | model={config.MAIN_LLM.model} "
        f"| workers={args.max_workers or config.MAX_CONCURRENT_TASKS}"
    )

    done = 0

    def progress(_record: dict[str, Any]) -> None:
        nonlocal done
        done += 1
        print(f"  [{done}/{len(items)}] done", end="\r", flush=True)

    results = mas_workflow.run_many(items, on_done=progress, max_workers=args.max_workers)
    print()

    out_path = config.RAW_DIR / f"{args.run_name}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_name": args.run_name,
        "model": config.MAIN_LLM.model,
        "dataset": str(args.data),
        "n": len(results),
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


if __name__ == "__main__":
    raise SystemExit(main())
