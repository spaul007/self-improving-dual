"""CLI: load dataset + run the MAS over a task/dataset -> results/raw/.

Each run gets an isolated copy of the pristine level database (mirroring
the benchmark's run.sh isolation), so runs are repeatable and carts from
different runs never collide:

    results/raw/<run_name>/
        database/case_{id}/...   # working copy incl. final cart.json + messages.json
        results.jsonl            # one MAS result record per case
        manifest.json

Usage:
    python run_inference.py --level 1 --n 3            # smoke run
    python run_inference.py --level 2                  # full level
    python run_inference.py --level 1 --case-ids 1 7
"""

import argparse
import json
import shutil
import time
from pathlib import Path

import config
from config import MASConfig, RESULTS_RAW
from mas_workflow import MASWorkflow


def parse_args():
    ap = argparse.ArgumentParser(description="Run the Shopping MAS")
    ap.add_argument("--level", type=int, default=None, choices=[1, 2, 3])
    ap.add_argument("--n", type=int, default=None, help="first N cases only")
    ap.add_argument("--case-ids", nargs="+", default=None, help="specific case ids")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--run-name", type=str, default=None)
    ap.add_argument("--max-llm-calls", type=int, default=None)
    ap.add_argument("--temperature", type=float, default=None)
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = MASConfig()
    if args.level is not None:
        cfg.level = args.level
    if args.workers is not None:
        cfg.workers = args.workers
    if args.max_llm_calls is not None:
        cfg.max_llm_calls = args.max_llm_calls
    if args.temperature is not None:
        cfg.temperature = args.temperature

    samples = json.loads(config.level_query_meta(cfg.level).read_text(encoding="utf-8"))
    if args.case_ids:
        wanted = {str(c) for c in args.case_ids}
        samples = [s for s in samples if str(s["id"]) in wanted]
    elif args.n:
        samples = samples[: args.n]
    if not samples:
        raise SystemExit("no samples selected")

    run_name = args.run_name or f"level{cfg.level}_n{len(samples)}"
    run_dir = RESULTS_RAW / run_name
    db_root = run_dir / "database"
    if db_root.exists():
        shutil.rmtree(db_root)
    db_root.mkdir(parents=True)

    master = config.level_database_dir(cfg.level)
    for s in samples:
        src = master / f"case_{s['id']}"
        if not src.exists():
            raise SystemExit(f"missing pristine case dir: {src}")
        shutil.copytree(src, db_root / f"case_{s['id']}")

    print(f"run={run_name}  level={cfg.level}  cases={len(samples)}  "
          f"workers={cfg.workers}  model={cfg.server.served_model_name}")
    t0 = time.time()
    results = MASWorkflow(cfg).run_many(samples, db_root)
    elapsed = time.time() - t0

    with open(run_dir / "results.jsonl", "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    ok = sum(1 for r in results if r["status"] == "ok")
    manifest = {
        "run_name": run_name, "level": cfg.level, "n": len(samples),
        "model": cfg.server.served_model_name,
        "temperature": cfg.temperature,
        "max_tokens": cfg.max_tokens,
        "max_llm_calls": cfg.max_llm_calls, "workers": cfg.workers,
        "executor_ok": ok, "executor_failed": len(samples) - ok,
        "elapsed_seconds": round(elapsed, 1),
        "avg_llm_calls": round(sum(r.get("llm_calls", 0) for r in results) / len(results), 1),
    }
    with open(run_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(json.dumps(manifest, indent=2))
    print(f"\nnext: python evaluate.py --run {run_name}")


if __name__ == "__main__":
    main()
