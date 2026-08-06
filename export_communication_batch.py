#!/usr/bin/env python3
"""Batch-run `platform_core.runner --export-communication` across every case in a
benchmark, for one --agent-dir. A thin parallel driver around the existing
single-case CLI (see platform_core/communication_instrumentation.py) -- no new
instrumentation logic here, just fan-out with a bounded concurrency of subprocess
invocations (each case is fully independent, so plain thread-pool-driven
`subprocess.run` calls parallelize cleanly without needing asyncio).

Usage:
    PYTHONPATH=. python3 export_communication_batch.py \\
        --agent-dir projects/math_mas/math_mas \\
        --benchmark projects/math_mas/benchmark \\
        --out-dir communication_traces/math_mas_seed \\
        --patch-agent-run "agents.base:BaseAgent.arun" \\
        --patch-transform "tools.mutable.compress:compress" \\
        --concurrency 10

Each case's trace is written to `<out-dir>/<sanitized_case_id>.json` (case ids
containing "/" are flattened to "__" so they're safe filenames). Prints a running
progress line to stderr; failures are reported but don't stop the batch.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def _sanitize(case_id: str) -> str:
    return case_id.replace("/", "__").replace("\\", "__")


def _load_case_ids(benchmark_dir: Path) -> list[str]:
    ids: list[str] = []
    with (benchmark_dir / "cases.jsonl").open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            case = json.loads(line)
            ids.append(str(case.get("id") or case.get("case_id")))
    return ids


def _run_one(
    agent_dir: Path,
    benchmark_dir: Path,
    case_id: str,
    out_dir: Path,
    patch_agent_run: list[str],
    patch_transform: list[str],
    context_param: str,
    communication_id_suffix: str,
    timeout_s: int,
) -> tuple[str, bool, float, str]:
    out_path = out_dir / f"{_sanitize(case_id)}.json"
    cmd = [
        sys.executable, "-m", "platform_core.runner",
        "--agent-dir", str(agent_dir),
        "--benchmark", str(benchmark_dir),
        "--case-id", case_id,
        "--export-communication", str(out_path),
        "--context-param", context_param,
    ]
    if communication_id_suffix:
        cmd += ["--communication-id", f"{case_id}_{communication_id_suffix}"]
    for target in patch_agent_run:
        cmd += ["--patch-agent-run", target]
    for target in patch_transform:
        cmd += ["--patch-transform", target]

    started = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return case_id, False, time.time() - started, f"timed out after {timeout_s}s"
    elapsed = time.time() - started
    ok = proc.returncode == 0 and out_path.exists()
    err = "" if ok else (proc.stderr or "")[-800:]
    return case_id, ok, elapsed, err


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--agent-dir", required=True, type=Path)
    p.add_argument("--benchmark", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--patch-agent-run", action="append", default=[])
    p.add_argument("--patch-transform", action="append", default=[])
    p.add_argument("--context-param", default="context")
    p.add_argument(
        "--communication-id-suffix",
        default="",
        help="If set, communication_id becomes '<case_id>_<suffix>' instead of the "
        "default '<case_id>_thread_1' -- useful to distinguish two variants "
        "(e.g. 'seed' vs 'round012') of a run over the same benchmark.",
    )
    p.add_argument("--concurrency", type=int, default=10)
    p.add_argument("--timeout", type=int, default=180, help="per-case subprocess timeout, seconds")
    p.add_argument("--limit", type=int, default=None, help="only run the first N cases (debug)")
    p.add_argument(
        "--case-ids-file", type=Path, default=None,
        help="Text file, one case id per line, to run instead of the benchmark's full "
        "cases.jsonl (e.g. a config's held-out eval split, from meta_agent.config.compute_split) "
        "-- --limit still applies on top of this list if both are given.",
    )
    args = p.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.case_ids_file:
        case_ids = [
            line.strip() for line in args.case_ids_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        case_ids = _load_case_ids(args.benchmark)
    if args.limit:
        case_ids = case_ids[: args.limit]

    print(
        f"Running {len(case_ids)} cases against {args.agent_dir} "
        f"(concurrency={args.concurrency}, out_dir={args.out_dir})",
        file=sys.stderr,
    )

    ok_count = 0
    failures: list[dict[str, Any]] = []
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futures = {
            ex.submit(
                _run_one, args.agent_dir, args.benchmark, cid, args.out_dir,
                args.patch_agent_run, args.patch_transform, args.context_param,
                args.communication_id_suffix, args.timeout,
            ): cid
            for cid in case_ids
        }
        for i, fut in enumerate(concurrent.futures.as_completed(futures), 1):
            case_id, ok, elapsed, err = fut.result()
            if ok:
                ok_count += 1
            else:
                failures.append({"case_id": case_id, "elapsed_s": round(elapsed, 1), "error": err})
                print(f"[{i}/{len(case_ids)}] FAILED {case_id} ({elapsed:.1f}s): {err[:200]}", file=sys.stderr)
            if i % 25 == 0 or i == len(case_ids):
                rate = i / (time.time() - t0)
                eta_s = (len(case_ids) - i) / rate if rate > 0 else 0
                print(
                    f"[{i}/{len(case_ids)}] ok={ok_count} fail={len(failures)} "
                    f"eta={eta_s/60:.1f}min",
                    file=sys.stderr,
                )

    total_s = time.time() - t0
    print(
        f"Done in {total_s/60:.1f}min. ok={ok_count} fail={len(failures)} out_dir={args.out_dir}",
        file=sys.stderr,
    )
    if failures:
        failures_path = args.out_dir / "_failures.json"
        failures_path.write_text(json.dumps(failures, indent=2), encoding="utf-8")
        print(f"Wrote failure details to {failures_path}", file=sys.stderr)
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
