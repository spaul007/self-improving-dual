"""Derives this project's cases.jsonl from projects/db_mas/benchmark/cases.jsonl
(read-only, never modified) filtered to cases whose real, measured per-case
latency was under 5 minutes on a full 100-case timing probe run against the
pristine db_mas seed.

Source of the timing data: a dedicated `evaluate_task_agent.py` run
(parallelism=8, no HGM/editing involved) against every case in db_mas's own
benchmark, whose real per-case `agent_metadata.timing.total_s` this script
reads directly from the persisted per-case logs:

    runs/adhoc_eval/full_latency_probe_full_benchmark/run_1/logs/case_*.json

3 of the 100 cases timed out entirely (3600s cap) and are correctly excluded
(no timing data to filter on). Of the remaining 97, 54 completed in under
300s -- that's what this project's cases.jsonl contains. Regenerate with:

    python3 projects/db_mas_small_latency/benchmark/generate_cases.py
"""
from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SOURCE_CASES = REPO_ROOT / "projects" / "db_mas" / "benchmark" / "cases.jsonl"
PROBE_LOGS_DIR = (
    REPO_ROOT
    / "runs"
    / "adhoc_eval"
    / "full_latency_probe_full_benchmark"
    / "run_1"
    / "logs"
)
OUT_PATH = Path(__file__).resolve().parent / "cases.jsonl"
LATENCY_CUTOFF_S = 300.0


def load_timing() -> dict[str, float]:
    timing: dict[str, float] = {}
    for f in sorted(PROBE_LOGS_DIR.glob("case_*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        cid = d.get("case_id")
        t = d.get("details", {}).get("agent_metadata", {}).get("timing", {}).get("total_s")
        if cid is not None and t is not None:
            timing[str(cid)] = float(t)
    return timing


def main() -> int:
    timing = load_timing()
    source_cases = [
        json.loads(line) for line in SOURCE_CASES.read_text(encoding="utf-8").splitlines() if line.strip()
    ]

    qualifying = [
        c for c in source_cases if timing.get(str(c.get("id") or c.get("case_id")), float("inf")) < LATENCY_CUTOFF_S
    ]
    qualifying.sort(key=lambda c: timing[str(c.get("id") or c.get("case_id"))])

    with OUT_PATH.open("w", encoding="utf-8") as fh:
        for c in qualifying:
            fh.write(json.dumps(c) + "\n")

    print(f"source cases: {len(source_cases)}")
    print(f"cases with real timing data: {len(timing)}")
    print(f"cases under {LATENCY_CUTOFF_S:.0f}s: {len(qualifying)}")
    print(f"wrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
