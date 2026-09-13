"""Standalone full-120-case audit of round_013's finalized task_agent state
from the 20260912_001239_travel_mas_refactored_gemma_no_backbone_selection_X100Y180
run -- to get an unbiased read on round_013's real quality, since its
official in-run score (~0.49-0.53 across repeat 32-case batches) has been
shown to be significantly deflated by the OpenRouter status="failed"
retriable-but-sometimes-unrecoverable bug: splitting round_013's own
32-case batch into bug-hit (n=8, mean 0.047) vs clean (n=44, mean 0.570)
cases showed the bug, not the edit, is dragging the official score down.

Parallelism is reduced (8, vs the config's usual 32), AND cases are run
in sequential chunks of 8 with a pause between chunks -- both on the
theory that lower/spaced-out concurrent load against OpenRouter may
reduce how often the status="failed"/invalid_prompt condition fires --
per the user's request. This is exploratory, not confirmed: an earlier
controlled test this session (32-way concurrent, 96 calls) did NOT
reproduce the failure at all, so load alone may not be the actual
trigger -- but it costs nothing to test here since we're already paying
for a full 120-case pass.

Single pass (not 3x) -- one clean read on the full benchmark, not a
statistical-confidence exercise. Never touches the live run's own
round_013 directory -- runs against its own fresh copy.

Usage: source /groups/AIC-MV/v.kulkarni1/.env && python3 eval_round013_full_benchmark.py
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

from meta_agent import config as cfg_mod
from meta_agent import runtime_env
from analyze_llm_call_failures import _iter_trace_files, _analyze_trace_file

REPO_ROOT = Path(__file__).resolve().parent
CONFIG = REPO_ROOT / "configs/hgm_travel_gemma_no_backbone_selection_X100Y180.yaml"
SOURCE_TASK_AGENT = (
    REPO_ROOT / "runs"
    / "20260912_001239_travel_mas_refactored_gemma_no_backbone_selection_X100Y180"
    / "round_013" / "task_agent"
)
REDUCED_PARALLELISM = 8
INTER_CHUNK_SLEEP_S = 8.0  # "wait 5-10s after every case" -- applied per
                           # chunk of REDUCED_PARALLELISM cases, since we
                           # run a chunk at a time rather than one case at
                           # a time (still spaces out request bursts to
                           # OpenRouter without going fully serial).
OUT_DIR = (REPO_ROOT / "round013_full_eval_120").resolve()

cfg = cfg_mod.load(str(CONFIG))
runtime_env.apply_all(cfg)
fw = cfg_mod.build_components(cfg)

assert SOURCE_TASK_AGENT.exists(), f"missing: {SOURCE_TASK_AGENT}"

print(f"Evaluator parallelism: {fw.evaluator.parallelism} -> {REDUCED_PARALLELISM}", flush=True)
fw.evaluator.parallelism = REDUCED_PARALLELISM

all_cases = sorted(list(fw.train_case_ids) + list(fw.eval_case_ids), key=int)
print(f"Evaluating round_013 on {len(all_cases)} cases, "
      f"in chunks of {REDUCED_PARALLELISM} with {INTER_CHUNK_SLEEP_S}s between chunks",
      flush=True)

if OUT_DIR.exists():
    shutil.rmtree(OUT_DIR)
OUT_DIR.mkdir(parents=True)

chunks = [
    all_cases[i:i + REDUCED_PARALLELISM]
    for i in range(0, len(all_cases), REDUCED_PARALLELISM)
]

all_per_case: list[dict] = []
crashed_any = False
t0 = time.time()
for idx, chunk in enumerate(chunks):
    chunk_dir = (OUT_DIR / f"round_{idx:02d}").resolve()
    chunk_dir.mkdir(parents=True)
    shutil.copytree(
        SOURCE_TASK_AGENT, chunk_dir / "task_agent",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    (chunk_dir / "logs").mkdir(exist_ok=True)

    t_chunk = time.time()
    result = fw.evaluator.run(chunk_dir, fw.benchmark_dir, case_ids=chunk)
    chunk_elapsed = time.time() - t_chunk
    crashed_any = crashed_any or result.crashed

    for c in result.per_case:
        all_per_case.append({
            "case_id": c.case_id, "passed": c.passed, "score": c.score,
            "error": c.error, "details": c.details,
        })
    print(
        f"chunk {idx + 1}/{len(chunks)} (cases {chunk}): "
        f"mean={result.score:.4f} passed={result.passed} failed={result.failed} "
        f"crashed={result.crashed} elapsed={chunk_elapsed:.0f}s",
        flush=True,
    )
    if idx < len(chunks) - 1:
        time.sleep(INTER_CHUNK_SLEEP_S)

elapsed = time.time() - t0
composite_score = sum(c["score"] for c in all_per_case) / len(all_per_case)
passed = sum(1 for c in all_per_case if c["passed"])
failed = len(all_per_case) - passed

(OUT_DIR / "result.json").write_text(
    json.dumps(
        {
            "node_id": 13,
            "source_run": "20260912_001239_travel_mas_refactored_gemma_no_backbone_selection_X100Y180",
            "parallelism_used": REDUCED_PARALLELISM,
            "inter_chunk_sleep_s": INTER_CHUNK_SLEEP_S,
            "composite_score": composite_score,
            "passed": passed, "failed": failed,
            "crashed": crashed_any, "elapsed_s": elapsed,
            "n_cases": len(all_per_case),
            "per_case": all_per_case,
        },
        indent=2,
    ),
    encoding="utf-8",
)
print(
    f"\ncomposite_score={composite_score:.4f} passed={passed} "
    f"failed={failed} crashed={crashed_any} elapsed={elapsed:.0f}s",
    flush=True,
)

# --- status="failed" bug health, across all chunks' trace.jsonl files ---
totals = {
    "n_llm_calls": 0, "n_status_failed_retries": 0,
    "n_exception_retries": 0, "n_terminal_failed_responses": 0,
}
cases_hit_retry: dict[str, int] = {}
cases_hit_terminal: dict[str, int] = {}
error_codes: dict[str, int] = {}
for trace_path in _iter_trace_files(OUT_DIR):
    r = _analyze_trace_file(trace_path)
    totals["n_llm_calls"] += r["n_llm_calls"]
    totals["n_status_failed_retries"] += r["n_status_failed_retries"]
    totals["n_exception_retries"] += r["n_exception_retries"]
    totals["n_terminal_failed_responses"] += r["n_terminal_failed_responses"]
    for case_id, n in r["cases_with_status_failed_retry"].items():
        cases_hit_retry[case_id] = cases_hit_retry.get(case_id, 0) + n
    for case_id, n in r["cases_with_terminal_failure"].items():
        cases_hit_terminal[case_id] = cases_hit_terminal.get(case_id, 0) + n
    for code, n in r["error_codes"].items():
        error_codes[code] = error_codes.get(code, 0) + n

n_cases_ever_hit = len(set(cases_hit_retry) | set(cases_hit_terminal))
print("\n=== status=\"failed\" bug health (this evaluation) ===")
print(f"total LLM calls: {totals['n_llm_calls']}")
print(f"status=failed retries (recovered): {totals['n_status_failed_retries']}")
print(f"exception retries (unrelated, transient network/etc): {totals['n_exception_retries']}")
print(f"terminal failures (retries exhausted): {totals['n_terminal_failed_responses']}")
print(f"cases with >=1 retry or terminal failure: {n_cases_ever_hit} / {len(all_per_case)} "
      f"({100.0 * n_cases_ever_hit / len(all_per_case):.1f}%)")
print(f"cases with a TERMINAL failure (score likely tanked): {sorted(cases_hit_terminal, key=int)}")
print(f"cases with any retry (recovered, likely fine): "
      f"{sorted(set(cases_hit_retry) - set(cases_hit_terminal), key=int)}")
print(f"error codes given by the API: {error_codes}")

(OUT_DIR / "failure_health.json").write_text(
    json.dumps(
        {
            "totals": totals,
            "cases_with_retry": cases_hit_retry,
            "cases_with_terminal_failure": cases_hit_terminal,
            "n_cases_ever_hit": n_cases_ever_hit,
            "n_cases_total": len(all_per_case),
            "error_codes": error_codes,
        },
        indent=2,
    ),
    encoding="utf-8",
)

print(f"\nOutput saved under: {OUT_DIR}")
