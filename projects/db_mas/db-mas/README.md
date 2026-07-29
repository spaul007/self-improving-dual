# db-mas

Standalone multi-agent system for the database root-cause diagnosis benchmark
(reimplemented from MARBLE's `multiagentbench/database` task, independent of
the MARBLE codebase).

## Architecture

Fixed **Coordinator + 5 Specialists** star topology, identical for every task:

- 5 statically-named specialist agents (`agents/specialists/<name>/`), one per
  real anomaly type used in the benchmark: `insert_large_data`,
  `lock_contention`, `vacuum`, `redundant_index`, `fetch_large_data`. Each has
  its own folder with a hardcoded prompt (verbatim from the original benchmark
  profiles) and investigates independently via `query_db`, then submits
  structured findings via `report_findings`.
- 1 Coordinator agent (`agents/coordinator/`) that receives all 5 specialists'
  findings, can run its own verification queries directly via `query_db` at
  any time, may ask **one** follow-up question to a specific specialist via
  `ask_specialist`, then submits a structured final verdict via
  `submit_verdict`.
- No peer-to-peer chat between specialists.

Tool schemas are organized by scope first, then (for shared tools) by whether
an automated prompt/tool optimizer is allowed to rewrite them:
- `common_tools/` -- tools shared by *multiple* agents (all 5 specialists,
  plus the Coordinator itself, use the identical `query_db` schema/handler --
  not several near-duplicate copies):
  - `common_tools/immutable/query_db.py` -- the system's actual interface to
    the benchmark environment (matches the original MARBLE contract). An
    optimizer must never touch this -- rewriting it would risk changing what
    the benchmark tests, not just how well the agents perform it.
  - `common_tools/mutable/report_findings.py` -- the internal protocol every
    specialist uses to hand its result to the Coordinator. Not part of the
    benchmark's environment contract, so an optimizer is free to retune it.
- `agents/coordinator/tools.py` -- `ask_specialist` and `submit_verdict`.
  These live *inside* the Coordinator's own folder rather than in
  `common_tools/`, since no other agent uses them -- they aren't "common,"
  just this one agent's own tools (still mutable in the same optimizer sense,
  just scoped to a single agent instead of shared across several).

The environment (`environment/`) spins up a real Postgres via Docker Compose,
runs the task's schema/seed SQL, then injects the task's anomalies using logic
ported from MARBLE's `anomaly_trigger/anomaly.py`.

**Setup vs. harness**: `environment/task_setup.py` (`setup_task_environment`/
`teardown_task_environment`) does the one-time, fixed environment prep --
bringing up Postgres, loading the schema, injecting anomalies. This defines
*what's being tested* (same reasoning as `query_db` being immutable) and must
never be touched by an optimizer. `mas_workflow.run_task` calls it, then
everything after that call is the harness: building and running the 5
specialists and the Coordinator -- the actual multi-agent system, and the
part that's fair game to change/tune.

**Parallel task execution**: `mas_workflow.run_many(..., max_workers=N)` runs
up to `N` tasks concurrently, each in its own OS process with its own Docker
Compose project and its own host port (`config.PARALLEL_DB_PORT_BASE + slot`,
a bounded pool of slots handed out via a `multiprocessing.Manager` queue so
concurrency never exceeds `N` and ports never collide). `max_workers=1` is the
original strictly-sequential behavior on the default port 5432. Note this only
parallelizes the per-task wall time that's local (Docker/Postgres/anomaly
injection); the specialist/coordinator phases still call the same shared LLM
endpoint, so heavy concurrency there just means more requests in that
endpoint's queue rather than a clean N&times; speedup.

## Setup

```bash
pip install -e .
cp .env.example .env   # edit if your vLLM endpoint differs
export $(cat .env | xargs)
```

Requires rootless Docker + Docker Compose v2 (no `sudo`).

## Usage

```bash
# Step 1: environment-only smoke test (no LLM)
python scripts/verify_env_standalone.py

# Step 2: single task end-to-end
python scripts/run_single_task.py --task-id 1

# Step 3: small batch (default: one task per anomaly type + a couple 3-label tasks)
# Runs up to 8 tasks concurrently by default (--max-workers 8); each gets its own
# Docker container on its own host port (config.PARALLEL_DB_PORT_BASE + slot).
python scripts/run_batch.py

# Strictly sequential (original behavior, everything on the default port 5432)
python scripts/run_batch.py --max-workers 1

# Run all 100 benchmark tasks, 8 at a time
python scripts/run_batch.py --all

# Score whatever's in results/raw/
python score.py
```

Results land in `results/raw/<task_id>.json` (transcripts + verdict) and, after
`score.py`, `results/scored/<task_id>.json` + `results/summary.json`.

## Notes on fidelity vs. the original MARBLE benchmark

- Labels/specialist profiles are identical across all 100 benchmark tasks, so
  they're hardcoded per-agent rather than templated from task data.
- `len(root_causes)` vs `number_of_labels_pred` is always `(1,2)` or `(2,3)` --
  agents are structurally required to name at least one non-root-cause label,
  so `precision`/`exact_match` are capped below 1.0 by design. `recall`/`F1`
  are the more meaningful metrics.
- Anomaly injection uses the full per-task spec (`threads`, `ncolumn`, `nrow`,
  `colsize`), unlike MARBLE's own runtime which silently drops `nrow`.
