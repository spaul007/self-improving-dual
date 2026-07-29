# Skill: Vacuum Investigator

## What it does
Investigates **VACUUM** — inappropriate/aggressive vacuum or autovacuum
activity — by querying PostgreSQL diagnostic views and reporting concrete
evidence plus a high/medium/low likelihood verdict for its assigned candidate.

## Position in the MAS
Stage 1 of 2. Runs in parallel with the four other investigators, with **no
context** from any of them. Its compressed report is one of the five briefings
the lead DBA weighs. It never makes the final decision.

## Inputs
| Field | Source |
|-------|--------|
| `question` | The task's `problem` field (scenario + output format + case id) |
| `context`  | Always empty for this agent |

## Output contract
Free-text evidence report ending with a high/medium/low likelihood verdict for
VACUUM. The report is compressed by `tools/mutable/compress.py` before the
lead sees it, so concrete numbers (VACUUM call counts, exec time) should
appear explicitly.

## Capabilities
- `query_db(sql)` tool calls (up to `MAS_TOOL_MAX_ROUNDS` ReAct rounds),
  replayed from the task's recorded snapshot — recommended reads:
  `pg_stat_all_tables` / `pg_stat_user_tables` (vacuum counters, dead tuples)
  and `pg_stat_statements` filtered for `VACUUM%` statements.

## Limits
- Snapshot replay, not a live DB: SQL that matches a recorded battery query
  returns its exact result; anything else referencing a known diagnostic table
  returns that table's **full recorded dump** (the agent's WHERE/ORDER/LIMIT
  are not applied) — the agent must filter the dump by reasoning. Unknown
  tables miss entirely.
- **Environment quirk (verified across all 100 snapshots):** table-level
  vacuum counters are freshly reset at recording time — `n_dead_tup` = 0 and
  `vacuum_count` = 0 everywhere, and `pg_stat_progress_vacuum` is always empty
  (transient view). "Dead tuples are 0, so not VACUUM" is therefore a false
  inference. The durable signal for injected VACUUM anomalies is `VACUUM FULL`
  entries in `pg_stat_statements` (large `calls` / `total_exec_time`).
- Single pass, no peer chat; sees only its own query results.
