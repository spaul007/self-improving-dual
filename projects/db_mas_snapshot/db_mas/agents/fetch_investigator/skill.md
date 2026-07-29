# Skill: Fetch Investigator

## What it does
Investigates **FETCH_LARGE_DATA** — SELECT statements scanning/returning very
large amounts of data — by querying PostgreSQL diagnostic views and reporting
concrete evidence plus a high/medium/low likelihood verdict for its assigned
candidate.

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
FETCH_LARGE_DATA. The report is compressed by `tools/mutable/compress.py`
before the lead sees it, so concrete numbers (rows, total_exec_time of the
offending SELECTs) should appear explicitly.

## Capabilities
- `query_db(sql)` tool calls (up to `MAS_TOOL_MAX_ROUNDS` ReAct rounds),
  replayed from the task's recorded snapshot — primary table:
  `pg_stat_statements`, filtering for SELECT statements with high `rows` or
  `total_exec_time`.

## Limits
- Snapshot replay, not a live DB: SQL that matches a recorded battery query
  returns its exact result; anything else referencing a known diagnostic table
  returns that table's **full recorded dump** (the agent's WHERE/ORDER/LIMIT
  are not applied) — the agent must filter the dump by reasoning. Unknown
  tables miss entirely.
- Large-SELECT evidence overlaps with POOR_JOIN_PERFORMANCE / CPU_CONTENTION
  narratives in the task text; only FETCH_LARGE_DATA is a scored label — the
  verdict must speak to it.
- Single pass, no peer chat; sees only its own query results.
