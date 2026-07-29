# Skill: Lock Investigator

## What it does
Investigates **LOCK_CONTENTION** — blocked/waiting queries and lock waits — by
querying PostgreSQL diagnostic views and reporting concrete evidence plus a
high/medium/low likelihood verdict for its assigned candidate.

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
LOCK_CONTENTION. The report is compressed by `tools/mutable/compress.py`
before the lead sees it, so concrete findings (wait_event_type, blocked pids,
ungranted locks) should appear explicitly.

## Capabilities
- `query_db(sql)` tool calls (up to `MAS_TOOL_MAX_ROUNDS` ReAct rounds),
  replayed from the task's recorded snapshot — primary tables: `pg_locks`
  (ungranted locks) and `pg_stat_activity` (waiting/blocked backends and
  their queries).

## Limits
- Snapshot replay, not a live DB: SQL that matches a recorded battery query
  returns its exact result; anything else referencing a known diagnostic table
  returns that table's **full recorded dump** (the agent's WHERE/ORDER/LIMIT
  are not applied) — the agent must filter the dump by reasoning. Unknown
  tables miss entirely.
- The snapshot is one moment in time: short-lived lock waits that resolved
  before recording are invisible; sustained contention shows as waiting
  backends / ungranted locks in the dumps.
- Single pass, no peer chat; sees only its own query results.
