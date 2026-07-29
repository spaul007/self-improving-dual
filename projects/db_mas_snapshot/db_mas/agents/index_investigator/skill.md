# Skill: Index Investigator

## What it does
Investigates **REDUNDANT_INDEX** — unused or duplicate indexes adding write
overhead — by querying PostgreSQL diagnostic views and reporting concrete
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
REDUNDANT_INDEX. The report is compressed by `tools/mutable/compress.py`
before the lead sees it, so concrete findings (duplicate index definitions,
idx_scan = 0 indexes) should appear explicitly.

## Capabilities
- `query_db(sql)` tool calls (up to `MAS_TOOL_MAX_ROUNDS` ReAct rounds),
  replayed from the task's recorded snapshot — primary tables:
  `pg_stat_user_indexes` (usage: `idx_scan`) and `pg_indexes` (definitions;
  duplicate/overlapping column sets on the same table).

## Limits
- Snapshot replay, not a live DB: SQL that matches a recorded battery query
  returns its exact result; anything else referencing a known diagnostic table
  returns that table's **full recorded dump** (the agent's WHERE/ORDER/LIMIT
  are not applied) — the agent must filter the dump by reasoning. Unknown
  tables miss entirely.
- Distinguish REDUNDANT_INDEX (duplicate/unused indexes exist) from
  MISSING_INDEXES (not in the scored label set — findings of absent indexes
  must not be reported as a separate candidate; the lead maps them per its
  instructions).
- Single pass, no peer chat; sees only its own query results.
