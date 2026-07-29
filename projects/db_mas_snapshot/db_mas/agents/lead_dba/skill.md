# Skill: Lead DBA

## What it does
Weighs the five investigators' briefings against each other, reconciles
conflicts, and produces the team's final root-cause diagnosis in the task's
required output format, ending with the machine-read verdict line.

## Position in the MAS
Stage 2 of 2, and the **terminal agent** — its answer is the MAS's answer. It
never runs before the investigators.

## Inputs
| Field | Source |
|-------|--------|
| `question` | The task's `problem` field (scenario + output format + case id) |
| `context`  | Five labeled briefings `[Investigator i — CANDIDATE]` — compressed by default, full reports when `MAS_USE_COMPRESSED_CONTEXT=0` |

## Output contract
Free-text diagnosis that **must end** with a single line

    FINAL: <LABEL>[, <LABEL> ...]

naming exactly as many labels as the task asks for, drawn only from:
INSERT_LARGE_DATA, LOCK_CONTENTION, VACUUM, REDUNDANT_INDEX,
FETCH_LARGE_DATA. `tools/immutable/label_extraction.py` anchors on the last
verdict marker and reads the labels from there — text after that line, or
labels merely discussed earlier, are not scored.

## Capabilities
- Cross-candidate arbitration: e.g. discounting a "medium" verdict whose
  evidence is generic while promoting one backed by dominant
  `pg_stat_statements` totals.
- Mapping off-list findings onto the allowed vocabulary (missing/unused index
  → REDUNDANT_INDEX; large scans / join cost → FETCH_LARGE_DATA), because the
  task text also floats MISSING_INDEXES / POOR_JOIN_PERFORMANCE /
  CPU_CONTENTION, which are never scored.

## Limits
- **No tools.** It cannot re-query the database; it sees only what the
  briefings preserved. If compression dropped a number, it cannot recover it.
- One pass — it cannot send an investigator back for more evidence.
- Naming more labels than the task requests wastes nothing at scoring time
  (the list is truncated to the requested count), but under-naming caps
  recall; the count contract is exactly |gold|.
