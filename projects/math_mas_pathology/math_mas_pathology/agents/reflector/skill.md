# Skill: Reflector

## What it does
Critically reviews a proposed solution, points out where the reasoning may
be wrong, and produces the corrected final answer.

## Position in the MAS
Stage 3 of 3, and the **terminal agent** — its answer is the MAS's answer.
It never runs before the predictor or verifier.

## Inputs
| Field | Source |
|-------|--------|
| `question` | The math-500 `problem` field |
| `context`  | Whichever context `mas_workflow.py` hands it — by default (pathology 2, stale context injection) this is the **predictor's original first-draft solution**, not the verifier's real final conclusion |

## Output contract
Critique followed by the corrected answer wrapped in `<answer>...</answer>`
tags. `tools/immutable/answer_extraction.py` reads that span.

## Capabilities
- Error detection in multi-step derivations (sign slips, dropped terms,
  arithmetic mistakes, misread problem statements)
- Confirming a correct solution rather than changing it for its own sake

## Limits
- One pass — it does not hand work back to the predictor or verifier.
- **Selective deafness (pathology 3)**: before building its prompt, it
  deterministically truncates whatever `context` string it receives down to
  only the last sentence (`tools/mutable/deafen.py`), dropping every earlier
  sentence — including any caveats, hedges, or corrections they contained.
  Toggle off with `MAS_ENABLE_SELECTIVE_DEAFNESS=0`.
- **Stale context (pathology 2)**: by default it never even sees the
  verifier's conclusion — see `mas_workflow.py`. Toggle off with
  `MAS_ENABLE_STALE_CONTEXT_PATHOLOGY=0`.
- Can introduce errors by "correcting" an already-correct solution.
