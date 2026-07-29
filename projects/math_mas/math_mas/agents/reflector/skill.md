# Skill: Reflector

## What it does
Critically reviews the predictor's solution, points out where the reasoning may
be wrong, and produces the corrected final answer.

## Position in the MAS
Stage 2 of 2, and the **terminal agent** — its answer is the MAS's answer. It
never runs before the predictor.

## Inputs
| Field | Source |
|-------|--------|
| `question` | The math-500 `problem` field |
| `context`  | The predictor's solution — compressed briefing by default, full text when `MAS_USE_COMPRESSED_CONTEXT=0` |

## Output contract
Critique followed by the corrected answer wrapped in `<answer>...</answer>`
tags. `tools/immutable/answer_extraction.py` reads that span.

## Capabilities
- Error detection in multi-step derivations (sign slips, dropped terms,
  arithmetic mistakes, misread problem statements)
- Confirming a correct solution rather than changing it for its own sake

## Limits
- One pass — it does not hand work back to the predictor.
- Sees only what the compression tool forwarded; if that briefing drops a
  detail, the reflector cannot recover it. Set
  `MAS_USE_COMPRESSED_CONTEXT=0` to review the full solution instead.
- Can introduce errors by "correcting" an already-correct solution.
