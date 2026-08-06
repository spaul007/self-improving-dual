# Skill: Predictor

## What it does
Produces a first, independent solution to a math problem: step-by-step
reasoning followed by a final answer.

## Position in the MAS
Stage 1 of 2. Runs first, with **no context** from any other agent. Its output
is the only input the reflector reviews.

## Inputs
| Field | Source |
|-------|--------|
| `question` | The math-500 `problem` field |
| `context`  | Always empty for this agent |

## Output contract
Free-text reasoning ending with the final answer wrapped in
`<answer>...</answer>` tags. `tools/immutable/answer_extraction.py` reads that
span — text outside the tags is not scored.

## Capabilities
- Multi-step arithmetic and algebraic manipulation
- LaTeX-formatted answers (fractions, radicals, intervals, coordinates)

## Limits
- Single pass, no self-correction — catching its own mistakes is the
  reflector's job.
- No tool calls: it does not execute code or query anything external.
