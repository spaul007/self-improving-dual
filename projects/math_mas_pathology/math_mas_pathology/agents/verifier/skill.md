# Skill: Verifier

## What it does
Independently re-derives a proposed solution to a math problem from scratch
and states whether it agrees with it.

## Position in the MAS
Stage 2 of 3, inserted between the predictor and the reflector. Runs
`MAS_VERIFY_ROUNDS` times per problem when `MAS_ENABLE_REPETITION_PATHOLOGY`
is on (see README.md "Communication Pathologies" — pathology 1).

## Inputs
| Field | Source |
|-------|--------|
| `question` | The math-500 `problem` field |
| `context`  | The predictor's first-draft solution — **identical on every one of the `MAS_VERIFY_ROUNDS` calls**, never updated with the verifier's own prior turns |

## Output contract
Free-text re-derivation ending with the final answer wrapped in
`<answer>...</answer>` tags. `tools/immutable/answer_extraction.py` reads
that span.

## Capabilities
- Independent re-derivation of a proposed solution
- Explicit agree/disagree judgment against the proposed answer

## Limits
- Stateless and single-pass per call — has no memory of its own earlier
  turns, so repeated calls with the same input can only vary by sampling
  temperature (`MAS_VERIFIER_TEMPERATURE`), not by learning from itself.
- Only its *last* turn's output is ever read by the pipeline (pathology 1);
  earlier turns are computed but discarded.
- Its final conclusion may not even reach the reflector — see pathology 2
  (stale context injection) in `mas_workflow.py`.
