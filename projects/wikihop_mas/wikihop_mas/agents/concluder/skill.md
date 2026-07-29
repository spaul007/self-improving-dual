# Skill: Concluder / Reflector

## What it does
Aggregates every hop's (answer, quote, source), judges whether each hop's
evidence is actually grounded for what the question needs, and produces the
final answer. For comparisons between two hop answers, calls `compare_values`
rather than trusting its own date/number arithmetic.

## Position in the MAS
Terminal stage. Called once per question (phase 1); if any hop is judged
ungrounded, the controller reruns that hop once and calls the Concluder a
second, final time (phase 2) — whose output is accepted unconditionally.

## Inputs
| Field | Source |
|-------|--------|
| `question` | The original question |
| `hops_summary` | Every completed hop's sub-question, answer, quote, source, and deterministic quote-verification result |

## Output contract
`{"hop_grounding": [{"hop_id", "grounded", "reason"}, ...], "final_answer", "reasoning"}`.
`tools/immutable/answer_extraction.py` pulls `final_answer` from this (falling
back to an `<answer>` regex only if JSON parsing failed).

## Capabilities
- Genuine LLM tool-calling via `compare_values` (date/number comparator),
  bounded to `config.CONCLUDER_MAX_ROUNDS` rounds.
- Drives the pipeline's one bounded retry loop via `hop_grounding`.

## Limits
- Its second call's output is accepted unconditionally — it cannot keep
  looping even if it still reports a hop as ungrounded. This is a deliberate
  termination guarantee, not a bug.
