# Skill: Decomposer

## What it does
Classifies a multi-hop question into one of 2WikiMultihopQA's four reasoning
types and breaks it into 1–2 sub-questions, marking whether they are
independent (order-agnostic) or dependent (hop 2 needs hop 1's answer).

## Position in the MAS
Stage 1, always runs first. Its `dependency` field is what the controller in
`mas_workflow.py` dispatches on to choose the independent-vs-chained execution
path.

## Inputs
| Field | Source |
|-------|--------|
| `question` | The dataset's `question` field |

## Output contract
A JSON object: `{"type", "dependency", "sub_questions": [...]}`. See
`mas_prompt_cfg.yaml`'s `agents.decomposer.task` for the exact shape and
examples. `agents/decomposer/workflow.py::parse_hop_plan` degrades to a single
independent hop over the raw question on a parse failure.

## Capabilities
- Classifies into `comparison | inference | compositional | bridge_comparison`.
- Emits a `{hop1_answer}` placeholder in hop 2's text for dependent chains.

## Limits
- Classifies from the question text alone — never sees the dataset's gold
  `type` label (that would be an oracle shortcut; see `config.ORACLE_TYPE`
  for the explicit ablation toggle).
- No tool calls, no retries of its own — a bad decomposition propagates
  downstream and can only be caught by the Concluder's grounding check.
