# Skill: Extractor

## What it does
Given a sub-question and the Retriever's paragraphs, extracts the specific
fact answering that hop, grounded in a verbatim quote from the source
sentence.

## Position in the MAS
Runs once per hop, immediately after that hop's Retriever call.

## Inputs
| Field | Source |
|-------|--------|
| `sub_question` | Same sub-question passed to the Retriever |
| `context` | The Retriever's accumulated retrieved sentences, formatted with `(title, sent_id)` |

## Output contract
`{"answer", "quote", "source_title", "source_sent_id"}` JSON. The `quote`
field is expected to be a verbatim substring of one retrieved sentence — the
controller checks this automatically via
`tools/immutable/grounding_check.py::verify_quote` after every call.

## Capabilities
- Single-pass structured extraction with source attribution, useful directly
  for supporting-fact scoring.

## Limits
- No tool calls — if the Retriever's paragraphs don't actually contain the
  answer, the Extractor can only report its best guess; catching that is the
  Concluder's grounding-check job, not the Extractor's.
