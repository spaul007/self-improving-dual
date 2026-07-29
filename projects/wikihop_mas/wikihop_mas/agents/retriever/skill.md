# Skill: Retriever

## What it does
Given one sub-question, searches the question's own ~10 context paragraphs for
relevant sentences via the `search_context` tool (BM25, closed-book) and hands
the accumulated hits to the Extractor. Does not answer the sub-question.

## Position in the MAS
Runs once per hop (1 or 2 hops depending on the Decomposer's plan), and again
for a single retried hop if the Concluder judges its evidence ungrounded.

## Inputs
| Field | Source |
|-------|--------|
| `sub_question` | One entry from the Decomposer's hop-plan (post entity-substitution for a dependent hop 2) |
| `retry_hint` | Empty on the first attempt; a rephrase hint if this is a grounding retry |

## Output contract
No strict output schema — the deliverable is the accumulated tool trace
(`ToolAgentOutput.retrieved_paragraphs`), not the free-text closing note.

## Capabilities
- Genuine LLM tool-calling: can call `search_context` more than once with a
  different phrasing if the first results look thin (up to
  `config.RETRIEVER_MAX_ROUNDS` rounds).

## Limits
- Closed-book: can only search the paragraphs shipped with this question, no
  open-domain Wikipedia access.
- If the round cap is hit while still calling tools, a final tool-free turn is
  forced — it does not loop indefinitely.
