# Qwen3.5-35B-A3B (node-1) tool-calling failure — investigation notes

## Summary

The single-agent travel workflow (`projects/travel/seed/workflow.py`) is
**effectively non-functional** against the vLLM server at
`http://gpu-aic-mv-02-st-p5-node-1:8000/v1` (`Qwen/Qwen3.5-35B-A3B`),
regardless of which client-side sampling settings are used. Full 120-case
confirmation: **`no_plan_rate = 1.0`, `mean_score = 0.0`**
(`runs/adhoc_eval/baseline_35b_final_full_full_benchmark/summary.json`).

The same model/endpoint works fine for `projects/travel_mas` (the 4-stage
multi-agent decomposition built this session) — degraded, but nowhere near
as badly (~40% no-plan on a 10-case sample, real scoreable plans on the
rest). This points to something specific to the single agent's long,
complex, many-tool, many-turn conversation pattern rather than a total
server outage.

## Root cause (confirmed via live trace, not inferred)

Manually stepped through the single agent's tool loop call-by-call against
this endpoint (raw `call_llm` responses, not just the evaluator's summary).
At iteration 3 of a real case, the model's own visible text says:

> "The tool name seems to be `query_train_info` but it's not being
> recognized. Let me check the available tools again..."

This is **not true** — the tool call executed successfully and a real
result was fed back into the conversation. The model's own perception that
its tool call wasn't recognized is wrong. After a few rounds of this
confused self-correction, the model degrades into emitting
content-less turns (`function_call` with `content_len=0`, no visible text
at all) until the tool loop's "no more tool_calls" exit condition fires
with nothing usable to return.

This is the **same symptom** originally found with `reasoning_effort:
"medium"` (garbled pseudo-tool-call text emitted as plain content instead
of real function calls) — just a different downstream manifestation of the
same underlying tool-call-recognition confusion.

## What was tried, and what happened

All against the same endpoint/model unless noted. `no_plan_rate` = fraction
of cases where the agent's final output had no usable `<plan>` content at
all (scorer's `plan conversion failed: agent produced no plan` path).

| Config | Agent | Sample size | no_plan_rate | mean score |
|---|---|---|---|---|
| `reasoning_effort: "medium"` (explicit) | single agent | 28 | 27/28 (96%) | ~0 |
| implicit mode, `temperature: 0.0` | single agent | 30 | 29/30 (97%) | 0.025 |
| implicit mode, `temperature: 0.2` | single agent | 10 | 9/10 (90%) | 0.0875 |
| implicit mode, default `temperature: 1.0` | single agent | 10 | 10/10 (100%) | 0.0 |
| implicit mode, `temperature: 0.2` + `max_output_tokens: 16384` | single agent | 10 | 9/10 (90%) | 0.0 |
| implicit mode, `temperature: 0.2` + `max_output_tokens: 16384` | single agent | **120 (full)** | **120/120 (100%)** | **0.0** |
| implicit mode, `temperature: 0.2` + `max_output_tokens: 16384` | **travel_mas** (4-stage MAS) | 10 | **4/10 (40%)** | **0.38125** |

Every single-agent setting tried lands in the same 90-100% no-plan range.
**Temperature and `max_output_tokens` were both conclusively ruled out** as
the variable — the failure signature (`budget_exhausted: False`, stopping
after only 3-9 of a possible 100 tool-loop iterations, empty final
`response.content`) is identical across temperature 0.0, 0.2, and 1.0, and
across a missing vs. an explicit generous `max_output_tokens` cap.

## Why `travel_mas` is much less affected

Same server, same model, same `call_llm` function, same tool-dispatch
mechanism (`platform_core.llm_wrapper.call_llm`,
`projects/travel_mas/seed/tool_wrapper.py`) — but `travel_mas` splits the
single agent's one long, complex conversation (all 9 tools declared every
turn, a huge system prompt with a full worked multi-day example, ~dozens of
turns needed for a real multi-day trip) into 4 short, narrow-scope stages
(Flight: 2 tools, Train: 1 tool, Sightseeing: 7 tools, Accounting: 0
tools/no-tool-loop-at-all), each usually finishing in far fewer turns.

Working hypothesis: the underlying tool-call-recognition quirk may be
present in both, but the single agent's long conversation gives it much
more room to trigger and then compound turn-over-turn into total
breakdown, while each MAS stage's short, focused conversation gives it far
less opportunity to do so. Not independently confirmed (would need a live
trace of a failing MAS stage to see if the same "not being recognized"
phrasing appears there too, just recovering before it cascades) — this is
the leading explanation, not a certainty.

## Likely underlying cause (not fixable client-side)

Everything tried is a **client-side** request parameter
(`reasoning.effort`, `temperature`, `max_output_tokens`) and none of them
changed the outcome. That points at something **server-side**: the node-1
vLLM deployment's tool-call-parser / chat-template configuration
(`--tool-call-parser`, possibly `--chat-template`) may not correctly match
how `Qwen/Qwen3.5-35B-A3B` actually emits tool calls, so the model's own
view of "did my tool call get recognized" doesn't line up with what
actually happened in the conversation history it's echoed back. This is
speculative (no access to how the node-1 server was actually launched to
confirm the flags used) but is the explanation best supported by the
evidence: identical symptom regardless of every client-side lever tried.

## Practical recommendation

- **Don't use the single agent (`projects/travel/seed`) against
  node-1/Qwen3.5-35B-A3B** — it is not currently usable there under any
  setting tried.
- **`projects/travel_mas` (the 4-stage MAS) is usable there**, at
  `temperature: 0.2` + `max_output_tokens: 16384`, implicit mode (no
  `reasoning_effort`) — degraded relative to the 122B endpoint but
  functional, and the decomposition itself appears to be a real robustness
  advantage against this specific server's quirk, not just a
  coincidence.
- For a clean single-agent 35B baseline (if one is ever needed), use
  `Qwen/Qwen3.5-122B-A10B` at `gpu-aic-mv-02-st-p5-node-2:8000` instead,
  which has no such issue (confirmed: `mean_score = 0.663` full 120-case,
  `no_plan_rate = 0.83%`).
- If someone wants to actually root-cause and fix the server side: check
  how the node-1 vLLM instance was launched (`--tool-call-parser` value)
  against Qwen3.5's actual documented tool-call output format, and compare
  against how node-2 (122B, working fine) was launched.

## Relevant files

- `projects/travel/seed/workflow.py` — single agent, the one affected.
- `projects/travel_mas/seed/workflow.py` — 4-stage MAS, meaningfully more
  robust to this on the same endpoint.
- `platform_core/llm_wrapper.py` — `call_llm`, `LLM_REASONING_EFFORT` /
  `LLM_TEMPERATURE` / `LLM_MAX_OUTPUT_TOKENS` env-var resolution (all
  three were tested exhaustively via this session's config changes).
- `configs/eval_local_travel_qwen35b_implicit.yaml` — the single-agent
  config used for all the implicit-mode/temperature/max-tokens tests above.
- `configs/travel_mas_qwen35b_implicit.yaml` — the equivalent `travel_mas`
  config.
- `runs/adhoc_eval/baseline_35b_final_full_full_benchmark/summary.json` —
  the full 120-case confirmation run.
