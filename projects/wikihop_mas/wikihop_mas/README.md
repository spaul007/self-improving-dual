# wikihop_mas

A four-agent multi-agent system for multi-hop QA, evaluated on
**2WikiMultihopQA**. Structurally mirrors
`rsi/agents/math_mas` (config/LLM-client/prompt-YAML/agent-directory/tools/eval
layout), but with a non-sequential, controller-driven topology instead of a
fixed pipeline, and genuine LLM tool-calling instead of plain Python function
calls.

**Decomposer → (independent hops | chained hops) → Concluder**, with a bounded
grounding-retry loop back to the Retriever. The Concluder's answer is the
system's answer.

---

## MAS setup

```
                          question
                             │
                             ▼
                      ┌──────────────┐
                      │  DECOMPOSER  │  classifies reasoning type, emits hop-plan
                      └──────┬───────┘
                             │
              ┌──────────────┴───────────────┐
              │ independent                    │ dependent
              │ (comparison /                  │ (inference /
              │  bridge_comparison)            │  compositional)
              ▼                                ▼
      hop1 --------------- hop2         hop1 --> entity_substitution --> hop2
      (order doesn't                    (hop2's question gets hop1's answer
       matter, run one                   substituted in before it runs)
       after another)
              │                                │
              └──────────────┬─────────────────┘
                             ▼
                      ┌──────────────┐
                      │  CONCLUDER   │  phase 1: final_answer + per-hop grounding
                      └──────┬───────┘
                             │
                  any hop ungrounded? ── yes ──> rerun ONLY that hop
                             │                          │
                             │ no                        ▼
                             │                   ┌──────────────┐
                             │                   │  CONCLUDER   │  phase 2 (mandatory,
                             │                   └──────┬───────┘   final, accepted
                             │                          │           unconditionally)
                             └──────────────┬───────────┘
                                            ▼
                                     final_answer  →  scored
```

Each hop is **Retriever** (tool-calling: `search_context`) → **Extractor** →
a deterministic quote-verification check.

| | Decomposer | Retriever | Extractor | Concluder |
|---|---|---|---|---|
| LLM calls | 1 | multi-turn (tool loop) | 1 | multi-turn (tool loop), 1–2× |
| Tool | — | `search_context` (BM25) | — | `compare_values` (date/number) |
| Output | hop-plan JSON | tool trace + note | `{answer, quote, source}` JSON | `{hop_grounding, final_answer}` JSON |

- **Non-sequential, controller-driven.** Independent-type questions run their
  two hops one after another (no data dependency, but also no concurrency —
  kept deliberately simple); dependent-type questions chain hop1's answer into
  hop2 via `{hop1_answer}` template substitution.
- **Bounded, not unbounded.** The Concluder is called at most twice; only the
  specific ungrounded hop is rerun (not the whole pipeline); every tool-calling
  agent is capped at a fixed number of LLM↔tool rounds with a forced final
  turn if the cap is hit. No `while True` gated on model output anywhere.
- **Closed-book retrieval.** The Retriever's `search_context` tool only
  searches the ~10 context paragraphs shipped with the current question — no
  open-domain Wikipedia access.
- **Fully synchronous — no asyncio.** Unlike math_mas, every LLM call here is
  a plain blocking `OpenAI` client call. Batch throughput across questions (if
  wanted) uses a plain `concurrent.futures.ThreadPoolExecutor`.
- **Structured JSON hand-offs**, not `<answer>` tags — every agent's task
  prompt in `mas_prompt_cfg.yaml` embeds a literal JSON example of its
  expected output shape.

### Prompt convention

Prompts live in `mas_prompt_cfg.yaml`, split per agent (same convention as
math_mas):

- `role` — **frozen.** The agent's identity. An optimizer must not rewrite it.
- `task` — **editable.** The instruction. Fair game for prompt optimization.

---

## Code structure

```
wikihop_mas/
├── config.py                  # env vars, model/endpoint, limits, prompt loader
├── mas_prompt_cfg.yaml   (E)  # role (frozen) + task (editable) per agent
├── llm_client.py               # sync OpenAI wrapper: call() + call_messages() (tools), retry/backoff
├── mas_state.py                # blackboard dataclasses: HopPlan, HopResult, MASState
├── mas_workflow.py       (E)  # run_task (one question) + run_many (batch) + the controller
├── run_inference.py            # CLI: load jsonl → run MAS → results/raw/
├── evaluate.py                 # CLI: score → results/scored/
│
├── scripts/
│   └── convert_parquet_to_jsonl.py   # ONE-OFF: pyarrow-only, not part of the runtime system
│
├── agents/
│   ├── base.py                 # BaseAgent (single-turn JSON) + ToolAgent (multi-turn/tools)
│   ├── decomposer/              # prompt.py, skill.md, workflow.py (DecomposerAgent)
│   ├── retriever/                # same 3 files (RetrieverAgent, tool-calling)
│   ├── extractor/                 # same 3 files (ExtractorAgent)
│   └── concluder/                  # same 3 files (ConcluderAgent, tool-calling)
│
├── tools/
│   ├── immutable/               # benchmark contract + environment definition — do not tune
│   │   ├── search_context.py     # per-question BM25 index + tool schema
│   │   ├── compare_values.py      # date/number comparator + tool schema
│   │   ├── grounding_check.py      # verify_quote() — deterministic, not an LLM tool
│   │   └── answer_extraction.py    # pulls final_answer out of Concluder's JSON
│   └── mutable/
│       └── entity_substitution.py  # hop2 {hop1_answer} templating
│
├── eval/
│   └── metrics.py               # normalize, EM/F1, supporting-fact EM/F1, joint EM/F1, retry_delta
│
├── data/2wikimultihopqa/
│   ├── {train,dev,test}.parquet  # downloaded once (see Setup) — not read by the runtime system
│   └── {train,dev,test}.jsonl     # produced once by scripts/convert_parquet_to_jsonl.py — this is what run_inference.py reads
└── results/{raw,scored}/          # run outputs
```

**(E)** = editable/tunable by a prompt optimizer.

### Why tools are split

- `tools/immutable/` — defines what counts as evidence/the answer/the search
  space. Rewriting these changes what the benchmark measures or what agents
  can even find, not how well they reason. Never tune.
- `tools/mutable/` — `entity_substitution.py` is an internal hand-off
  protocol, not part of the benchmark contract. Free to retune.

---

## Setup

Requires Python 3.10+ and an OpenAI-compatible endpoint (vLLM).

```bash
pip install openai pyyaml python-dateutil
```

**No pyarrow/pandas dependency for the runtime system** — parquet is only
touched once, by `scripts/convert_parquet_to_jsonl.py`.

### 1. Download the dataset (once)

```bash
mkdir -p data/2wikimultihopqa
for split in train dev test; do
  curl -L -o data/2wikimultihopqa/${split}.parquet \
    https://huggingface.co/datasets/xanhho/2WikiMultihopQA/resolve/main/${split}.parquet
done
```

### 2. Convert parquet → jsonl (once)

```bash
pip install pyarrow   # only ever needed for this step
python3 scripts/convert_parquet_to_jsonl.py
```

This writes `data/2wikimultihopqa/{train,dev,test}.jsonl`. `run_inference.py`
reads only these `.jsonl` files from here on.

Defaults (all overridable by env var — see `config.py`):

| Setting | Env var | Default |
|---|---|---|
| Model | `MAS_MODEL` | `Qwen/Qwen3.5-35B-A3B` |
| Endpoint | `MAS_BASE_URL` | `http://gpu-aic-mv-02-st-p5-node-1:8000/v1` |
| Max tokens | `MAS_MAX_TOKENS` | `8192` |
| Temperature | `MAS_TEMPERATURE` | `0.0` |
| Concurrent questions (run_many) | `MAS_MAX_CONCURRENT_TASKS` | `8` |
| Retriever tool-call round cap | `MAS_RETRIEVER_MAX_ROUNDS` | `3` |
| Concluder tool-call round cap | `MAS_CONCLUDER_MAX_ROUNDS` | `3` |
| Per-hop grounding-retry cap | `MAS_MAX_HOP_RETRIES` | `1` |
| Oracle type (ablation/debug) | `MAS_WIKIHOP_ORACLE_TYPE` | `0` (off) |

Check the endpoint is reachable:

```bash
python3 -c "import llm_client; print(llm_client.get_client().call('Say OK'))"
```

---

## How to run

All commands from this directory.

### Quick smoke test (3 questions)

```bash
python3 run_inference.py --limit 3 --run-name smoke --evaluate
```

### Full dev set, inference + scoring in one command

```bash
python3 run_inference.py --run-name dev_full --evaluate
```

Detached, since it takes a while:

```bash
nohup python3 run_inference.py --run-name dev_full --evaluate \
  > results/dev_full.log 2>&1 &
tail -f results/dev_full.log
```

### Separate steps

```bash
python3 run_inference.py --run-name dev_full     # → results/raw/
python3 evaluate.py      --run-name dev_full     # → results/scored/
```

### Useful flags

```bash
# a slice: questions 100–199
python3 run_inference.py --start 100 --limit 100 --run-name slice_100 --evaluate

# more throughput
python3 run_inference.py --run-name fast --evaluate --max-workers 16

# inspect the first 10 wrong answers
python3 evaluate.py --run-name dev_full --show-errors 10

# re-score an existing raw file without re-running inference
python3 evaluate.py --raw results/raw/dev_full.json --run-name rescored
```

`run_inference.py` flags: `--data --limit --start --run-name --max-workers
--evaluate --show-errors`
`evaluate.py` flags: `--run-name --raw --show-errors`

---

## Output

`results/raw/<run>.json` — per question: the prediction, gold answer, the full
hop-plan and per-hop trajectory (retriever tool trace, extractor output,
grounding verdicts), the Concluder's call(s), and elapsed time.

`results/scored/<run>.json` — the same records plus per-sample metrics and a
`summary`:

```
answer_em / answer_f1              # final MAS answer quality
sp_em / sp_f1                      # supporting-fact quality (title, sent_id) sets
joint_em / joint_f1                # combined answer+supporting-fact metric
decomposer_type_accuracy           # Decomposer's own classification vs. gold `type` (eval-only)
avg_retriever_rounds               # avg tool-call rounds per hop
pct_hops_retried                   # fraction of hops that needed a grounding retry
fixed_by_grounding_retry           # retry rescued a wrong pre-retry answer
broken_by_grounding_retry          # retry broke a correct pre-retry answer
errors / avg_elapsed_s
```

`fixed_by_grounding_retry` vs `broken_by_grounding_retry` is the metric that
matters for this topology's one non-trivial control-flow decision — overall
accuracy alone can't tell you whether the retry loop is earning its cost.

---

## Scoring

Standard HotpotQA/2WikiMultihopQA-style scoring:

1. `tools/immutable/answer_extraction.py` pulls the answer from the
   Concluder's JSON (`final_answer` field), falling back to an
   `<answer>...</answer>` regex only if JSON parsing failed.
2. `eval/metrics.py::normalize_answer` lowercases, strips articles/punctuation,
   fixes whitespace.
3. **Answer**: exact match + token-overlap F1 on normalized strings.
   **Supporting facts**: precision/recall/F1 over predicted vs. gold
   `(title, sent_id)` sets (predicted = deduplicated `extractor_source` across
   hops). **Joint**: the standard combination of the two.

### Known failure modes

- If the Concluder doesn't emit valid JSON, extraction falls through to the
  `<answer>` regex and may return an empty string, scoring as wrong even if
  the reasoning was right. Check `_parse_error` in the raw trajectory before
  assuming a reasoning failure — it may be a prompt-formatting problem in
  `mas_prompt_cfg.yaml` (an `(E)` file).
- A high `broken_by_grounding_retry` count means the retry loop is net-harmful
  for some question shape — worth inspecting before tuning anything else,
  since it's the one place this topology can make things worse.
