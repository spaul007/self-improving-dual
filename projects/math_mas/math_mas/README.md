# math_mas

A two-agent multi-agent system for math problem solving, evaluated on
**MATH-500**. Reimplemented from MASPO's `reflect` topology as a standalone
repo — it does not import MASPO.

**Predictor → Reflector**, run sequentially. The reflector's answer is the
system's answer.

---

## MAS setup

```
        question
           │
           ▼
     ┌───────────┐
     │ PREDICTOR │  solves from scratch, step by step
     └─────┬─────┘
           │ full solution
           ▼
     ┌───────────┐
     │ compress  │  mutable tool: ~30-word briefing
     └─────┬─────┘
           │ short context
           ▼
     ┌───────────┐
     │ REFLECTOR │  critiques, corrects, emits final answer
     └─────┬─────┘
           ▼
      <answer>...</answer>  →  scored
```

| | Predictor | Reflector |
|---|---|---|
| Stage | 1 (first) | 2 (terminal) |
| Context in | none | predictor's solution |
| Output | full solution | critique + final answer |

- **Sequential, not parallel.** The reflector never runs before the predictor.
- **One pass.** No loop back to the predictor, no peer-to-peer chat.
- **Compression is on by default** — the reflector sees a short briefing, not
  the full chain of thought. Set `MAS_USE_COMPRESSED_CONTEXT=0` to pass the
  full solution instead.
- Every agent must wrap its final answer in `<answer>...</answer>`; that span
  is the only thing scored.

### Prompt convention

Prompts live in `mas_prompt_cfg.yaml`, split per agent:

- `role` — **frozen.** The agent's identity. An optimizer must not rewrite it.
- `task` — **editable.** The instruction. Fair game for prompt optimization.

---

## Code structure

Paths below are relative to this directory itself (no extra nesting level —
this folder IS the root):

```
├── config.py                  # env vars, model/endpoint, limits, prompt loader
├── mas_prompt_cfg.yaml   (E)  # role (frozen) + task (editable) per agent
├── llm_client.py              # async LLM wrapper: retry/backoff, concurrency
├── mas_workflow.py       (E)  # run_task (one problem) + run_many (batch)
├── run_inference.py           # CLI: load MATH-500 → run MAS → results/raw/
├── evaluate.py                # CLI: score → results/scored/
│
├── agents/
│   ├── base.py                # shared prompt assembly + LLM call + extraction
│   ├── predictor/
│   │   ├── prompt.py     (E)  # role/task accessors
│   │   ├── skill.md      (E)  # what this agent can and cannot do
│   │   └── workflow.py   (E)  # PredictorAgent
│   └── reflector/             # same three files
│
├── tools/
│   ├── immutable/answer_extraction.py   # benchmark contract — do not tune
│   └── mutable/compress.py         (E)  # predictor→reflector hand-off
│
├── eval/
│   └── metrics.py             # normalize, is_correct, accuracy, reflector_delta
│
├── data/math-500/math_500.jsonl   # 500 problems
└── results/{raw,scored}/          # run outputs
```

**(E)** = editable/tunable by a prompt optimizer.

### Why tools are split

- `tools/immutable/` — `answer_extraction.py` defines *what counts as the
  answer*. Rewriting it changes what the benchmark measures, not how well the
  agents perform. Never tune it.
- `tools/mutable/` — `compress.py` is an internal hand-off protocol, not part
  of the benchmark contract. Free to retune.

---

## Setup

Requires Python 3.10+ and an OpenAI-compatible endpoint (vLLM).

```bash
pip install openai pyyaml
```

Defaults (all overridable by env var — see `config.py`):

| Setting | Env var | Default |
|---|---|---|
| Model | `MAS_MODEL` | `Qwen/Qwen3.5-35B-A3B` |
| Endpoint | `MAS_BASE_URL` | `http://gpu-aic-mv-02-st-p5-node-1:8000/v1` |
| Max tokens | `MAS_MAX_TOKENS` | `8192` |
| Temperature | `MAS_TEMPERATURE` | `0.0` |
| Concurrent tasks | `MAS_MAX_CONCURRENT_TASKS` | `16` |
| LLM calls in flight | `MAS_LLM_CONCURRENCY` | `60` |
| Compress context | `MAS_USE_COMPRESSED_CONTEXT` | `1` (on) |

Check the endpoint is reachable:

```bash
python3 -c "
import asyncio, llm_client
print(asyncio.run(llm_client.get_client().acall('Say OK', max_tokens=10)))"
```

---

## How to run

All commands from this directory.

### Quick smoke test (3 problems)

```bash
python3 run_inference.py --limit 3 --run-name smoke --evaluate
```

### Full MATH-500, inference + scoring in one command

```bash
python3 run_inference.py --run-name math500_full --evaluate
```

Detached, since it takes a while:

```bash
nohup python3 run_inference.py --run-name math500_full --evaluate \
  > results/math500_full.log 2>&1 &
tail -f results/math500_full.log
```

### Separate steps

```bash
python3 run_inference.py --run-name math500_full     # → results/raw/
python3 evaluate.py      --run-name math500_full     # → results/scored/
```

### Useful flags

```bash
# a slice: problems 100–199
python3 run_inference.py --start 100 --limit 100 --run-name slice_100 --evaluate

# more throughput
python3 run_inference.py --run-name fast --evaluate --max-concurrent 32

# inspect the first 10 wrong answers
python3 evaluate.py --run-name math500_full --show-errors 10

# re-score an existing raw file without re-running inference
python3 evaluate.py --raw results/raw/math500_full.json --run-name rescored
```

`run_inference.py` flags: `--data --limit --start --run-name --max-concurrent
--evaluate --show-errors`
`evaluate.py` flags: `--run-name --raw --show-errors`

---

## Output

`results/raw/<run>.json` — per problem: the prediction, gold answer, the full
two-stage trajectory (prompt + raw output per agent), and elapsed time.

`results/scored/<run>.json` — the same records plus `correct`, normalized
answers, and a `summary`:

```
accuracy            # final MAS accuracy
predictor_accuracy  # stage 1 alone
fixed_by_reflector  # predictor wrong → reflector right
broken_by_reflector # predictor right → reflector wrong
errors              # tasks that raised
avg_elapsed_s
```

`fixed_by_reflector` vs `broken_by_reflector` is the metric that matters for
this topology — overall accuracy alone can't tell you whether stage 2 is
earning its cost.

---

## Scoring

Exact match on normalized answers, matching MASPO's non-judge MATH rule, so
numbers are directly comparable:

1. `tools/immutable/answer_extraction.py` pulls the answer —
   `<answer>` tags, else `\boxed{}`, else a trailing-sentence fallback.
2. `eval/metrics.py::normalize_answer` canonicalizes LaTeX (`\frac{1}{2}` →
   `1/2`, strips `$`, `\text{}`, whitespace).
3. Correct iff the normalized strings are equal.

`extract_answer` and `normalize_answer` are kept byte-compatible with MASPO's
`utils.py`.

### Known failure mode

If an agent doesn't emit `<answer>` tags, extraction falls through to the
trailing-sentence heuristic and returns prose fragments, which score as wrong
even when the reasoning was right. This is inherited MASPO behavior. It shows
up as a high `broken_by_reflector` count. If you see that, check whether the
reflector is running past its answer before fixing anything else — it is a
prompt problem in `mas_prompt_cfg.yaml` (an `(E)` file), not a math problem.
