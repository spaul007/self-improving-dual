# math_mas_pathology

A sibling of `math_mas` (same MATH-500 benchmark, same scoring contract),
modified to deliberately exhibit three communication pathologies at its
agent-to-agent hand-offs. It exists as a negative-control test-bed: it lets
us check whether the self-improving loop (and its diagnostic/instrumentation
tooling) can actually detect and attribute *broken hand-offs* between
agents, rather than only ever observing generic model weakness.

**Predictor → Verifier (×N, repeated) → Reflector**, run sequentially. The
reflector's answer is the system's answer.

---

## Communication Pathologies (deliberate)

Three independently-toggleable pathologies, all ON by default so the
project exhibits all of them out of the box. Each is implemented directly
in the vendored MAS code (not a wrapper), and each degrades to sane
behavior when its toggle is off — see `config.py` for every env var.

### 1. Repetition-then-ignore (`MAS_ENABLE_REPETITION_PATHOLOGY`)

The Verifier is asked the **exact same** re-verification question
`MAS_VERIFY_ROUNDS` times in a row (default 20) — byte-identical
`(question, context)` on every call, no turn index or prior answer ever fed
back in, so only `MAS_VERIFIER_TEMPERATURE` sampling can make one turn
differ from another. All `N` turns are computed and kept (see
`agents/verifier/workflow.py::VerifierResult.turns`), but **only the last
turn is ever used downstream** — turns 1..N-1 are pure wasted cost with no
guaranteed information value.

Toggle off: verifier runs once (`n=1`).

### 2. Stale context injection (`MAS_ENABLE_STALE_CONTEXT_PATHOLOGY`)

At the Verifier → Reflector hand-off (`mas_workflow.py::run_task`), the
Reflector is deliberately given the Predictor's **original first-draft**
output instead of the Verifier's real, freshest conclusion — even though
the latter was fully computed one line earlier. The entire Verifier stage's
work is computed but never actually reaches the Reflector.

Toggle off: Reflector gets `verifier_final_context` (the verifier's real
last-turn conclusion) instead of `first_draft`.

### 3. Selective deafness (`MAS_ENABLE_SELECTIVE_DEAFNESS`)

Before building its prompt, the Reflector deterministically truncates
whatever context string it receives down to **only its last sentence**
(`tools/mutable/deafen.py`, pure Python, no LLM call) — dropping every
earlier sentence, including any caveats, hedges, or corrections they
contained.

Toggle off: Reflector uses the full context string unchanged.

### Why these three, together

They're deliberately independent and located at different points, so they
can be attributed separately:
- Pathology 1 lives *inside* the Verifier's own repeated calls to itself.
- Pathology 2 lives at the Verifier → Reflector *hand-off*.
- Pathology 3 lives *inside* the Reflector's own message consumption,
  regardless of which context it was handed.

`agents/verifier/*` is left on the same **editable** surface as
`predictor`/`reflector` (see `mutable_exclude` in the HGM configs) — an HGM
run against this project is effectively asking "can the self-improvement
loop detect and fix a broken agent hand-off?"

`platform_core/communication_instrumentation.py` (already verified against
math_mas) can be pointed at this project unchanged: every agent, including
the new Verifier, still goes through the same `BaseAgent.arun` call site;
the only schema delta is the additive, backward-compatible
`AgentOutput.meta` field the Reflector uses to record its deafness
diagnostics.

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
     │ compress  │  mutable tool: ~30-word briefing → first_draft
     └─────┬─────┘
           │ first_draft  (also pathology 2's "stale" artifact)
           ▼
     ┌───────────┐
     │ VERIFIER  │  × MAS_VERIFY_ROUNDS, IDENTICAL input every turn  (pathology 1)
     └─────┬─────┘  only the LAST turn's output is ever used
           │ verifier_final_context = compress(last turn)
           ▼
   context_for_reflector = first_draft, NOT verifier_final_context   (pathology 2)
           │
           ▼
     ┌───────────┐
     │ REFLECTOR │  deafens its context to the last sentence only    (pathology 3)
     └─────┬─────┘  critiques, corrects, emits final answer
           ▼
      <answer>...</answer>  →  scored
```

| | Predictor | Verifier | Reflector |
|---|---|---|---|
| Stage | 1 (first) | 2, repeated ×N | 3 (terminal) |
| Context in | none | predictor's first draft, identical every turn | `context_for_reflector` (stale by default), deafened to its last sentence |
| Output | full solution | re-verification (only last turn used) | critique + final answer |

- **Sequential, not parallel.** No stage runs before its predecessor.
- **One pass overall** — no loop back to an earlier stage, no peer-to-peer
  chat — but the Verifier stage itself makes `MAS_VERIFY_ROUNDS` identical,
  independent calls (pathology 1).
- **Compression is on by default** — mirrors math_mas's
  `MAS_USE_COMPRESSED_CONTEXT` convention, now applied at two hand-offs
  (predictor→verifier, verifier→[diagnostic] context).
- Every agent must wrap its final answer in `<answer>...</answer>`; that
  span is the only thing scored.

### Prompt convention

Prompts live in `mas_prompt_cfg.yaml`, split per agent (unchanged from
math_mas):

- `role` — **frozen.** The agent's identity. An optimizer must not rewrite it.
- `task` — **editable.** The instruction. Fair game for prompt optimization.

The Verifier's `task` template is identical on every one of its
`MAS_VERIFY_ROUNDS` calls, by design — that sameness is what makes it "the
same question asked N times" rather than N different questions.

---

## Code structure

Paths below are relative to this directory itself (no extra nesting level —
this folder IS the root):

```
├── config.py                  # env vars, model/endpoint, limits, prompt loader,
│                               # + pathology toggles / MAS_VERIFY_ROUNDS /
│                               # MAS_VERIFIER_TEMPERATURE
├── mas_prompt_cfg.yaml   (E)  # role (frozen) + task (editable) per agent,
│                               # now including `verifier`
├── llm_client.py               # async LLM wrapper: retry/backoff, concurrency
├── mas_workflow.py       (E)  # run_task (one problem) + run_many (batch);
│                               # pathologies 1 and 2 are wired here
├── run_inference.py            # CLI: load MATH-500 → run MAS → results/raw/
├── evaluate.py                  # CLI: score → results/scored/
│
├── agents/
│   ├── base.py                 # shared prompt assembly + LLM call + extraction
│   │                            # (+ AgentOutput.meta, additive)
│   ├── predictor/               # unchanged from math_mas
│   │   ├── prompt.py     (E)
│   │   ├── skill.md      (E)
│   │   └── workflow.py   (E)
│   ├── verifier/                # NEW — pathology 1 (repetition-then-ignore)
│   │   ├── prompt.py     (E)
│   │   ├── skill.md      (E)
│   │   └── workflow.py   (E)   # VerifierAgent.arun_repeated
│   └── reflector/
│       ├── prompt.py     (E)
│       ├── skill.md      (E)
│       └── workflow.py   (E)   # build_prompt applies pathology 3 (deafen)
│
├── tools/
│   ├── immutable/answer_extraction.py   # benchmark contract — do not tune
│   └── mutable/
│       ├── compress.py            (E)  # hand-off summarizer
│       └── deafen.py              (E)  # NEW — pathology 3's sentence-splitter
│
├── eval/
│   └── metrics.py              # normalize, is_correct, accuracy, stage_delta
│                                # (generalizes math_mas's reflector_delta),
│                                # reflector_delta, verifier_delta
│
├── data/math-500/math_500.jsonl   # 500 problems, same set math_mas uses
└── results/{raw,scored}/          # run outputs
```

**(E)** = editable/tunable by a prompt optimizer.

---

## Setup

Same as `math_mas` — Python 3.10+, an OpenAI-compatible endpoint (vLLM).

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
| Verifier temperature | `MAS_VERIFIER_TEMPERATURE` | `0.2` |
| Verify rounds | `MAS_VERIFY_ROUNDS` | `20` |
| Concurrent tasks | `MAS_MAX_CONCURRENT_TASKS` | `16` |
| LLM calls in flight | `MAS_LLM_CONCURRENCY` | `60` |
| Compress context | `MAS_USE_COMPRESSED_CONTEXT` | `1` (on) |
| Repetition pathology | `MAS_ENABLE_REPETITION_PATHOLOGY` | `1` (on) |
| Stale-context pathology | `MAS_ENABLE_STALE_CONTEXT_PATHOLOGY` | `1` (on) |
| Selective-deafness pathology | `MAS_ENABLE_SELECTIVE_DEAFNESS` | `1` (on) |

To run something close to plain math_mas's topology for comparison, set all
three `MAS_ENABLE_*_PATHOLOGY`/`MAS_ENABLE_SELECTIVE_DEAFNESS` toggles to
`0` (the Verifier stage still runs once, just with no pathological effect).

---

## How to run

Same interface as `math_mas` — all commands from this directory.

```bash
# quick smoke test (3 problems)
python3 run_inference.py --limit 3 --run-name smoke --evaluate

# full MATH-500, inference + scoring in one command
python3 run_inference.py --run-name math500_full --evaluate

# ablate one pathology at a time
MAS_ENABLE_STALE_CONTEXT_PATHOLOGY=0 python3 run_inference.py --limit 20 --run-name no_stale --evaluate
```

`run_inference.py`/`evaluate.py` flags are unchanged from math_mas's own
(`--data --limit --start --run-name --max-concurrent --evaluate
--show-errors` / `--run-name --raw --show-errors`).

---

## Output

`results/raw/<run>.json` — per problem: the prediction, gold answer, the
full three-stage trajectory (prompt + raw output per agent, including every
discarded verifier turn), `first_draft`, `verifier_final_context`,
`context_used_by_reflector`, the active `pathology_flags`, and elapsed time.

`results/scored/<run>.json` — the same records plus `correct`, normalized
answers, and a `summary`:

```
accuracy             # final MAS accuracy
predictor_accuracy   # stage 1 alone
fixed_by_reflector    # predictor wrong -> final right   (combined verifier+reflector effect)
broken_by_reflector   # predictor right -> final wrong
fixed_by_verifier      # predictor wrong -> verifier right (verifier stage alone, in principle)
broken_by_verifier     # predictor right -> verifier wrong
errors                # tasks that raised
avg_elapsed_s
```

`fixed_by_verifier`/`broken_by_verifier` vs. `fixed_by_reflector`/
`broken_by_reflector` is the key diagnostic pair for this project: it shows
whether the verifier stage would have helped *in principle*, independent of
whether pathology 2 ever let that help reach the final answer.

---

## Scoring

Identical to math_mas — exact match on normalized answers
(`tools/immutable/answer_extraction.py` → `eval/metrics.py::normalize_answer`
→ string equality), unmodified so the two projects stay directly
comparable.

### Known failure modes

- Inherited from math_mas: if an agent doesn't emit `<answer>` tags,
  extraction falls through to a trailing-sentence heuristic and returns
  prose fragments, scored as wrong even when the reasoning was right.
- New to this project: even a *correct* verifier conclusion can be
  invisible in the final accuracy number if pathology 2 (stale context) is
  on — check `fixed_by_verifier`/`broken_by_verifier` against
  `fixed_by_reflector`/`broken_by_reflector` before concluding the verifier
  stage "doesn't help."
