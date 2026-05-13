# Experiment log

Authoritative record of every optimization / evaluation run on this
repo. Append-only. Each entry is self-contained — setup, config,
results, run-dir paths, diagnosis.

`DEV_LOG.md` is for *code* and *decision* history; this file is for
*experiment* history. Don't duplicate result tables in both.

## Entry template

Copy this when adding a new run.

```
## YYYY-MM-DD — short title (project · what changed)

Job: <SLURM jobid or "local">           Wall: <HH:MM>
Run dir: <runs/... or /groups/AIC-MV/n.tzou/evaluations/...>
Config: <configs/foo.yaml @ commit short-sha>
Models: task=<>  editor=<>  strategy=<>  scorer-plan-convert=<>
Split / scope: <max_rounds N · train_size T · max_cases M · parallelism P>

### Results
| ... per-round / per-level table ... |

### Observations / diagnosis
- ...

### Links
- Authoritative summary (if external): /groups/AIC-MV/n.tzou/...
- Related commits: short-sha, short-sha
```

---

# Travel project

## 2026-05-05 — first live evolution loop (smoke, 5 cases)

Job: 137618 · gpu-aisystem-queue-st-p5-node-11    Wall: 44:19
Run dir: `runs/20260505_*_travel_default/` (deleted in 2026-05-06 cleanup)
Config: `configs/travel.yaml` (`max_cases: 5`, `max_rounds: 5`, reasoning="high")

### Results
| Round | Score | Notes |
|---|---|---|
| 0 | **0.138** ← best | 1/5 plans emitted; case 4 hit composite 0.6875 |
| 1 | 0.000 | Editor added "final repair pass" → 5/5 plans but hallucinated facts → all 0 |
| 2 | 0.087 | Branched from round 0; mild variation |
| 3-5 | 0.138 | No further improvement; round 4 edit failed validation |

### Observations
- First confirmation that the end-to-end loop works under SLURM (cgroup tear-down clean).
- Plan-conversion call to `gpt-5-2025-08-07` runs end-to-end against live API.
- Sparse signal at N=5 cases — strategy could not find a real lift.

---

## 2026-05-06 — seed full-120 attempts (token-cap bug discovery)

Sequence of partial / aborted full-benchmark seed evals while diagnosing
why the seed scored ~0.05 on 111 cases.

| Attempt | Setup | Result | Verdict |
|---|---|---|---|
| 1 | `wall_time=600`, MAX_ITER=100, reasoning=high, p=4 | All cases timed out @ 600s | Wall too short, cancelled |
| 2 | `wall_time=1800`, MAX_ITER=40, reasoning=medium, p=4 | Cancelled (no per-case persistence yet) | — |
| 3 | `wall_time=1800`, MAX_ITER=40, reasoning=medium, p=16 | 111/120 complete, mean composite **0.0535** | Killed connection mid-run |
| 4 (first high-reasoning) | reasoning=high, MAX_ITER=100, parity-ported seed | Plan rate **18 %** | Token cap bug discovered |
| 5 (after token-cap fix) | `DEFAULT_MAX_OUTPUT_TOKENS: 8192 → 32768`, 20 cases | **0.3312** (plan rate 100 %) | Hit ≥0.30 target ✓ |

### Root cause of partial 4
`reasoning=8192 out=8192` → reasoning consumed full output budget,
visible content was empty. Fixed by bumping
`platform_core/llm_wrapper.py::DEFAULT_MAX_OUTPUT_TOKENS` from 8192 →
32768.

### Result file (keep)
- `runs/eval_20260506_062620_travel_baseline/` — parity-port baseline 0.331

---

## 2026-05-06 — travel optimization, parity fixes applied (138368)

Job: 138368 · gpu-aic-mv-01-st-p5-node-1    Wall: ~1h 08m
Run dir: `runs/20260506_203919_travel_default/`
Config: `configs/travel.yaml` (3 rounds × 10 train + 10 eval, reasoning=medium)
Parity fixes active: `messages.extend(raw_output)`, plan-converter retries 3→10, JSON tool error envelope

### Results
| Round | Train | Eval | Notes |
|---|---|---|---|
| 0 | **0.431** | 0.469 | Post-fix seed (pre-fix was 0.175 / 0.331) |
| 1 | 0.000 | 0.362 | Edit failed validation — eval ran against unchanged base |
| 2 | 0.344 | 0.400 | Mutation applied but regressed (rejected by branch policy) |
| 3 | **0.525** | 0.444 | Branched from round 0; **+22 % over seed** |
| Final | **Best round: 3, score: 0.525** | | |

### A/B impact of message-history fix
- Round-0 train: **0.175 → 0.431** (2.5×) vs pre-fix run 138354
- LLM calls per case: ~57 → ~17
- Median latency: 15.7 s → 6.8 s

### Observations
- First time the optimizer LLM read per-case roll-ups
  (`top_failed_checks`: Business Hours, Cost Calculation Accuracy)
  and proposed targeted edits.
- Round 1's edit broke `run_task` signature → surfaced framework gap:
  no fast-fail when an edit fails validation, no `edit_errors` in next
  round's prompt. Both fixed in subsequent patch (`LoadTestValidator`
  + `edit_errors` rendering).

---

## 2026-05-06 — 20-round full-split run (138386, cancelled at round 5)

Job: 138386    Wall: 5h 24m (cancelled)
Run dir: `runs/20260506_220659_travel_default/`
Config: `loop.max_rounds: 20`, full 60/60 split, reasoning=medium

### Results
| Round | Train | Eval |
|---|---|---|
| 0 | 0.484 | 0.457 |
| 1 | 0.449 | 0.410 |
| 2 | 0.517 | 0.482 |
| 3 | 0.493 | 0.504 |
| 4 | 0.473 | **0.511** (best held-out) |
| 5 | incomplete | — |

### Observations
- Modest +5pt lift on held-out over seed in 5 rounds.
- `edit_errors` / `load_test` / fast-fail-eval-skip plumbing all
  worked correctly — no infrastructure regressions.

---

## 2026-05-07 — travel optimization, 6 rounds (138556)

Job: 138556    Wall: ~8h (COMPLETED, ended early at round 5)
Run dir: `runs/20260507_033117_travel_default/`
Config: 20-round target (full split, reasoning=medium)

### Results
| Round | Train | Eval |
|---|---|---|
| 0 | 0.490 | 0.466 |
| 1 | 0.405 | 0.452 |
| 2 | 0.492 | 0.447 |
| 3 | 0.406 | 0.444 |
| 4 | **0.173** | **0.178** ← collapse |
| 5 | 0.197 | 0.227 |

### Observations / diagnosis
- Round 4's mutation broke something material — score fell to ~0.18
  on both halves. With `branch_policy: best`, round 5 should have
  forked from round 0 or 2 but stayed low.
- Worth investigating: did the manager re-pick the collapsed agent
  as parent? `round_004/strategy.json` + `round_005/feedback.json`.

---

## 2026-05-07 — scorer-change A/B (138903 vs 138914)

Standalone full-120 seed evals around the
`projects/travel/benchmark/scorer.py` edit at 17:10:32 UTC. **Owner
did not document the change in DEV_LOG — flagged as open.**

| Eval | Job | Wall | Score |
|---|---|---|---|
| Pre-change | 138903 | 22 m | ~0.46 |
| Post-change | 138914 | 16 m | ~0.59 |
| Δ | | | **+13pp** |

### Observations
- Seed code unchanged between the two runs → the +13pp is purely a
  scorer recalibration, not an agent improvement.
- All subsequent travel numbers must be tagged "scorer v2" if the
  comparison crosses this line.

---

## 2026-05-07 — verification-pass pattern works (138929)

Job: 138929 · gpu-aic-mv-01-st-p5-node-1
Run dir: `runs/20260507_181012_travel_default/`
Config: 3 rounds × full 60/60 split, reasoning=high everywhere,
`score_target: 0.95`, post-scorer-v2

### Results
- Round 0: train=0.602 eval=0.586 (post-scorer-v2 seed)
- Round 1: train=**0.697** (held-out partial when last checked) —
  **+9.5pt train lift**

### Round 1 strategy
> Add a post-processing verification/rewrite step after the main
> tool loop produces a draft answer. If the draft contains a
> `<plan>...</plan>` block, make one additional `call_llm` pass with
> a strict reviewer prompt that: (1) recomputes any totals from
> itemized prices, (2) checks every transfer for feasible time
> buffers, (3) deletes unsupported claims, (4) returns only a
> corrected `<plan>` block.

First time the "verification pass" pattern paid off measurably on
travel.

---

## 2026-05-12 — bake-off round 1 (OpenAI vs Qwen 122B)

Authoritative summary: `/groups/AIC-MV/n.tzou/evaluations/SUMMARY_20260512.md`

### Scope changes vs plan
- 397B dropped pre-flight (~800 GB fresh download too expensive).
- gpt-oss-120b dropped mid-flight (pre-release wheel torch nightly
  unobtainable).

### Results
| Eval | Job | Wall | r0 train/eval | best train (rd) | best eval (rd) |
|---|---|---|---|---|---|
| OpenAI gpt-5.4-mini medium | 140839 | 1h 28m | 0.581 / 0.669 | **0.594 (2)** | **0.769 (3)** |
| Qwen3.5-122B-A10B (local vLLM) | 140883 | 5m 39s | 0.000 / 0.000 | 0.000 (0) | 0.000 (0) |

Run dirs:
- `/groups/AIC-MV/n.tzou/evaluations/20260512_202232_eval_openai_medium/`
- `/groups/AIC-MV/n.tzou/evaluations/20260512_214621_eval_local_qwen3_5_122b_a10b/`

### Qwen root cause
Qwen3.5 emits tool calls in its native `<tool_call><function=...>`
Hermes-style text format; vLLM's `/v1/responses` did not parse
these into `function_call` items (`num_tool_calls: 0` everywhere).
The wrapper's `_extract_output` saw no tool calls, so task_agent
got nothing dispatched and produced no plan.

Fix scheduled for next round: add `--tool-call-parser qwen3_xml`
(after one wrong-guess at `hermes`).

---

## 2026-05-13 — bake-off round 2 (infrastructure fixed, plan-emit gap)

Authoritative summary: `/groups/AIC-MV/n.tzou/evaluations/SUMMARY_20260513.md`

Curl probes confirmed both servers now serve structured tool_calls:

```
Qwen3.5-122B-A10B (--tool-call-parser qwen3_xml):    tool_calls: [...]  ✓
openai/gpt-oss-120b (mainline vLLM, TP=2, --tool-call-parser openai):  tool_calls: [...]  ✓
```

### Live eval result
Both local-model evals still scored 0/10. `details.error = "plan
conversion failed: agent produced no plan"`.

### Diagnosis (per trace inspection)
- Tools dispatched correctly (~30 successful calls, 0 errors).
- Models exit the agent loop after 2-3 turns with brief reasoning
  fragments ("Now East Lake.", "Good, now I have the coordinates...").
- Never write the formal `<plan>` block expected by the scorer.
- **Direct single-shot curl probes against the same servers DO
  produce proper `<plan>` blocks** — the early exit only happens
  inside the agent loop after N tool turns.

User direction: stop iterating on infrastructure; remaining work is
prompt/workflow level. Two follow-ups landed:
- Force-final-`<plan>` retry when loop exits without one (commit
  `4a35655`/`b18b4ec`).
- Reasoning-item strip env-conditional via
  `META_AGENT_STRIP_REASONING=1` (commit `9196ec5`).

---

## 2026-05-13 — bake-off round 3 (all three runs complete end-to-end)

Authoritative summary: `/groups/AIC-MV/n.tzou/evaluations/SUMMARY_20260513_v2.md`

### Results
| Eval | r0 train/eval | best train (rd) | best eval (rd) |
|---|---|---|---|
| OpenAI gpt-5-mini medium | 0.569 / 0.794 | **0.794 (r3)** | 0.750 (r3) |
| Qwen3.5-122B-A10B | 0.312 / 0.306 | **0.550 (r1)** | **0.656 (r1)** |
| openai/gpt-oss-120b | 0.125 / 0.006 | **0.169 (r1)** | **0.125 (r1)** |

Qwen seed → best: **+76 % relative on train, +114 % on held-out** —
first successful end-to-end optimization run on a locally-hosted
open-weights model.

### Root cause of round-2 plan-emit gap (fix in round 3)
The seed's `messages.extend(raw_output)` echoed `reasoning` items
back as input for the next call.
- **Helps OpenAI**: gpt-5-mini's reasoning chain encodes encrypted
  state; stripping → SDK 400s.
- **Hurts local models**: echoed intermediate fragments read as
  "conversation almost done" → premature loop exit, no `<plan>`
  written.

Split via env var (`META_AGENT_STRIP_REASONING=1` in
`eval_local_*.yaml`'s `env:` block); OpenAI configs leave it unset.

Lenient `_extract_plan` (returns ≥200-char prose when no `<plan>`
tags match) salvages plan-shaped output from local models that
ignore the tag instruction even after the force-plan retry.

### OpenAI baseline regression check
Round-0 train 0.569 vs prior 0.581 (Δ = −0.012, within ±5pp).
Held-out at round 0 was 0.794, **better** than the prior 0.669.

---

# Shopping project

## 2026-05-13 — shopping seed full-120 baseline

Job: 140996 · 16-CPU node    Wall: 11m 59s
Run dir: `runs/eval_20260513_042151_seed/`
Config: `configs/shopping.yaml`, gpt-5.4-mini medium, max_cases unset (all 120)

### Results
- Overall composite_score: **0.7737** (target was ~0.65)
- 0 crashes, 120/120 non-zero, 46 strict-pass (case_score == 1.0)

Per level:
| Level | n | Mean | Perfect |
|---|---|---|---|
| L1 (no budget) | 50 | 0.815 | 21 |
| L2 (budget-constrained) | 50 | 0.764 | 19 |
| L3 (coupon optimization) | 20 | 0.694 | 6 |

### Observations
- Target was ~0.65; actual baseline is +12pp above that.
- L3 is the weakest level — coupon stacking is hard.

---

## 2026-05-13 — shopping optimization, 5 rounds 60/60 split (141028)

Job: 141028    Wall: 1h 34m 54s
Run dir: `runs/20260513_050054_shopping_default/`
Config: `configs/shopping.yaml` @ commit `20691cf` (max_rounds 3 → 5),
        split seed=42, train_size=60 → 60 train + 60 held-out per round

### Per-round results
| Round | Train | Eval (held-out) |
|---|---|---|
| 0 | 0.784 | 0.781 |
| 1 | 0.751 | 0.764 |
| **2** | **0.798** ← best | 0.765 |
| 3 | 0.783 | 0.771 |
| 4 | 0.772 | 0.760 |
| 5 | 0.772 | 0.737 |

Best by train: round 2 (+1.4pp over seed). Held-out at round 2:
−1.6pp vs seed (mild overfitting visible during the run).

---

## 2026-05-13 — shopping round-2 full-120 verification (141091)

Job: 141091    Wall: 13m 34s
Run dir: `runs/eval_20260513_063623_task_agent/`
Agent: `runs/20260513_050054_shopping_default/round_002/task_agent`

### Headline
- Seed full-120 (baseline 140996): **0.7737**
- Round 2 full-120 (141091): **0.7664**
- **Δ overall: −0.0073 (−0.7pp)** — round 2 underperforms the seed
  on the full benchmark.

### Per-level (seed → round 2)
| Level | n | Seed | Round 2 | Δ |
|---|---|---|---|---|
| L1 (no budget) | 50 | 0.815 | 0.782 | **−3.3pp** |
| L2 (budget) | 50 | 0.764 | 0.770 | +0.6pp |
| L3 (coupon opt) | 20 | 0.694 | 0.718 | +2.4pp |

### Diagnosis (textbook train/eval overfit)
Round 2's edit appended a four-bullet "coupon safety" rule set to
every level's system prompt — see
`runs/20260513_050054_shopping_default/round_002/strategy.json`
("add_coupon_to_cart error rate was 0.31; reduce hallucinated
coupon calls").

Effects break down cleanly by level:
- L1 has **no coupons** → extra paragraph is wasted context and
  distracts the model on product selection (−3.3pp, largest single
  regression).
- L2 has minor coupon use → small win (+0.6pp).
- L3 is exactly the bucket the rule targeted → meaningful win
  (+2.4pp).

Train/held-out story explained:
- Train half (60 cases by `split.seed=42`) had a slightly
  coupon-heavier mix → appendix helped on train (+1.4pp).
- Manager picks best round by **train score**, so round 2 was named
  "best" — but held-out already showed −1.6pp.
- Full 120 collapses to −0.7pp because L1's regression outweighs
  L2/L3's gains by absolute case count.

### Open follow-ups
1. **Weight held-out into best-round selection.**
   `HillClimbingManager._best_round` picks max(train); a variant
   could pick max(0.5·train + 0.5·held_out) so edits that overfit
   get penalized.
2. **Per-level scoring signal to the strategy proposer.** The
   gatherer already collects per-level numbers via
   `ShoppingScorer.aggregate`; the strategy prompt could render
   those breakdowns so the proposer sees L1 regressing and avoids
   global edits that hurt one bucket.
3. **Longer optimization runs.** 5 rounds reached a local max at
   round 2; rest drifted down. Worth trying `max_rounds: 10-15`
   with `branch_policy: "best"` so the optimizer continues to
   branch off round 2 instead of the latest.

---
