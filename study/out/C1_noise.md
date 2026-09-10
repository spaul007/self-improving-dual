# C Tier 1 — what the current measurement can detect

## 1. Noise floor

5 repeat evaluations of the **same agent code** on the **same 120 cases**.

| repeat | eval_20260828_150213_seed | eval_20260828_150218_seed | eval_20260828_150223_seed | eval_20260828_150229_seed | eval_20260828_150235_seed |
|---|---|---|---|---|---|
| score | 0.6089 | 0.4896 | 0.5870 | 0.5724 | 0.5401 |

- run-to-run sd at n=120: **0.0464**
- between-case variance (true difficulty): 0.0393 (sd 0.1982)
- within-case variance (same code, same case, different run): 0.0505 (sd 0.2248)
- **ICC = 0.437** — only 44% of per-observation variance is case difficulty; the rest is the agent being stochastic
- cases fully deterministic across all repeats: 0/120

### Standard error by design

| n cases | SE of one mean (fresh cases) | SE of paired Δ (same cases) | SE of unpaired Δ |
|---|---|---|---|
| 16 | ±0.0749 | ±0.0795 | ±0.1060 |
| 32 | ±0.0530 | ±0.0562 | ±0.0749 |
| 60 | ±0.0387 | ±0.0410 | ±0.0547 |
| 120 | ±0.0274 | ±0.0290 | ±0.0387 |

## 2. Power — cases needed to resolve an effect (80% power, α=0.05)

| effect Δ | paired cases | unpaired cases |
|---|---|---|
| 0.02 | 1,984 | 3,526 |
| 0.03 | 882 | 1,567 |
| 0.05 | 318 | 565 |
| 0.10 | 80 | 142 |

The verdict band in `edit_outcome.NEUTRAL_BAND` is ±0.02.

## 3. Discriminability of the search's own node scores

- 29 evaluated nodes
- observed spread of node means: sd **0.0794**
- spread expected from sampling error alone: sd **0.0613**
- residual real between-node signal: sd **0.0505** → **40%** of the observed spread
- seed node 0: 0.5979 over 60 evals
- best node 27: 0.7109 over 16 evals
- **best − seed = +0.1130**

## 4. Winner's curse — the null model

Every node given **identical true quality**, drawn with its own real `n_evals` and the measured noise; take the max over the tree.

- observed best − seed: **+0.1130**
- null distribution of that same statistic: median +0.1286, p90 +0.2000, p95 +0.2223, p99 +0.2659
- **P(null gap ≥ observed) = 0.616** (20,000 trials)

> The headline improvement is **not distinguishable** from what the selection rule manufactures out of noise alone.

## 5. Published arm comparison, with the noise attached

| arm | run | nodes | best node | best mean | n evals | SE |
|---|---|---|---|---|---|---|
| control-no-editmem | `20260818_230318_travel_hgm_1000_qwen122b_node5_no_errcond_no_behsum` | 33 | 33 | 0.7823 | 60 | ±0.0387 |
| editmem | `20260821_000745_travel_hgm_1000_qwen122b_node5_editmem` | 31 | 8 | 0.7635 | 60 | ±0.0387 |
| OR-no-editmem | `20260827_003441_travel_hgm_1000_or_medium_no_editmem` | 31 | 31 | 0.7135 | 60 | ±0.0387 |
| OR-editmem | `20260827_005240_travel_hgm_1000_or_medium_editmem` | 30 | 27 | 0.6885 | 60 | ±0.0387 |
| belief-gpt54 | `20260908_225039_gpt54_beliefs2stage` | 29 | 27 | 0.7109 | 16 | ±0.0749 |

- widest gap: **control-no-editmem − OR-editmem = +0.0938** ± 0.0547 → **1.7σ** before any winner's-curse correction, and each arm's *best* node is itself a max over ~30 nodes.

> These are best-node scores, each already selected as a maximum. The winner's-curse section above applies to every one of them, so the true arm gap is smaller than the nominal one.
