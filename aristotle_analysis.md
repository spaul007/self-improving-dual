# HGM limitation: node-selection reliability degrades with small train sets

## The finding

HGM's final node selection (`hgm.py::_finalize` → `hgm_tree.py::lcb_select`) picks
the best node via a Bayesian lower-confidence-bound over each node's own
`Beta(1 + n_success, 1 + n_failure)` posterior, accumulated purely from
**train-split** evaluations (`epsilon` quantile, 0.25 in all our configs — see
`lcb_select`'s docstring). It is explicitly designed to discount thin evidence:
a node with a higher raw mean but few evals loses to a node with a slightly
lower mean but many evals, at any reasonable epsilon (< ~0.5).

That mechanism is statistically sound *given enough data* — but it cannot
manufacture certainty the data doesn't contain. Across the three projects run
this session, reliability visibly degrades as the train split shrinks:

| Project | Train size | Outcome |
|---|---|---|
| math_mas | 100 | Selection held up: best node (12) genuinely improved on the full 500-case benchmark (0.780 → 0.894 accuracy), no contradiction found when independently verified. |
| wikihop_mas | 50 | Selection **failed**: best node (18) scored *worse* than the untouched seed on the full 200-case benchmark (0.631 → 0.627). A statistically-tied sibling (node 17, exact same train mean to floating-point precision) was the real winner (0.680 full-benchmark), invisible to the selection mechanism. |
| db_mas | 20-27 | Selection dominated by noise: per-node timeout rates ranged 4-19% (vs. an intended <5%), and excluding timeout-affected cases from each node's mean closed roughly 40-50% of the apparent gap between children and the seed — meaning a meaningful fraction of the "children are worse" conclusion was host-contention roulette, not real edit quality. |

## The wikihop_mas smoking gun

Nodes 17, 18, and 19 (three genuinely different code edits, different
lineages: 9→17, 9→14→18, 17→19) produced **exactly identical** train-split
scores: `n_success=35.1169, n_failure=14.8831` for all three, matching to
floating-point precision, over the same 50-case train pool. Their Beta
posteriors are therefore mathematically identical (`variance=0.004003` for
all three) — `lcb_select`'s choice among them came down to Monte Carlo
sampling-order noise (a single `np.random.default_rng(0)` generator advancing
per-node, not reseeded), not a real signal. Confirmed by reproducing the exact
selection computation offline.

## Does using the eval set instead of train fix it?

Tested directly: recomputed what selection would have picked if `lcb_select`
had run on eval-set (50 held-out cases) performance instead of train, for the
three tied candidates:

| Node | Train mean | Eval-set score (n=50) | Full 200-case score |
|---|---|---|---|
| 17 | 0.702 (tied) | 0.660 | 0.680 |
| 18 (actual pick) | 0.702 (tied) | 0.609 | 0.627 |
| 19 | 0.702 (tied) | **0.694** (would win) | 0.653 |

Eval-based selection would have picked node 19 instead of node 18 — a real
improvement (0.653 vs. 0.627 full-benchmark, and both beat the untouched
seed's 0.631... only node 19 does, 18 does not). But it still would not have
found the actual best node (17, 0.680 full-benchmark), because the eval set
is the same size and same noisiness as train — the problem doesn't go away,
it just relocates to a different 50-case sample. Rank order flips between
eval (19 > 17) and the full 200 (17 > 19), the same statistical-power
problem in a different place.

## Takeaway

The eval set is only 50 cases here (and db_mas's is smaller still) — the
same order of magnitude as train, so switching *which* split drives
selection doesn't add information, it just moves where the noise lands. The
fix that would actually help is a larger held-out set, not changing which
existing split is used. **Decision (explicit, this session): keep the
current train-based `lcb_select` logic as-is** — not switching to
eval-based selection, given it only partially helps and the codebase/config
surface would grow for a fix that's known to be incomplete.

## Quantifying the sampling variance directly (not just qualitative)

Rather than reason qualitatively about "small samples are noisy," measured it
directly. For wikihop_mas's original 200-case dataset, reused the real
per-case scores already computed for the full-benchmark baseline eval
(zero new LLM calls -- pure resampling) to draw 1000 independent random
subsets at several sizes and compute the resulting sampling std of the mean
score:

| n | empirical std (1000 draws) | theoretical SE (finite-pop-correction) | realistic range |
|---|---|---|---|
| 20 | 0.094 | 0.096 | 0.32 – 0.90 |
| 50 (wikihop's actual train size) | 0.055 | 0.055 | 0.46 – 0.81 |
| 100 (math_mas's actual train size) | 0.033 | 0.032 | 0.52 – 0.74 |
| 150 | 0.019 | 0.018 | 0.59 – 0.69 |

Empirical Monte Carlo and the textbook finite-population-correction formula
(`SE = (pop_std/sqrt(n)) * sqrt((N-n)/(N-1))`) agree almost exactly -- good
validation that this is a real, well-behaved statistical effect, not an
artifact.

**Comparing signal to noise directly**: wikihop_mas's real best-node edge
was +0.049 (0.680 vs. baseline 0.631) -- *smaller than one full standard
deviation of sampling noise at n=50* (0.055). math_mas's real edge was
+0.114 against an n=100 noise std of only 0.033 -- a ~3.5x signal-to-noise
ratio, comfortably detectable. This is the precise, quantified reason
math_mas's selection held up and wikihop_mas's didn't: not an algorithm
difference, a signal-to-noise difference.

Ran the identical analysis for db_mas (population from the full 100-case
latency probe, timeouts excluded: N=97, mean=0.477, std=0.288): at
`train_size=27` (the small_latency run's actual size), sampling std=0.047 --
larger than the 0.042 gap observed between the best child (node 3,
timeout-adjusted) and the seed. The "no edit beat the seed" conclusion from
that run sits entirely inside the noise floor; the data cannot actually
distinguish that edit from neutral-or-better.

**Why is wikihop_mas's variance so high in the first place?** Checked
directly rather than assumed: two compounding causes. (1) The score
distribution is strongly bimodal -- across all 200 cases, 30% score exactly
0.0 and 55.5% score exactly 1.0, only 14.5% land in between (typical of
exact-entity-answer QA scored via F1: right entity -> near-perfect token
overlap, wrong entity -> zero overlap, partial credit is the exception). A
near-Bernoulli variable at ~60% "pass rate" sits close to its
theoretical-maximum variance. (2) Real difficulty heterogeneity by question
type: "comparison" questions score 0.958 mean (nearly solved) vs.
"compositional" questions (the largest category, 45% of the dataset) at
0.457 mean -- a random sample's score swings with how many of each type it
happens to draw.

## Fix: a 2000-case dataset, verified to actually reduce the noise

Checked whether our 200-case wikihop_mas benchmark was even representative
of the full source data. It is: verified byte-for-byte against the
official 2WikiMultihopQA release (Dropbox-hosted, same 2020-10-29 release
as the paper) -- all 200 cases match by `_id`/question/answer exactly. But
we were only using the first 200 rows (in file order, not randomized) of a
12,576-row dev split; train (167,454 rows) and test (12,576 rows) were
never touched at all.

Built `projects/wikihop_mas_2k/` -- same vendored MAS code and adapter glue
reused via symlink (`db-mas`-style pattern), only `benchmark/cases.jsonl`
differs: a genuinely random sample (seeded shuffle, not "first N") of 2000
rows from the same 12,576-row dev split. Ran the pristine seed on all 2000
cases for real (score=0.5949, zero crashes, wall_time_s=2858.8 at
parallelism=16), then repeated the same resampling methodology:

| n | empirical std (1000 draws) | realistic range |
|---|---|---|
| 50 | 0.062 | 0.38 – 0.79 |
| 200 | 0.032 | 0.49 – 0.73 |
| 500 | 0.018 | 0.53 – 0.65 |
| **1000** | **0.011** | **0.56 – 0.63** |

Requested check -- 10 independent random 1000-case draws, real numbers, not
simulated: means of 0.593, 0.581, 0.596, 0.594, 0.605, 0.604, 0.587, 0.602,
0.601, 0.590 -- **std=0.0074, range=0.0238**. Scaling total dataset size
200 -> 2000 (and per-sample size 50 -> 1000) cut sampling variance by
roughly **7-8x**, landing well below every real edit-quality signal found
this session (smallest was ~0.05). This is the regime where a genuine
improvement is actually distinguishable from sampling luck. Config:
`configs/hgm_wikihop_mas_2k.yaml` (manager/eval_budget not yet re-tuned for
a real HGM search at this scale -- placeholder copied from the x20/y50
config, sized only for the variance-check split of 1000/1000).

## Train/eval split mechanism (confirmed, not assumed)

`meta_agent/config.py::compute_split` uses `random.Random(seed).shuffle(...)`
— a genuine Fisher-Yates random permutation of every case id in the
benchmark (seeded only for reproducibility, not for bias), then simple
slicing into `train`/`eval`. None of db_mas/math_mas/wikihop_mas's configs
set `stratify_by`, so all three use this plain random-shuffle path (the
`stratify_by` branch, when set, does a *stratified* proportional random
split by a chosen field — still random within each stratum, just balanced
across it; not used here).
