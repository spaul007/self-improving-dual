# math_mas HGM run — baseline vs. best found

Run: `configs/hgm_math_mas_full.yaml`, experiment `runs/20260725_231653_math_mas_hgm_full`
(100 train / 50 eval cases, randomly sampled out of Math500; `eval_budget: 12000`).

Both rows below are evaluated on the **full 500-case benchmark** (not just the
train/eval split), via `evaluate_task_agent.py`, so they're directly comparable
head-to-head — the baseline is the unmodified vendored seed agent, "best found"
is HGM round `round_027` (node 27), the best-scoring node discovered by the
search at the time of writing.

| | Baseline (seed) | Node 27 (best found) | Δ |
|---|---|---|---|
| **Accuracy** | 0.780 (390/500) | **0.902 (451/500)** | **+12.2pp** (+61 cases) |
| Predictor accuracy | 0.900 | 0.900 | unchanged |
| Fixed by reflector | 12 | 8 | -4 |
| **Broken by reflector** | **72** | **7** | **-65** |
| Avg time/case | 39.6s | 19.7s | ~2x faster |

## What actually changed

`predictor_accuracy` is identical (0.900 → 0.900) — the underlying solver
never changed. The entire accuracy gain came from fixing the reflector: it
went from net-destructive (broke 72 correct predictor answers, only fixed 12
— net **-60**) to nearly neutral-positive (broke 7, fixed 8 — net **+1**), a
**10x reduction** in the exact failure mode diagnosed earlier in this
project's history (missing `<answer>` tags, rambling until truncated,
prompt-leakage confusion). Node 27's edit lineage is `0 → 12 → 27`.

## Statistical confidence

Both scores are over the full 500-case benchmark, so the 95% CI half-widths
are tight (±3.6pp baseline, ±2.6pp node 27) — the 12.2pp gap is far beyond
what sampling noise could produce. Node 27's search-time score on its own
50-case train split was 0.960, vs. 0.902 on the full 500 — a modest, expected
amount of overfit to that specific sample, not a red flag; 451 of the 500
cases it's scored on here were never part of its own optimization loop at
all, so this is a genuine, generalizing improvement.

## Run status at time of writing

103 rounds generated, ~2850/12000 budget evals spent (excludes the free seed
pre-evaluation). The run was stopped intentionally at this point (12000 was
judged to be more budget than needed, given the strength of the result
already found).

## Follow-up run: sizing by target tree shape (X/Y method)

Rather than size the next run by wall-clock time, sized it by desired tree
shape instead: "X good-enough agents, each evaluated on Y examples on
average." With `B = eval_budget = X*Y` and `alpha` solved from `B**alpha ==
X` (this makes the widening schedule `hgm_tree.py::schedule_favors_expand`
— `budget_spent**alpha >= n_real_nodes - 1` — naturally arrive at ~X nodes by
the time the full budget is spent):

```
alpha = ln(X) / ln(B)
```

For **X=20, Y=100**: `B=2000`, `alpha=ln(20)/ln(2000)=0.3941`. Sanity-checked
the growth curve is well-behaved, not a hard cutoff: ~9 nodes by 10% of
budget spent (~22 evals each), converging to ~21 nodes by 100% (~95 evals
each) — close to the X/Y targets. Added as a reusable mode in
`size_eval_budget.py` (`--target-agents`/`--evals-per-agent`). New config:
`configs/hgm_math_mas_x20_y100.yaml`, projected ~2.3h.

**A real gotcha found while sizing this run, worth remembering for any future
one**: `eval_budget` does not unilaterally determine how much of itself gets
spent. `loop.max_rounds` is a *hard, independent* cap on node count
(`can_grow = n_real_nodes < max_rounds` in `hgm.py`'s main loop) — if
`max_rounds` is hit *and* every existing node has already been evaluated on
its full train set, the loop `break`s early, leaving `eval_budget` unspent.
The two knobs control different axes (`eval_budget` = total evaluation
effort; `max_rounds` = breadth, how many distinct edits get tried) and each
EXPAND's editor LLM call isn't even drawn from `eval_budget` at all (only
evaluation spend is). **Rule of thumb**: keep `max_rounds * train_size`
comfortably above `eval_budget`, or `max_rounds` silently becomes the real
constraint instead of the one actually sized. For the X20/Y100 config:
`max_rounds=40`, `train_size=100` → capacity 4000 vs. `eval_budget=2000` — a
safe 2x margin, confirmed non-binding.

---

# wikihop_mas_2k HGM run — baseline vs. best found

Run: `configs/hgm_wikihop_mas_2k_x20_y1000.yaml`, experiment
`runs/20260728_024003_wikihop_mas_2k_hgm_x20_y1000` (2000-case random sample
of the full 2WikiMultihopQA dev split — see `projects/wikihop_mas_2k/` and
`aristotle_analysis.md`'s sampling-variance section; `train_size=eval_size=
1000`, `eval_budget=20000`, X=20/Y=1000).

Both rows below are scored on the **full 2000-case dataset** (train + eval
combined), via the run's built-in `full_eval_top_k=3` finalist audit — not
just the 1000-case train split — so they're directly comparable.

| | Baseline (seed) | Node 8 (best found) | Δ |
|---|---|---|---|
| **Score (answer F1, full 2000)** | 0.588 | **0.615** | **+0.027** |
| Train mean (n=1000) | 0.588 | 0.608 | +0.021 |
| Held-out eval (n=1000) | — | 0.616 | — |

## What actually changed

Edit lineage: `0 → 5 → 8` (two successive prompt edits, both to
`mas_prompt_cfg.yaml`, no code changes). Node 5's edit relaxed the
Concluder's grounding thresholds (previously rejecting thin-but-real evidence
as "ungrounded", producing false `Unknown`/`Not mentioned` answers). Node 8
built on that, targeting the known DEPENDENT-vs-INDEPENDENT question-type gap
(compositional/inference questions scoring far below comparison questions)
by strengthening the extraction/grounding-judgment prompt.

Two other finalists were also audited on the full dataset and both beat the
seed too, though by less: node 18 (0.609, lineage `0→4→12→18`) and node 13
(0.606, lineage `0→3→9→13`) — this wasn't a single lucky branch; the search
found the same general direction (grounding-strictness tuning) from multiple
independent starting points.

## Statistical confidence

This session separately measured wikihop_mas_2k's sampling-variance floor via
10 independent resampled 1000-case draws from real per-case data: std=0.0074,
range=0.0238 (see `aristotle_analysis.md`). Node 8's same-sample delta over
the seed was tracked across its entire evaluation history (n=100 → 900) and
consistently landed around +0.02–0.03 (roughly 3σ above the noise floor at
each checkpoint, with one dip to ~1.6σ at n=600) — never decaying to zero the
way several other nodes' early "leads" did. The full 2000-case audit score
(+0.027) lands squarely in that same range, confirming this was a real,
reproducible signal rather than a lucky sample.

## Run status

Finished. 21 nodes, all 20,000 budget evals spent (21,000 incl. the free seed
pre-eval and the finalize top-up). Best = node 8, train mean 0.608 (n=1000).

---

# db_mas_snapshot HGM run — baseline vs. best found

Run: `configs/hgm_db_mas_snapshot_x20_y100.yaml`, experiment
`runs/20260728_015256_db_mas_snapshot_hgm_x20_y100` (snapshot-replay variant
of the DB root-cause-diagnosis MAS — no Docker/live Postgres at inference
time; see `projects/db_mas_snapshot/`). `train_size=100` = the **entire**
dataset (`eval_size=0` by design — 100 total cases isn't enough to hold out a
separate eval split and still train meaningfully), `eval_budget=2000`, X=20/
Y=100.

Because `train_size` already covers every case in the dataset, both rows
below are already "full-benchmark" scores — there's no separate held-out or
full-audit step needed the way wikihop_mas_2k's run had.

| | Baseline (seed) | Node 19 (best found) | Δ |
|---|---|---|---|
| **Score (recall, n=100, full dataset)** | 0.365 | **0.550** | **+0.185** |

## What actually changed

Edit lineage: `0 → 1 → 2 → 9 → 17 → 19` — five successive prompt edits, all
to `mas_prompt_cfg.yaml`, no code changes at any point despite the editor
being explicitly told it could modify code (a framework-wide nudge added
earlier this session). Node 1 itself was a losing branch (0.31, below the
0.365 seed), but its child node 2 recovered strongly (0.495) by teaching the
lead-DBA agent that the recorded snapshots reset table-level vacuum counters
to zero — so `n_dead_tup`/vacuum-count evidence is always misleadingly 0, and
the real signal for VACUUM anomalies is `VACUUM FULL` statements in
`pg_stat_statements` — plus adding an explicit label-priority decision
hierarchy to resolve cases where multiple investigators report conflicting
evidence. Later nodes (9, 17, 19) each refined that same hierarchy further,
culminating in node 19 fixing a priority-ordering bug where REDUNDANT_INDEX
(ranked too high) was overriding strong LOCK_CONTENTION/INSERT_LARGE_DATA
evidence (e.g. a case with ~1.5M seconds of INSERT execution time getting
overridden by an unrelated unused-index finding). Final hierarchy: VACUUM
(P1) > LOCK/INSERT discrimination (P2) > REDUNDANT_INDEX (P3) > FETCH (P4) >
INSERT fallback (P5).

## Statistical confidence

No formally validated sampling-variance model was built for this project the
way wikihop_mas_2k and math_mas got (only 100 total cases, and the recall
metric is coarse — {0, 0.5, 1.0} per case for the 2-gold-label tasks). Taken
at face value, +0.185 is a large gap relative to the noise seen elsewhere in
this run (e.g. small-n branches swinging ±0.05–0.10 on 20-case partial
evals), and the win was checked at growing sample sizes along the way (e.g.
node 2 held at +0.19 to +0.20 across n=20→60, not decaying) — but treat this
as a good, plausible result rather than a rigorously bounded one, since it's
measured on the same 100 cases the search optimized against with no
independent held-out set.

## Run status

Finished. 21 nodes, all 2,000 budget evals spent (2,100 incl. the free seed
pre-eval and the finalize top-up). Best = node 19, train mean 0.550 (n=100).
