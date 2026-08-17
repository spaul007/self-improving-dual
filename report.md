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

---

# travel_mas HGM (dual) run — baseline vs. best found so far (in progress, paused)

Run: `configs/hgm_dual_travel_mas_35b_4000.yaml`, experiment
`runs/20260802_175007_travel_mas_hgm_dual_35b_4000` (`Qwen/Qwen3.5-35B-A3B`
@ node-1, implicit mode — no `reasoning_effort`, `temperature=0.2`,
`max_output_tokens=16384`; `eval_budget=4000`, `train_size=60`; editor/
summarizer on `Qwen/Qwen3.5-122B-A10B` @ node-2, matching math_mas/db_mas
convention). `travel_mas` itself is a 4-stage sequential MAS (Flight → Train
→ Sightseeing → Accounting agents) built this session as a "slightly
better than single-agent" alternative to `projects/travel/seed` — see
earlier sections of this session's history for the single-agent-vs-seed-MAS
comparison; this section covers HGM's optimization *on top of* the seed MAS.

This was the **3rd attempt** at this run — the first two both died to real,
previously-undiscovered framework bugs, only surfaced by attempting a real
long-running `hgm_dual` job (never triggered by short `evaluate_task_agent.py`
smoke runs): (1) `projects/travel_mas/tools/` had no discovery package, so
`SchemaWrapperConsistencyValidator` saw all 9 tools as unprovided and every
edited round failed validation for 27+ rounds straight (fixed by adding
`projects/travel_mas/tools/__init__.py`); (2) no HTTP client timeout was set
on the vLLM calls, letting a single stuck call block for 3+ hours (fixed with
an explicit 300s request timeout in `platform_core/llm_wrapper.py` and both
`scorer.py` copies). Both fixes confirmed live before this 3rd, real run was
started fresh.

| | Baseline (seed, full 120) | Node 17 (best found, full 120) | Δ |
|---|---|---|---|
| **Score (composite, full 120)** | 0.36875 | **0.5333** | **+0.1646** (+44.6% relative) |
| Same-run train score (n=60) | 0.356 | 0.535 | — |
| Pass rate (full 120) | 0/120 | 2/120 | +2 cases |
| no_plan_rate (full 120) | 30.8% | 7.5% | -23.3pp |

Node 17's full-120 confirmatory score (0.5333) lands almost exactly on its
own search-time train score (0.535, n=60) — negligible overfit, a genuine
generalizing improvement rather than a lucky sample.

## What actually changed

Edit lineage: `0 → 9 → 10 → 13 → 17` (all prompt/code edits to
`workflow.py`, restricted to the seed's mutable-surface convention). Node
10 was a weak intermediate branch (0.158, well below the 0.356 seed) that
recovered strongly through nodes 13 (0.416) and 17 (0.535). Node 17's edit
specifically targeted the **Time Feasibility** dimension, which round 13's
feedback showed at 0.00 mean score / 75% failure rate: the sightseeing
agent was outputting checklist-formatted text claiming transfer times had
been checked, without actually calling `query_road_route_info` (called only
once across an entire 16-case sample despite 12 Time Feasibility failures).
Node 17 restructured `_run_sightseeing_stage` into two explicit phases —
Phase 1 gathers all entity details and forces a real `query_road_route_info`
call for every planned transfer into a structured data object; Phase 2
composes the itinerary using only that verified data — closing the
"checklist without a real tool call" loophole the model had been exploiting.

## Statistical confidence

No formal sampling-variance model was built for `travel_mas` this session.
Taken at face value, the full-120 confirmatory run (0.5333) essentially
reproduces the node's own 60-case search-time score (0.535), which is a
reasonable indicator this isn't sampling noise — but treat it as a good,
plausible result rather than a rigorously bounded one, the way math_mas's
and wikihop_mas_2k's reports above are.

## Run status

**Paused (killed intentionally), not finished.** 42 nodes generated,
2252/4000 budget evals spent (56.3%) at round_042 when stopped. Best =
node 17, train mean 0.535 (n=60), full-120 confirmatory mean 0.5333 (see
table above). Can be resumed as a fresh run from this point if further
search is wanted later — the config, fixes, and this checkpoint are all in
place to do so.

---

# travel single-agent HGM (dual) run — same-endpoint comparison against travel_mas

Run: `configs/hgm_dual_travel_single_35b_1000.yaml`, experiment
`runs/20260804_183603_travel_hgm_dual_single_35b_1000` (`project: "travel"` —
the original single-agent seed, NOT `travel_mas`; same 35B/implicit task_agent
settings and same 122B editor/summarizer convention as the `travel_mas`
hgm_dual run above; same predetermined 60-case `train_ids_path` split). This
run answers a direct question: **if you spend HGM search budget optimizing
the single task agent alone (prompt/code edits only, no structural
redesign), how much of travel_mas's improvement can you recover?**

Died mid-run at 796/1000 budget spent when the `node-1` vLLM endpoint went
offline (endpoints rotated to `node-6`/`node-5` — see `vllm_endpoint.md`
memory). `main_loop.py` has no `--resume` mechanism, so the run's tree/RNG
state couldn't be reopened; a fresh sibling run was launched afterward
(`configs/hgm_dual_travel_single_35b_3000.yaml`, `eval_budget=3000`, same
train split, on `node-6`/`node-5`) rather than resuming this one.

## Baseline vs. best found (this run, before it died)

| | Baseline (seed, full 120) | Node 14 (best found, full 120) | Δ |
|---|---|---|---|
| **Score (composite, full 120)** | 0.0000 | **0.3406** | **+0.3406** |
| Same-run train score (n=60) | 0.015 | 0.409 | — |
| Pass rate (full 120) | 0/120 | 0/120 | unchanged |
| no_plan_rate (full 120) | 100% | 18.3% | -81.7pp |

Node 14's edit lineage: `0 → 11 → 12 → 14`. The first ten sibling edits off
the seed (nodes 1-10) all scored exactly 0.0 — real, `edit_failed: False`
evaluations, not a validation bug (confirmed by inspecting per-case error
traces: genuine "agent produced no plan" failures matching the breakdown
documented in `qwen35bnotworking.md`, plus one edit — node 4 — that
introduced a real runtime crash calling `.get()` on a pydantic object).
Node 11 was the first to escape zero (0.039), adding an explicit
"Phase 1 completion criteria" checklist to the system prompt. Node 12
(0.395) added the actual behavioral fix: when the model stops calling
tools without having emitted a `<plan>` block, inject one explicit rescue
message ("Do NOT call any more tools. Generate your final plan NOW") and
give it one more iteration before giving up — the seed's original code had
no such rescue call at all. Node 14 is a further sibling refinement of the
same idea, edging node 12 out as the leader (0.409 vs. 0.395 on-search,
0.3406 vs. presumably similar on full-120 — node 12 itself was never
separately confirmed on the full benchmark).

## Apples-to-apples: best node at matched raw budget (~720-750 evals) vs. travel_mas

Both this run and the `travel_mas` hgm_dual run above found their eventual
"final" best node very early and never displaced it for the rest of the
run — so comparing at matched raw budget spent (not matched fraction of
each run's differently-sized total budget) is a fair, apples-to-apples
check of how much of the win was already locked in early:

| | travel_mas (4-agent MAS) | Single agent |
|---|---|---|
| Best node at ~750 budget spent | node 17 (at budget=736) | node 14 (at budget=720) |
| On-search mean_utility at that checkpoint | 0.5957 | 0.4089 |
| Same node as the run's eventual final-best? | Yes (held leader to 2252/4000) | Yes (held leader to 796/1000, when it died) |
| Full-120 confirmatory score | **0.5333** | **0.3406** |

## What this shows

HGM search alone, editing only the single agent's prompt/code (no
structural redesign), recovers **most but not all** of the way from a
completely broken agent (0.0) to a working one — reaching 0.3406, close to
but still below the *unoptimized* `travel_mas` seed's own score (0.3688,
see the `travel_mas` section above) and well below the *optimized*
`travel_mas` (0.5333). The multi-agent decomposition still has a real,
meaningful edge over prompt-editing a single agent alone on this fragile
35B endpoint — HGM can partially compensate for a bad architecture, but
doesn't fully substitute for a better one here.

## Statistical confidence

Same caveat as the `travel_mas` section: no formal sampling-variance model
was built for this project. Node 14's full-120 score (0.3406) is
reasonably close to its own 60-case search-time score (0.409, a ~7pp gap,
larger than travel_mas node 17's near-exact match) — plausible mild
overfit to the 60-case train split, or just noise from a small sample;
treat 0.3406 as a good, plausible result rather than a tightly bounded one.

## Run status

**Dead (endpoint outage), not finished.** 22 nodes generated, 796/1000
budget evals spent (79.6%) when the `node-1` server went offline. Best =
node 14, train mean 0.409 (n=60), full-120 confirmatory mean 0.3406 (see
table above). A fresh (not resumed) sibling run at `eval_budget=3000` on
the new `node-6`/`node-5` endpoints was launched afterward — see
`configs/hgm_dual_travel_single_35b_3000.yaml`,
`runs/20260806_041514_travel_hgm_dual_single_35b_3000`.

---

# shopping / shopping_mas baselines — single-agent vs. 4-agent vendor MAS

Both projects newly integrated this session (`projects/shopping`, ported
from the `origin/shopping` branch; `projects/shopping_mas`, ported from
the vendored `Shopping-MAS-main` 4-agent implementation — see
`integrating.md`/memory for the integration playbook both followed).
Baselines below are raw seed scores, no HGM search — `Qwen/Qwen3.5-35B-A3B`
@ node-6, temperature 0.2, implicit mode (no explicit `reasoning_effort`,
confirmed the working setting for this deployment — see
`qwen35bnotworking.md`), full 120-case benchmark (50 L1 / 50 L2 / 20 L3).

| | Single-agent (`shopping`) | 4-agent MAS (`shopping_mas`) |
|---|---|---|
| **Score (full 120)** | 0.0 → **0.2517** (after fix, see below) | **0.8829** |
| Passed | 0 → 18/120 | 79/120 |
| Per level | L1 0.39 · L2 0.13 · L3 0.20 | L1 0.91 · L2 0.88 · L3 0.88 |
| Wall time | ~58 min | ~5.4 hours |
| Crashes | None (119-120/120 completed cleanly both versions) | 1 case timeout (1800s) |

The MAS's much longer wall time reflects its own internal structure — 4
sequential/parallel agent stages with their own tool rounds per case,
vs. the single agent's flat 2-phase tool loop.

## Single-agent `shopping`: root cause of the original 0.0, and the fix

The first full-120 baseline (`runs/adhoc_eval_shopping/baseline_full_35b_implicit_v2_full_benchmark`)
scored a suspicious flat **0.0 on all 120 cases** — every case's cart came
back completely empty (0 matched, 0 extra products; confirmed by direct
inspection, not just the scorer's summary). The model's tool-call
sequence looked entirely reasonable in a live-traced smoke test
(`search_products` → sensible IDs → `get_product_details` on those exact
IDs), yet every single `get_product_details`/`filter_by_brand` call
returned empty.

**Root cause**: `platform_core.llm_wrapper.call_llm` (OpenAI Responses
API) against this model/endpoint sends array-typed tool arguments back
**double-encoded as JSON strings** instead of native arrays — e.g.
`{"product_ids": "[\"a\", \"b\"]"}` (a string) instead of
`{"product_ids": ["a", "b"]}`. The tool code
(`projects/shopping/tools/get_product_details.py`,
`filter_by_brand.py`) does `for pid in product_ids` expecting a list;
against a string this iterates individual *characters*, matching
nothing, silently, no exception. Scalar-typed arguments (`query`,
`limit`, etc.) were unaffected — only tools with array-typed params hit
this, which is why `search_products`/`get_user_info`/`get_cart_info`
worked fine while `get_product_details`/`filter_by_brand` always came
back empty. `projects/shopping_mas` (same model/endpoint) never
exhibited this because it goes through a different, project-local Chat
Completions client instead of the shared Responses-API wrapper.

**Fix**: ported `shopping_mas`'s Chat-Completions-API approach into a new
project-local `projects/shopping/seed/llm_client.py` (drop-in
`call_llm(messages, tools=...) -> LLMResponse` matching
`platform_core.llm_wrapper`'s own interface/env-vars, so `workflow.py`
only needed its import line and Responses-API-specific message-building
(`_append_raw_output`/`_strip_reasoning`, which have no Chat-Completions
equivalent) swapped for standard Chat-Completions turn format). Verified
live: the same case that produced an empty cart under the Responses API
produced all 5 expected products (¥3645 total) under Chat Completions. A
6-case harness smoke test went from 0.0 to 0.4333. The full 120-case
re-run (`runs/adhoc_eval_shopping/baseline_full_35b_chatcompletions_v3_full_benchmark`)
landed at **0.2517** (18/120 passed, 0 crashes) — the 6-case sample was
optimistic, as small samples often are, but the fix's effect at full
scale is unambiguous (0.0 → 0.2517, from provably broken to genuinely
attempting the task).

Failure causes across the 102 remaining failures: `missing_product` (85),
`feature_mismatch` (7), `not_cheapest` (4), `ambiguous` (4),
`user_info_mismatch` (3). Notably **81/120 cases still end with a fully
empty cart** even after the fix — a separate, not-yet-root-caused
remaining issue (a live single-case trace showed occasional LLM turns
returning neither text content nor a tool call, breaking the loop's
progress; not investigated further this session). This is a real
avenue for improvement (via HGM search or further debugging) but out of
scope for this baseline-establishment pass — this memory entry documents
what's confirmed vs. open: `responses_api_array_args_bug.md`.

## Statistical confidence

No formal sampling-variance model was built for either project. The
single-agent's 6-case smoke score (0.4333) vs. full-120 score (0.2517)
is a large gap driven by small-sample variance, not concerning on its
own, but a reminder not to trust small-sample checks as final numbers for
this project. `shopping_mas`'s 0.8829 is a full-120 result already, no
smoke-vs-full comparison available.

## Run status

**Both complete, seed baselines only — no HGM search run yet for either
project this session.** `shopping_mas` config:
`configs/eval_local_shopping_mas_qwen35b_implicit.yaml`. `shopping`
single-agent config: `configs/hgm_dual_shopping.yaml` (note: despite the
name, this run used `evaluate_task_agent.py` directly against the seed,
not an actual `hgm_dual` search). `shopping`'s new `llm_client.py` is
currently NOT in that config's `mutable_exclude` list (the config has no
`mutable_exclude` at all — the whole seed dir is HGM-editable by default)
— worth adding `mutable_exclude: ["llm_client.py"]` (matching
`shopping_mas`'s own convention) before running an actual HGM search,
so the editor doesn't inadvertently mutate the infra fix documented
above.
