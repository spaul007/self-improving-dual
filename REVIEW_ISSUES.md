# Issues for review (not applied) — 2026-09-08

Written while watching `runs/20260907_212018_travel_hgm_1000_qwen122b_gpt54_beliefs2stage`
(meta agents GPT-5.4 medium, task agent local Qwen3.5-122B). **Nothing in
section 1 has been applied.** Section 2 lists what *was* changed in the repo
since that run started, so it can be vetoed. Times are UTC.

---

## 1. The "≥ 8 shared cases" requirement starves attribution (and belief scoring)

> **Resolved 2026-09-08 (user decision):** the LLM judge is the outcome signal
> — see `PLAN_judge_signal.md`. The analysis runs on a node's own
> evaluations (`analysis_min_own_evals`), its effect verdict labels strategy
> beliefs (`strategy_label: judge`), and the score Δ is shown with its
> standard error as context (records carry an `unpaired` line). Options A–E
> below are kept for the record; `strategy_label: delta` keeps the old rule.

### What happens today
- Δ vs parent is computed **only on cases both nodes ran**
  (`meta_agent/edit_outcome.py::compute_outcome`); with fewer than
  `min_shared` (8) the verdict is `inconclusive`. The per-node analysis call
  (which produces the implementation verdict) is gated on the same count,
  and a belief prediction is scored only when both exist.
- Each node is evaluated on **16 cases drawn at random from the 60-case train
  split, independently per node** (`hgm._evaluate`, `self._task_rng.sample`).
  The overlap of a child's first batch with its parent is hypergeometric:

  | parent evals | expected shared (batch 16) | P(shared ≥ 8) |
  |---:|---:|---:|
  | 16 | 4.3 | 0.02 |
  | 32 | 8.5 | 0.73 |
  | 48 | 12.8 | 1.00 |
  | 60 (root) | 16.0 | 1.00 |

  So only depth-1 nodes (parent = root) are measurable after their first
  batch. A depth-2 node needs its parent or itself to be re-evaluated first.
- Observed in this run: nodes 4/5/7 (children of node 2, 16 evals) share
  5/3/4 cases with it → no Δ, no analysis, no score, 15–50 min after being
  measured. 3 scored predictions in 3 h; the guidance optimizer needs 8 for
  its first step.

### What the run's numbers say about precision (the real constraint)
Per-case score sd ≈ 0.33 (nodes 2–7 pooled; the task is noisy run-to-run,
not just case-to-case). Standard error of Δ under different designs:

| design | SE of Δ | example from this run |
|---|---:|---|
| paired, 16 shared cases | ±0.06–0.09 | node 2 vs 0: +0.094 ± 0.088 |
| paired, 32 shared cases | ±0.04 | node 3 vs 0: +0.057 ± 0.043 |
| unpaired, 16 own vs 16 own | ±0.12–0.15 | node 4 vs 2: −0.17 ± 0.14 (paired on 5 shared: −0.225) |
| root-adjusted (score − root's score on the same case), 16 vs 16 | ±0.13–0.15 | node 7 vs 2: −0.07 ± 0.13 (paired on 4 shared: +0.047) |

The verdict band is ±0.02 (`NEUTRAL_BAND`). **No 16-case design resolves a
0.02 effect**; effects around 0.1 or more (node 4) are visible under any of
them. "8 shared cases" is a proxy for precision — and a weak one, since a
paired Δ at exactly 8 shared cases still has SE ≈ ±0.1.

### Options (none applied)

**A. Unpaired fallback verdict, with the standard error shown.**
When `n_shared < min_shared` but both nodes have ≥ `min_own` own evaluations
(e.g. 16), take the verdict from `delta_all` (child mean over its own cases −
parent mean over its own cases; already computed in `EditOutcome`), and render
`Δ ±SE (unpaired, 16 vs 16)`. Parent-independent; sampling untouched; every
node is measurable after its first batch. Change points: `compute_outcome`
(verdict rule + `se` field), the record's performance line
(`edit_memory_render` / `edit_memory.py`), the analysis gate in
`edit_memory.py` (`oc.n_shared >= min_shared` → a `measurable` flag),
`belief_scoring.resolve` (same flag). Cost: unpaired is ~1.5–2× noisier than
paired at equal n, and the label stays binary.

**B. Uncertainty-aware belief labels** (complements A).
Instead of a binary y from a ±0.02 band, score a strategy belief against
y = P(Δ > 0 | data) ≈ Φ(Δ̂ / SE) (or a paired/unpaired bootstrap). A node
measured at +0.05 ± 0.12 yields y ≈ 0.66 rather than "helped"; a coin-flip
node yields y ≈ 0.5 and neither rewards nor punishes anyone. Brier (p−y)²
stays a proper score. Optionally re-score as n grows (keep the latest).
Change points: `belief_scoring` (`Scored` gains `se`, `n`; soft y),
calibration report wording, `EditOutcome.se`.

**C. Bigger batches** — config only: `eval_batch_size` 24 or 32.
P(≥ 8 shared with a 16-eval parent) becomes 0.25 / 0.73; a 32-vs-32 paired
Δ has SE ≈ ±0.04–0.06. Cost: about half as many nodes per 1000 evals
(~30 instead of ~60); the HGM reference used random batches of 16.

**D. Common evaluation order** (global, not parent-dependent).
One seeded permutation of the 60 cases per run; every node consumes it in
order, so batch 1 is the same 16 cases for everyone and
shared = min(n_child, n_parent) always. Risk: the search sees the same 16
cases first and selection overfits that fold (mitigated by HGM's
re-evaluation of promising nodes and `finalize_top_k` on the full split;
the permutation could also be re-drawn every N expansions). Change point:
`hgm._evaluate` (batch = the first `n_take` not-yet-run cases of the
permutation), ~10 lines behind a config flag.

**E. Additive model** score(node, case) = node effect + case effect, fitted
by least squares over *all* observations in the tree; gives Δ̂ ± SE for any
pair using every evaluation. Helps most where case difficulty dominates
(modest here). Larger change (`edit_outcome`, a new module, numpy). Later.

**Rejected / not proposed:** lowering `min_shared` (paired SE at 4 shared
≈ ±0.17); re-evaluating the parent or child to force overlap, or drawing a
child's batch from its parent's cases (my `eval_pair_with_parent`, written
and **reverted**) — both bend the search policy to attribution needs;
root-only anchoring (no real gain, see table).

**Recommendation:** A + B now — parent-independent, no sampling change,
every node measurable after its first batch, and the noise made explicit
instead of hidden behind a binary verdict. C (32) as a config experiment if
fewer, better-measured nodes is acceptable. D only if common folds are.

---

## 3. Judge-first prompts (2026-09-10) — applied, 498 tests green

Audit finding: the scoring arithmetic was judge-driven but every prompt still
led with the score. Changed so the judge's verdicts come first everywhere and
the benchmark score is one labelled context line:

- editor system prompt: `editor.config.objective` (`agent_editor.py::OBJECTIVES`),
  auto `judge` under belief steering (`config.py::editor_objective`); the
  legacy sentence stays for `full` mode and the control (byte-identical).
- belief steering (`steering.py`): objective "Fix what the judge found", a
  `## What the judge found on this parent` section, judge-first sibling lines
  (verdict + targets, regressions, then score); the planning pass predicts
  `expected_effect` / `expected_targets` (`edit_prediction.json` v2) and its
  registry block shows verdicts before Δ.
- belief maintainer: judge-first run context, `judge_ledger_lines` ledger
  (verdicts, targets, regressions, implementation; Δ trailing), grammar and
  example cite by verdict; optimizer prompt says the verdicts are the labels
  and quotes the score only when well measured; the agreement section is a
  diagnostic (≥ 16 shared OR |unpaired Δ| > 2×SE, parent-independent).
- loose ends: `label_source` defaults to `judge` everywhere (pre-v7 rows still
  load as delta); `first_edit` falsy-zero guard; `targets_by_edit` /
  `regressions` now surface in steering, the planner and the ledger; stale
  two-stage docstring; `configs/…_gpt54_beliefs2stage_delta.yaml` for the
  label-source ablation.
- residuals, by design: the judge reads the edit's own description (bias
  mitigated by its count/case-id requirements and "never fired → no_effect");
  rows are scored once (flips reported); the GPT-5.4 run's 46 rows remain
  Δ-labelled; HGM search/selection is score-only and untouched.
- docs: `EDIT_MEMORY_SPEC.md` rewritten as the spec of the implemented system;
  `EDIT_MEMORY.md` regenerated from the GPT-5.4 run by
  `study/render_edit_memory_example.py` (re-run after the first judge-mode run).

## 2. Applied in the repo since the run started (revertable — say which)

| change | why | tests |
|---|---|---|
| `edit_usage.added_surface`: None-safe sort of `(label, name)` pairs | node 6 lost its usage store (`str` vs `None`) → never analysed/scored | `MixedNameSortTests` |
| `edit_usage._log_call_re`: lookahead window instead of a consumed one | a `trace.log(` within ~3 lines of the previous one was invisible to the surface (cost 3 nodes one log point each here) | `test_adjacent_call_sites_are_all_detected` |
| tagger `NODE_SYSTEM_TMPL`: instrumentation is never its own sub-edit | every node was tagged with `add-decision-telemetry` as a second "strategy" | existing tagger tests |
| `hgm.py` `eval_pair_with_parent` | **reverted** — see 1; config edits were rejected before being applied | — |

The live run is unaffected by all of these (module loaded at start, config
snapshotted); node 6's missing `edit_usage.json` was regenerated offline
before its first evaluation.
