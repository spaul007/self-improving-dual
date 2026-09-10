# Plan: LLM feedback as the optimization signal

**Status (2026-09-08): approved and implemented** with the proposed answers to
all five decisions (judge default, skip unclear/weak, score once with flips
reported, implementation view, parent rows for shared cases only). Tests: 477
green (`tests/test_judge_signal.py` covers the new behaviour). The live
GPT-5.4 run keeps its loaded code; a new seeded run is needed to use this.
Deviations from the text below: the judge's code budget is its own key
(`analysis_code_char_budget`, 20000) rather than the tagger's diff cap;
`unclear`/weak nodes are *pending* (re-checked after later analyses) rather
than skipped for good; `judge_min_evidence` is the config name of decision 2.

**Follow-up 2026-09-10 (applied):** the prompts were made judge-first as well
— steering objective + a parent-verdict section, editor objective sentence
(`editor.config.objective`, auto `judge` in belief mode), planning-pass
prediction (`expected_effect` / `expected_targets`), belief run context and
per-strategy ledger, contract grammar/example, optimizer prompt, and the
judge-vs-score section reframed as a parent-independent diagnostic. A
`…_delta.yaml` ablation config isolates the label source. See
`REVIEW_ISSUES.md` §3. No run has exercised judge mode yet.

Date 2026-09-08. Repo `rsi/edit-memory-sep6/self-improving-dual`. Companion to
`REVIEW_ISSUES.md` §1 (the "8 shared cases" problem).

## Interpretation — please confirm
"The LLM feedback should be the focus of the optimization" is read as: the
per-node **analysis call becomes the judge**, and its verdicts are the outcome
labels that (a) score belief predictions (Brier), (b) drive the guidance
optimizer, (c) fill the ledger, the records and the steering lines. The
benchmark score stays what HGM searches on and is shown as *noisy context*
(with its standard error), but it no longer gates or labels anything in the
belief loop. The judge already reads traces, constraint outcomes, usage counts
and the diff; what is missing is (1) an explicit effect verdict, (2) evidence
that does not need parent overlap, (3) removing the shared-case gate, and (4)
rewiring scoring/reporting to the verdict.

## 1. The judge — `meta_agent/edit_usage.py` (analysis v7)

**Inputs** (`build_analysis_prompt`, deterministic; kept: description with
sub-edits, diff, usage counts with scorer agreement, surface, sampled events,
ground-truth excerpts):
- *Performance*: child and parent own-case means with n and SE; paired Δ only
  when ≥ 8 shared, else "no paired estimate (k shared)". Explicit rule: a score
  difference under ~2×SE is noise and is not evidence.
- *Per-check table*: today over shared cases only. Add an **unpaired** table:
  failure rate per check over the child's own cases vs the parent's own cases
  (k/n each), labelled unpaired.
- *Per-case rows*: today only cases the components touched (≤ `max_cases`).
  Give one row per child case in the latest batch (16): score, failed checks
  (compact), component verdicts fired on that case (from the usage store's
  events), error/timeout; parent rows for the shared cases plus its aggregate.
- *Within-case evidence*: nothing new to capture; a rule that a component
  changing the output and the final check then passing on the same case is
  the strongest evidence, cross-node score deltas the weakest.

**Outputs** (`ANALYSIS_TOOL`), per sub-edit in `sub_edit_verdicts[i]`:
- `implementation_sound` + reason (existing);
- `effect`: `improved | no_effect | regressed | unclear` on the behaviour the
  sub-edit targets; `effect_reason` anchored to counts/case ids;
- `evidence`: `strong` (within-case or component-level counts on ≥ 8 cases)
  | `moderate` (unpaired per-check movement) | `weak` (diff only / < 4 cases);
- `targeted_checks`: check names the sub-edit aims at.
Node-level: `regressions` (checks that got worse, with counts) + existing
summary/components/targets/collateral.

**Rules** added to `ANALYSIS_SYSTEM`: the judge never sees belief predictions
(true today; keep it so); "never fired" → `no_effect`, not `unclear`;
`improved` needs evidence beyond the diff; the score Δ counts only if
|Δ| > 2×SE. `ANALYSIS_VERSION = 7` (re-runs each node's analysis once).
`render_analysis` adds `- **effect (edit N)**: improved (strong) — reason`.

**Diff shown to the judge**: switch from the 6000-char middle-truncated diff to
the implementation view (added lines per definition, whole units, no
elision) under the same char budget — decision 4 below.

## 2. Gating and records — `meta_agent/edit_memory.py`, `edit_outcome.py`
- The analysis gate `oc.n_shared >= min_shared` (edit_memory.py:837) becomes
  `child.n_evals >= analysis_min_own_evals` (default 16 = one batch).
  Re-analysis per new batch stays as today (`analysis_sig` over batches).
- `EditOutcome` gains `se_all` (unpaired) and `se_shared`; the paired verdict
  stays computed for context.
- Record performance line: `child 0.660/16 vs parent 0.672/16 — unpaired
  Δ −0.012 ± 0.12; 3 shared cases: no paired estimate` (or the paired Δ ± SE
  when ≥ 8 shared). `## Analysis` carries the effect lines.
- `edit_memory_render._load_records` parses `effect_by_edit`,
  `evidence_by_edit`, `effect_reason_by_edit`, `se`, next to `impl_by_edit`.
- New config keys under `edit_memory`: `analysis_min_own_evals: 16`,
  `strategy_label: judge` (`delta` = current behaviour, kept for ablation),
  `judge_min_evidence: moderate`.

## 3. Belief scoring — `meta_agent/belief_scoring.py`
- Strategy kind: y from the matched sub-edit's `effect`: improved → 1;
  no_effect or regressed → 0; `unclear` → skipped ("judge unclear"); evidence
  below `judge_min_evidence` → skipped ("weak evidence"). Still scored only
  when that sub-edit is sound.
- Implementation kind unchanged (y = sound) — now available after one batch.
- `Scored` gains `effect`, `evidence`, `effect_reason`, `delta`, `se`.
- Score once at first eligibility (as now). When a later analysis flips the
  verdict, the calibration report lists it ("judge flipped on node N after
  batch 2") so drift is visible — decision 3 below.
- Calibration report adds a **judge-vs-Δ agreement** line over nodes where
  the paired Δ is well measured (≥ 16 shared): the check that the judge tracks
  the benchmark rather than the diff's narrative.

## 4. Contract, citations, ledger — `belief_contract.py`, `edit_beliefs.py`, `edit_memory_render.py`
- `CONTRACT_SYSTEM`: strategy p = P(the judge finds the mechanism improved its
  target | sound); implementation p = P(sound); the score is noisy context.
- Citation grammar: `[node N: improved|no_effect|regressed|unclear]`,
  `[node N: Δ+0.05±0.12]`, `[node N: unmeasured]`. `verify_citations` checks
  the effect word against the record (SOFT) and Δ within tolerance.
- `build_ledger`: per strategy — nodes judged, improved / no_effect /
  regressed counts, sound / unsound counts, mean unpaired Δ ± SE as context.
  Heading "Deterministic per-strategy ledger (ground truth)" → "Per-strategy
  outcomes (judge verdicts; score Δ is context)".

## 5. Steering and planning pass — `steering.py`, `agent_editor_two_stage.py`
- Sibling line: `node 8: improved (strong) · impl sound · Δ +0.03 ± 0.12
  unpaired · "goal"`; flags unchanged.
- Planning-pass measured-node lines use the same verdict wording.

## 6. Guidance optimizer — `belief_optimizer.py`
Scored-prediction lines include effect, evidence strength and the judge's
reason, so the critique can name systematic mismatches (e.g. "beliefs predict
`helps` for verifiers that never fire"). Brier stays unweighted; weak evidence
is skipped rather than down-weighted (decision 2).

## 7. Config, README, tests
- Belief configs: the three keys from §2. README belief section updated.
- Tests: v7 render/parse; own-eval gate; `resolve` in judge mode (improved →
  1, regressed → 0, unclear/weak skipped) and delta mode unchanged; citation
  verifier; ledger counts; steering line; optimizer prompt content;
  `EditOutcome` SE; `_load_records` parsing; adjusted fixtures for the new
  performance line.

## Cost and risk
- Analysis prompt grows from ~18k chars to ~30–40k (16 per-case rows +
  unpaired table): ≈ $0.03–0.05 per analysis on GPT-5.4 medium; ≈ 100–150
  analyses per 1000-eval run → ≈ $5–8 extra.
- Judge bias: it sees the edit's own description and diff (it must, to judge
  mechanism). Mitigations: counts required for every claim, "never fired →
  no_effect", per-sub-edit verdicts, the judge-vs-Δ agreement line.
- The belief loss now measures "predicts the judge"; the agreement line is
  what tells you whether the judge is tracking the benchmark.

## Not changed
HGM search and selection; evaluation sampling; the belief document format
apart from citations; the no-edit-mem control and `full` steering mode.
A new seeded run (`run_seeded.py`, round 0 reused) is needed to see it live;
the current run keeps its loaded code.

## Decisions needed
1. Default `strategy_label: judge` in the belief configs, `delta` kept only
   for ablation?
2. Nodes with `unclear` or weak evidence: skip (proposed) or score at y = 0.5?
3. Score once at first eligibility (proposed) or re-score when the judge's
   verdict changes with more batches?
4. Judge prompt diff: implementation view (proposed) or keep the truncated
   unified diff?
5. Per-case rows: all 16 child cases + parent rows for shared cases only
   (proposed), or the parent's 16 as well (+ ~5k chars)?

Effort: about one day including tests.
