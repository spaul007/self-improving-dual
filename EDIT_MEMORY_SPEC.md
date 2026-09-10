# Spec — the edit memory as implemented

Record format 6 · analysis v7 (the judge) · belief format 2 · retrieval manifest v2.
Every section names the code it describes as `file.py::symbol`; when the two
disagree, the code is right and this file is stale. `README.md` ("Edit memory",
"Belief mode") is the user-facing summary; `EDIT_MEMORY.md` is a rendered
example from a real run (regenerate it with
`study/render_edit_memory_example.py`).

## 1. Purpose and units

The edit memory is the run's tree-global account of every edit attempted:
what was changed, what the judge found, and what the score did — across all
branches, not one lineage. Facts (records, registry, usage, outcomes) are
deterministic or LLM-authored once; interpretation lives in a separate,
scored belief document (§4). `meta_agent/edit_memory.py` module docstring.

Units:

- **Node record** — one `round_NNN/edit_memory.md` per expanded node
  (`edit_memory.py::RECORD_NAME`), written at expand time and rewritten in
  place on every refresh (§2.4).
- **Sub-edit** — a record holds `## Edit N` blocks, one per distinct mechanism
  the edit introduced, at most `max_subedits` (default 3). The tagger splits
  by *what was done*, never by area; two sub-edits may not share a strategy;
  instrumentation (trace.log, telemetry, counters) is never its own sub-edit
  (`edit_memory.py::NODE_SYSTEM_TMPL`).
- **Registry** — `edit_memory_registry.json` (`edit_memory.py::REGISTRY_NAME`):
  two axes, `strategies` (level 1, *how* it was built) and `areas` (level 2,
  *what* it aimed at), each `{id: {definition, first_node, edits: [{node,
  edit_index, name}]}}`. Ids are promoted at first use; the setup pass's
  candidates live apart in `edit_memory_candidates.json` and never enter the
  registry unused (`edit_memory.py::EditMemory.setup`, `_admit`).
- **Fit** — how a sub-edit's strategy id was assigned
  (`edit_memory.py::EditMemory._fit`): `exact` (an established id), `folded`
  (an undefined id folded into an existing one at Jaccard ≥
  `MIN_FIT_SIMILARITY` = 0.5 over hyphen tokens), or `forced` (the strategy
  axis is at `max_strategies` and the nearest id by shared token was
  imposed). Rendered as `- **fit**: …` (§2.1); forced tags are never matched
  to a belief and never charged (§5).

## 2. The record

### 2.1 Layout (`edit_memory.py::render_record`, `render_edits`)

```
---
node: 30
parent: 9
depth: 3
lineage: 0 > 3 > 9 > 30
---

## Edit 1
- **name**: `add-plan-tool-shim`
- **category level 1 (strategy)**: `add-tool-compatibility-shim`
- **category level 2 (area)**: `finalization-tool-compatibility`
- **what**: <one sentence, paths stripped>
- **why**: <one sentence, specific to THIS sub-edit>
- **fit**: forced by the registry cap — …   (only for folded / forced)

## Outcome
- **performance**: child 0.7073 over 60 evaluated cases (vs parent on 60 shared: child 0.7073, parent 0.6396, Δ +0.0677)
- **unpaired**: child 0.7073/60 vs parent 0.6396/60 · Δ +0.0677 ± 0.0612 · paired SE ±0.0431
- **generalization**: seen 0.7073/60 (Δ +0.0677)
- **new tools** (3 batches, 60 cases): `plan` 13 calls / 13 cases
- **new log point `decision_branch/plan_tool_shim`**: fired ≥13x (pass 13) (…) -> scorer on those cases: 2 pass / 11 fail · SUSPECT VERIFIER

## Analysis
- **`plan shim`** (gate) — <activated>; <verdict behaviour>
  - agreement: …
  - likely cause: …
- **target `commonsense:…`** — remaining k/n (was j/n, +d) — …
- **collateral**: …
- **implementation**: sound — <reason>
- **implementation (edit 1)**: sound — <reason>
- **effect (edit 1)**: improved (strong; targets: opening_hours) — <reason>
- **regressions**: budget 1->3 fails (-2)
```

Frontmatter carries human keys only (`node/parent/depth/lineage`); machine
state lives in the sidecar (§2.4). `split_record` cuts the body at
`\n## Outcome`, so a refresh can regenerate everything below it without
re-parsing model prose.

### 2.2 Outcome lines (`edit_memory.py::render_record`, `edit_outcome.py::compute_outcome`)

- `performance`: the child's mean over ALL its evaluated cases, then the
  paired comparison on the cases both parent and child ran (`n_shared`,
  means, `delta_shared`). Variants: `(no cases shared with parent yet)`,
  `not yet measured`.
- `unpaired` (format 6): each side's mean over its OWN cases, `delta_all` ±
  `se_all` (`edit_outcome.py::se_unpaired`), and the paired standard error
  `se_shared` when cases are shared. Exists so a node is comparable to its
  parent after one batch: a 16-case batch from the 60-case split shares
  ~4 cases with a 16-eval parent. At ~16 cases every SE is near ±0.1, five
  times the ±0.02 band, so readers treat |Δ| < 2·SE as noise.
- `generalization`: seen vs unseen split of the child's score
  (`edit_memory.py::render_generalization`, `edit_outcome.py::seen_split`);
  "seen" = the parent's evaluated cases at edit time, captured in
  `edit_usage.json::seen_case_ids`.
- usage lines (`edit_usage.py::usage_lines`): per new tool `k calls / n
  cases` or `**0 calls**`; per new log point `fired kx (verdict counts)` or
  `**never fired**`, with the deterministic scorer cross-tab `-> scorer on
  those cases: X pass / Y fail` and `SUSPECT VERIFIER` when passes land
  mostly on scorer-failed cases. "No usage data" and "0 calls" are never
  conflated: a store with no consumed batch renders nothing.
- The delta verdict (`helped` ≥ `verdict_threshold` 0.02, `hurt`, `neutral`,
  `inconclusive` below `min_shared` 8 shared cases) is derived at render
  time (`edit_memory_render.py::_verdict`, `perf_text.py::classify_delta`),
  never stored.

### 2.3 Analysis lines (`edit_usage.py::render_analysis`)

Component bullets (role `gate | detector | other`, activation, verdict
behaviour, agreement, ≤ 3 likely-cause lines), `target` lines (failure
counts after vs before), `collateral`, the node-level `implementation`
verdict (v5), per-sub-edit `implementation (edit N)` (v6), per-sub-edit
`effect (edit N)`: `improved | no_effect | regressed | unclear` with the
evidence grade `strong | moderate | weak` and up to 4 `targets:` (v7), and
the node-level `regressions` line (v7). Older payloads render no line for
fields they lack.

### 2.4 Refresh and sidecar (`edit_memory.py::EditMemory.refresh_outcomes`, `_refresh_one`)

Refresh is radius 1 (the node, its parent, its children). A record is
rewritten only when `case_sig(child) / case_sig(parent) / threshold /
min_shared / RECORD_FORMAT` changed, tracked in `round_NNN/edit_memory_state.json`
(`STATE_NAME`; keys `child_case_sig, parent_case_sig, threshold, min_shared,
fmt, analysis_sig`). Bumping `RECORD_FORMAT` rewrites every record exactly
once. The `## Analysis` section is carried forward verbatim unless the judge
re-ran (§3.4).

### 2.5 What `_load_records` exposes (`edit_memory_render.py::_load_records`)

Per node: `fm, body, text, usage, suspect, analysis, delta, n_shared, n_abs,
parent_abs, child_abs, delta_all, se_all, se_shared, parent_abs_all,
parent_n_all, tags [{edit, name, strategy, area, what, why, fit}], impl_sound,
impl_reason, impl_by_edit, impl_reason_by_edit, effect_by_edit,
evidence_by_edit, targets_by_edit, effect_reason_by_edit, effect, evidence,
effect_reason (the first judged sub-edit's), regressions`. Everything is
parsed back from the markdown by the regexes at the top of that module.

## 3. The judge (per-node analysis, v7)

### 3.1 Inputs (`edit_usage.py::build_analysis_prompt`, called from `edit_memory.py::EditMemory._analyze`)

1. The record body (sub-edits with strategy/area/what/why).
2. The implementation view (`edit_code.py::render_implementation_view`):
   added lines grouped by top-level definition, full source of new
   definitions, whole units only, under `analysis_code_char_budget`; the
   middle-truncated unified diff is used only when sources are missing.
3. Performance with standard errors: each side's own-case mean (SE), the
   unpaired Δ ± SE, the paired Δ ± SE when ≥ `min_shared` cases are shared
   (else flagged as not a usable estimate). Header rule: a Δ smaller than
   2×SE is noise.
4. Per-check failure tables: PAIRED (shared cases with check data, via the
   run's `per_check_recipe`) and UNPAIRED (each side over its own cases).
5. Deterministic usage lines with scorer agreement, the surface (new tools /
   log points), and up to `MAX_ANALYSIS_EVENT_LINES` sampled runtime events
   (stratified per label/name/verdict from the `MAX_EVENTS`-capped store).
6. One row per evaluated case, shared cases first with the parent's row
   beside: score, failed checks, which of this edit's components fired with
   what verdict, error/timeout (`MAX_ANALYSIS_CASE_ROWS` = 32).
7. Ground-truth excerpts for the cases the components touched
   (`MAX_ANALYSIS_CASES`).

The judge never sees the belief document or any registered prediction.

### 3.2 Instructions (`edit_usage.py::ANALYSIS_SYSTEM`)

"You are the JUDGE: your verdicts are the outcome labels the run learns
from." Evidence hierarchy: (1) within-case before/after, (2) component
counts joined to the scorer over ≥ 8 cases, (3) per-check movement on the
targeted checks (paired, else unpaired), (4) the score Δ (noise under 2×SE),
(5) the implementation alone (never enough for `improved`). Rules: gates and
detectors are judged in both directions incl. the remediation join; "never
fired" is always `no_effect`; every claim cites counts, case ids or diff
lines; the description states intent, not what happened.

### 3.3 Output (`edit_usage.py::ANALYSIS_TOOL`)

`components[]`, `targeted_constraints[]` (remaining `k/n`, `was`),
`collateral`, node-level `implementation_sound` + `implementation_reason`,
`sub_edit_verdicts[]` = `{edit, sound, reason, targeted_checks[], effect,
evidence, effect_reason}`, `regressions`.

### 3.4 Gate, versioning, cost (`edit_memory.py::EditMemory._refresh_one`, `edit_usage.py::analysis_sig`)

The judge runs at refresh when the usage store has consumed at least one
batch, `analysis_mode` is `refresh` (or `final` during finalize), and the
node is *measurable*: in judge mode `child_n_all ≥ analysis_min_own_evals`
(16 — one batch of the node's OWN cases, no parent overlap needed); in delta
mode `n_shared ≥ min_shared` (the pre-v7 gate). It re-runs only when
`analysis_sig` (the node's own case signature + consumed batches, salted
with `ANALYSIS_VERSION`) changes, so each new batch re-buys one call and a
version bump re-buys every node once. The exact prompt is dumped to
`round_NNN/edit_analysis_prompt.txt`.

## 4. Beliefs

### 4.1 Contract (`belief_contract.py::GRAMMAR`, `EXAMPLE`, `parse_document`, `validate_document`)

```
## Summary                          optional, first, ≤ SUMMARY_CHAR_CAP (1200)
### belief:<slug> — <title>
- kind: strategy | implementation
- scope: strategy=<registry id> [area=<registry id>]
- predict: p=<P_MIN 0.05 .. P_MAX 0.95>
- evidence: <why; citations>
- next: <one concrete move for the editor>
- track: <CODE-GENERATED; stripped on parse, never accepted from the model>
```

Citations (`CITE_RE`): `[node N: improved|no_effect|regressed|unclear]`,
`[node N: improved; Δ+0.0310/12]` (verdict + paired Δ over 12 shared),
`[node N: Δ-0.17±0.14]` (unpaired Δ ± SE), `[node N: unmeasured]`. The
grammar and example are judge-first: cite by verdict, add a Δ only when it
is well measured (≥ 16 shared, or |unpaired Δ| > 2×SE). HARD violations
(prose above the first belief, unknown field, unexpected section, missing
field, bad kind/predict/p range, unknown strategy/area id, duplicate slug or
scope, summary/document over cap, no beliefs) cost the submission; SOFT
violations (`verify_citations`: no record, verdict/Δ misquotes, stale
`unmeasured`) are reported and flagged in the `- track:` line but never
reject. `match_belief` picks, per kind, the belief whose scope matches a
sub-edit's (strategy, area) most specifically (area match beats
strategy-only); a strategy-wide belief covers every area.

### 4.2 The maintainer (`edit_beliefs.py::BeliefStore.update`)

Triggered by the manager after every expand and every evaluation batch; an
evidence signature (per-node case/analysis sigs + registry bytes +
proposal joins, salted with `BELIEF_FORMAT`) makes a no-change trigger cost
zero calls. Order: (1) score newly measurable predictions (§5) and refresh
track lines; (2) build the calibration report and let the optimizer step
(§6); (3) evidence gate; (4) one rewrite call — system prompt =
`CONTRACT_SYSTEM` with `SCORING_JUDGE` (or `SCORING_DELTA` under
`label_source: delta`) + the learned guidance; user prompt = run context
(judge-first: "the run optimizes what the judge finds", score as
orientation), registry ids, the current document with track lines, the
calibration report, the per-strategy outcomes ledger
(`edit_memory_render.py::judge_ledger_lines`: verdict tally, targets,
regressions, implementation verdicts, score as trailing context), and the
full records that changed since the last update (whole records, newest
first, oldest dropped whole under `evidence_char_budget`). The response is
validated; one retry names the violations; a still-hard result keeps the
previous document. Prompts are dumped to
`edit_memory_beliefs_prompts/update_NNNN.txt`; the previous document is
archived to `edit_memory_beliefs_archive/`.

### 4.3 Registration (`edit_beliefs.py::BeliefStore.register`, called from `managers/hgm.py::HGMManager._expand` right after `record_node`)

Freezes, per kind, the belief that covered the new node's usable (non-forced)
tags and its `p` into `round_NNN/belief_prediction.json` (`belief_version`,
`instruction_version`, `tags`, `coverable`, `matched_edit`, the section
text). `coverable` is false when no earlier node used any of the node's
strategies — no belief could have existed, so silence is not charged.

## 5. Scoring (`belief_scoring.py`)

A prediction is scored once, at first eligibility, with Brier loss
`(p − y)²`; 0.25 is what an uninformative p = 0.5 scores.

- `is_measurable`: judge mode — the analysis has run (`impl_sound` present);
  delta mode — `delta` over ≥ `min_shared` shared cases.
- implementation kind: `y = 1` iff the matched sub-edit (else the node) was
  judged sound.
- strategy kind: scored only when that sub-edit is sound. Judge mode
  (`label_source: judge`, the default): `y = 1` iff the judge's effect for
  the matched sub-edit (else the first judged sub-edit) is `improved`;
  `no_effect`/`regressed` → 0; `unclear` or evidence below
  `judge_min_evidence` (`perf_text.py::EVIDENCE_RANK`) → pending, re-checked
  after later analyses. Delta mode: `y = 1` iff Δ ≥ threshold.
- Uncovered but coverable predictions score at p = 0.5; uncoverable ones are
  skipped for good with `UNCOVERABLE_REASON`.
- `Scored` rows persist in `edit_memory_beliefs_state.json::scored` with
  `label_source, effect, evidence, effect_reason, matched_edit, delta,
  n_shared, delta_all, se_all`; rows from before v7 load as `delta`.
- Track lines (`track_lines`): `n=k · Brier x (0.25 = uninformative) ·
  outcomes: <node yes/no …> [· k misquoted citation(s)]`.
- Calibration report (`render_calibration_report`): labels line, totals and
  per-guidance-version Brier, per belief, worst misses (judge verdict and
  reason before the score), uncovered nodes with the near-miss hint,
  not-scored reasons, open predictions, **Judge vs score — a diagnostic, not
  a yardstick** (rows whose score is well measured: ≥ 16 shared or |unpaired
  Δ| > 2×SE; disagreement flags a node worth re-reading), **verdict changes
  since scoring** (rows are never re-scored), citation checks.

## 6. Guidance optimizer (`belief_optimizer.py::InstructionOptimizer`)

The maintainer's system prompt is a fixed contract plus a learned guidance
text (`belief_instruction.md`, seeded with `SEED_INSTRUCTION`, archived as
`belief_instruction_archive/vNNN.md`). Every `optimize_every` (8) newly
scored predictions: rollback check (revert to an earlier version that beats
the current one by more than `optimize_rollback_margin` 0.02 when both have
≥ `optimize_min_scored` 8 rows), then one call (`OPTIMIZER_SYSTEM`) that
sees the current and earlier guidance with their Brier, the predictions
scored since the last step (belief text at registration, the judge's
verdict, evidence grade, reason and targets; the score only when well
measured), and the calibration report, and returns a critique + revised
guidance ≤ `instruction_char_cap`. Prompts/responses:
`step_NNN_prompt.txt` / `step_NNN_response.json`. Caveat (run issue #10):
`scored_since_step` counts all newly scored rows while per-version Brier
attributes by registration version, so a version is usually replaced before
it reaches `optimize_min_scored` of its own rows.

## 7. Steering

### 7.1 Belief mode (`steering.py::render_belief_steering`, selected in `managers/hgm.py::HGMManager._render_expand_context` when `steering_mode == "belief"`)

Exactly these sections, nothing appended, nothing truncated:

1. `## Objective` — `OBJECTIVE_TEXT` ("Fix what the judge found …") and one
   `Score context (a noisy reference, not the objective): seed …· best so far
   … · this parent …` line.
2. `## What the judge found on this parent (node N)` — the parent's verdict
   with targets, regressions, implementation verdict and reason
   (`parent_judge_line`); or "not yet judged" / "node 0 is the seed".
3. `## Scope of this edit` — `SCOPE_TEXT` (one strategy in one area; instrument
   decision points with trace.log; do not bundle mechanisms).
4. `## Edits already applied along this lineage (root → parent)` (omitted at
   depth 0).
5. `## Edits already tried directly off this parent (node N)` —
   `SIBLING_FRAMING` then one line per sibling (`compact_sibling_lines`):
   `judge <verdict (grade; targets)> · regressions: … · score <paired Δ, else
   unpaired Δ ± SE> · child m/n · "goal" · flags: dead component, suspect
   verifier, implementation unsound, edit failed`.
6. `## Belief document` — `BELIEF_FRAMING`, the calibration line, the document
   verbatim.

The editor's system prompt carries `agent_editor.py::OBJECTIVE_JUDGE` in this
mode (`config.py::editor_objective` injects `objective="judge"`; a YAML
`editor.config.objective` wins).

### 7.2 Full mode and the control

`steering_mode: full` renders the legacy block
(`edit_memory_render.py::render_edit_memory`: run context with "the highest
ABSOLUTE score", reading guide, BUILD ON / REPAIR / DIVERSIFY, the
per-strategy ledger, the focus block, every record oldest first, compact
fallback over budget) and the editor keeps `OBJECTIVE_SCORE`; with no edit
memory the manager's own legacy text applies. Both are byte-identical to
their pre-belief form and are the ablation controls.

## 8. Retrieval and the two-stage editor (`agent_editor_two_stage.py`, `edit_archive.py`)

Stage 1 (`PROPOSE_SYSTEM`, `PROPOSAL_TOOL`): from the steering context, the
registry block (`_render_registry_ids`: strategy/area ids with nodes,
measured nodes as `judge … · regressions … · implementation … · score …`, and
what the parent's own edit already retrieved), the feedback and the parent's
sources, it returns 1–3 candidate edits, a memory query (nodes / strategies /
areas / keywords / include_code) and a prediction: `belief_id`,
`expected_effect` (improved / no_effect / regressed), `expected_targets`
(legacy `expected_direction` / `expected_delta` still accepted). The
prediction is written to `round_NNN/edit_prediction.json` (v2) and joined
into the calibration report as "cited by N proposal(s)"; it is not a scoring
input. Retrieval (`edit_archive.py::resolve_query`, explicit nodes > category
> keyword, `DEFAULT_MAX_NODES` 4, `DEFAULT_CHAR_BUDGET` 60000) returns whole
records plus the implementation view from sources (fallback: the
`edit_code.md` slice when it fits), never cut mid-text; `retrieval_manifest.json`
(v2) records what was shown and omitted. Stage 2 is the single-call editor
with the advisory proposal and the retrieved memory appended.

## 9. Tagger (`edit_memory.py::EditMemory.setup`, `record_node`)

Setup (once per run, `SETUP_SYSTEM`): candidate strategies/areas from the
seed and the per-check recipe (`edit_outcome.py::validate_recipe`). Per node
(`NODE_SYSTEM_TMPL`, `NODE_TOOL`): the editor's intent, changed files, the
established/suggested categories and the diff (`diff_char_cap`, the only
middle-truncated text on any LLM path) → sub-edits with name/what/why/
strategy/area; `_absorb` caps `max_subedits`, `_fit` assigns ids (§1),
`_admit` promotes ids into the registry at first use. The prompt is dumped
to `round_NNN/edit_memory_prompt.txt`. Records are presence-idempotent: a
second `record_node` for the same round makes no call.

## 10. Configuration

`edit_memory.config` (`edit_memory.py::EditMemory.__init__`): `model`,
`reasoning_effort`, `base_url` (pin it — see `config.py::meta_base_url_warnings`),
`steering` (true), `steering_token_budget` (48000, full mode only),
`verdict_threshold` (0.02), `min_shared` (8), `top_k_checks` (40),
`max_strategies` (18; 30 in the travel belief configs), `max_subedits` (3),
`diff_char_cap` (6000), `setup_pass` (true), `per_check_recipe` (null = let
setup choose), `usage_tracking` (true), `usage_max_events` (400),
`analysis_mode` (refresh | final | off), `analysis_max_cases` (10),
`analysis_max_event_lines` (120), `analysis_max_case_rows` (32),
`analysis_min_own_evals` (16), `analysis_code_char_budget` (20000),
`strategy_label` (judge | delta), `judge_min_evidence` (strong | moderate |
weak), `code_record` (true), `code_diff_char_cap` (20000), `steering_mode`
(full | belief), `beliefs` (dict → `BeliefStore`, or omit to disable).

`beliefs` (`edit_beliefs.py::BeliefStore.__init__`): `enabled`,
`doc_char_cap` (40000), `max_delta_records` (12), `evidence_char_budget`
(60000), `threshold` / `min_shared` / `label_source` / `min_evidence`
(default from the parent block), `optimize_enabled`, `optimize_every` (8),
`optimize_min_scored` (8), `optimize_rollback_margin` (0.02),
`instruction_char_cap` (2500), `optimize_model`, `optimize_reasoning_effort`.
An unknown key raises at startup in belief mode.

`editor.config` (`agent_editor_two_stage.py::TwoStageEditor.__init__`):
`type: two_stage`, `model`, `reasoning_effort`, `base_url`, `max_attempts`,
`propose_model`, `propose_reasoning_effort`, `retrieval_char_budget` (60000),
`max_retrieved_nodes` (4), `propose_enabled`, `objective` (auto: judge under
belief steering).

`task_agent` (`config.py::TaskAgentSpec`): `model`, `reasoning_effort`,
`base_url`, `temperature` (0.2), `timeout_s` and `max_output_tokens` — the
last three are exported only on the evaluator's case subprocesses
(`LLM_TEMPERATURE`, `LLM_TIMEOUT_S`, `LLM_MAX_OUTPUT_TOKENS`,
`evaluator.py::SubprocessEvaluator._child_env`), never to the meta agents;
`timeout_s` must stay well below `evaluator.wall_time_s_per_case` (600 vs
1800 in the travel configs, so a stalled request costs one retry, not the
case) and `max_output_tokens` bounds reasoning + output together (65536).

Reference configs: `configs/hgm_travel_1000_qwen122b_gpt54_beliefs2stage.yaml`
(judge), `…_delta.yaml` (label-source ablation), the node-5 / local Qwen
and DeepSeek variants, and `configs/hgm_travel_smoke_beliefs2stage.yaml`.
`run_seeded.py --config … --donor runs/<run>/round_000 [--force]` reuses a
donor's seed evaluation.

## 11. Determinism, LLM boundary, invariants

LLM-authored: the tagger's sub-edits, the judge's analysis, the belief
document, the guidance text, the planning pass, the edit itself.
Deterministic: outcomes and standard errors, usage capture and the scorer
cross-tab, the implementation view, retrieval, registration, scoring, track
lines, the ledger, every steering block.

Invariants: records are refreshed in place (no history beyond the belief
archives); the judge is re-bought only when a node's own evidence changes;
predictions are frozen at expand, before any outcome exists; rows are scored
once (later verdict changes are reported, not re-scored); the `full` mode
and the no-edit-memory control are byte-identical to their pre-belief text;
HGM expansion, evaluation sampling and final selection never read the belief
layer; nothing on the editor or maintainer path is truncated mid-text.

## 12. Artefact map

Run root: `edit_memory_registry.json`, `edit_memory_candidates.json`,
`edit_memory_beliefs.md`, `edit_memory_beliefs_state.json`,
`edit_memory_beliefs_archive/beliefs_NNNN.md`,
`edit_memory_beliefs_prompts/update_NNNN.txt`, `belief_instruction.md`,
`belief_instruction_archive/{vNNN.md, step_NNN_prompt.txt, step_NNN_response.json}`,
`run_summary.md`, `config.snapshot.yaml`, `README_ISSUES.md` (human log).

`round_NNN/`: `edit_memory.md` (§2), `edit_memory_state.json` (§2.4),
`edit_memory_prompt.txt` (§9), `edit_usage.json` (surface, consumed batches,
tool counts, events, `seen_case_ids`), `edit_analysis_prompt.txt` (§3),
`edit_code.md` (verbatim diff + full source of new definitions),
`belief_prediction.json` (§4.3), `edit_prediction.json` (§8),
`retrieval_manifest.json` (§8), `verbose/` (every prompt and response when
`verbose: true`), `hgm_node.json`, `eval_result.json`, `logs/`.
