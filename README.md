# meta-agent

A framework for **self-evolving agents**: a meta-agent that mutates a task
agent's source code, evaluates the result on a benchmark, gathers feedback,
and iterates. Modular and config-driven so you can swap managers, evaluators,
seeds, and benchmarks without touching the loop.

## Layout

```
projects/<name>/     Everything specific to one task agent in one place:
  seed/                round-0 starting agent (workflow.py, tool_wrapper.py,
                       tools_schema.json, mutable_tools/)
  benchmark/           cases.jsonl + scorer.py (+ optional _eval/ helpers)
  tools/               immutable tools registered with platform_core.tools
  data/                per-sample data the tools consult (gitignored)

meta_agent/          The meta-agent — managers, editor, evaluator,
                     validators, gatherer, registry, config loader.
platform_core/       Immutable platform: LLM wrapper (OpenAI Responses API),
                     trace emitter, runner contract, tool dispatcher. The
                     editor cannot modify these files.
configs/             YAML configs that wire a project + framework knobs together.
runs/                Per-experiment folders: each round is a self-contained,
                     browsable snapshot of the task agent + logs + scores.
tests/               Smoke tests that run without an API key.
main_loop.py         Optimization entry point: load config → assemble → manager.evolve().
evaluate.py          Standalone evaluation entry point: run a specific
                     task_agent against the full benchmark.
```

Three projects ship with the repo: `projects/math/` (a tiny arithmetic
benchmark + a single-shot seed), `projects/travel/` (a multi-day
trip-planning benchmark with a tool-loop seed), and `projects/shopping/`
(a tool-loop shopping-cart benchmark). Add a new project by creating
`projects/<name>/{seed,benchmark,tools,data?}/` and pointing a YAML at it
via `project: "<name>"`.

## Setup

```bash
pip install -r requirements.txt
source /users/n.tzou/api.sh        # exports OPENAI_API_KEY
```

The framework requires Python 3.10+ (uses `type | None` syntax and Pydantic v2).

## Run

```bash
PYTHONPATH=. python3 main_loop.py --config configs/default.yaml
```

This evolves the math project's seed against its benchmark for up to 5
rounds using the `hill_climbing` manager and the `subprocess` evaluator —
all with `gpt-5.4-mini` at reasoning effort `high`. Each round the manager
picks what to branch from, and the editor makes **one self-improvement
call** that diagnoses *and* rewrites the agent's code in a single step
(see "Optimization managers" below).

Output lands under `runs/<timestamp>_<experiment_name>/`:

```
runs/20260504_153012_math_default/
├── config.snapshot.yaml          # exact config that drove this run
├── round_000/
│   ├── task_agent/               # seed verbatim
│   ├── logs/
│   │   ├── trace.jsonl           # JSONL of llm_call/llm_response/tool_call/...
│   │   ├── case_<id>.json        # per-case CaseResult, written as each finishes
│   │   └── case_<id>.stderr      # only present when a case timed out / crashed
│   ├── strategy.json             # null for round 0
│   ├── eval_result.json          # full EvaluationResult (per-case + aggregate)
│   ├── eval_score.json           # only when split: enabled — held-out composite
│   ├── feedback.json
│   └── hgm_node.json             # HGM only — authoritative per-node tallies
├── round_001/
│   ├── task_agent/               # editor's mutation of round_000
│   ├── behavior_memory.md        # only when summarizer: enabled
│   ├── behavior_aggregate.json   # ditto — the structured pre-aggregation
│   ├── edit_memory.md            # only when edit_memory: enabled
│   └── ...
├── edit_memory_registry.json     # only when edit_memory: enabled (see below)
├── edit_memory_candidates.json   # ditto
└── ...
```

## Optimization managers

The `manager` chosen in the YAML decides the search regime. Two ship:

- **`hill_climbing`** — a linear trajectory: each round branches from the
  best round so far, the editor makes one self-improvement, evaluate, repeat
  for `loop.max_rounds`.
- **`hgm`** — a **Huxley-Gödel-Machine** tree search (arXiv 2510.21614):
  keeps a *tree* of agents, decouples expansion from evaluation under an
  adaptive schedule, picks which node to expand by Thompson sampling over
  *clade metaproductivity*, and counts its budget in agent-task evaluations
  (`eval_budget`) rather than rounds. `configs/hgm_{math,travel,shopping}.yaml`
  wire it up. Before the final pick it re-evaluates the top finalists on the
  full train split so a thinly-evaluated fluke can't win.

In **both** managers a self-modification is **one editor call**: the manager
selects what to work on and hands the editor a cheap steering `context`
string; the editor's single `submit_self_improvement` call diagnoses the
agent *and* rewrites its code, emitting an `EvolutionStrategy` summary
(logged to `strategy.json`). There is no separate "propose a strategy" LLM
call.

### What the editor sees (information gathering)

To diagnose well, the editor is given, in addition to the agent's own code:

- **Example-driven failure analysis** — the feedback gatherer turns each node's
  per-case results into a compact report: the top recurring failure categories
  (from the project's `categorize_errors`), a *diverse* pair of representative
  cases per category (a near-miss + a severe failure) shown as **query → agent
  plan → what failed**, and the hardest (lowest-scoring) cases. This is generic:
  the gatherer reads only a contract (`details["query"]`, `details["raw_result"]`,
  score/passed/error) plus the project categorizer's output — all domain parsing
  stays in the project folder. Stored on `AgentFeedback.failure_report`.
- **Tool implementations** (`projects/<p>/tools/*.py`) and a **database schema**
  (`projects/<p>/db_schema.md`) — so the editor understands what each tool does
  and what the data looks like when generating/modifying tool calls.
- **Evaluation scoring code** — only when `eval_visibility: whitebox` (see below).

Set per-run via a top-level key:

```yaml
eval_visibility: "blackbox"   # default: behavioral feedback + tools + DB schema
# eval_visibility: "whitebox" # also inject projects/<p>/benchmark/scorer.py (+ _eval/)
```

Ground-truth data (`data/`, `cases.jsonl`, validation files) is **never** exposed
in either mode. To enable the failure report, set the gatherer's
`error_categorizer` to the project's categorizer (same `module:func` value the
dual manager uses); without it the report degrades to hardest-cases-only.

To run HGM:

```bash
# Locally (math is a fast smoke target)
PYTHONPATH=. python3 main_loop.py --config configs/hgm_math.yaml

# On SLURM — wrapper with HGM-sized resource defaults (see slurm/README.md)
slurm/run_hgm.sh travel        # configs/hgm_travel.yaml
slurm/run_hgm.sh shopping      # configs/hgm_shopping.yaml
```

### Agentic editor (optional)

`editor.type: "agentic"` replaces the single-shot editor with an HGM-seed-style
**coding agent**: a tool-use session (one `call_llm` per iteration, all tool
calls in a response processed) that works directly on the round's
`task_agent/` copy and ends with a structured `submit_self_improvement` call
carrying only a summary — the edits are already on disk. It is **one agentic
flow**: no proposal / retrieval stage, no feedback digest or steering context
in the prompt; the agent reads the parent node's evidence (and, when the run
has edit memory, the memory / diff / belief / registry files) from disk
itself. No docker. The meta agent never writes a belief prediction — the
belief layer registers its own (see "Edit memory").

Prompts: one fixed system prompt (`meta_agent/agentic/session.py`) and one
instruction message per session. Paths are expressed through four roots —
`$RUN_DIR`, `$NODE_DIR`, `$PARENT_DIR`, `$REPO_DIR` — that are environment
variables in every bash call and are expanded by the editor tool, so no
experiment path ever appears in the prompt. A run **without** edit memory
gets an instruction that never mentions memory or beliefs; a run **with** it
additionally gets the run-level memory files in its workspace map and one
procedure step telling it to consult the belief document (accumulated
understanding of previous edits) as guidance, follow its node references,
and diversify.

Tools:

- `bash` — a fresh `bash -c` per call (cwd = `task_agent/`), inside a
  **bubblewrap** (`bwrap`) allow-list sandbox: no network, environment scrubbed
  (no API key, no `LLM_*`, no `*_DATABASE_ROOT`), only the paths below exist.
  `sandbox: auto` probes bwrap once and falls back to a cwd-only subprocess
  with a printed warning; `sandbox: bwrap` refuses to run unconfined.
- `editor` — `view` (line-numbered, optional `view_range`), `create`,
  `str_replace` (`old_str` must match exactly once), `insert`. No whole-file
  overwrite: the model edits, it does not rewrite. Paths: absolute, `$VAR/...`,
  or relative to `task_agent/`.
- `validate` — runs the configured validators on demand (the fast "unit test";
  does not count as a submission).
- `submit_self_improvement` — `optimization_goal`, `proposed_changes`,
  `rationale`. Validators run on submit; on errors the workspace is **kept**
  and the error list goes back to the model (up to `max_attempts` rounds).

Read/write surface (`meta_agent/agentic/policy.py`; the editor tool enforces
it in-process, the sandbox mirrors it as binds):

| access | paths |
|---|---|
| writable | `round_NNN/task_agent/{workflow.py,tool_wrapper.py,tools_schema.json}`, `task_agent/mutable_tools/*.py` (new files allowed), `round_NNN/agentic/scratch/` (the agent's own throwaway scripts; never validated) |
| read-only | the whole run directory — every `round_NNN/` (`hgm_node.json`, `strategy.json`, `feedback.json`, `eval_result.json`, `logs/`, `edit_memory.md`, `edit_code.md`, `task_agent/`) and the run-level `edit_memory_registry.json`, `edit_memory_candidates.json`, `edit_memory_beliefs*.{md,json}`, `tree_snapshots.jsonl` — plus `platform_core/`, `projects/<p>/tools/`, `projects/<p>/db_schema.md` |
| never | `projects/<p>/benchmark/` (scorer, `_eval/`, cases.jsonl), `projects/<p>/data/`, the categorizer, `meta_agent/`, other runs. `eval_visibility` is ignored by this editor. |

Because bwrap binds are live views, files the framework writes to the run
root later (new memory records, belief updates) are visible without any
rebinding. The task agent can *not* be run on cases inside the sandbox (no
model access, no database) — by design; checks are `validate`, import checks
and scratch scripts. Budget reminders are injected at a third and at the
halfway point of `max_llm_calls` (while nothing has been edited) and in the
final stretch.

Per-session artifacts: `round_NNN/agentic/transcript.jsonl` (every LLM call,
tool call with input/result, validation round) and `agentic/session.json`
(`end_reason`, `n_llm_calls`, per-tool counts, `validation_rounds`,
`sandbox_mode`, `memory_enabled`, roots, token usage). With
`META_AGENT_VERBOSE=1` the exact system prompt, instruction and final message
history land in `verbose/`. In `hgm_dual` mode these stay under
`variants/var_k/` (only `task_agent/` and `logs/` are promoted), and the dual
manager's per-variant category focus is only delivered when
`include_manager_context: true`.

```yaml
editor:
  type: "agentic"
  config:
    model: "deepseek/deepseek-v4-pro-0813"   # meta-model; any Responses-API endpoint works
    reasoning_effort: "low"
    base_url: "https://openrouter.ai/api/v1"
    api_key_env: "OpenRouter_API_KEY"        # key for THIS editor's calls (source api.sh);
                                             # task agent / edit memory keep OPENAI_API_KEY
    llm_timeout_s: 600                       # per-request cap so a stall can't eat the session
    max_attempts: 3            # submission / validation rounds
    max_llm_calls: 150         # tool-loop iterations (history re-sent each call)
    timeout_s: 5400            # wall-clock per session; stops at 90%
    bash_timeout_s: 120
    max_tool_output_chars: 20000
    max_view_chars: 40000
    sandbox: "auto"            # auto | bwrap | none
    include_manager_context: false
```

`api_key_env` (and `timeout_s`) are per-call parameters of `call_llm`, so the
editor can sit on a second provider while everything else in the run keeps
the global `OPENAI_API_KEY`. To move the *whole* run (task-agent subprocesses
included) to another provider's key, export `LLM_API_KEY_ENV: "<VAR>"` from the
YAML `env:` block — the run-wide default for `api_key_env`. The
`hgm_travel_100_dsv4pro_agentic_{editmem,no_editmem}.yaml` pair does this
(DeepSeek V4 Pro for meta and task agent, task reasoning `none`,
`finalize_top_k: 0` = no end-of-run top-k fill-up, `verbose: true` so every
session's full message history is kept, `seed_round_dir` to reuse a previous
run's seed pre-evaluation).

```bash
source /groups/AIC-MV/sudipta.paul/code/random/api.sh   # exports OpenRouter_API_KEY
PYTHONPATH=. META_AGENT_VERBOSE=1 python3 main_loop.py --config configs/hgm_travel_smoke_agentic.yaml
PYTHONPATH=. python3 main_loop.py --config configs/hgm_travel_100_dsv4pro_agentic_editmem.yaml
```

## Standalone evaluation

`evaluate.py` runs a specific task_agent (the seed, a saved round, or any
hand-edited copy) against the **full** benchmark — ignoring any `split:`
block in the YAML. Use this to measure baselines and to compare any round
of an optimization run on the held-out evaluator.

```bash
# Score the unedited seed
PYTHONPATH=. python3 evaluate.py \
    --config configs/travel.yaml \
    --agent projects/travel/seed

# Score round 3 of an optimization run
PYTHONPATH=. python3 evaluate.py \
    --config configs/travel.yaml \
    --agent runs/20260506_140000_travel_default/round_003/task_agent
```

Results land in `runs/eval_<stamp>_<agent_basename>/round_eval/`. Per-case
results are persisted to `logs/case_<id>.json` as each case finishes, so
you can inspect partial scores while the run is still going.

## Tree snapshots (best-at-budget analysis)

The optimization managers evaluate nodes dynamically, so "which node is the
best so far" keeps changing as the budget is spent. To support budget-vs-budget
method comparison, any manager (`hgm`, `hgm_dual`, `hill_climbing`) can record a
**time series** of the whole tree. Enable it per run with an opt-in manager key:

```yaml
manager:
  type: "hgm"          # or hgm_dual, hill_climbing
  config:
    snapshot_tree: true
```

A snapshot is appended after every EXPAND/EVALUATE to
`runs/<exp>/snapshots/tree_snapshots.jsonl` — one JSON line per step holding the
budget spent, the full node roster (`node_id`, `parent_id`, `round_dir`,
`mean_utility`, `n_evals`, …) and a pointer to the current best node. This is the
same shape as the original HGM's `hgm_metadata.jsonl`. Off by default (no
behavior change, no `snapshots/` dir written).

`snapshot_eval.py` then picks the best agent at any budget level and
re-evaluates it (reusing the `evaluate.py` machinery):

```bash
# List the best agent at every recorded budget (no evaluation, no API key):
PYTHONPATH=. python3 snapshot_eval.py \
    --experiment-dir runs/<exp> --all --list

# Re-evaluate the best agent at budgets 100 and 200 on the held-out split:
PYTHONPATH=. python3 snapshot_eval.py \
    --config configs/hgm_travel.yaml \
    --experiment-dir runs/<exp> --budgets 100,200 --eval-split
```

Per-budget results are written to `runs/<exp>/snapshots/eval_at_budget_<B>.json`.

## Edit memory (optional)

A tree-global record of **what edits were attempted and what they did to the
score**. Distinct from the behavior summarizer: that describes how an agent
*behaved at runtime* along one lineage; this describes *what was changed and
whether it paid off*, across every branch. Both can run together or apart.

```yaml
edit_memory:
  type: "default"
  config:
    model: "gpt-5.4"
    reasoning_effort: "medium"
    steering: true              # inject the accumulated memory into the editor
    steering_token_budget: 48000
    verdict_threshold: 0.02     # helped/hurt boundary — see the caveat below
    min_shared: 8
    max_strategies: 30          # ceiling on the category vocabulary (code default 18; the
                                # belief configs raise it — cap-forced fits are never scored)
    max_subedits: 3             # a node may bundle several distinct changes
    setup_pass: true            # one call per run: proxy categories + recipe
```

**Off by default.** Omit the block and nothing changes: no LLM calls, no files
written, and the editor's prompt is byte-identical to before.

Cost is **one LLM call per node**, plus one per run for setup. Outcomes are
recomputed continuously but always deterministically — no LLM sits in that path.

Artifacts:

- `round_NNN/edit_memory.md` — one record per node: the sub-edits it made, each
  with a name, a two-level category, what it did and which failure it targeted;
  then the measured outcome.
- `edit_memory_registry.json` — the run-global category vocabulary. Contains
  **only categories some edit actually used**, so it is safe to show the editor.
- `edit_memory_candidates.json` — proxy categories proposed by the setup pass.
  **Tagger-only, never shown to the editor**: they are hypotheses, and letting
  the editor read them as though they were tried history biases the tree search.

Two things to know before enabling it:

- **`verdict_threshold` assumes scores on `[0,1]` where higher is better.** It
  is calibrated against a travel run; on a different scoring scale the
  helped/hurt/neutral split becomes meaningless until re-checked. The same
  threshold applied to one reference run gives "31 of 75 edits hurt" at 0.02
  and "2 hurt" at 0.10.
- **Edit memory requires the per-round `round_NNN/task_agent/` layout the
  managers write**, since it diffs a node against its parent. A manager that
  does not produce that layout silently produces no edit memory. It is
  supported by `hgm` only — `hgm_dual` raises, because it makes more than one
  editor call per node and cannot honour one-call-per-node.

### Belief mode (learnable beliefs + two-stage editor)

`steering_mode: "belief"` replaces the record dump with a **belief document**
under a predictive contract, and the two-stage editor retrieves whole records
and implementation slices on demand:

```yaml
editor:
  type: "two_stage"
  config:
    base_url: "https://api.openai.com/v1"   # pin it — see the base_url warning under Configuration
    propose_reasoning_effort: "medium"
    retrieval_char_budget: 60000
    max_retrieved_nodes: 4
    # objective: judge          # injected automatically under steering_mode "belief";
                                # "score" restores the legacy objective sentence
edit_memory:
  type: "default"
  config:
    code_record: true
    steering_mode: "belief"
    strategy_label: judge            # what strategy beliefs are scored against (below)
    analysis_min_own_evals: 16       # the judge runs after one batch of the node's OWN cases
    judge_min_evidence: moderate     # weak-evidence verdicts stay pending
    analysis_code_char_budget: 20000 # implementation view shown to the judge
    beliefs:
      enabled: true
      doc_char_cap: 40000            # HARD cap ~10k tokens; reject + one retry, never truncated
      optimize_enabled: true         # online guidance optimization
      optimize_every: 8              # step after this many newly scored predictions
      optimize_min_scored: 8
      optimize_rollback_margin: 0.02
      instruction_char_cap: 2500
```

Contract (`meta_agent/belief_contract.py`): the document is only an optional
`## Summary` (≤ 1200 chars) plus `### belief:<slug>` sections, each with
`kind` (`strategy` | `implementation`), `scope` (`strategy=<registry id>
[area=<registry id>]`), `predict` (`p=0.05..0.95`), `evidence` (with verified
citations: `[node N: improved]` — the judge's verdict —, `[node N: improved;
Δ+0.0310/12]`, `[node N: Δ-0.0117±0.1461]` — unpaired Δ ± SE — or `[node N:
unmeasured]`) and `next`. Anything else is a violation: one retry names the
violations, then the previous document is kept.

The judge (`meta_agent/edit_usage.py`, analysis v7): the per-node analysis
call is the outcome oracle. It reads the edit's implementation view (added
lines per definition, whole units), the usage counts with scorer agreement,
the runtime logs, a **paired** per-check table (shared cases, when any) and an
**unpaired** one (each side over its own cases), one row per evaluated case
(score, failed checks, which components fired with what verdict), and the
performance numbers with their standard errors. Per sub-edit it returns the
implementation verdict (`sound|unsound`) and an **effect verdict** —
`improved | no_effect | regressed | unclear` with an evidence grade
(`strong` = within-case or component counts over ≥ 8 cases, `moderate` =
per-check movement, `weak` = score Δ or code only) and the targeted checks —
rendered as `- **effect (edit N)**: improved (strong; targets: …) — reason`,
plus a node-level `- **regressions**:` line. It runs once the node has
`analysis_min_own_evals` evaluations of its own: no shared cases with the
parent are needed (a random 16-case batch shares ~4 cases with a 16-eval
parent, which is why the old `min_shared` gate left most nodes unjudged).
Records also carry `- **unpaired**: child m/n vs parent m/n · Δ ± SE` (fmt 6);
at ~16-case batches every SE is near ±0.1, so a Δ under 2×SE is noise —
readers see the judge's verdict first and the score as context.

Scoring (`meta_agent/belief_scoring.py`): when a node is expanded, the belief
covering its registry tags is pre-registered per kind in
`round_NNN/belief_prediction.json`. Once the analysis has run (requires
`analysis_mode: refresh` + `usage_tracking`), each kind pays Brier loss:
implementation beliefs predict P(sound); strategy beliefs predict
P(the judge finds the mechanism improved its target | sound) — `improved` is
y=1, `no_effect`/`regressed` y=0, `unclear` or evidence below
`judge_min_evidence` stays pending — and are scored only on sound sub-edits.
`strategy_label: delta` restores the pre-v7 rule (y = Δ ≥ threshold over
≥ `min_shared` shared cases) for ablation. A belief matched to a sub-edit is
scored on that sub-edit's verdicts, so one broken mechanism in a bundled edit
no longer vetoes the others. A node no belief covers scores at p=0.5 (loss
0.25) — unless no belief *could* have covered it (first node of a new
strategy), or its strategy id was force-fitted at the registry cap
(`- **fit**: forced …` in the record): those are retired unscored. Per-belief
calibration is written back into the document as a code-generated `- track:`
line and into the maintainer's calibration report, which in judge mode also
shows a **judge-vs-score diagnostic** on rows whose score is well measured
(≥ 16 shared cases, or |unpaired Δ| > 2×SE — a flag for nodes worth
re-reading, not a yardstick for the judge) and any **verdict changes** since
a row was scored (rows are scored once). `strategy_label: delta` (the
`…_delta.yaml` config) restores the pre-judge Δ labels for ablation; the
maintainer's, optimizer's and planner's prompts are judge-first — the judge's
verdicts, targeted checks and regressions come first, the benchmark Δ is
quoted only as context and only when well measured.

Guidance optimization (`meta_agent/belief_optimizer.py`): the maintainer's
system prompt is a fixed contract plus a learned guidance text
(`belief_instruction.md`, seeded short). Every `optimize_every` scored
predictions one LLM call critiques the misses and rewrites the guidance
(`belief_instruction_archive/vNNN.md`, `step_NNN_prompt.txt`); a version that
scores worse than an earlier one by more than `optimize_rollback_margin` is
rolled back.

Steering (`meta_agent/steering.py`): the editor's context in belief mode is one
block — the objective ("fix what the judge found"; the seed / best / parent
scores are demoted to one `Score context` line), the judge's verdict, targeted
checks and regressions for the parent, the scope rule, lineage, one judge-first
line per sibling already tried off the parent (verdict and targets, regressions,
then the score as context), and the belief document verbatim. The editor's
system prompt sentence on what to target follows (`editor.config.objective`,
auto-set to `judge` in belief mode, `score` elsewhere so the `full` mode and
the no-edit-memory control stay byte-identical). The planning pass predicts
the judge's verdict and the checks it expects to move
(`edit_prediction.json` v2: `expected_effect`, `expected_targets`;
`expected_direction` / `expected_delta` kept as legacy). No ledger, nothing
truncated; full records and implementations reach the editor only through
retrieval (`retrieval_manifest.json` v2 records what was shown and what was
omitted whole).

Artifacts: `edit_memory_beliefs.md`, `edit_memory_beliefs_state.json`,
`edit_memory_beliefs_archive/`, `edit_memory_beliefs_prompts/update_NNNN.txt`,
`belief_instruction.md`, `belief_instruction_archive/`,
`round_NNN/belief_prediction.json` (also records `coverable` — the first node
of a brand-new strategy is never charged the silence loss, since no belief
could have covered it), `round_NNN/edit_prediction.json`,
`round_NNN/retrieval_manifest.json`, `round_NNN/edit_analysis_prompt.txt` (the
judge's exact prompt), `round_NNN/edit_usage.json`, `round_NNN/edit_code.md`.
`EDIT_MEMORY_SPEC.md` specifies every format; `EDIT_MEMORY.md` is a worked
example assembled from one run by
`PYTHONPATH=. python3 study/render_edit_memory_example.py runs/<run> --out EDIT_MEMORY.md`.

Restarting without re-paying the seed evaluation: `run_seeded.py --config
<same config> --donor runs/<donor_run>/round_000` copies the donor's
`round_000` logs, replays its per-case results into node 0 and continues from
round 1 (the donor's `config.snapshot.yaml` must match on `task_agent`,
`split` and `project`; `--force` overrides).

## Train/eval split (optional)

Add a top-level `split:` block to the YAML to hold out a deterministic
fraction of cases as a validation set during optimization:

```yaml
split:
  seed: 42
  train_size: 60      # 60 train, rest = held-out eval
```

When set:
- The strategy and "best round" selection see only the train half.
- A held-out composite score is computed and printed per round, and
  persisted as `round_<NNN>/eval_score.json`. It is *not* fed back to
  the strategy.

`evaluate.py` always runs the full benchmark regardless of `split:`.

## Tests

```bash
PYTHONPATH=. python3 -m unittest tests.test_smoke
# agentic editor (policy, tools + bwrap confinement, scripted end-to-end sessions)
PYTHONPATH=. python3 -m unittest tests.test_agentic_policy tests.test_agentic_tools tests.test_agentic_editor
```

Smoke tests do not hit OpenAI — they exercise validators, the subprocess
evaluator (with stub agents), and the config loader. Always run before
committing.

## Configuration

`configs/default.yaml` is the reference. Every swappable component is selected
by name from a registry (`meta_agent/registry.py`) and uses the same
`{type, config}` shape:

```yaml
project:    "math"                  # filesystem layout: projects/math/{seed,benchmark,tools,data}/
manager:    { type: "hill_climbing",   config: { branch_policy: "best",
                                                 strategy_history_window: 5 } }
evaluator:  { type: "subprocess",      config: { wall_time_s_per_case: 120, parallelism: 1, ... } }
gatherer:   { type: "default",         config: {} }
editor:     { type: "default",         config: { model: "gpt-5.4-mini", reasoning_effort: "high", max_attempts: 2 } }
validators: [ {type: "syntax"}, {type: "signature"}, ... ]

task_agent: { model: "gpt-5.4-mini", reasoning_effort: "high" }
env:        {}                        # optional: project-specific env-var overrides
split:      { seed: 42, train_size: 60 }  # optional — see "Train/eval split" above
runs_root:  "runs"                     # optional — where run folders go (default "runs")
```

`runs_root` sets where per-experiment run folders are written — it
defaults to the repo-local `runs/` directory. Run folders are large
(per-round `task_agent/` copies + traces + per-case JSON), so on a host
with a small local disk point them at a bigger filesystem — either set
`runs_root:` in the YAML, or export the `META_AGENT_RUNS_ROOT` env var
(an explicit YAML value wins over the env var). SLURM job logs are
separate — redirect those with `SLURM_LOG_DIR`. `slurm/run_hgm.sh` sets
both to group storage automatically; `configs/default.yaml` carries a
commented sample.

The YAML is the source of truth — every component (`manager`,
`evaluator`, `editor`, `gatherer`, `validators`) must declare its
`type:` explicitly. There are no code-level defaults; a missing
component fails Pydantic validation up-front. `project: "<name>"`
resolves only the *filesystem* paths (seed dir, benchmark dir, tools
package, data dir) and auto-imports the project's `benchmark/scorer.py`
so its `@register` decorators run before the YAML's named lookup.

`gatherer.type` is always `"default"` — there's only one gatherer
implementation. Project-specific roll-ups live on the *scorer* class
(see "Project-specific feedback" under Developing). Travel example:

```yaml
project: "travel"
gatherer: { type: "default", config: {} }
evaluator:
  type: "subprocess"
  config:
    scorer: "travel_default"     # registered class with score() + aggregate()
    parallelism: 16
    # ...
```

`task_agent` settings flow into the evaluator subprocesses via `LLM_MODEL` and
`LLM_REASONING_EFFORT` env vars; the seed workflow picks them up automatically.
`META_AGENT_PROJECT=<project>` is exported so child subprocesses load only
that project's tools (via `projects.<project>.tools`).

Three `task_agent` keys are task-agent-only by construction — the evaluator
sets them on each case subprocess's environment, never globally, so the meta
agents keep their own budgets (`meta_agent/evaluator.py::_child_env`):
`temperature` (`LLM_TEMPERATURE`), `timeout_s` (`LLM_TIMEOUT_S`) and
`max_output_tokens` (`LLM_MAX_OUTPUT_TOKENS`). `timeout_s` must sit well
below `evaluator.wall_time_s_per_case` — the wrapper's own default is 3600 s,
longer than a typical 1800 s case, so a stalled request could never be
retried and killed its case (the 2026-09-07 travel run lost 9% of its case
evaluations that way); at 600 s a stall costs one retry. `max_output_tokens`
bounds a runaway generation and counts reasoning and visible output together;
the travel configs use 65536, about 1.6× the historical p99.9. The evaluator
warns at startup when `timeout_s` is not below the case limit.

`build_components` also prints `[config] warning: <component> names model …
but no base_url` for any editor / summarizer / edit_memory that names a
`model` without a `base_url` while the task agent has one
(`meta_agent/config.py::meta_base_url_warnings`): `call_llm` falls back to
the task agent's `LLM_BASE_URL`, which once sent gpt-5.4 requests to a local
vLLM server. Pin `base_url` on every meta component.

The `env:` block is the only place project-specific environment goes.
Each `key: value` is exported with `os.environ[key] = value` before the
evaluator spawns child subprocesses; the children inherit it. The
framework knows zero project-specific keys — projects own their own
defaults inside their tools (e.g. travel's `_csv.database_root()` falls
back to `projects/travel/data/database_en` when `TRAVEL_DATABASE_ROOT`
isn't set; you only need an `env:` override if your data lives
elsewhere).

## Developing

### Add a new manager (changes the optimization regime)

The manager owns the optimization regime end-to-end: bootstrapping round
0, deciding what to branch from, calling the editor, evaluator, and
gatherer, and deciding when to stop. It does **not** write code or make a
"propose a strategy" LLM call — the editor's single self-improvement call
does the diagnosis and the rewrite; the manager just selects and hands the
editor an optional steering `context` string. `HillClimbingManager`
(linear) and `HGMManager` (tree search) are the references — do whatever
fits your regime (random search, beam search, genetic, etc.).

```python
# meta_agent/managers/random_search.py
from meta_agent.registry import register

@register("manager", "random_search")
class RandomSearchManager:
    def __init__(self, *, sample_corpus: str): ...
    def evolve(self, editor, evaluator, gatherer, seed_dir, benchmark_dir,
               experiment_dir, max_rounds, score_target,
               train_case_ids=None, eval_case_ids=None):
        # own the entire round loop; write per-round folders matching the
        # disk layout contract (see meta_agent/managers/__init__.py for
        # the EvolutionManager Protocol). `round_NNN/task_agent/` is the
        # load-bearing part: the behavior summarizer and edit memory both
        # diff a node against its parent through it, and silently produce
        # nothing for a manager that does not write it.
        # Accept (and ignore, if unused) `summarizer=` and `edit_memory=`:
        # main_loop passes both unconditionally.
        ...
```

YAML:
```yaml
manager: { type: "random_search", config: { sample_corpus: "..." } }
```

Make sure your module is imported by
`meta_agent/config.py::_ensure_builtins_loaded` (or list it under
`plugins:` in the YAML).

### Add a new project

A project is a directory under `projects/` that bundles every asset
specific to one task agent. Layout:

```
projects/<name>/
├── seed/
│   ├── workflow.py         # def run_task(task: Task) -> AgentOutput
│   ├── tool_wrapper.py     # ToolWrapper(execute, get_schema)
│   ├── tools_schema.json   # OpenAI / Anthropic tool-schema format both accepted
│   └── mutable_tools/
│       └── __init__.py     # empty; editor may add files here
├── benchmark/
│   ├── cases.jsonl         # {"id": "...", "input": "...", optional "context", "env", ...}
│   └── scorer.py           # def score(case, agent_output) -> {score, passed, details}
├── tools/
│   ├── __init__.py         # imports each sub-module so register_tool runs
│   └── *.py                # one immutable tool per file
└── data/                   # optional; per-sample CSVs etc., gitignored
```

`Task` and `AgentOutput` are imported from `platform_core.runner`. A
`Task` carries `description: str`, `case_id: str`, and a free-form
`context: dict[str, Any]` that benchmarks can populate per case (default
`{}`). An `AgentOutput` wraps the `result` (whatever the scorer
consumes — usually a string) plus optional `metadata: dict` for the
agent's own annotations.

`agent_output` reaches the scorer as an `AgentOutput` instance — read
`.result` for the agent's primary payload.

Pick a project from a YAML config:

```yaml
project: "<name>"
```

Standalone debug of a single case (no evaluator needed):

```bash
source /users/n.tzou/api.sh
python -m platform_core.runner \
    --agent-dir projects/<name>/seed \
    --benchmark projects/<name>/benchmark \
    --case-id 0
```

The seed must satisfy all default validators on its own. Run a smoke
test by pointing a YAML at the project and running `main_loop.py` with
`loop.max_rounds: 0`.

### Add project-specific feedback (optional)

If your project's scorer attaches structured roll-ups to `details` that
the optimizer should see — e.g. counts of failed checks, per-dimension
scores, "no plan emitted" rates — add an `aggregate()` method to your
scorer class. The framework's `DefaultFeedbackGatherer` calls it once
per round and lands the result on `AgentFeedback.project_metrics`.

```python
# projects/<name>/benchmark/scorer.py
from meta_agent.registry import register


@register("scorer", "<name>_default")
class MyScorer:
    def score(self, case, agent_output) -> dict:
        # Per-case: returns {score, passed, details}.
        # Whatever you put on `details` is what aggregate() will see
        # later as case.details for that case.
        ...

    def aggregate(self, per_case, trace_events) -> dict:
        # Round-level: walk per_case (list of CaseResult), read the
        # `details` keys you wrote in score(), and return a flat dict
        # of name → scalar / list-of-(name,count) / dict-of-name-to-number.
        # The framework's prompt renderers walk this generically.
        return {
            "no_plan_rate": ...,
            "top_failed_checks": [(name, count), ...],
            "dimension_means": {dim: mean, ...},
        }
```

Then point your YAML at it:

```yaml
gatherer: { type: "default", config: {} }
evaluator:
  type: "subprocess"
  config:
    scorer: "<name>_default"
```

`projects/travel/benchmark/scorer.py::TravelCompositeScorer` is the
live reference (per-case `score()` + round-level `aggregate()`).
Math's scorer has no `aggregate()` method, so its `project_metrics`
is `{}` — trace stats and tool-error rate are still surfaced for
free by the framework gatherer.

### Add a new immutable tool to a project

```python
# projects/<name>/tools/my_tool.py
from platform_core.tools import register_tool

NAME = "my_tool"
SCHEMA = {
    "name": NAME,
    "description": "...",
    "input_schema": {"type": "object", "properties": {...}, "required": [...]}
}

def run(**kwargs) -> str:
    ...

register_tool(NAME, SCHEMA, run)
```

Then add `from . import my_tool` to `projects/<name>/tools/__init__.py`
so the registration runs when the project is loaded.

## How the meta-agent and the task agent talk to the LLM

Both go through `platform_core.llm_wrapper.call_llm`. It uses OpenAI's
**Responses API** (`client.responses.create`), accepts tool schemas in any of
three shapes (Responses-API, Chat-Completions, Anthropic), and emits trace
events that the feedback gatherer reads back. With reasoning effort set the
wrapper sends `reasoning={"effort": ...}`; a temperature rides alongside it
only when sourced from the `LLM_TEMPERATURE` env var (which the evaluator
sets per case subprocess from `task_agent.temperature`, default 0.2 =
low-variance sampling) and the model tolerates the combination — OpenAI first-party
reasoning models (gpt-5 family, o-series) always get `reasoning` without
`temperature`.

Defaults are sourced from `LLM_MODEL` and `LLM_REASONING_EFFORT` env vars so
the same workflow code runs in the meta-agent and in evaluator subprocesses
without threading config through.

## Constraints the editor cannot violate

The agent editor may only:
- Modify `workflow.py`, `tool_wrapper.py`, `tools_schema.json` in the round folder.
- Add or modify `*.py` files under `mutable_tools/`.

It may **not**:
- Touch any file under `platform_core/`.
- Change `run_task`'s signature (`def run_task(task)` — one positional
  arg named `task`).
- Import any `platform_core.*` module other than `platform_core.llm_wrapper`
  or `platform_core.runner` from `workflow.py`/`tool_wrapper.py`, or
  anything other than `platform_core.tools` from `mutable_tools/*.py`.

These are enforced by seven validators that run before evaluation —
six static (AST/regex/byte-comparison) plus one `load_test` validator
that spawns a subprocess to actually import the agent's mutable
modules (catches `NameError`, `ImportError`, and any exception raised
at module load that the static checks can't see). A violation
short-circuits the round, the eval split is skipped (saves compute),
and the validator errors land in the next round's `feedback.edit_errors`
where the strategy and editor LLMs can see them.
