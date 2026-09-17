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
PYTHONPATH=. python3 main_loop.py --config configs/hgm_math.yaml
```

This evolves the math project's seed against its benchmark with the `hgm`
manager, the `subprocess` evaluator and the **agentic editor** — a coding
agent (bash + editor tools in a sandbox) that diagnoses the parent node from
its evidence on disk and edits the agent's code in place (see "Optimization
manager" and "Agentic editor" below).

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
│   ├── agentic/                  # the editor session: transcript.jsonl, session.json, scratch/
│   └── ...
└── ...
```

## Optimization manager

The `manager` chosen in the YAML decides the search regime. One ships:

- **`hgm`** — a **Huxley-Gödel-Machine** tree search (arXiv 2510.21614):
  keeps a *tree* of agents, decouples expansion from evaluation under an
  adaptive schedule (or pairs them — see `expand_eval_size` below), picks
  which node to expand by Thompson sampling over *clade metaproductivity*,
  and counts its budget in agent-task evaluations (`eval_budget`) rather than
  rounds. `configs/hgm_{math,travel,shopping}.yaml` wire it up. Before the
  final pick it re-evaluates the top finalists on the full train split so a
  thinly-evaluated fluke can't win.

A self-modification is **one editor call**: the manager selects the parent
node; the editor's session diagnoses the agent *and* edits its code, ending
with a `submit_self_improvement` call whose summary becomes the node's
`EvolutionStrategy` (logged to `strategy.json`). There is no separate
"propose a strategy" LLM call. The manager also builds a cheap steering
`context` string (lineage, scores, sibling goals), which the agentic editor
only includes when `include_manager_context: true`.

### What the editor sees (information gathering)

To diagnose well, the editor can read, in addition to the agent's own code:

- **Example-driven failure analysis** — the feedback gatherer turns each node's
  per-case results into a compact report: the top recurring failure categories
  (from the project's `categorize_errors`), a *diverse* pair of representative
  cases per category (a near-miss + a severe failure) shown as **query → agent
  plan → what failed**, and the hardest (lowest-scoring) cases. This is generic:
  the gatherer reads only a contract (`details["query"]`, `details["raw_result"]`,
  score/passed/error) plus the project categorizer's output — all domain parsing
  stays in the project folder. Stored on `AgentFeedback.failure_report`.
  The gatherer writes it (with the raw metrics) to each node's `feedback.json`,
  which is the first thing the agentic editor is told to read.
- **Tool implementations** (`projects/<p>/tools/*.py`) and a **database schema**
  (`projects/<p>/db_schema.md`) — so the editor understands what each tool does
  and what the data looks like when generating/modifying tool calls.

Ground-truth data (`data/`, `cases.jsonl`, validation files) and the scoring
code are **never** exposed (the top-level `eval_visibility` key is accepted for
compatibility but ignored by the agentic editor). To enable the failure report,
set the gatherer's `error_categorizer` to the project's categorizer
(`module:func`); without it the report degrades to hardest-cases-only.

To run HGM:

```bash
# Locally (math is a fast smoke target)
PYTHONPATH=. python3 main_loop.py --config configs/hgm_math.yaml

# On SLURM — wrapper with HGM-sized resource defaults (see slurm/README.md)
slurm/run_hgm.sh travel        # configs/hgm_travel.yaml
slurm/run_hgm.sh shopping      # configs/hgm_shopping.yaml
```

### Agentic editor

`editor.type: "agentic"` (the only editor) is an HGM-seed-style **coding
agent**: a tool-use session (one `call_llm` per iteration, all tool calls in a
response processed) that works directly on the round's `task_agent/` copy and
ends with a structured `submit_self_improvement` call carrying only a summary
— the edits are already on disk. It is **one agentic flow**: no proposal
stage, no feedback digest or steering context in the prompt; the agent reads
the parent node's evidence from disk itself. No docker.

Prompts: one fixed system prompt (`meta_agent/agentic/session.py`) and one
instruction message per session. Paths are expressed through roots —
`$RUN_DIR`, `$NODE_DIR`, `$PARENT_DIR`, `$REPO_DIR` — that are environment
variables in every bash call and are expanded by the editor tool, so no
experiment path ever appears in the prompt. The prompt never mentions memory
of any kind.

**Read scope** (`editor.config.read_scope`) decides how far the agent may look:

- `"run"` (default) — the whole run directory: every node's code, evidence
  and logs, so the agent can compare siblings and ancestors itself.
- `"parent"` — only `$PARENT_DIR` (the parent node's round dir) and its own
  `$NODE_DIR`. There is no `$RUN_DIR` root at all: the prompt, the tool
  descriptions, the bash environment and the sandbox binds all omit it, so
  sibling and ancestor nodes are neither named nor reachable.

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
| read-only (`read_scope: run`) | the whole run directory — every `round_NNN/` (`hgm_node.json`, `strategy.json`, `feedback.json`, `eval_result.json`, `logs/`, `agentic/`, `task_agent/`) and run-level files such as `tree_snapshots.jsonl` — plus `platform_core/`, `projects/<p>/tools/`, `projects/<p>/db_schema.md` |
| read-only (`read_scope: parent`) | `$PARENT_DIR/` and `$NODE_DIR/` only, plus the same `platform_core/` and project tool/schema paths |
| never | `projects/<p>/benchmark/` (scorer, `_eval/`, cases.jsonl), `projects/<p>/data/`, the categorizer, `meta_agent/`, other runs, and the run's `edit_memory/` directory (masked with a tmpfs; only the with-memory arm gets one file back, see "Edit memory"). `eval_visibility` is ignored by this editor. |

Because bwrap binds are live views, files the framework writes to the bound
directories later (new rounds under `run` scope) are visible without any
rebinding. The task agent can *not* be run on cases inside the sandbox (no
model access, no database) — by design; checks are `validate`, import checks
and scratch scripts. Budget reminders are injected at a third and at the
halfway point of `max_llm_calls` (while nothing has been edited) and in the
final stretch.

Per-session artifacts: `round_NNN/agentic/transcript.jsonl` (every LLM call,
tool call with input/result, validation round) and `agentic/session.json`
(`end_reason`, `n_llm_calls`, per-tool counts, `validation_rounds`,
`sandbox_mode`, `read_scope`, roots, token usage). With
`META_AGENT_VERBOSE=1` the exact system prompt, instruction and final message
history land in `verbose/`.

```yaml
editor:
  type: "agentic"
  config:
    model: "deepseek/deepseek-v4-pro-0813"   # meta-model; any Responses-API endpoint works
    reasoning_effort: "low"
    base_url: "https://openrouter.ai/api/v1"
    api_key_env: "OpenRouter_API_KEY"        # key for THIS editor's calls (source api.sh);
                                             # the task agent keeps OPENAI_API_KEY
    llm_timeout_s: 600                       # per-request cap so a stall can't eat the session
    max_attempts: 3            # submission / validation rounds
    max_llm_calls: 150         # tool-loop iterations (history re-sent each call)
    timeout_s: 5400            # wall-clock per session; stops at 90%
    bash_timeout_s: 120
    max_tool_output_chars: 20000
    max_view_chars: 40000
    sandbox: "auto"            # auto | bwrap | none
    include_manager_context: false
    read_scope: "run"          # run | parent (see "Read scope" above)
```

**Provider pin.** Every block that names `deepseek/deepseek-v4-pro-0813` on
OpenRouter — editor, `edit_memory`, and the task agent — carries
`extra_body: {provider: {order: ["StreamLake", "Alibaba"], allow_fallbacks: false}}`:
OpenRouter tries the listed providers in that order and never routes to any
provider outside the list (a single provider's tokens-per-minute cap took a
run down on 2026-09-16; Baidu was then dropped after a 40-case provider A/B —
StreamLake 0.666, Alibaba 0.613, Baidu 0.580 with 135 rate-limit retries; the
generation record of each call names the provider that served it). `extra_body` is merged verbatim into
the request body: for the meta agents it is a per-call `call_llm` kwarg; for
the task agent it is exported child-only as `LLM_EXTRA_BODY` (JSON) like the
other `task_agent` knobs. Keep the pin on any new config that uses this model.

`api_key_env` (and `timeout_s`) are per-call parameters of `call_llm`, so the
editor can sit on a second provider while everything else in the run keeps
the global `OPENAI_API_KEY`. To move the *whole* run (task-agent subprocesses
included) to another provider's key, export `LLM_API_KEY_ENV: "<VAR>"` from the
YAML `env:` block — the run-wide default for `api_key_env`. The
`hgm_travel_{100,1000}_dsv4pro_agentic_no_editmem.yaml` configs do this
(DeepSeek V4 Pro for meta and task agent, task reasoning `none`,
`verbose: true` so every session's full message history is kept,
`seed_round_dir` to reuse a previous run's seed pre-evaluation).

```bash
source /groups/AIC-MV/sudipta.paul/code/random/api.sh   # exports OpenRouter_API_KEY
PYTHONPATH=. META_AGENT_VERBOSE=1 python3 main_loop.py --config configs/hgm_travel_smoke_agentic.yaml
PYTHONPATH=. python3 main_loop.py --config configs/hgm_travel_1000_dsv4pro_agentic_no_editmem.yaml
```

#### Expansion-paired evaluation (`manager.config.expand_eval_size`)

By default (0) an expansion only creates the child; whether and when it is
evaluated is the bandit's later decision, so some nodes end a run with no
evaluation at all. `expand_eval_size: 16` evaluates every freshly expanded
child on 16 random train cases immediately (charged to `eval_budget` and to
the widening counter) and refuses to expand when the paired batch is
unaffordable. At 1000 evals with `alpha 0.5` this leaves the tree width
unchanged (32 nodes / 63 batches) but makes 32 of the batches mandatory child
evaluations, so every editor session is measured; the bandit keeps the other
31 batches. The agentic configs enable it; set it to 0 for the reference's
fully decoupled behaviour. The `init_expansions` always branch from the
seed root (node 0) — with paired evaluation a fresh child would otherwise
be expandable immediately and could win the clade bandit for the next
init expansion; the regular loop's parent choice is unchanged.

## Edit memory (optional)

`edit_memory: {type: agentic}` adds a learned memory of *what edits were
tried and what the evidence says about them* to the HGM loop, written by
agentic curators and read by the editor on a bandit-chosen fraction of
expansions. Absent, nothing changes (prompts are pinned byte for byte by
`tests/test_agentic_golden.py`).

```
C_{i+1} <- Expand(C_i, L_i, B_j)  or  Expand(C_i, L_i)     bandit over the two arms
Z       <- Curation(last m nodes and their logs)          agentic curator
B_{j+1} <- Generation(Z, B_j, I_k)                        one LLM call
Q       <- Curation(tau_meta, tau_task+/-, B_j, I_k)      agentic curator
I_{k+1} <- InstructionUpdate(Q, I_k)                      one LLM call
```

- **Cadence.** Every `window_size` (m) successful, paired-evaluated
  expansions the **memory curator** — an agentic session with the same
  bash/editor tools as the editor, confined to the window's nodes, their
  parents and `edit_memory/` — reads diffs, editor transcripts, per-case
  outcomes and traces itself and writes `window_NNN/curation.md` (Z):
  per node *what changed / intent / editor process / what the evaluation
  shows / shortcomings (strategy vs implementation) / usefulness verdict*,
  cross-node patterns, and a gradient against the current memory. One
  call then rewrites the memory minimally: `edit_memory_vNNN.md` with
  *ranked edits / usefulness / strategy vs implementation / guidance*.
  Nothing is pre-digested for the curators; node mean scores appear as
  context only and are never passed to the generator.
- **Checks never discard content.** A curator's document is checked on
  submit (per-node sections, cross-window + gradient sections; the five Q
  sections for the audit); a failing check is quoted back and the curator
  may fix and resubmit, but on its last attempt — and at wrap-up — the
  document is accepted as it is, with any still-missing sections inserted
  as `(not provided by the curator)` placeholders (`salvaged` in
  `session.json`). The memory and the addendum are checked for structure,
  size (`memory_max_chars` 40k, `instruction_addendum_max_chars` 10k) and
  score-prediction language; a failing draft is regenerated once with the
  findings quoted back, and the final draft is used either way, findings
  recorded in `memory_call.json` / `update_call.json` and `state.json`.
  Only an LLM failure keeps the previous version.
- **Arms.** Before the first memory every expansion is arm `none`. After
  it, a Thompson bandit picks `with` or `without` per expansion from the
  two arms' pooled HGM tallies (`Beta(Σ n_success + prior, Σ n_failure +
  prior)` over every node generated under the arm; pre-memory nodes count
  for `without`; edit-failed nodes add no mass), after `arm_min_pulls`
  forced pulls of `with`. `selection: always | never` are ablations
  (`never` = the memory is written and nobody reads it). The arm and
  memory version are recorded in `hgm_node.json` and the tree snapshots.
- **What the with arm sees.** Exactly one extra file, `$EDIT_MEMORY_FILE`
  (the versioned memory), one extra procedure step, and — because the
  memory cites node ids — the **whole run readable** regardless of
  `read_scope`; the configured scope governs the without arm only. The
  `edit_memory/` directory itself is masked for both arms.
- **Instruction.** Every `instruction_every` (n) memory versions — checked
  when a window closes, *before* that window's memory is generated — if any
  node was expanded with the memory since the last update, the
  **instruction curator** audits those nodes (did the editor read and use
  the memory — τ_meta; did the guided mechanisms fire and help at task-agent
  runtime — τ_task±; is the representation serving the editor) and writes
  `instruction_update_NNN/q.md`; one call then revises the generator's
  addendum (`instruction_vNNN.md`, size-capped, never the fixed core), and
  the window's memory is generated under the revised addendum. With
  `instruction_every: 1`: window 1 → B₁ (no audit possible yet); window 2 →
  audit of the nodes that used B₁ → I₁ → B₂ under I₁; and so on. This is a
  nested TextGrad: Z and Q are textual gradients, the two calls are
  minimal-change optimizer steps.
- **Requires** `manager.config.expand_eval_size > 0` (every window node
  needs per-case evidence). Run dir: `edit_memory/{state.json,
  edit_memory.md, edit_memory_vNNN.md, instruction.md, instruction_vNNN.md,
  window_NNN/, instruction_update_NNN/}`.

```yaml
edit_memory:
  type: "agentic"
  config:
    window_size: 4              # m successful expansions per memory version
    instruction_every: 2        # n memory versions per instruction update
    selection: "bandit"         # bandit | always | never
    arm_min_pulls: 2
    beta_prior: 1.0
    seed: 42
    model: "deepseek/deepseek-v4-pro-0813"
    reasoning_effort: "low"
    base_url: "https://openrouter.ai/api/v1"
    api_key_env: "OpenRouter_API_KEY"
    llm_timeout_s: 600
    curator: { max_llm_calls: 60, timeout_s: 2400, bash_timeout_s: 120, sandbox: "auto", max_attempts: 2 }
    memory_max_chars: 40000
    instruction_addendum_max_chars: 10000
```

`configs/hgm_travel_1000_dsv4pro_agentic_editmem.yaml` is the no-editmem
1000-eval config plus this block; `configs/hgm_travel_smoke_agentic_editmem.yaml`
exercises a memory, a with-arm expansion and an instruction update in one
smoke run. The design page `EDIT_MEMORY_ARCHITECTURE.html` has the diagrams.

## Standalone evaluation

`evaluate.py` runs a specific task_agent (the seed, a saved round, or any
hand-edited copy) against the **full** benchmark — ignoring any `split:`
block in the YAML. Use this to measure baselines and to compare any round
of an optimization run on the held-out evaluator.

```bash
# Score the unedited seed
PYTHONPATH=. python3 evaluate.py \
    --config configs/hgm_travel.yaml \
    --agent projects/travel/seed

# Score round 3 of an optimization run
PYTHONPATH=. python3 evaluate.py \
    --config configs/hgm_travel.yaml \
    --agent runs/20260506_140000_travel_default/round_003/task_agent
```

Results land in `runs/eval_<stamp>_<agent_basename>/round_eval/`. Per-case
results are persisted to `logs/case_<id>.json` as each case finishes, so
you can inspect partial scores while the run is still going.

## Tree snapshots (best-at-budget analysis)

The optimization managers evaluate nodes dynamically, so "which node is the
best so far" keeps changing as the budget is spent. To support budget-vs-budget
method comparison, the manager can record a **time series** of the whole tree.
Enable it per run with an opt-in manager key:

```yaml
manager:
  type: "hgm"
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

## Dashboard

`hgm_dashboard.py` is a Streamlit viewer for live or finished runs under
`runs/`. It needs `streamlit` + `pandas` (listed at the bottom of
`requirements.txt`; the `hgm-dual` conda env does not have them, the base
miniconda python does):

```bash
# from the base miniconda python (has streamlit), or `pip install streamlit pandas` into hgm-dual first
/users/sudipta.paul/miniconda3/bin/python3 -m streamlit run hgm_dashboard.py \
    --server.port 8502 --server.address 0.0.0.0 --server.headless true
```

Three views (sidebar radio; the experiment picker lists every
`runs/<exp>/` that has a `config.snapshot.yaml` and at least one `round_*/`,
newest first, defaulting to the newest live one — `runs/eval_*` dirs from
`evaluate.py` are ignored):

- **Run** — header + budget bar, a diagnostics panel (edit-failed nodes,
  editor sessions that ended without submitting, per-case errors/timeouts,
  crashed evals, zero-mean nodes), the search tree (fill = train mean;
  border = memory arm when the edit-memory layer is on; double border =
  current best), a nodes table with diff line counts and editor-session
  stats, and a per-round drill-down: **Strategy**, **Agentic session** (the
  editor's `agentic/transcript.jsonl` grouped by LLM call — assistant text,
  every bash/editor/validate/submit call with its result, editor calls shown
  as diffs, token curve; prompts from `verbose/` when present),
  **Evaluation** (per-case table with dimension columns, dimension means,
  hard-constraint and failed-check counts, case inspector), **Diff vs
  parent** (mutable surface only), **Feedback**.
- **Edit memory** (only for runs with an `edit_memory/` dir) — layer state
  and arm pulls, per-arm train-mean summary, event log, memory / instruction
  version browser with diff-vs-previous, each curation window (nodes,
  `curation.md`, generator-call attempts, the curator's own transcript) and
  each instruction update (`q.md`).
- **Compare** — pick several runs: best-train-mean-vs-budget step curves
  from `snapshots/tree_snapshots.jsonl` (needs `snapshot_tree: true`), with
  held-out `eval_at_budget_*.json` points overlaid when present; a summary
  table; the best node's dimension means side by side.

The data layer is `meta_agent/run_inspect.py` (rounds, diffs, diagnostics,
snapshots, cross-run summary) and `meta_agent/run_inspect_agentic.py`
(transcripts, sessions, `edit_memory/`) — pure Python, no Streamlit import,
reusable from analysis scripts, tested in `tests/test_run_inspect*.py`. The
dashboard never reads `logs/trace.jsonl` or `feedback.json`'s `log_excerpt`;
everything else is cached per file mtime, so auto-refresh on a live run only
re-parses what changed. Times are shown in America/Los_Angeles.

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
# agentic editor (policy, tools + bwrap confinement, scripted end-to-end sessions, golden prompts)
PYTHONPATH=. python3 -m unittest tests.test_agentic_policy tests.test_agentic_tools tests.test_agentic_editor tests.test_agentic_golden
# edit-memory layer (bandit, cadence, curators, validators; scripted LLM)
PYTHONPATH=. python3 -m unittest tests.test_edit_memory_layer tests.test_edit_memory_curator tests.test_edit_memory_generator
```

Smoke tests do not hit OpenAI — they exercise validators, the subprocess
evaluator (with stub agents), and the config loader. Always run before
committing.

## Configuration

`configs/hgm_math.yaml` is the reference. Every swappable component is selected
by name from a registry (`meta_agent/registry.py`) and uses the same
`{type, config}` shape:

```yaml
project:    "math"                  # filesystem layout: projects/math/{seed,benchmark,tools,data}/
manager:    { type: "hgm",             config: { eval_budget: 400, init_expansions: 5, alpha: 0.6, ... } }
evaluator:  { type: "subprocess",      config: { wall_time_s_per_case: 120, parallelism: 1, ... } }
gatherer:   { type: "default",         config: {} }
editor:     { type: "agentic",         config: { model: "gpt-5.4-mini", reasoning_effort: "high",
                                                 max_llm_calls: 40, sandbox: "auto", read_scope: "run", ... } }
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
both to group storage automatically.

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
but no base_url` for an editor that names a `model` without a `base_url`
while the task agent has one
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
"propose a strategy" LLM call — the editor's session does the diagnosis and
the edit; the manager just selects and hands the editor an optional steering
`context` string. `HGMManager` (tree search) is the reference — do whatever
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
        # load-bearing part: the editor copies the parent's and diffs the
        # child's against it.
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
