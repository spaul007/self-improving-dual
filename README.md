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
│   └── feedback.json
├── round_001/
│   ├── task_agent/               # editor's mutation of round_000
│   └── ...
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
        # the EvolutionManager Protocol).
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
events that the feedback gatherer reads back. Reasoning models (gpt-5 family)
get `reasoning={"effort": ...}` instead of `temperature`.

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
