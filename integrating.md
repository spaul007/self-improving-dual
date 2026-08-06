# Integrating a vendored agent into the framework

A step-by-step playbook, distilled from two real integrations (`db_mas`,
`math_mas` — see `projects/db_mas/`, `projects/math_mas/`, and the plan file
`/users/v.kulkarni1/.claude/plans/let-us-try-to-tender-fiddle.md` Addenda 3-8
for the full narrative/rationale behind each decision below). Follow this
when wiring up a new raw multi-agent system as an HGM task-agent project.

## Ground rule, stated up front

**Never restructure or flatten the vendor's own code to fit the framework.**
If the framework's assumptions don't fit a project's real shape, widen the
framework (a new config knob, a new generic mechanism) — don't distort a
working, independently-meaningful codebase to satisfy a narrow assumption
baked into the editor. Every mechanism below (`mutable_exclude`,
`seed_dir_name`, `tool_source_dirs`) exists because an earlier project's real
structure didn't fit the original hardcoded 3-file convention, and the fix
each time was to generalize the framework, not to reshape the project.

---

## Step 0 — Understand the raw implementation before touching anything

Read (don't assume) the vendor's own code and answer:
- Entry point: is there one `run_task(item) -> result` function? Sync or
  async? What's the exact input/output shape?
- Agent topology: fixed roles, or dynamic? Where do per-agent prompts live?
- LLM client: OpenAI-compatible? What env vars does it read (its own
  convention, e.g. `MAS_MODEL`/`MAS_BASE_URL` — almost never the framework's
  `LLM_MODEL`/`LLM_BASE_URL`)?
- Tools: does it use real function-calling, or are "tools" just plain
  Python helpers?
- Benchmark data: where does it live, what's the per-record schema, how many
  records?
- Scoring: deterministic, LLM-judge, or both? Where does the logic live —
  reuse it verbatim, never reimplement it.
- Infra dependency: does it need Docker/a database/other real external
  state per task (like db_mas's Postgres), or is it pure in-process LLM
  calls (like math_mas)? This drives evaluator timeout/memory/parallelism
  tuning later.
- Does the project already have its own mutable/immutable convention (e.g.
  `tools/immutable/` vs `tools/mutable/`, a `role`/`task` split in a prompt
  config)? If so, it maps directly onto `mutable_exclude` later — don't
  invent a different boundary.

## Step 1 — Vendor the code, verbatim

Copy the raw implementation into `projects/<project>/<seed_dir_name>/`
(straight copy, minus `results/` and `__pycache__`). Two choices:
- If the project has no natural single top folder, use a `seed/` wrapper
  (the framework's default `seed_dir_name`).
- If the project's own top-level folder can double as the seed dir directly
  (most cases), set `seed_dir_name` to that folder's actual name in the
  YAML config and skip the wrapper entirely — there's no reason for an
  empty pass-through folder.

Nothing inside this folder should differ from the original, except the one
new file in the next step.

## Step 2 — The framework-mandated entry point: `workflow.py`

Add `workflow.py` at the **top level of the vendored folder** (a sibling of
the project's own main orchestration file). It must expose exactly:

```python
def run_task(task: Task) -> AgentOutput
```

Rules, learned the hard way (see plan Addenda 3/5/6 for the two designs that
were tried and abandoned first):

- **Make it fully self-contained.** Import the project's own orchestration
  module directly and translate `Task`/`AgentOutput` inline — do **not**
  route through a separate `adapter/task_translation.py`. An earlier
  design tried exactly that (plus a read-only "reference" for the editor,
  plus a renamed-file + configurable-validator variant) and all were
  abandoned once it became clear the editor should simply never see this
  file at all (see Step 4) — so there's no benefit to indirection here.
- **If the raw orchestration is async, bridge it**: `asyncio.run(mas_workflow.run_task(item))`.
  Confirmed the framework's own invocation (`platform_core.runner._invoke_workflow`)
  is purely synchronous — it never runs its own event loop.
- **Never let ground truth reach the agent.** When building the item/task
  dict from `Task.context`, do not set any answer/label field the raw
  agent's own signature might accept for bookkeeping — leave it unset so it
  comes back empty rather than leaking the real answer through
  `Task.context` → `AgentOutput` → logs.
- This file must land on the editor's **exclude list** (Step 5) — it's
  fixed framework glue, not agent behavior, and it already satisfies the
  signature validator's AST check on its own.

## Step 3 — The benchmark contract: `benchmark/`

Lives as a **project-level sibling** of the seed dir — confirmed structurally
unreachable from the editor (`_copy_workspace`/`copy_seed_to` only ever copy
the seed dir's own contents into `round_NNN/task_agent/`), so nothing here
needs runtime protection, only naming-collision defense-in-depth (see
`mutable_exclude`'s `benchmark/` entry below).

- **`generate_cases.py`** — one-off generator reading the project's real
  benchmark data (external file or the vendored copy's own `data/`) and
  emitting `cases.jsonl`, one JSON object per line:
  ```json
  {"id": "...", "input": "<prose fed to Task.description>",
   "context": {"...": "<dict fed to Task.context, no ground truth>"},
   "meta_info": {"...": "<ground truth + any scorer-only metadata>"}}
  ```
  Ground truth goes **only** in `meta_info`. Confirmed via
  `meta_agent/evaluator.py`: `Task.description = case["input"]`,
  `Task.context = case["context"]` — `meta_info` is never passed to the
  agent, only read back by the scorer.
- **`scorer.py`** — a thin, framework-mandated-path shim (this exact
  filename/location is required — `meta_agent.evaluator`/
  `meta_agent.config._load_project_components` import it by convention).
  For a fresh project, it's fine for this file to hold the real scorer
  class directly (see `adapter/` below for when to split it out).

## Step 4 — The adapter/glue package: `adapter/`

Also a project-level sibling of the seed dir (never copied per round —
permanently safe from the editor without needing an exclude entry). Holds:

- **`<project>_path.py`** — single source of truth locating the vendored
  seed dir's absolute path, resolved relative to *this file's own*
  `__file__` (not the seed dir's, since only the seed dir gets copied
  per-round), overridable via an env var. Used so the scorer can import the
  raw project's own scoring helper functions unmodified.
- **`scorer_impl.py`** — a registered class,
  `@register("scorer", "<project>_default")`. **Reuse the vendor's own
  scoring function/logic unchanged** — never reimplement it. Implements:
  - `.score(case, agent_output) -> {"score": ..., "passed": ..., "details": {...}}`
  - optionally `.aggregate(per_case, trace_events) -> dict` — round-level
    rollups (mean score, project-specific breakdowns). Picked up
    automatically by `DefaultFeedbackGatherer._project_metrics` as long as
    the scorer is a registered **class instance**, not a bare function.
- **`gatherer_impl.py`** — `@register("gatherer", "<project>_default")`,
  a passthrough subclass of `DefaultFeedbackGatherer`. Only override
  `pass_threshold` (via a small `gatherer_config.json` or YAML config) if
  the project's `score` is continuous and structurally can't reach `1.0`
  (e.g. F1) — leave it as a pure no-op subclass if score is strictly binary.
- **`summarizer_impl.py`** — `@register("summarizer", "<project>_default")`,
  a subclass of `BehaviorSummarizer` overriding **only**
  `_extract_failure_hint` to read the project's own scorer-specific
  `details` keys. The base class's generic hint-sniffing (`failed_checks`,
  `missing_*`/`extra_*`, a bare `error` string) won't match any
  project-specific key names, so without this override the per-case line in
  every behavior memo comes out empty.
- **Do not add a `task_translation.py` here.** That pattern is superseded
  (Step 2) — kept in `db_mas` only as dead code the user explicitly asked
  not to delete, not something to replicate in a new project.

## Step 5 — Config: `configs/hgm_<project>_sanity.yaml`

- `project`, `seed_dir_name` (skip the `seed/` wrapper when the vendored
  folder can serve directly, per Step 1).
- `mutable_exclude` (exclude-list, not the legacy 3-file include-list):
  - the project's own immutable-tool folder(s) (from Step 0's audit)
  - the scoring code the adapter imports directly — a **measurement-integrity**
    boundary: if the editor could rewrite this, it could redefine its own
    grading function
  - `workflow.py` — the framework glue from Step 2; excluded so the editor
    never even sees it
  - `benchmark/` — defense-in-depth; already structurally unreachable
    (Step 3), named explicitly anyway
- `tool_source_dirs` — directories (relative to the seed dir) to scan
  recursively for tool-defining `.py` files, when the project's tool code
  isn't one `projects/<name>/tools/` package.
- `manager: hgm`, with a **train_size large enough that an all-zero seed is
  improbable**. `hgm.py::_expandable()` only allows expansion from nodes
  with `mean_utility > 0` (faithful to the reference HGM algorithm) — a
  too-small sample (e.g. 2 cases) has a real chance of landing the seed at
  exactly 0, which silently prevents the editor from ever being called, no
  error raised. 6 cases worked well for a ~5-agent-accuracy-range model;
  size it to the model's real observed accuracy if known.
- `evaluator` — timeout/memory/parallelism tuned to whether the project
  needs real infra per case (Docker: minutes, generous timeout, low
  parallelism, watch for the confirmed SIGKILL-on-timeout container-leak
  risk) or is pure in-process LLM calls (seconds, tighter timeout is fine,
  higher parallelism is safe).
- `editor`/`summarizer` (+ `failure_summarizer`, optional — see "Reusable
  framework capabilities" below) — point at the meta-agent model.
  `env:` — set the **project's own** LLM env-var names (check Step 0's
  audit; never the `task_agent:` YAML block, which only wires
  `platform_core.llm_wrapper`-based projects and is irrelevant to a vendor
  with its own `llm_client.py`).
- `validators: [syntax, signature, immutable_files, load_test]` — omit
  `schema_wrapper_consistency`/`mutable_tool_imports`/`mutable_tool_routing`/
  `imports`, which all assume the legacy `tool_wrapper.py`/
  `tools_schema.json`/`mutable_tools/` convention a freshly-vendored project
  doesn't use.

## Step 6 — Verify offline before spending a single live eval

No live LLM/HGM run needed for most of this:

1. Load `benchmark/scorer.py` via `importlib.util.spec_from_file_location`
   (the same mechanism the framework itself uses) and confirm
   `meta_agent.registry.available("scorer"/"gatherer"/"summarizer")` show
   the new `<project>_default` names in all three buckets.
2. Call the scorer's `.score()`/`.aggregate()` directly against a few
   hand-built fake `agent_output`s; compare arithmetic against the raw
   project's own scoring function for the same inputs — must match exactly
   since it's meant to be unmodified.
3. Simulate the editable surface: instantiate
   `AgentEditor(mutable_exclude=[...])` against the **real** vendored
   folder, confirm every excluded path is absent from
   `_read_mutable_sources`'s output and rejected by `_is_path_allowed`, and
   everything else is present/allowed.
4. Run one real case through `workflow.py` directly against a live LLM
   endpoint (no framework machinery involved), confirm a well-formed
   `AgentOutput`.

Only after all four pass, run a small live HGM sanity check
(`PYTHONPATH=. python3 main_loop.py --config configs/hgm_<project>_sanity.yaml`).

## Step 7 — Live sanity check: what to actually watch for

- Confirm the editor genuinely gets invoked (a real `EXPAND`, not just the
  seed's free pre-evaluation) — see Step 5's note on seed sample size.
- Confirm the resulting diff stays entirely inside the mutable surface, and
  is a real, sensible, in-scope edit (not a reinvention of excluded glue —
  a symptom of the editor not being able to see something it needed to;
  widen what's shown, don't just exclude more).
- **Never trust the editor's self-reported `rationale`/`optimization_goal`/
  `proposed_changes` at face value.** This session repeatedly found
  confident, specific, *wrong* claims: a "malformed tag" that was never
  malformed, a scoring quirk misattributed as a behavioral bug, a file path
  hallucinated from a stale README diagram. Only `target_files` is
  structurally grounded (derived from the real edited paths); the prose
  fields have zero code-level verification. Spot-check any specific claim
  against the actual raw case data before believing it — even after adding
  an explicit "double-check yourself" instruction to the editor's system
  prompt, it only helped inconsistently.
- **Audit the vendored project's own README/docs for a stale directory-tree
  diagram** that shows the project's own folder name as a redundant nested
  root (common when a README was written before the project was vendored).
  The editor reads these docs as part of "current sources," and can copy a
  bogus path prefix from a tree diagram into its own edit, silently writing
  to a duplicate file that never actually executes — the real target file
  is left untouched and the "edit" is a no-op. Fix: keep such diagrams
  relative-to-self, no named root label.
- If a case's evaluation reveals the raw agent doing something surprising,
  verify it directly against `eval_result.json`'s `details.raw_result`
  before concluding anything — e.g. a rambling agent that never emits a
  required output tag will get its answer replaced by a garbage fallback
  string; this looks like a random bug until you read the actual raw
  completion.

## Reusable framework capabilities (already built, just wire them in)

- **`mutable_exclude` + `seed_dir_name` + `tool_source_dirs`** — the three
  `FrameworkConfig` fields this whole playbook depends on (Steps 1, 5).
- **`meta_agent/failure_summarizer.py`** (`failure_summarizer:` config
  block, opt-in, off by default) — an LLM-synthesized cross-case failure
  digest (main patterns + hardest cases, grounded in near-full raw text, not
  the small char-capped sample `failure_report.py` renders for direct
  display). Fires on every evaluation batch including the root/seed's.
  Reduce `gatherer.config.n_hard_cases` (e.g. to 2) once this is on, so the
  editor sees a narrative plus a couple of grounding examples, not a large
  redundant raw sample.
- **`hgm_dashboard.py`** (`streamlit run hgm_dashboard.py`) — a live/
  post-hoc viewer for any run using this exclude-list convention: tree
  diagram, diagnostics panel (auto-flags edit-failed nodes, per-case
  errors, crashed evals, zero-mean nodes — including runs that crashed
  before ever writing `eval_result.json`, via a `logs/case_*.json`
  fallback), nodes table with diff line-counts, per-round drill-down.
- **`meta_agent/run_inspect.py`** — the pure-Python data layer behind the
  dashboard; reusable directly for any offline analysis script.
- **`platform_core/communication_instrumentation.py`** +
  `platform_core/runner.py`'s `--export-communication` flag — capture a
  per-task-instance communication trace (every agent's exact prompt, every
  tool call, every inter-agent hand-off) from a vendored project **without
  changing a single line of its own code**. See the dedicated section below.

## Instrumenting agent communication (tool calls, messages, prompts)

For evaluating communication quality separately from task accuracy: a
generic, non-invasive mechanism to capture, per task instance, (a) every
tool call each agent makes, (b) every message passed between agents
(including one-way pipeline hand-offs, not just live back-and-forth), and
(c) each agent's exact prompt — exported as one JSON object. Built and
live-verified against `math_mas` and `db_mas_snapshot` (see plan file
Addendum 10 for the full design rationale); documented here so the same
approach generalizes to any future project without rederiving it.

**How it works, in one sentence**: `platform_core/runner.py --agent-dir
<project> --benchmark <dir> --case-id <id> --export-communication out.json
--patch-agent-run "module:Class.method"` monkey-patches the named method at
runtime (reading the `prompt`/`tool_calls` the *unmodified* code already
computes and returns, and detecting hand-offs by matching one agent's
previously-returned text against a later call's `context` argument),
records everything into a `CommunicationRecorder`, then restores the
original method once the task finishes. Nothing is added to the project's
own files, ever — so this also works unchanged against any HGM round's
`task_agent/` copy, not just the seed, and can't be stripped out by an
editor edit.

Verified command shape (identical for both target projects, only the paths/
module targets change):
```
python -m platform_core.runner --agent-dir projects/<project>/<project> \
  --benchmark projects/<project>/benchmark --case-id <id> \
  --export-communication /tmp/comm.json \
  --patch-agent-run "agents.base:BaseAgent.arun" \
  --patch-transform "tools.mutable.compress:compress"
```

**A real bug found during live verification, now fixed**: `db_mas_snapshot`'s
`AgentOutput.answer = (raw or "").strip()` often equals `raw` byte-for-byte
(no surrounding whitespace to strip), so naively registering both `raw` and
`answer` as separate "producer texts" double-counted the same output —
showing up as a duplicate `compress` tool-call entry and a duplicate
`communications` entry per investigator. Fixed by making
`CommunicationRecorder.register_output` idempotent per `(agent_id, text)`
and capping `find_producers` at one match per producing agent — a project
with two identical-content attributes on its result object is not a
special case to special-case per-project, it's the general rule.

**Applying this to the two projects not yet instrumented**:
- **`wikihop_mas`** — two agent-invocation shapes need separate
  `--patch-agent-run` targets: `agents.base:BaseAgent.run` (Decomposer/
  Extractor) and `agents.base:ToolAgent.run_with_tools` (Retriever/
  Concluder — its tool-call ledger is `result.trace`, already handled by
  the same defensive `tool_calls`-or-`trace` check in
  `_record_from_agent_call`). Its Extractor hand-off
  (`retriever_out.retrieved_paragraphs`, a `list[Paragraph]`, not a flat
  string) doesn't fit the default string-`context_param` matching — either
  add a small adapter that joins paragraph text before matching, or accept
  that this one hand-off isn't auto-detected for this project. Its other
  three hand-offs (Decomposer→hop dispatch, hop1→hop2 entity substitution,
  hops→Concluder summary) are plain strings and fit the generic pattern
  directly.
- **`db_mas`** (Docker) — has *genuine* bidirectional messaging via
  `ask_specialist` (`agents/coordinator/workflow.py:112-126` /
  `agents/specialists/_base.py:74-122`), a direct in-process method call
  (`specialist.answer_followup(question)`) not visible through the outer
  `run()` boundary alone. Needs an additional
  `--patch-agent-run "agents.specialists._base:SpecialistAgent.answer_followup"`
  target (recording `question` as an incoming message, its return value as
  the reply), plus tracking "which coordinator is currently active" (there
  is exactly one per task) to attribute sender/receiver correctly — not
  handled by the generic patcher as built. **Concurrency caveat, must be
  solved before instrumenting this project**: its 5 specialists run via
  `ThreadPoolExecutor` (`mas_workflow.py:71-76`), not `asyncio.gather` —
  `contextvars` (what `recording_scope` uses to find the active recorder)
  do **not** automatically propagate into new OS threads the way they do
  into `asyncio` tasks. Each submitted specialist task needs the active
  recorder explicitly carried across the thread boundary (e.g.
  `contextvars.copy_context().run(...)` per submitted task). Not an issue
  for `math_mas`/`db_mas_snapshot`/`wikihop_mas`, all of which use
  `asyncio.gather`.

## Still-open / deferred (not blockers, just known gaps)

- `error_categorizer` — richer failure-category grouping; neither db_mas
  nor math_mas has one configured yet.
- Tool-call/LLM-call tracing instrumentation feeding the
  BehaviorSummarizer's `tool_usage`/`llm_calls` project_metrics (a
  different mechanism from the per-task communication export above —
  `platform_core.trace.log(...)` calls inside the vendor's own tool/
  decision points, wired into `SubprocessEvaluator`'s round-level
  aggregation, not a standalone `--export-communication` run). Without it,
  those two fields stay empty and the summarizer can only say "no
  instrumentation fired."
- Wiring `platform_core.communication_instrumentation` into
  `SubprocessEvaluator` so a communication trace is captured automatically
  during normal HGM/eval rounds (today it's opt-in, standalone-only, via
  `runner.py --export-communication`) — the module's `recording_scope` +
  no-op-when-inactive design supports this without rework, just not built.
  Also still open: an LLM-judge "communication quality" score consuming
  this JSON (analogous to the `communication_score`/`coordination_score`
  judge already designed for the Docker `db_mas` project's `score.py`).
- A "duplicate basename" write-time check (catch a new file whose name
  already exists elsewhere in the workspace, the exact bug the README-diagram
  issue above causes) — designed, not yet built.
- The confirmed SIGKILL-on-timeout Docker-container-leak risk for any
  infra-heavy project (`SubprocessEvaluator`'s per-case timeout kills the
  child uncatchably, so its `finally:`-block container teardown never runs).
