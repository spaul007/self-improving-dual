# db-mas: Design Document

## 1. What this is

`db-mas` is a standalone multi-agent system (MAS) that performs the **database root-cause
diagnosis** benchmark originally defined in MARBLE (`multiagentbench/database`, 100 curated tasks
in `database_main.jsonl`). It is a clean reimplementation, not a wrapper: it does not import
MARBLE or depend on its `Engine`/`AgentGraph` framework. It reuses only the benchmark's *data*
(task schemas, anomaly specs, ground truth) and the *logic* of MARBLE's anomaly-injection scripts
(ported, not imported).

**The task**: a Postgres database backing some application (a healthcare system, a music-streaming
platform, an e-commerce site, etc.) has 1-2 performance anomalies injected into it. A team of
agents must investigate the live database via SQL and correctly name the injected anomaly type(s)
out of a fixed 5-label set: `INSERT_LARGE_DATA`, `LOCK_CONTENTION`, `VACUUM`, `REDUNDANT_INDEX`,
`FETCH_LARGE_DATA`.

**Why a rebuild instead of reusing MARBLE directly**: MARBLE's DB task is entangled in a generic
multi-agent engine built for many unrelated benchmarks (coding, negotiation, Minecraft, ...), uses
`sudo docker compose`, and its scoring pipeline uses a second, fragile LLM call just to extract
predicted labels from free text. This project fixes all three: independent codebase, rootless
Docker, and agents that emit structured JSON directly via tool calls (see §4.3) so scoring never
has to parse prose.

## 2. Architecture: a fixed star topology

```
                    ┌──────────────┐
                    │  Coordinator │
                    └──────┬───────┘
           ┌────────┬──────┼──────┬────────┐
           │         │      │      │        │
       Insert     Lock    Vacuum  Redundant  Fetch
       LargeData  Cont.            Index     LargeData
      (specialist)(specialist)(specialist)(specialist)(specialist)
```

This shape is **identical for every one of the 100 benchmark tasks** — it is hardcoded, not
configurable per task. This was verified directly against the data before building: every task
has exactly 5 agent profiles and exactly 5 candidate labels, and the profile text for each of the
5 roles is byte-for-byte identical across all 100 tasks (only the domain narrative, DB schema, and
injected anomalies vary). So instead of a single generic "specialist" class templated from
per-task data, there are 5 statically-named specialist classes, each hardcoded to its role.

Key rules of the topology:
- **No peer-to-peer communication between specialists.** All communication is specialist ↔
  Coordinator only.
- Each specialist investigates **independently and in parallel**, with its own DB connections and
  its own message history.
- The Coordinator may ask **at most one** follow-up question, ever, to exactly one specialist
  (`ask_specialist`) — not per-round, not per-specialist: one, total, for the whole task.
- The Coordinator can also run its **own** direct verification queries (`query_db`) at any time,
  independent of the one-follow-up budget.
- The Coordinator's decision is final and structured (`submit_verdict`) — never free text.

## 3. Code structure

```
db-mas/
├── config.py                 # env vars, DB connection defaults, tunable limits
├── llm_client.py              # vLLM Chat Completions wrapper (retry/backoff)
├── benchmark.py               # loads database_main.jsonl, by task_id or all
├── mas_workflow.py            # orchestrates one task (harness) / many tasks (parallel)
├── score.py                   # deterministic metrics + 2 LLM-judge passes, CLI
│
├── common_tools/               # tools shared by MULTIPLE agents, split by whether an
│   │                          #   automated optimizer may rewrite them
│   ├── immutable/query_db.py  # shared by all 5 specialists AND the Coordinator
│   └── mutable/report_findings.py  # shared by all 5 specialists
│
├── agents/
│   ├── base.py                # shared LLM-call/tool-dispatch loop + dataclasses
│   ├── coordinator/
│   │   ├── prompt.py          # system prompt template
│   │   ├── tools.py           # ask_specialist + submit_verdict (Coordinator-only, not "common")
│   │   └── workflow.py        # CoordinatorAgent
│   └── specialists/
│       ├── _base.py           # SpecialistAgent shared mechanics (run loop, answer_followup)
│       ├── insert_large_data/{prompt.py, workflow.py}
│       ├── lock_contention/{prompt.py, workflow.py}
│       ├── vacuum/{prompt.py, workflow.py}
│       ├── redundant_index/{prompt.py, workflow.py}
│       └── fetch_large_data/{prompt.py, workflow.py}
│
├── environment/
│   ├── docker-compose.yml     # single postgres_db service, host port parameterized
│   ├── docker_lifecycle.py    # compose up/down, wait_for_ready, run_init_sql
│   ├── db_conn.py             # psycopg2 connection helper
│   ├── anomaly_injection.py   # ported from MARBLE's anomaly_trigger/anomaly.py
│   └── task_setup.py          # setup_task_environment / teardown_task_environment
│
├── scripts/
│   ├── verify_env_standalone.py  # Docker+SQL+anomaly smoke test, no LLM
│   ├── run_single_task.py        # one task end-to-end + score it
│   └── run_batch.py               # N tasks (sequential or parallel) + score them all
│
└── results/
    ├── raw/<task_id>.json      # full transcript + verdict, written by mas_workflow
    ├── scored/<task_id>.json   # score.py output per task
    └── summary.json            # aggregate across the last scored batch
```

### Why tools are split into `common_tools/` vs. per-agent files

Two independent axes are in play, and the folder structure encodes both:

1. **Scope** — is a tool used by *one* agent or shared by *several*? `query_db` and
   `report_findings` are used identically by all 5 specialists (and `query_db` by the Coordinator
   too) — genuinely common code, so they live in `common_tools/`. `ask_specialist` and
   `submit_verdict` are used by exactly one agent (the Coordinator) — they live in
   `agents/coordinator/tools.py`, not dressed up as "common" when they aren't.
2. **Mutability to an automated optimizer** — within `common_tools/`, `immutable/` holds
   `query_db`: the system's actual interface to the benchmark's environment (matches the original
   MARBLE contract). An automated prompt/tool optimizer must never rewrite this — doing so would
   risk changing *what the benchmark tests*, not just how well the agents perform it.
   `mutable/report_findings.py` (and the Coordinator's own two tools) are internal
   coordination-protocol tools, not part of the benchmark's environment contract, so they're fair
   game for an optimizer to retune.

## 4. The multi-agent system in detail

### 4.1 Environment setup (not part of the harness)

`environment/task_setup.py`'s `setup_task_environment(task, project_name, port)` brings up a fresh
Postgres via Docker Compose, loads the task's schema (`init_sql`), and injects the task's
anomalies (`environment/anomaly_injection.py`, ported logic from MARBLE, using the task's full
per-task spec — `threads`, `ncolumn`, `nrow`, `colsize` — for fidelity to what each task actually
specifies). `teardown_task_environment(project_name)` always tears the container down (`compose
down -v`), even on failure.

This is deliberately **not** part of `mas_workflow.run_task`'s own body — it's called once at the
top and once in the `finally`. The reasoning: this setup defines *what is being tested* (the
injected anomaly is the ground truth), exactly analogous to why `query_db` is an immutable tool.
Everything in `run_task` *after* that call — building and running the specialists and Coordinator
— is the harness: the actual multi-agent system, and the part that's meant to be inspected, tuned,
or swapped out.

### 4.2 Specialists (`agents/specialists/`)

Each of the 5 specialist classes (`InsertLargeDataSpecialist`, `LockContentionSpecialist`,
`VacuumSpecialist`, `RedundantIndexSpecialist`, `FetchLargeDataSpecialist`) is a thin subclass of
`SpecialistAgent` (`agents/specialists/_base.py`) supplying only its own hardcoded
`SYSTEM_PROMPT_TEMPLATE`/`LABEL` — the verbatim MARBLE profile text for that role (e.g. the vacuum
specialist's prompt always points at `pg_stat_all_tables`/`pg_stat_progress_vacuum`).

Each specialist's `run()`:
1. Gets a system prompt (fixed role) + a user message ("begin your investigation...").
2. Loops (`agents/base.py::run_tool_loop`, up to `MAX_SPECIALIST_TOOL_CALLS=8` turns) calling
   `query_db` freely with `tool_choice="auto"`.
3. Terminates by calling `report_findings` — a structured tool call
   (`label`, `supports_label`, `evidence`, `confidence`) — never free text. `label` is always
   forced to the specialist's own fixed `LABEL` regardless of what the model's tool-call arguments
   say, since which label a specialist covers is a system fact, not something to trust the model
   to restate correctly.
4. If a non-terminal tool (`query_db`) and the terminal tool (`report_findings`) are called in the
   *same* turn (parallel tool calls), the terminal call is **deferred** rather than accepted — it
   would have been decided before that query's result was seen. The model gets another turn.
5. If the turn budget is exhausted without a `report_findings` call, one forced turn pins
   `tool_choice` to it so the specialist can't dodge producing a structured answer
   (`forced_fallback=True` is recorded).

All 5 specialists for a task run **in parallel** via `ThreadPoolExecutor` (`mas_workflow.run_task`)
— independent DB connections, no shared state, matching the "no peer chat" rule.

### 4.3 Coordinator (`agents/coordinator/`)

Built only after all 5 specialists finish. Its prompt (`agents/coordinator/prompt.py`) embeds the
task narrative, the candidate labels, `number_of_labels_pred`, and every specialist's structured
findings. It has three tools:

- **`query_db`** (shared, from `common_tools/immutable/`) — direct, repeatable verification, up to
  `MAX_COORDINATOR_QUERY_CALLS` worth of turn budget (this bounds LLM round trips, not raw
  `query_db` call count — a single turn can bundle several parallel `query_db` calls).
- **`ask_specialist`** — usable **at most once, total**. When used, the Coordinator looks up the
  actual `SpecialistAgent` object by `agent_id` and calls its `answer_followup(question)` method,
  which re-enters *that specialist's own accumulated conversation* (including everything it
  already investigated), lets it optionally run one more `query_db`, and returns a plain-text
  answer. This is the only place any "conversation" crosses agent boundaries. Once used, the tool
  is removed from the toolset for the rest of the run — a second attempt is structurally
  impossible, not just discouraged by the prompt.
- **`submit_verdict`** — terminal: `{predicted_root_causes: [...], reasoning}`. Validated for
  exactly `number_of_labels_pred` labels drawn from the candidate set, with one corrective re-ask
  if the model gets the count or labels wrong.

Same anti-staleness rule as specialists: if `submit_verdict` is bundled in the same turn as
`query_db` or `ask_specialist`, it's deferred rather than accepted, since it would predate that new
information.

`answer_followup` has one extra defensive layer: some served models occasionally leak a raw,
unparsed tool-call template (e.g. `<tool_call><function=query_db>...`) into the plain-text content
channel instead of a structured tool call (observed live against a Qwen3.5 deployment). If detected
(`_looks_like_unparsed_tool_call`), that turn is not accepted as a real answer — the specialist's
own malformed turn is recorded and it's nudged to retry properly, rather than handing the
Coordinator garbage.

### 4.4 Tool-calling contract

Every terminal action in this system (`report_findings`, `submit_verdict`, and the judges'
`submit_judgment`/`submit_coordination_judgment` in `score.py`) is a forced function call with a
JSON Schema, never a request to "answer in JSON" as free text. This is the direct fix for the
original benchmark's fragile approach of parsing predicted labels out of prose with a second LLM
call — since every agent's output is already structured at the source, no such parsing step exists
anywhere in this codebase.

## 5. Execution model

`mas_workflow.run_task(task, port=None)` runs one task end-to-end and writes
`results/raw/<task_id>.json` (a `TaskResult`: verdict, reasoning, full transcript, token usage,
per-phase timing). Any exception anywhere in the harness is caught and recorded as
`error=str(e)` rather than propagating — one bad task never kills a batch.

`mas_workflow.run_many(task_ids, max_workers=1)`:
- `max_workers=1` (or omitted): strictly sequential, every task on the default port (5432) —
  original, simplest behavior.
- `max_workers>1`: runs up to that many tasks **concurrently**, each in its own OS process
  (`ProcessPoolExecutor`) with its own Docker Compose project and its own host port
  (`config.PARALLEL_DB_PORT_BASE + slot`, a bounded pool of slots handed out via a
  `multiprocessing.Manager` queue, so concurrency never exceeds `max_workers` and ports never
  collide). `config.DB_CONFIG["port"]` is mutated once per task at the start of `run_task` —
  safe only because each concurrently-differing-port task runs in its own process, so a single
  process only ever has one task's DB traffic in flight.

This only parallelizes the per-task wall time that's genuinely local (Docker startup, schema load,
anomaly injection). The specialist/coordinator phases all call the same shared vLLM endpoint, so
heavy concurrency there means more requests queued at that endpoint rather than a clean N×
speedup — live testing showed real wall-clock benefit (~1.8x on one 7-task batch), capped by
whichever single task's LLM calls happened to be slowest that run.

## 6. Evaluation (`score.py`)

Three independent things are measured per task, plus one always-on cheap check:

1. **`deterministic`** (no LLM) — precision/recall/F1/`exact_match` of `predicted_root_causes`
   against the task's true `root_causes`, computed directly as set operations. No fragile
   label-extraction step is needed because the Coordinator's verdict is already structured JSON.

   **Why precision and exact_match are structurally capped below 1.0, for every task, regardless
   of agent quality:** `number_of_labels_pred` (the required size of the predicted set —
   `submit_verdict` is validated to contain *exactly* this many labels, not "up to") is always,
   across all 100 tasks, exactly **one more** than `len(root_causes)` (the true count): 50 tasks
   are `(1 root cause → must predict 2)`, 50 are `(2 root causes → must predict 3)`. So even a
   hypothetically perfect agent that correctly names every true cause is still forced to name one
   additional label that is *not* a true cause, just to fill the required set size. Precision =
   `|predicted ∩ gold| / |predicted|`, so the best possible value is `len(root_causes) /
   number_of_labels_pred`: **0.5** for 1-cause tasks, **0.667** for 2-cause tasks — never 1.0, no
   matter how good the diagnosis is. `exact_match` (`predicted_set == gold_set`) is even more
   strongly ruled out: since `|predicted|` is always `|gold| + 1`, the two sets can never even be
   the same *size*, so set equality is mathematically impossible for every task in this benchmark
   — `exact_match_rate` reading `0.0` across a batch is not a sign of a problem.

   `recall` (`|predicted ∩ gold| / |gold|`) is unaffected by this and *can* reach 1.0 if the agent
   includes every true cause among its predictions, regardless of the one forced extra — it's the
   metric that actually reflects diagnostic accuracy here, along with `F1`.

   This is an artifact of the original MARBLE benchmark data itself (confirmed directly against all
   100 tasks in `database_main.jsonl`; not something introduced by this reimplementation), and no
   rationale for choosing "exactly one more than the truth" is documented anywhere in the MARBLE
   codebase — no README, no code comment, nothing in `engine.py` beyond consuming the value as-is.
   Plausible guesses (forcing agents to also rank/prioritize a plausible-but-wrong candidate; a
   dataset-generation heuristic without deeper justification) are only guesses, not confirmed intent.
2. **`judge`** (one LLM call, `judge_task`) — quality of the *answer*: `evidence_grounding`,
   `reasoning_soundness`, `investigation_thoroughness`, `label_justification`, each 1-5, plus a
   short feedback string. Given the task, gold labels, the verdict, and every specialist's
   `report_findings` evidence.
3. **`coordination`** (one LLM call, `judge_coordination`, **togglable** — see below) — quality of
   the *process*, adapted from MARBLE's original (but never actually exercised, see §6.1)
   Communication/Planning rubrics:
   - `communication_score` (1-5): if `ask_specialist` was used, was it well-targeted and did the
     answer meaningfully inform the verdict? If not used, was skipping it reasonable given the
     specialist reports already available, or a missed chance to resolve real conflict between
     them?
   - `coordination_score` (1-5): does the final verdict draw on and synthesize *all five*
     specialists' reports (and the Coordinator's own `query_db` checks), or does it look like a
     rubber-stamp of the single highest-confidence report?
   - Built from `_format_coordinator_activity`, which pairs every Coordinator tool call with its
     matching tool-result message (by `tool_call_id`) so the judge sees exactly what was asked/
     queried and what came back.

`aggregate(scored)` produces `results/summary.json`: `mean_precision`/`mean_recall`/`mean_f1`/
`exact_match_rate`, `mean_judge` (per criterion), `recall_by_anomaly_type` (true per-label recall,
computed from each task's actual gold/predicted sets — not by reusing a task's overall recall for
every one of its gold labels, which would conflate multi-label tasks), and `mean_coordination`
(`None` if no task was coordination-scored).

Every scored dict keeps `"coordination"` present (as `None` when skipped) rather than omitting the
key, so downstream consumers always see a predictable shape. `n_scoring_errors` in the summary
counts tasks that failed to score at all (e.g. a judge call that never produced a tool call after
retries) — these are isolated per task, not fatal to the batch.

### 6.1 Why `coordination`/`communication` scoring exists and how it was designed

MARBLE's `evaluator_prompts.json` defines a `Communication` rubric (Information Exchange, Clarity,
Task Assistance, Efficiency) and a `Planning` rubric (Role Clarity, Task Alignment, Autonomy), each
a single LLM call parsed to a 1-5 `{"rating": X}`. But in `engine.py`'s `graph_coordinate()` — the
only coordination mode the DB benchmark ever uses (all 500 original configs set
`coordinate_mode: graph`) — both calls are commented out and replaced with a hardcoded `-1`. The
reference implementation defines the rubric but never actually scores it for this task.

MARBLE's verbatim "Planning" rubric doesn't fit this system's fixed star topology: Role Clarity
and Autonomy are true by construction here (roles are hardcoded per specialist folder, specialists
always run independently in parallel) — they'd score 5/5 every time and add no discriminating
signal. The rubric was adapted instead of copied verbatim: `communication_score` keeps the
Communication rubric's spirit (it maps well onto the one `ask_specialist` exchange),
`coordination_score` replaces Planning with criteria that are actually variable in this system
(did the verdict synthesize all 5 reports, or rubber-stamp one). Verified live: the judge
genuinely discriminates — it penalized tasks where the Coordinator skipped its one follow-up
despite real conflict between specialists, and rewarded it for correctly skipping the follow-up
when specialist reports already agreed.

### 6.2 CLI

```bash
python score.py                        # both judges run (default)
python score.py --no-coordination-eval # skip the coordination/communication judge pass
python scripts/run_batch.py --no-coordination-eval  # same flag, forwarded to score.py
```

## 7. Key design decisions and known limitations

- **Chat Completions, not the Responses API** (`llm_client.py`): chosen for broad vLLM
  compatibility with tool calling, since Responses-API support varies by served model.
- **Full per-task anomaly spec used, deviating from MARBLE's actual runtime**: MARBLE's own
  `db_env.py` silently drops the `nrow` field when invoking anomaly injection (always defaults to
  100 rows) even though every task's config specifies a much larger, evidently deliberate value
  (e.g. 20000). This project uses the full spec instead of reproducing that gap.
  `duration`/`nindex`/`table_name` aren't present in any task's spec either way, so those use fixed
  project constants (`ANOMALY_DURATION_S=60`, matching the anomaly script's own default).
- **Known race-condition-shaped edge case in parallel execution**: if a worker process in
  `run_many`'s `ProcessPoolExecutor` hard-crashes (not a clean Python exception — e.g. OOM-killed),
  its held port slot is never returned to the queue, permanently shrinking available concurrency
  for the remainder of that batch by one. Low probability given how stable the Docker/Postgres
  path has been in testing, but a real gap if hardening further is ever warranted.
- **`MAX_COORDINATOR_QUERY_CALLS` bounds round trips, not literal call count** — a single LLM turn
  can bundle multiple parallel `query_db` calls, so the effective number of individual queries is a
  soft, not a hard, budget. Intentional: bounding round trips controls wall-clock/cost more
  directly than bounding raw tool-call count would.
