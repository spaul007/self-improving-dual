# db_mas

A six-agent multi-agent system for database root-cause analysis, evaluated on
**MARBLE / MultiAgentBench's database benchmark** (100 tasks). Reimplemented
from MASPO_v2's `database_team` topology as a standalone repo — it does not
import MASPO or MARBLE at inference time.

**Five specialist investigators (parallel, each with the `query_db` tool) →
one lead DBA (terminal).** The lead's diagnosis is the system's answer.

---

## MAS setup

```
    problem  (+ its recorded DB snapshot, bound per-task)
       │
       ├─────────┬─────────┬─────────┬─────────┐
       ▼         ▼         ▼         ▼         ▼
   ┌────────┐┌────────┐┌────────┐┌────────┐┌────────┐
   │ INSERT ││  LOCK  ││ VACUUM ││ INDEX  ││ FETCH  │   query_db ReAct loop
   │ invest.││ invest.││ invest.││ invest.││ invest.│   over the snapshot
   └───┬────┘└───┬────┘└───┬────┘└───┬────┘└───┬────┘
       ▼         ▼         ▼         ▼         ▼
   compress  compress  compress  compress  compress     mutable tool: ≤120-word
       │         │         │         │         │        briefings
       └─────────┴────┬────┴─────────┴─────────┘
                      ▼
                ┌───────────┐
                │ LEAD DBA  │  weighs evidence, no tools
                └─────┬─────┘
                      ▼
        diagnosis ending "FINAL: <LABEL>[, ...]"  →  scored
```

| | Investigators (×5) | Lead DBA |
|---|---|---|
| Stage | 1 (parallel) | 2 (terminal) |
| Assigned candidate | one each: INSERT_LARGE_DATA, LOCK_CONTENTION, VACUUM, REDUNDANT_INDEX, FETCH_LARGE_DATA | all |
| Tools | `query_db` (snapshot replay) | none |
| Context in | none | five labeled briefings |
| Output | evidence report + high/med/low verdict | diagnosis + `FINAL:` line |

- **Parallel then sequential.** The five investigators never see each other;
  the lead never runs before them. One pass, no loop back, no peer chat.
- **Compression is on by default** — the lead sees ≤120-word briefings, not
  full query transcripts. Set `MAS_USE_COMPRESSED_CONTEXT=0` to pass full
  reports instead.
- The lead must end with a `FINAL: <LABEL>[, <LABEL> ...]` line; that line is
  what gets scored.

### Prompt convention

Prompts live in `mas_prompt_cfg.yaml`, split per agent:

- `role` — **frozen.** The agent's identity, including its assigned candidate
  root cause. An optimizer must not rewrite it (e.g. must not turn an
  investigator into the decision-maker).
- `task` — **editable.** The strategy/instruction. Fair game for prompt
  optimization.

Prompt text is ported from MASPO_v2's proven `DATABASE_TEAM_PROMPTS`.

---

## The database environment: record-and-replay

MARBLE's real environment is a live Postgres-in-Docker stack with a per-task
anomaly injected at init. It is single-instance, fixed-port and strictly
serial — incompatible with concurrent batch inference. This repo therefore
ships **per-task snapshots** and replays them:

- `data/marble-db/db_cache/<task_id>.json` — one snapshot per task
  (`queries`: a battery of canonical diagnostics recorded verbatim; `tables`:
  full dumps of the 8 diagnostic views). **All 100 snapshots are included**,
  so inference starts from the snapshots directly — no Docker, no Postgres.
- `tools/immutable/query_db.py` replays them: an agent's SQL is served
  **exact** (normalized match on a battery query), by **table_fallback** (the
  referenced view's full dump; the agent's WHERE/ORDER/LIMIT are NOT applied),
  or **miss** (an error pointing at the available views). In practice nearly
  all agent queries are served by table_fallback — agents must reason over the
  top-N dumps themselves.
- Each concurrent task binds its own snapshot via a `ContextVar`, so batches
  are concurrency-safe.
- Coverage counters (exact / table_fallback / miss / unbound) are **persisted
  into the raw results payload** (`db_coverage`) and summarized at scoring
  time from the saved trajectories.

### Regenerating (or extending) the snapshots

Only needed if MARBLE's tasks change or you want different battery queries:

```bash
# needs docker + the MARBLE package + psycopg2 (conda env `crew_ai` is known-good)
python snapshot/prepare_dataset.py        # MARBLE jsonl -> data/marble-db/database_tasks.jsonl
python snapshot/record_db_cache.py        # live Docker DB once per task -> data/marble-db/db_cache/
python snapshot/record_db_cache.py --task-id 7   # re-record a single task
```

`record_db_cache.py` is the ONLY thing that touches Docker and runs tasks
serially (each task: compose up → init + inject anomaly → warm-up → snapshot →
compose down). MARBLE's default location is
`/groups/AIC-MV/sudipta.paul/code/rsi/MARBLE`; override with `MARBLE_ROOT`.

---

## Code structure

```
Paths below are relative to this directory itself (no extra nesting level --
this folder IS the root):

├── config.py                  # env vars, model/endpoint, limits, prompt loader
├── mas_prompt_cfg.yaml   (E)  # role (frozen) + task (editable) per agent
├── llm_client.py              # async LLM wrapper + query_db ReAct tool loop
├── mas_workflow.py       (E)  # run_task (one case) + run_many (batch)
├── run_inference.py           # CLI: load tasks → run MAS over snapshots → results/raw/
├── evaluate.py                # CLI: score → results/scored/
│
├── agents/
│   ├── base.py                # shared prompt assembly + LLM/tool-loop call
│   ├── insert_investigator/   # INSERT_LARGE_DATA
│   │   ├── prompt.py     (E)  # role/task accessors + assigned CANDIDATE
│   │   ├── skill.md      (E)  # what this agent can and cannot do
│   │   └── workflow.py   (E)  # the agent class
│   ├── lock_investigator/     # LOCK_CONTENTION   (same three files)
│   ├── vacuum_investigator/   # VACUUM
│   ├── index_investigator/    # REDUNDANT_INDEX
│   ├── fetch_investigator/    # FETCH_LARGE_DATA
│   └── lead_dba/              # terminal decision-maker
│
├── tools/
│   ├── immutable/query_db.py         # snapshot replay of the DB env — do not tune
│   ├── immutable/label_extraction.py # FINAL:-line parser — do not tune
│   └── mutable/compress.py      (E)  # investigator→lead briefing
│
├── eval/
│   └── metrics.py             # recall/precision/f1/exact-match + summary
│
├── snapshot/
│   ├── prepare_dataset.py     # flatten MARBLE jsonl (offline, docker-free)
│   └── record_db_cache.py     # record snapshots from the live Docker DB (offline, serial)
│
├── data/marble-db/
│   ├── database_tasks.jsonl   # 100 flattened tasks (problem + gold root_causes)
│   └── db_cache/<id>.json     # 100 recorded snapshots — inference starts here
└── results/{raw,scored}/      # run outputs
```

**(E)** = editable/tunable by a prompt optimizer.

### Why tools are split

- `tools/immutable/` — `query_db.py` IS the benchmark environment, and
  `label_extraction.py` defines *what counts as the answer*. Rewriting either
  changes what the benchmark measures, not how well the agents perform. Never
  tune them.
- `tools/mutable/` — `compress.py` is an internal hand-off protocol, not part
  of the benchmark contract. Free to retune.

---

## Setup

Requires Python 3.10+ and an OpenAI-compatible endpoint (vLLM) **started with
tool-calling enabled**:

```bash
pip install openai pyyaml
# vLLM server must run with:  --enable-auto-tool-choice --tool-call-parser hermes
```

`run_inference.py` preflights the endpoint and aborts with that hint if
tool-calling is rejected.

Defaults (all overridable by env var — see `config.py`):

| Setting | Env var | Default |
|---|---|---|
| Model | `MAS_MODEL` | `Qwen/Qwen3.5-35B-A3B` |
| Endpoint | `MAS_BASE_URL` | `http://gpu-aic-mv-02-st-p5-node-1:8000/v1` |
| Max tokens | `MAS_MAX_TOKENS` | `4096` |
| Temperature | `MAS_TEMPERATURE` | `0.0` |
| Concurrent tasks | `MAS_MAX_CONCURRENT_TASKS` | `8` |
| LLM calls in flight | `MAS_LLM_CONCURRENCY` | `60` |
| Compress context | `MAS_USE_COMPRESSED_CONTEXT` | `1` (on) |
| Tool loop on | `MAS_TOOLS_ENABLED` | `1` (on) |
| Max ReAct rounds | `MAS_TOOL_MAX_ROUNDS` | `5` |

Check the endpoint is reachable:

```bash
python3 -c "
import asyncio, llm_client
print(asyncio.run(llm_client.get_client().acall('Say OK', max_tokens=10)))"
```

---

## How to run

All commands from this directory. No Docker needed — inference replays the
shipped snapshots.

### Quick smoke test (3 tasks)

```bash
python3 run_inference.py --limit 3 --run-name smoke --evaluate
```

### Full 100-task benchmark, inference + scoring in one command

```bash
python3 run_inference.py --run-name database_full --evaluate
```

Detached, since it takes a while:

```bash
nohup python3 run_inference.py --run-name database_full --evaluate \
  > results/database_full.log 2>&1 &
tail -f results/database_full.log
```

### Separate steps / useful flags

```bash
python3 run_inference.py --run-name database_full     # → results/raw/
python3 evaluate.py      --run-name database_full     # → results/scored/

# specific cases only
python3 run_inference.py --task-ids "1,42,87" --run-name three_cases --evaluate

# inspect the first 10 imperfect diagnoses
python3 evaluate.py --run-name database_full --show-errors 10

# re-score an existing raw file without re-running inference
python3 evaluate.py --raw results/raw/database_full.json --run-name rescored

# ablation: no evidence gathering
python3 run_inference.py --no-tools --run-name no_tools --evaluate
```

`run_inference.py` flags: `--data --limit --start --task-ids --run-name
--max-concurrent --no-tools --evaluate --show-errors`
`evaluate.py` flags: `--run-name --raw --show-errors`

---

## Output

`results/raw/<run>.json` — per task: the lead's full diagnosis, the parsed
labels, gold `root_causes`, the six-agent trajectory (prompt + raw output +
every `query_db` call with its replay kind), elapsed time, plus the run-level
`db_coverage` counters.

`results/scored/<run>.json` — the same records plus per-task
recall/precision/f1/exact-match and a `summary`:

```
task_score / recall   # headline: mean root-cause recall
precision, f1         # diverge from recall only when the team under-names
exact_match(_rate)    # recall == 1 is an exact match (count contract below)
extraction_failed     # diagnoses with no parseable FINAL: verdict
over_named            # verdicts naming more labels than requested
per_label             # recall broken down by gold label (30 tasks each)
tool_usage            # query_db calls by replay kind, from trajectories
errors, avg_elapsed_s
```

---

## Scoring

Deterministic — no judge LLM anywhere in the scoring path:

1. `tools/immutable/label_extraction.py` reads the labels from the diagnosis:
   it anchors on the **last** verdict marker (`FINAL:`, "final diagnosis", …)
   and collects allowed labels in order of appearance (else scans the closing
   lines).
2. The list is truncated to k = |gold set| (over-naming is recorded as
   `n_named` but not scored).
3. `eval/metrics.py` computes set metrics case-insensitively. **Task score =
   recall** — the fraction of gold root causes recovered — matching MASPO_v2's
   `DatabaseTaskJudge` (which surfaces it as `accuracy`). A crashed task
   counts 0.

Known divergences, both inherited from MASPO_v2 (numbers comparable with
MASPO_v2, NOT with upstream MARBLE/crewai):

- **Count contract:** the task text asks for exactly |gold| labels instead of
  MARBLE's |gold|+1 (see `snapshot/prepare_dataset.py` for the rationale).
- MASPO_v2 additionally falls back to an LLM extractor for unparseable
  answers; this repo stays deterministic and reports `extraction_failed`
  instead.

### Known failure modes

- **Missing `FINAL:` line.** Extraction falls back to scanning the closing
  lines; if those don't name allowed labels either, the task scores 0 with
  `extraction: "none"`. A rising `extraction_failed` count is a prompt problem
  in `mas_prompt_cfg.yaml` (`lead_dba.task`), not a diagnostic one.
- **Wrong-vocabulary answers.** The task text floats MISSING_INDEXES /
  POOR_JOIN_PERFORMANCE / CPU_CONTENTION, which are never scored; the lead's
  prompt maps them onto the allowed five. Watch `n_named` < requested.
- **Vacuum false negatives.** In the snapshots, `n_dead_tup`/`vacuum_count`
  are freshly reset (all zeros) and `pg_stat_progress_vacuum` is empty in all
  100 tasks; real VACUUM anomalies show up as `VACUUM FULL` in
  `pg_stat_statements`. See `agents/vacuum_investigator/skill.md`.
- **Case marker.** Each problem ends with `[Diagnosis case #<id>]` — many
  MARBLE cases share identical text but different gold, distinguishable only
  via query_db. Don't strip the marker.
