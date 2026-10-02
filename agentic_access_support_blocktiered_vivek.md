# Agentic Access in Block-Tagged HGM — Editor & Block Suggester

## What it is

Two independent opt-in features, one per meta-agent role, that replace a single "dump everything upfront, get one big response back" LLM call with a **multi-turn tool-calling loop**. Each was built in direct response to the same observed failure mode: large, bundled JSON responses (a full multi-file diff, or a long suggestion) becoming malformed/truncated as they grow. The editor's own code comment cites the exact number: **~65% of EXPANDs failed on malformed JSON over 81 rounds with DeepSeek v4** before this was added.

Both features are **fully additive and off by default** — enabling one never changes behavior for a config that doesn't set it.

## Agentic Editor (`meta_agent/agent_editor.py`)

**Config keys** (under `editor.config:` in the YAML):
```yaml
editor:
  type: "default"
  config:
    agentic_editing: true      # default: false
    agentic_max_turns: 20      # default: 20 (already the default — only need to override if changing it)
```

**What changes**: instead of one `submit_self_improvement` call carrying every changed file's full content in a single JSON `files` array, the editor gets a bounded loop (up to `agentic_max_turns` LLM round-trips) with four tools:

| Tool | Purpose |
|---|---|
| `read_file(path)` | Inspect a file's current content before editing it |
| `write_file(path, content)` | Submit **one file's full new content per call** — never bundled |
| `run_code_validators()` | Run the real validator suite (syntax, imports, signatures, etc.) against edits so far, before finalizing |
| `submit_self_improvement_summary(optimization_goal, proposed_changes, rationale)` | Finish — no file content in this call, just the write-up |

Real trials needed **3-6 turns** even for a genuine 4-file fix, well inside the 20-turn budget. Failure modes are explicit, not silent: exceeding the turn budget or a malformed final-summary call both produce a distinct marker goal string (`"(editor exceeded agentic turn budget without submitting a summary)"` / `"(summary call had malformed JSON; edits below were still applied)"`) rather than a generic failure — and critically, files already written via `write_file` before the failure are still kept, since each one succeeded independently.

## Agentic Block Suggester (`meta_agent/block_suggester.py`)

**Config keys** (under `block_suggester.config:`):
```yaml
block_suggester:
  type: "default"
  config:
    agentic_access: true       # default: false
    agentic_max_turns: 20      # default: 20
```

**What changes**: instead of a full mutable-source dump shown upfront, the model gets two **read-only** tools:

| Tool | Purpose |
|---|---|
| `read_file(path, offset?, limit?)` | Read a file on demand, with line-range support for large files |
| `grep(path, pattern, max_matches?)` | Regex search returning a window around each match (not just line starts — safe against one very long line) |

Paths are **alias-rooted**, not real filesystem paths:
- `harness/<rel>` — the current mutable source being diagnosed
- `logs/<rel>` — **this parent node's own real per-case evaluation logs**: each `logs/case_<id>.json` has `converted_plan` (structured scored output) and `raw_plan_text` (literal agent output), plus `logs/trace.jsonl` (real `tool_call`/`tool_result`/`llm_call` events)
- `eval_result.json` — every case's score and per-check pass/fail with violation messages

This is strictly more than the non-agentic path ever sees — the non-agentic block_suggester never gets per-case logs or traces at all, only the curated feedback digest. The code comment notes this was **confirmed live** to let the model independently derive root-cause diagnoses (a specific regex bug in a validator, an exact prompt-rule violation) that the digest alone never surfaced a concrete example of. The suggester must respond as plain markdown text (not a tool call) to finish, under 400 words; if evidence is too thin for a real diagnosis, it's explicitly allowed to say so rather than invent one.

Turn-budget exhaustion or malformed tool arguments are logged explicitly (`"[block_suggester] agentic suggestion for node {id} (block={block}) exhausted {N} turns..."`), same transparency convention as the editor.

## How they interact

Both flags are independent — you can enable one, both, or neither. They're read via the standard `ComponentSpec.config` → constructor-kwargs passthrough (`meta_agent/config.py::_build_with_injection`), so **any keyword either class's `__init__` accepts can be set directly in YAML** without needing a dedicated schema field — this is why `agentic_editing`/`agentic_access`/`agentic_max_turns` all just work as plain config keys, same as `model`/`base_url`/`reasoning_effort` elsewhere.

## Recommendation from live experience

Running both together (production node-32-restart run, `configs/hgm_travel_deepseek27b_from_node32_agentic_X100Y300.yaml`) has shown zero malformed-JSON failures and zero `edit_failed` nodes across the entire run (44 EXPANDs, 0 edit failures) — a sharp contrast to the same config's non-agentic predecessor (58% EXPAND failure rate, 45.9% specifically from malformed JSON). The tradeoff is wall-clock time per round (multiple LLM round-trips instead of one), not reliability.

## Result: node 21, the best-verified node this run produced

Node 21 (block=`foundation_capability`, implementation_strategy=`mixed`, single file `agents/common.py` — injects budget-awareness warnings into the shared tool-calling loop so the LLM self-terminates gracefully before exhausting its iteration budget, reducing `tool_calling_budget_exceeded` failures) was reached via 4 separate in-loop EVALUATE passes on the 60-case train split and settled at a stable **in-run composite of 0.683** (240 total evals) — the most-repeated node of the entire run.

To check whether that holds up independently, it was evaluated 3 fresh times against the **full 120-case pool** (train 60 + held-out eval 60 combined), on an otherwise idle GPU (no contention from other runs):

| Repeat | Composite | Case accuracy | No-plan rate | Conversion-error rate |
|---|---|---|---|---|
| 1 | 0.6646 | 5.00% | 0% | 0% |
| 2 | 0.6698 | 5.83% | 0% | 0% |
| 3 | 0.6750 | 5.00% | 0% | 0% |
| **Mean** | **0.6698** | **5.28%** | **0%** | **0%** |

This is the strongest verification result produced across the project: the independent full-120 mean (0.670) lands within 0.013 of the in-run train-only cmp (0.683), with zero no-plan cases and zero conversion failures across all 360 case-evaluations. Unlike several other single/low-repeat "leader" nodes seen during this run (e.g. node 6: 0.7125 on 1 pass → 0.6506 on a 2nd; node 28: 0.7104 on 2 passes → 0.6547 on a 3rd, later independently confirmed at a true mean of ~0.67), node 21's score did not regress on additional scrutiny — it is a genuine, reproducible improvement over both the seed (0.641) and the lineage it was restarted from (node 32, 0.639/0.648), not an artifact of sampling variance.

### Exact paths to this experiment

- **Run directory**: `runs/20260929_190918_travel_mas_refactored_deepseek27b_from_node32_agentic_X100Y300/`
- **Node 21's workspace**: `runs/20260929_190918_travel_mas_refactored_deepseek27b_from_node32_agentic_X100Y300/round_021/task_agent/`
- **Config**: `configs/hgm_travel_deepseek27b_from_node32_agentic_X100Y300.yaml`
- **Full-120 3× re-verification output**: `eval_node21_full120_3x_out/` (script: `eval_node21_full120_3x.py`)

### Bandit reward state at node 21's selection (Thompson-sampled, not the mean-leader)

*Tier 2 — block* (chosen: `foundation_capability`, which did NOT have the highest posterior mean — `mixed` did at 0.653 — but won the Thompson draw):

| Block | Mean | Sampled | n_evals |
|---|---|---|---|
| foundation_capability (chosen) | 0.644 | **0.752** | 2 |
| verifiers | 0.577 | 0.671 | 1 |
| mixed | 0.653 | 0.649 | 5 |
| individual_subagent | 0.642 | 0.602 | 3 |
| collaboration_workflow | 0.644 | 0.564 | 2 |

*Tier 1 — implementation_strategy* (chosen: `mixed`):

| Strategy | Mean | Sampled | n_evals |
|---|---|---|---|
| mixed (chosen) | 0.644 | **0.664** | 4 |
| llm_heavy | 0.657 | 0.629 | 6 |
| harness_heavy | 0.611 | 0.580 | 3 |

### Lineage: what changed from the original bare seed to node 21

Node 21 inherits two full lineages end to end: original bare seed → **node 11 → node 17 → node 26 → node 32** of the *earlier* production run (the X100Y300 run killed to restart this one) → this run's own seed (`seed_from_node32`) → **node 7 → node 12 → node 21**. Diffing node 21's full workspace directly against the true original seed (`projects/travel_mas_refactored/seed_qwen27b_nothink/`) shows exactly **6 files changed**, accumulated across 7 edits total:

| File | Changed by (chronological) | What changed |
|---|---|---|
| `tool_wrapper.py` | node 11 (prior run) | Pre-execution schema validation on tool calls, to prevent `TypeError` crashes from malformed LLM tool arguments |
| `agents/common.py` | node 11 (prior run) + **node 21 (this run)** | Prior run: shared tool-calling loop stabilized against malformed calls (same edit as tool_wrapper.py/mas_workflow.py above, one combined `mixed`/`harness_heavy` fix). This run: injects budget-awareness warnings into that same shared loop so the LLM self-terminates gracefully before exhausting its iteration budget (reduces `tool_calling_budget_exceeded`) |
| `agents/accounting.py` | node 17 then node 26 (prior run) | node 17: deterministic budget-consistency verifier — parses line-item prices from the sightseeing body, recomputes totals, patches mismatched Budget Summary numbers. node 26: fixed the verifier's regexes to handle the "×" multiplication sign and plural forms |
| `agents/sightseeing.py` | node 26 then node 32 (prior run) + **node 7 (this run)** | node 26: added a mandatory trip-requirements extraction step to the prompt (reduces constraint-misreading/apply-info-incorrectly). node 32: deterministic time-feasibility verifier patching travel_city spans to match tool-reported durations. This run's node 7: extended SELF-CHECK (meal/business-hours/attraction-coverage) + a deterministic ends-with-accommodation verifier |
| `mas_workflow.py` | node 11 (prior run) + **node 12 (this run)** | Prior run: part of the same schema-validation stabilization fix. This run: Train now reads Flight's output message and skips intercity legs Flight already booked (dedup fix) |
| `agents/train.py` | **node 12 (this run)** | Gains visibility into Flight's output message to prevent duplicate intercity legs (paired with the mas_workflow.py change above) |

No other files differ from the original seed — `workflow.py` and `agents/immutable/` remain untouched throughout, as required by `mutable_exclude`.
