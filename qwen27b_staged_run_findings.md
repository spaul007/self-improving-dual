# Qwen-27B staged run: findings on non-submission

Source data: `logs/harness_staged_run_27b_medium.log`, `logs/run_staged_27b_medium.sh`,
`harness_staged_run_27b_medium/stage_5/work/` (transcript.jsonl, develop.json). Meta-agent:
`Qwen/Qwen3.8-27B` on local vLLM (node-6). Task agent throughout: local `Qwen/Qwen3.5-35B-A3B`
(never changed, never at risk here). Starting workspace: `harness_redesign_run_1/work/workspace`.

## Config used (exact, from the launch script)

| Setting | Value |
|---|---|
| Meta-agent model | `Qwen/Qwen3.8-27B` |
| Endpoint | `http://gpu-aic-mv-02-st-p5-node-6:8001/v1` (local) |
| Reasoning effort | `medium` |
| Max output tokens | `49152` |
| Turn budget per stage | `200` (`--stage-max-turns`) |
| Eval calls per stage | `4` (default, not overridden) |
| In-loop sample size | `8` cases (default; `evaluate_variant` did not yet support `case_ids` at this point, so all 4 calls per stage were against the same 8-case sample, never the full 60) |

## Stage outcomes: two distinct non-submission failure modes

Of 6 stages, 4 ended without `submit_variant` ever being called. These split into two
genuinely different phenomena that shouldn't be conflated:

| Stage | Unit | Turns used | Eval calls used | Why it ended |
|---|---|---|---|---|
| 1 | Route Consistency | 55/200 | 0/4 | **API timeout** -- `call_llm timed out after 603s` on turn 54 |
| 2 | Route Consistency | 158/200 | 1/4 | Submitted; rolled back on the gate (not a submit failure) |
| 3 | hard constraints: restaurant | 55/200 | 0/4 | **API timeout**, turn 54 |
| 4 | hard constraints: restaurant | 34/200 | 0/4 | **API timeout**, turn 33 |
| **5** | **hard constraints: transport (train/flight)** | **200/200** | **2/4** | **Genuinely exhausted its turn budget** -- never called `submit_variant` |
| 6 | hard constraints: transport (train/flight) | 122/200 | 0/4 | **API timeout**, turn 121 |

- **Timeout stages (1, 3, 4, 6):** the meta-agent's own LLM call to node-6 timed out after
  603s mid-session. This ends the session early regardless of remaining turn/eval budget --
  an infra/congestion issue (node-6 is shared), not the model running out of allowance.
- **Stage 5 is the clean "ran out of calls without submitting" case:** it used its full
  200-turn allowance and still had not submitted. Its `develop.json`:
  ```json
  {
    "submitted": false, "summary": "", "turns_used": 200,
    "tool_usage": {"read_file": 46, "list_cases": 1, "grep": 10, "run_python": 100,
                   "write_file": 6, "str_replace_file": 37, "evaluate_variant": 2},
    "eval_calls_used": 2, "evaluated_in_loop": true, "meta_call_failed": false
  }
  ```
  Its last two recorded turns (198-199) show it still mid-debug, not winding down toward a
  decision: it tried to introspect its own code offline via `run_python` using `inspect` /
  `__code__` / `__doc__`, all rejected by the sandbox's allow-list (`run_python REJECTED
  (nothing executed): attribute '__code__' not allowed; attribute '__doc__' not allowed`).
  The turn budget ran out while it was still trying to figure out whether its own edit had
  actually taken effect.

## The prompt it used

The harness does not persist the literal system/user prompt text to disk (`transcript.jsonl`
only logs tool calls, not the system/user messages), and the prompt template
(`STAGE_SYSTEM` in `experiment_harness_staged.py`) has been revised multiple times since this
run (Sep 22, ~06:17-07:50) in the course of debugging a later DeepSeek session. What follows
is the **current** template rendered with stage 5's real values (unit, checkpoint composite,
budgets) -- structurally the same prompt family, but **not** a byte-exact historical copy.

Known, material differences from what stage 5 actually saw:

- **`check_workspace` did not exist yet.** No fast (near-instant) pre-check tool combining
  policy/syntax scan, HGM's real validator suite, and a real-plan-produced check. Only
  `run_python` (offline, no LLM) and the expensive, call-limited `evaluate_variant` existed.
- **The gate was described as composite-based, not unit-score-based.** The real line at the
  time would have read approximately *"ACCEPTS it only if the composite improves over the
  current checkpoint (currently 0.3563)"* -- the unit-score gate (and the more honest
  description of it below) was added later in the same debugging session after finding
  composite too noisy a signal for a single-unit stage.
- **Neither of two later guidelines existed:** (a) put a restriction in the tool schema, not
  just in the prompt text, if the LLM must not call a tool; (b) before submitting, verify your
  own summary's claims against what the code actually does. Both came directly out of
  debugging a DeepSeek session that submitted a summary overstating what its own code did.
- **`evaluate_variant` did not accept `case_ids` yet** and always ran the small 8-case
  in-loop sample -- never the full 60 by default. That default-to-60-by-default behavior is
  also a later change.

Rendered current template (stage 5's real numbers: unit = "hard constraints: transport
(train/flight)", checkpoint composite 0.3563, unit score 0.9423, `--min-gain` 0.005,
`{eval_rounds}`=4, `{max_turns}`=200):

```
You are a meta-agent continuing the redesign of a travel-planning harness around a WEAK runtime model, one focused stage at a time.

BACKGROUND. The task agent turns a traveler's request into a full day-by-day itinerary. A hidden grader scores every itinerary against many named checks;
you never see its source, only real graded runs. The workspace ALREADY contains a working code-owned pipeline (written in earlier sessions): code makes
all tool calls (through ToolWrapper), schedules the days and renders the required line format; the LLM is at most used for small validated subtasks.
The runtime model (Qwen3.5-35B-A3B, about 3B active parameters) is unreliable: it skips tool calls, fabricates names and prices, and cannot compose a
long itinerary in one generation. So keep everything deterministic in code, keep any LLM call small and constrained (validated by code, with a code
fallback), and never let the model write the itinerary.

GRADING STRUCTURE (visible in the logs' dimension_scores / hard_score): composite = (commonsense + hard) / 2. Each commonsense dimension counts (1/8) only
if EVERY check in it passes; the hard score counts only if EVERY hard constraint of the case passes (they depend on what the traveler asked for). A case
with no plan scores 0. Fixing one check rarely moves the score; whole dimensions must pass consistently, and a fix for one check must not break others.

THIS STAGE'S UNIT: hard constraints: transport (train/flight)
[real failing-check breakdown for this unit, computed from the checkpoint train evaluation -- see describe_unit() in experiment_harness_staged.py]

HOW YOUR WORK IS JUDGED. After your session the harness re-evaluates your workspace on ALL 60 train cases and ACCEPTS it only if THIS UNIT's own score
(currently 0.9423 on train) improves by at least 0.005 AND the no-plan rate does not get worse; otherwise your whole stage is rolled back.
(Overall composite is currently 0.3563 and is reported alongside, but it is NOT itself the accept/reject gate -- it blends 8 other units
this stage does not touch, so it is a much noisier signal for a single-unit fix than the unit's own pass rate.) So improve this unit WITHOUT regressing
what already works (the pipeline already satisfies most scheduling checks -- do not rewrite the scheduler; extend it). A crash or an empty result on any
case is a regression, and will worsen the no-plan guard even if it happens not to touch this unit's own checks.

RULES
- Keep the workflow.py -> mas_workflow.run_task(task) -> AgentOutput contract; the final result is the complete itinerary text. Extend the existing
  code in small, testable pieces (add functions/modules; do not restart from scratch). No hard-coding of case ids, cities or answers; general behavior only.
- No file/network/process/env access in the code you write. The model and endpoint are fixed (local Qwen3.5-35B): do not try to change them.
  workflow.py, tool_wrapper.py, tools_schema.json and agents/immutable/ are frozen.
- If you don't want the LLM to call a tool for a given case (e.g. because code already decided the answer deterministically), remove that
  tool from the schema you pass to run_tool_stage for that case -- do NOT just tell it not to call the tool in the prompt text. Confirmed
  on this exact codebase: told not to call a specific tool while that tool stayed in its schema, the weak model called it anyway in every
  sampled case, and at least once used the tool's own result over the value it was given, breaking a case that passed before. A schema the
  model literally cannot call is enforced; a request in the prompt is not, no matter how explicit.
- run_python is free and makes no LLM calls. check_workspace's static/policy checks are free too, but it also makes ONE real local-LLM run on a
  fixed train case to confirm a plan is actually produced (does not consume an evaluate_variant call). evaluate_variant evaluates ALL 60 train
  cases by default (takes a while) -- pass case_ids (train ids only) to check just a specific subset faster while iterating. You have
  4 evaluate_variant calls total. Turn budget: 200.

WORKFLOW
1. Read corpus/digest.md and the train cases where this unit fails (list_cases with failed_check=..., show_case) -- these are the CURRENT pipeline's own
   plans and the grader's messages. Read the relevant workspace code (start with mas_workflow.py and the planner).
2. Use run_python heavily: print REAL data (plans, tool outputs) first and parse those; do not assume formats from prompts (square brackets in the
   format spec are placeholders, not literal). Test offline on the stored train requests (tool lookups work offline through ToolWrapper after
   train_data.use_case(cid); LLM calls do not) before spending an evaluate_variant call.
3. After editing, call check_workspace (full policy/syntax scan, HGM's real validator suite, and a check that a real train case still produces a
   plan) to catch cross-file mistakes and no-plan regressions BEFORE spending a limited evaluate_variant call on something that would just crash,
   get rejected, or silently produce no plan.
4. Implement, then evaluate_variant -- pass case_ids=[the train ids where this unit currently fails] for a fast targeted check while you're
   still iterating, and call it with no case_ids (all 60) once you believe it's fixed, since that full-60 result is what decides accept/rollback.
   Read the per-case results, refine. Only submit_variant once the unit improves and nothing else got worse.
5. Before calling submit_variant, re-read your OWN changed code and check every claim you are about to write in the summary against what the
   code actually does line by line -- not what you intended it to do. A summary that overstates what a change does (e.g. "the LLM no longer
   decides X" when the LLM's tool schema still lets it decide X) is worse than an accurate but modest one: it hides exactly the failure mode
   most worth catching before this gets built on further.
6. submit_variant with a short summary of what you changed.
```

(Items 3 and 5 in WORKFLOW, and the `check_workspace` mentions throughout, are the parts that
did **not** exist when stage 5 actually ran -- included here so the template is complete and
usable for a future re-run, not because stage 5 saw them.)

## Bottom line

Qwen-27B's non-submissions in this run were roughly evenly split between infra timeouts
(node-6 congestion, not a capability signal) and one genuine turn-budget exhaustion while
stuck debugging (a real capability/tooling signal -- it needed a cheap way to verify its own
edit had taken effect, which `check_workspace` now provides). It was also working without the
unit-score gate, the tool-schema-vs-prompt-instruction guideline, and the
verify-summary-against-code guideline -- all added later in response to issues found in a
subsequent DeepSeek session. A re-run with the current scaffolding is not yet available for
comparison.
