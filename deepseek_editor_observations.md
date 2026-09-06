# DeepSeek v4 as `agent_editor` — observations

Model: `deepseek/deepseek-v4-pro-0813` via OpenRouter
(`https://openrouter.ai/api/v1`), `reasoning_effort: medium`. `block_suggester`
left on the unchanged local Qwen3.5-122B-A10B endpoint. Test bed:
`configs/hgm_travel_deepseek_editor_sanity_1case.yaml` — a single easy case
(case "1", 2-day Hefei→Nanjing trip), X=50/Y=5 sizing, `verbose: true`.

## Headline result — early sample (14 EXPANDs, 2026-09-04 evening)

| | |
|---|---|
| Edits that failed outright | **7 / 14 (50%)** |
| Edits that succeeded (parsed, validated, wrote real files) | 7 / 14 |
| Best resulting node | node 11, **0.9375** (single eval — noisy case, treat cautiously) |
| Other real improvements | node 3: 0.51 avg / 5 evals; node 7: 0.656 avg / 4 evals |
| Seed baseline | 0.37–0.75 across repeats (this case is itself noisy) |
| Process crashes | **0** |

## Headline result — full run to date (81 EXPANDs, 29 successful rounds, 2026-09-05)

The failure rate held at scale rather than improving with more samples:

| | |
|---|---|
| Total EXPAND attempts | 81 |
| Edits that failed outright | **53 / 81 (65%)** |
| Edits that succeeded | 28 / 81 |
| Best node | node 64, **0.825** avg / 5 evals (well-sampled) |
| Other strong, well-confirmed nodes | node 31: 0.754 / 15 evals; node 42: 0.715 / 9 evals; node 32: 0.633 / 8 evals |
| Process crashes | **0** (held at scale too) |

**Failure breakdown, counted precisely from the run log:**

| Signal | Count |
|---|---|
| Individual LLM responses with malformed/unparseable JSON (`_raw_arguments` fallback) | **48** |
| Individual LLM responses that skipped calling `submit_self_improvement` entirely | **71** |
| EXPANDs whose *final* (last-attempt) failure reason was malformed JSON | 23 |
| EXPANDs whose *final* failure reason was the generic "no file edits" (no-tool-call or empty-files) | 30 |

(Each EXPAND gets up to 2 attempts; a node can hit one failure mode on
attempt 1 and a different one on attempt 2, so the raw per-response counts
exceed the per-node breakdown.) Bottom line: DeepSeek is *capable* of real,
well-targeted fixes when it works — two of its best nodes (below) are
genuinely sharp diagnoses — but it fails to produce a usable tool call on
this task roughly two-thirds of the time, and that rate did not improve
as the search accumulated more attempts.

**What the best nodes actually did**, for context on the diagnosis quality
when it *does* work:
- **node 64** (0.825, `agents/sightseeing.py`): *"Eliminate the 0.0 no-plan
  failures by making the sightseeing wrap-up robust against
  stochastic/truncated `<itinerary>` tag emission (lenient extraction +
  escalating retry loop)."* — this independently rediscovered and targeted
  the exact wrap-up-retry reliability problem this session traced by hand
  earlier (the `task_failure`/`output_truncated` split work).
- **node 31** (0.754, `mas_workflow.py` + all three stage files): added a
  one-step recovery loop so a failed Sightseeing stage retries with
  diagnostic context instead of terminating empty, and propagates
  upstream Flight/Train failures for detection.

## The dominant failure mode

Almost every failure followed the same two-step pattern across independent
attempts (nodes 1, 4, 5, and others):

1. **Attempt 1**: the model responds with prose/reasoning and never calls
   `submit_self_improvement` at all (falls into `agent_editor.py`'s
   "model did not call submit_self_improvement" fenced-JSON-recovery path,
   which usually recovers 0 files).
2. **Attempt 2** (after the retry prompt): the model *does* call the tool,
   but its `arguments` string fails to parse as JSON — sizes observed:
   19817, 13099, 3256, 17182, 15402, 273 chars. `platform_core/llm_wrapper.py`
   wraps this as `{"_raw_arguments": <raw string>}`.

Manually reproducing the exact same prompt+context outside the live loop
(2026-09-04, earlier in this session) showed DeepSeek *can* produce a
complete, well-reasoned, multi-file fix (a new `agents/validator.py`
structural validator, correctly wired into `mas_workflow.py`, correctly
recognizing and routing around the `agents/immutable/` edit restriction) —
so the underlying reasoning/diagnosis quality is not the problem. The
JSON itself breaks, almost certainly from a single mis-escaped character
somewhere deep in tens of KB of embedded Python source (triple-quoted
strings, regex patterns, nested quotes all have to survive perfect JSON
escaping in one giant string; one slip breaks the whole parse).

## Fixes shipped in response (this session, `meta_agent/agent_editor.py`)

1. Detect the `_raw_arguments` fallback explicitly and treat it as a
   distinct, actionable retry case — the retry prompt now says outright:
   *"Make sure you return a valid JSON object: double-check that every
   string value — especially each file's `content` — has its quotes,
   backslashes, and newlines properly escaped."* (Previously this landed
   as the generic, unhelpful "editor returned no file edits.")
2. Also guard the case where the arguments parse as *valid* JSON but
   aren't an object (bare list/string/null) — this previously crashed the
   entire HGM process with an uncaught `AttributeError` (nothing above
   `editor.apply()` catches exceptions, all the way up to
   `main_loop.py`'s `fw.manager.evolve()` call). Confirmed via direct
   reproduction before the fix; confirmed silent/safe after.
3. Fixed two adjacent `target_files=["workflow.py"]` placeholders that
   misrepresented what actually happened on a failed/fallback parse —
   now `[]` when nothing was recovered, or the real recovered paths when
   fenced-JSON recovery did find something.

All three landed with unit tests (`tests/test_agent_editor_malformed_json_recovery.py`,
6 tests) and were verified against this exact sanity run: after the fix,
malformed-JSON attempts no longer crash the process and get a specific,
actionable retry message instead of the generic one.

## What this does and doesn't tell us

- **Does tell us**: DeepSeek v4 (via this OpenRouter route) has a real,
  reproducible JSON-formatting/tool-calling reliability gap specifically
  for this large, code-heavy multi-file editing task — not a one-off
  fluke, and not (based on the direct reproduction, and the two strong
  nodes it eventually did produce) a diagnosis-quality problem. The rate
  held at **65% over 81 attempts**, i.e. it did not self-correct or
  improve as the search accumulated more rounds/retry examples.
- **Does not tell us**: whether this is inherent to the model itself, an
  artifact of OpenRouter's routing/serving of it, or something a
  prompt-level mitigation (e.g. asking for one file at a time, or a
  stricter "no unescaped control characters" reminder) could reduce
  further. Not isolated in this pass.
- **Does not tell us**: how this compares like-for-like to the local
  Qwen3.5-122B-A10B endpoint on the *same* single-case test — no
  equivalent controlled n=81 run was done with Qwen as editor in this
  session for a direct comparison. Qwen has not shown this specific
  JSON-malformation failure mode anywhere else in this session's much
  heavier use as `block_suggester`/`failure_summarizer`/`editor`
  (including in the concurrently-running full-scale production run on the
  full 60-case train split), which is suggestive but not a controlled
  comparison.

## Root-cause debugging session (2026-09-05)

The sanity run was killed (81 EXPANDs was enough signal; further rounds
weren't adding new information) to spend the compute on isolating the
actual cause instead. All artifacts were already on disk (`verbose: true`
had been on the whole time), so this needed no new sanity-run traffic for
the first pass.

**Step 1 — mine the existing verbose artifacts** (`debug_deepseek_json_failures.py`,
committed to the repo root; scans every `round_NNN/verbose/
editor_attempt_N_response.json`, re-parses each captured `_raw_arguments`
string with Python's own `json` module to get the exact error/position):

- 49 malformed-JSON instances found across the run's verbose logs.
- Every single one is `"Unterminated string starting at"` — not truncation
  (only 2/49 broke in the last 5% of the string; 47/49 broke well before
  the end) and not a scattered mix of error types.
- **36 of 49 (73%) broke specifically while writing `agents/sightseeing.py`'s
  `content` field** — by far the single dominant file (next highest:
  `mas_workflow.py`/`agents/common.py` at 3 each). This is the largest,
  most quote-and-formatting-heavy file in the seed.
- A naive "append closing brackets" repair recovers *syntactically* valid
  JSON for 96% of instances — but this is a false promise: since the
  unterminated string swallows everything after the real break point, the
  "recovered" `content` for the broken file is truncated mid-file, not
  the intended content. `_write_edits` would write a syntactically broken
  Python file, which the `syntax` validator should then reject anyway —
  so a bracket-repair fallback would not actually be a useful recovery
  step, just a more expensive way to still fail. (Revises the original,
  more optimistic "consider a repair-parse fallback" note below.)
- The failure rate is **not concentrated on retries**: attempt 1 malformed
  rate (26/82 = 32%) and attempt 2 (23/66 = 35%) are statistically the
  same. The retry's added "make sure you return valid JSON" guidance
  neither helps nor hurts measurably.

**Step 2 — controlled A/B tests against the live OpenRouter endpoint**,
reproducing conditions directly (`deepseek_json_ab_test.py`,
`deepseek_json_ab_test2.py` in the scratch dir), each asking DeepSeek to
make one trivial, well-defined edit (add a docstring comment) and echo
back real seed file(s) unchanged otherwise:

| Config | Result |
|---|---|
| Single file (`sightseeing.py`, ~10KB), `reasoning_effort: medium` | **6/6 valid JSON** |
| Same, with `strict: true` schema (constrained-decoding shape) | **6/6 valid JSON** |
| Single file, low temperature (`0.1`), no `reasoning_effort` | 5/6 valid — worse, not better |
| Single file, `strict: true` + low temperature | 5/6 valid — worse, not better |
| **4 real files bundled** (`mas_workflow.py`, `flight.py`, `train.py`, `sightseeing.py`, ~30KB total), `reasoning_effort: medium` | **8/8 valid JSON** |

None of these reproduced *any* failure under `reasoning_effort: medium` —
ruling out both "large single file" and "multiple bundled files" as the
trigger by themselves, and ruling out `strict: true` mode and lower
temperature as fixes (temperature reduction measurably hurt).

**Step 3 — the decisive test**: reuse the REAL `agent_editor.py` code path
end-to-end (`deepseek_strict_real_prompt_test.py`) — the actual full
"hard rules" system prompt, a real saved `block_suggestion.md` from the
run, real project context, `strict: true` injected into
`SELF_IMPROVEMENT_TOOL` with `platform_core.llm_wrapper._normalise_tool_schema`
patched in-process to stop silently dropping it (confirmed live: it does
drop `strict` today, for every tool schema shape). Result on the first 5
of 8 trials: **5/5 still failed** (3 malformed JSON, 2 no-tool-call) —
`strict: true` provides no protection on the real prompt either.

**Conclusion (superseded below)**: the trigger is not payload size, file
count, schema strictness, or temperature — it only appears once the task
requires genuine diagnosis and generation of new content under the real,
full system prompt (9 hard rules, verification instructions, deviation
disclosure, etc.), never on simplified reproductions that just ask it to
echo back existing, already-valid text with a trivial addition. This
looked like a text-generation reliability issue specific to *composing*
non-trivial new content at this scale, not a structural/schema/decoding
knob — until the agentic experiment below overturned that.

## BREAKTHROUGH: agentic read_file/write_file/run_code_validators tools (2026-09-05)

Every fix attempted so far kept the single-shot "bundle the whole diagnosis
+ every changed file into one `submit_self_improvement` tool call" shape.
The one thing not yet tried was changing that shape itself: give the model
**agentic, multi-turn tool access** -- `read_file(path)`, `write_file(path,
content)` (one file per call, never bundled), `run_code_validators()` (the
REAL configured validator stack, run live against the actual on-disk
workspace via `fw.editor._run_validators`), and `submit_done(...)` to
finish -- instead of one large structured call carrying everything at once.
Prototype: `debug_deepseek_agentic_prototype.py` (repo root).

First attempt (8-turn cap, loose instructions) was inconclusive: it spent
its entire budget exploring the codebase (reading 4-5 files, some
nonexistent paths, one premature validator check) and never reached
`write_file` at all. Tightened the prompt (told it exactly which one file
to change, to read it once, then write immediately, and raised the turn
cap to 16) and re-ran:

| | |
|---|---|
| Trials | 8 |
| Trials that completed (`submit_done` reached) | **8 / 8** |
| Total `write_file` calls across all trials | 12 |
| `write_file` calls with valid JSON | **12 / 12 (100%)** |
| `write_file` calls with malformed JSON | **0** |

Zero JSON malformation across every single call -- a complete reversal of
the ~65% failure rate seen with the bundled shape. Even more notable: in
3 of 8 trials, the *first* `write_file` had a genuine bug in the generated
Python itself (a real `SyntaxError`, an unterminated string literal in the
written code, a smoke-test crash) -- caught by the real
`run_code_validators` tool, and DeepSeek successfully self-corrected with
a second `write_file` call, still perfectly valid JSON, until validators
passed. That's the full intended loop (diagnose -> write -> validate ->
self-correct -> finish) working end-to-end, with the framework's own real
validators in it, not a mocked check.

**Interpretation**: the earlier conclusion was too pessimistic. The
failure wasn't an unfixable text-generation reliability ceiling -- it was
specific to the *shape* of the single bundled call (multiple large,
quote-heavy strings packed into one JSON object alongside
optimization_goal/proposed_changes/rationale). Separating "decide what to
write" from "commit one file's content" into independent tool calls, each
carrying exactly one large string and nothing else, removed the failure
mode entirely in this sample. This also revives the "agentic access"
option that was earlier deferred as too big an architecture change for
`block_suggester`/`agent_editor` (see the original discussion above the
DeepSeek section) -- the payoff, at least for this model, turned out to be
much larger than the deferred cost estimate assumed.

**Caveats before generalizing**: n=8 on one easy case, one file
(`agents/sightseeing.py`) always being the one changed -- not yet tested
with multiple files needing changes in one EXPAND (a real, common case,
e.g. node 31's fix from the earlier sanity run touched 4 files), and not
yet wired into the real `agent_editor.py`/HGM loop (this is a standalone
prototype, not a production change).

**Follow-up: does it hold with genuinely multiple files? (2026-09-06)**
Re-ran the same architecture (`debug_deepseek_agentic_multifile_test.py`)
but replaced the single-file task with node 31's real 4-file diagnosis
reused verbatim from the earlier sanity run: make flight/train stage
failures explicit instead of silent, add a one-step diagnostic retry to the
sightseeing stage, and wire the failure signals through
`mas_workflow.py` -- a genuinely coordinated change requiring *different*
edits to `mas_workflow.py` + `agents/flight.py` + `agents/train.py` +
`agents/sightseeing.py` in the same trial (not the same comment pasted into
4 files, which the earlier bundled-call A/B test had already shown was
trivially easy).

Result: **8/8 trials completed (`submit_done` reached), 8/8 wrote all 4
target files, 32/32 total `write_file` calls returned valid JSON -- 0
malformed.** Every trial followed the intended shape: one `read_file` per
target file (often batched in a single turn), one `write_file` per file
(observed as either 4 sequential calls or a batch of several in one turn,
e.g. trial 7's turn 1 fired all 4 `write_file` calls together and every one
parsed clean), one `run_code_validators` call, then `submit_done` -- no
trial needed a validator-driven fix-up retry this time (unlike the
single-file batch's 3/8 self-corrections), consistent with this task's
edits being smaller/more mechanical per file even though there were more of
them.

This directly answers the caveat above: the fix is not specific to
`agents/sightseeing.py` being the one file in play, and per-file JSON
validity does not degrade when a trial has to touch multiple different
files -- each `write_file` call still carries exactly one file's content,
so the malformed-JSON failure mode (tied to the *bundled* multi-file
`files: [...]` array shape in the original `agent_editor.py`) simply never
has an opportunity to occur, regardless of how many files the underlying
fix needs. The updated, harder caveat: still n=8, still one specific task
family (silent-failure/retry plumbing), and still not wired into the real
`agent_editor.py`/HGM loop.

## Open follow-ups

- ~~Re-run with a larger sample to see whether the failure rate holds,
  worsens, or improves.~~ **Done**: held (50% → 65% over 81 attempts, if
  anything slightly worse at scale, not better).
- ~~Isolate root cause further.~~ **Done** — see "Root-cause debugging
  session" above: 73% of failures are `agents/sightseeing.py` specifically,
  always "unterminated string," not truncation.
- ~~Consider a repair-parse fallback.~~ **Downgraded to not worth building**:
  the recovered content would be truncated mid-file in most cases (the
  string swallows everything after the real break), so a repair fallback
  would mostly just relabel a failure, not recover a working fix.
- ~~Try `strict: true` / lower temperature / no reasoning_effort.~~ **Done,
  none helped**: `strict: true` gave 0/5 successes on the real prompt;
  lower temperature without `reasoning_effort` made the simplified tests
  measurably worse.
- Still no controlled Qwen-as-editor comparison on this exact single-case
  setup, to isolate "is this DeepSeek-specific" from "is this task-shape
  hard for any model here."
- Not yet tried: splitting the tool schema to request one file per call
  (forcing smaller individual string values) rather than bundling all
  changed files into one `files` array — untested because the size/count
  A/B tests never reproduced a failure to test a fix against; would need
  testing directly against the real prompt shape instead, following
  Step 3's methodology.
- Not yet tried: asking DeepSeek to return code base64-encoded or
  otherwise avoid embedding large, quote-heavy raw source as an inline
  JSON string value at all — a structural workaround rather than a
  prompt/schema tweak, bypassing the escaping problem instead of solving
  it.
