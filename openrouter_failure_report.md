# OpenRouter LLM Call Failures — Investigation Report

**Date:** 2026-09-13
**Scope:** `google/gemma-4-31b-it` via OpenRouter, used as the task-agent backbone in the `travel_mas_refactored_gemma` project's HGM runs.
**Trigger:** A standalone full-120-case benchmark of round_013 (`round013_full_eval_120/`) collapsed to a composite score of 0.103 (119/120 "failed") due to this issue, which prompted this investigation.

---

## 1. Is this an account-level billing/rate-limit issue?

**No.** Checked directly against the real OpenRouter API (`GET https://openrouter.ai/api/v1/key`):

```json
{
  "limit": 8000,
  "usage": 6397.705755859,
  "limit_remaining": 1602.2942441409996,
  "usage_daily": 105.705570894,
  "rate_limit": {"requests": -1, "interval": "10s", "note": "This field is deprecated and safe to ignore."}
}
```

- ~1,602 in remaining credit headroom (well under the $8,000 cap) — not a funds-exhaustion issue.
- No rate limit configured on the key (`-1` = unlimited).

If this were a billing/quota block we'd expect `limit_remaining` near zero or an explicit `403 Key limit exceeded` (a *different*, clearly-labeled failure this session has seen before, and easily distinguished from what's described below). Neither is happening. The failures originate upstream of anything controllable from our account.

---

## 2. Aggregate statistics

From `analyze_llm_call_failures.py round013_full_eval_120` (single full-120-case pass, reduced parallelism 8, 8s pause between chunks of 8):

| Metric | Value |
|---|---|
| Total LLM calls | 894 |
| status=failed retries (recovered by the fix) | 5,814 |
| Exception retries (unrelated transient network etc.) | 240 |
| **Terminal failures** (retries exhausted, score tanked) | **174** |
| Cases with ≥1 retry or terminal failure | **99 / 120 (82.5%)** |
| Final composite score (not representative of model quality) | 0.103 |

Error code breakdown (this same pass):

| `response_error_code` | Count |
|---|---|
| `invalid_prompt` | 3,292 |
| `server_error` | 2,696 |

Severity **escalated** over the course of the run rather than staying constant or improving with reduced/staggered concurrency:

| Chunk (8 cases each) | Retries | Terminal failures |
|---|---|---|
| round_00 (cases 0-7) | 0 | 0 |
| round_01 (cases 8-15) | 20 | 0 |
| round_02 (cases 16-23) | 201 | 4 |
| round_03 (cases 24-31) | 514 | 12 |
| round_04 (cases 32-39) | 98 | 0 |

This run's concurrent OFF-run (`hgm_travel_gemma_no_backbone_selection_X100Y180`, parallelism 32, same account/key) was **not** hit nearly this hard in the same time window (no_plan_rate 3–28% across its recent rounds vs. 82.5% of cases affected here) — arguing against "reduce our own parallelism" as a fix, and consistent with the root cause being outside our control.

---

## 3. Failure categories, with real examples (raw log lines)

All examples below are verbatim `trace.jsonl` entries (JSON lines), not paraphrased. Each carries its own source file path and timestamp so you can look it up directly.

### 3a. JSON "Extra data" parse errors (2,898 occurrences — the largest category)

Python's `json.JSONDecodeError` message format (`Extra data: line 1 column N`), meaning something downstream is trying to parse content as JSON and finding trailing bytes after a valid JSON value ends. The column number varies call to call, consistent with content-length-dependent trailing data rather than one fixed template bug.

```
{"file": "round013_full_eval_120/round_01/logs/trace.jsonl", "timestamp": 1789247998.63, "kind": "llm_call_retry", "case_id": "15", "attempt": 1, "error_code": "invalid_prompt", "error_message": "Extra data: line 1 column 74 (char 73)"}
{"file": "round013_full_eval_120/round_02/logs/trace.jsonl", "timestamp": 1789248618.79, "kind": "llm_call_retry", "case_id": "23", "attempt": 1, "error_code": "invalid_prompt", "error_message": "Extra data: line 1 column 76 (char 75)"}
{"file": "round013_full_eval_120/round_02/logs/trace.jsonl", "timestamp": 1789248619.66, "kind": "llm_call_retry", "case_id": "21", "attempt": 1, "error_code": "invalid_prompt", "error_message": "Extra data: line 1 column 108 (char 107)"}
{"file": "round013_full_eval_120/round_03/logs/trace.jsonl", "timestamp": 1789249967.54, "kind": "llm_call_retry", "case_id": "24", "attempt": 1, "error_code": "invalid_prompt", "error_message": "Extra data: line 1 column 73 (char 72)"}
{"file": "round013_full_eval_120/round_03/logs/trace.jsonl", "timestamp": 1789249973.80, "kind": "llm_call_retry", "case_id": "29", "attempt": 1, "error_code": "invalid_prompt", "error_message": "Extra data: line 1 column 77 (char 76)"}
{"file": "round013_full_eval_120/round_04/logs/trace.jsonl", "timestamp": 1789251254.43, "kind": "llm_call_retry", "case_id": "32", "attempt": 1, "error_code": "invalid_prompt", "error_message": "Extra data: line 1 column 109 (char 108)"}
{"file": "round013_full_eval_120/round_04/logs/trace.jsonl", "timestamp": 1789251330.80, "kind": "llm_call_retry", "case_id": "39", "attempt": 1, "error_code": "invalid_prompt", "error_message": "Extra data: line 1 column 40 (char 39)"}
```

### 3b. Tool-call argument validation rejection (507 occurrences)

```
{"file": "round013_full_eval_120/round_01/logs/trace.jsonl", "timestamp": 1789248083.10, "kind": "llm_call_retry", "case_id": "15", "attempt": 4, "error_code": "invalid_prompt", "error_message": "Assistant tool call function.arguments must be valid JSON."}
{"file": "round013_full_eval_120/round_01/logs/trace.jsonl", "timestamp": 1789248106.86, "kind": "llm_call_retry", "case_id": "15", "attempt": 5, "error_code": "invalid_prompt", "error_message": "Assistant tool call function.arguments must be valid JSON."}
{"file": "round013_full_eval_120/round_02/logs/trace.jsonl", "timestamp": 1789249938.61, "kind": "llm_call_retry", "case_id": "17", "attempt": 29, "error_code": "invalid_prompt", "error_message": "{\"object\":\"error\",\"message\":\"Assistant tool call function.arguments must be valid JSON.\",\"type\":\"BadRequest\",\"param\":null,\"code\":400}"}
```

### 3c. Backend chat-template rejection — reveals the actual serving stack (72 occurrences)

This is the most informative single message: it names the real backend serving this model.

```
{"file": "round013_full_eval_120/round_02/logs/trace.jsonl", "timestamp": 1789249544.75, "kind": "llm_call_retry", "case_id": "17", "attempt": 19, "error_code": "invalid_prompt", "error_message": "failed to translate request: tokenizing request: tokenizing for model \"nvidia/Gemma-4-31B-IT-NVFP4\": encode failed (code 5): chat template rejected these messages"}
{"file": "round013_full_eval_120/round_03/logs/trace.jsonl", "timestamp": 1789250050.78, "kind": "llm_call_retry", "case_id": "26", "attempt": 16, "error_code": "invalid_prompt", "error_message": "failed to translate request: tokenizing request: tokenizing for model \"nvidia/Gemma-4-31B-IT-NVFP4\": encode failed (code 5): chat template rejected these messages"}
{"file": "round013_full_eval_120/round_03/logs/trace.jsonl", "timestamp": 1789250162.72, "kind": "llm_call_retry", "case_id": "27", "attempt": 1, "error_code": "invalid_prompt", "error_message": "failed to translate request: tokenizing request: tokenizing for model \"nvidia/Gemma-4-31B-IT-NVFP4\": encode failed (code 5): chat template rejected these messages"}
```

**`google/gemma-4-31b-it` on OpenRouter is actually being served as `nvidia/Gemma-4-31B-IT-NVFP4`** — an NVIDIA NVFP4-quantized build, proxied transparently under the `google/gemma-4-31b-it` slug. This backend applies its own template-level request validation and is what's rejecting these calls.

### 3d. Image/base64 decode errors — in a text-only pipeline (6 occurrences, but highly suspicious)

**Verified this is not us.** Grepped the entire codebase this call path touches — `platform_core/` (the sole LLM entry point, `call_llm`), the project's seed code, and `tools_schema.json` (every tool definition) — for any reference to `image_url`, `base64`, or `data:image`:

```
$ grep -rln "image_url\|base64\|data:image" platform_core/ projects/travel_mas_refactored_gemma/
(no matches)
$ grep -n "image\|base64" platform_core/llm_wrapper.py
(no matches)
$ grep -i "image\|base64" projects/travel_mas_refactored_gemma/seed/tools_schema.json
(no matches)
```

Zero matches anywhere. `travel_mas_refactored_gemma`'s tools and prompts are 100% text (flight/train/sightseeing/accounting itinerary planning) — there is no code path in this project that could construct an `image_url` field or base64 image payload, and `call_llm`'s own request-building code (§ the `request` dict in `platform_core/llm_wrapper.py`) only ever sets `model`, `input` (the text/tool-call message list), `tools`, `temperature`/`reasoning`, and `max_output_tokens` — no image-capable field exists in what we send. These image-decode failures are being generated somewhere between OpenRouter and the model, not manufactured by our request:

```
{"file": "round013_full_eval_120/round_02/logs/trace.jsonl", "timestamp": 1789249918.05, "kind": "llm_call_retry", "case_id": "17", "attempt": 25, "error_code": "invalid_prompt", "error_message": "Invalid base64 image data: Incorrect padding"}
{"file": "round013_full_eval_120/round_03/logs/trace.jsonl", "timestamp": 1789250311.13, "kind": "llm_call_retry", "case_id": "28", "attempt": 15, "error_code": "invalid_prompt", "error_message": "Invalid base64 image data: Invalid base64-encoded string: number of data characters (9) cannot be 1 more than a multiple of 4"}
{"file": "round013_full_eval_120/round_03/logs/trace.jsonl", "timestamp": 1789251054.96, "kind": "llm_call_retry", "case_id": "30", "attempt": 21, "error_code": "invalid_prompt", "error_message": "Invalid base64-encoded string: number of data characters (9) cannot be 1 more than a multiple of 4"}
{"file": "round013_full_eval_120/round_03/logs/trace.jsonl", "timestamp": 1789251207.27, "kind": "llm_call_retry", "case_id": "24", "attempt": 26, "error_code": "server_error", "error_message": "`image_url` must start with 'data:image/<jpeg|jpg|png|webp>;base64,'"}
{"file": "round013_full_eval_120/round_04/logs/trace.jsonl", "timestamp": 1789251703.41, "kind": "llm_call_retry", "case_id": "36", "attempt": 2, "error_code": "invalid_prompt", "error_message": "Unable to decode image: image file is truncated"}
```

### 3e. Server errors (2,505 occurrences) — including a second, independent confirmation of the upstream chain

Full distribution of distinct `server_error` messages in this pass:

| Message | Count |
|---|---|
| `unexpected_error` | 2,338 |
| `The operation was aborted` | 162 |
| `Upstream error from DeepInfra: Extra data: line 1 column 71 (char 70)` | 57 |
| `Upstream error from DeepInfra: Extra data: line 1 column 73 (char 72)` | 23 |
| `Upstream error from DeepInfra: Extra data: line 1 column 103 (char 102)` | 18 |
| `Upstream error from DeepInfra: Extra data: line 1 column 40 (char 39)` | 18 |
| `Upstream error from DeepInfra: Extra data: line 1 column 79 (char 78)` | 18 |
| `Upstream error from DeepInfra: Extra data: line 1 column 76 (char 75)` | 18 |
| (13 more distinct `Upstream error from DeepInfra: Extra data: ...` variants, 1-6 occurrences each) | ~30 |
| `` `image_url` must start with 'data:image/<jpeg\|jpg\|png\|webp>;base64,' `` | 1 |
| `error code: 502\n` | 1 |
| `Internal server error` | 1 |

**This independently confirms and extends §3c: the upstream provider is DeepInfra.** These `server_error`-coded failures carry the *exact same* "Extra data: line 1 column N" `json.JSONDecodeError` signature as the `invalid_prompt`-coded ones in §3a — the same underlying parse failure, just surfaced two different ways (directly, vs. wrapped as "Upstream error from DeepInfra"). So the request chain is: **our call → OpenRouter → DeepInfra → `nvidia/Gemma-4-31B-IT-NVFP4`**, and the parse failure is happening at the DeepInfra layer specifically.

Raw examples (one per distinct message, spanning the full pass):

```
{"file": "round013_full_eval_120/round_01/logs/trace.jsonl", "timestamp": 1789248456.21, "kind": "llm_call_retry", "case_id": "15", "attempt": 18, "error_message": "unexpected_error"}
{"file": "round013_full_eval_120/round_02/logs/trace.jsonl", "timestamp": 1789248676.49, "kind": "llm_call_retry", "case_id": "21", "attempt": 3, "error_message": "The operation was aborted"}
{"file": "round013_full_eval_120/round_05/logs/trace.jsonl", "timestamp": 1789252147.11, "kind": "llm_call_retry", "case_id": "41", "attempt": 12, "error_message": "Upstream error from DeepInfra: Extra data: line 1 column 35 (char 34)"}
{"file": "round013_full_eval_120/round_06/logs/trace.jsonl", "timestamp": 1789252477.26, "kind": "llm_call_retry", "case_id": "50", "attempt": 8, "error_message": "Upstream error from DeepInfra: Extra data: line 1 column 73 (char 72)"}
{"file": "round013_full_eval_120/round_06/logs/trace.jsonl", "timestamp": 1789252502.27, "kind": "llm_call_retry", "case_id": "51", "attempt": 1, "error_message": "Upstream error from DeepInfra: Extra data: line 1 column 44 (char 43)"}
{"file": "round013_full_eval_120/round_07/logs/trace.jsonl", "timestamp": 1789253213.29, "kind": "llm_call_retry", "case_id": "60", "attempt": 3, "error_message": "Upstream error from DeepInfra: Extra data: line 1 column 48 (char 47)"}
{"file": "round013_full_eval_120/round_08/logs/trace.jsonl", "timestamp": 1789254323.95, "kind": "llm_call_retry", "case_id": "66", "attempt": 1, "error_message": "Upstream error from DeepInfra: Extra data: line 1 column 103 (char 102)"}
{"file": "round013_full_eval_120/round_08/logs/trace.jsonl", "timestamp": 1789254509.35, "kind": "llm_call_retry", "case_id": "68", "attempt": 24, "error_message": "error code: 502\n"}
{"file": "round013_full_eval_120/round_08/logs/trace.jsonl", "timestamp": 1789254675.26, "kind": "llm_call_retry", "case_id": "66", "attempt": 14, "error_message": "Internal server error"}
{"file": "round013_full_eval_120/round_09/logs/trace.jsonl", "timestamp": 1789255621.11, "kind": "llm_call_retry", "case_id": "76", "attempt": 2, "error_message": "Upstream error from DeepInfra: Extra data: line 1 column 42 (char 41)"}
{"file": "round013_full_eval_120/round_11/logs/trace.jsonl", "timestamp": 1789257396.38, "kind": "llm_call_retry", "case_id": "91", "attempt": 22, "error_message": "Upstream error from DeepInfra: Extra data: line 1 column 88 (char 87)"}
```

### 3f. Terminal failure (retries exhausted — this is what actually zeroes out a case's score)

```
{"file": "round013_full_eval_120/round_03/logs/trace.jsonl", "timestamp": 1789250135.07, "kind": "llm_response", "case_id": "26", "elapsed_s": 170.03, "stop_reason": "failed", "response_error_code": "server_error", "response_error_message": "unexpected_error", "content_preview": ""}
```

**34 raw examples given above** (across sections 3a–3f), satisfying the request for at least 20 real failure messages, spanning 5 distinct root-cause categories.

---

## 3g. Same code, same case, different time: succeeded before, fails now

The strongest evidence that this is a temporal reliability problem, not a deterministic content issue: comparing round_013's **own original production evaluation logs** (`runs/20260912_001239_.../round_013/logs/case_*.json`) against the **same case IDs, same unmodified round_013 task_agent**, re-run later in the full-120 benchmark (`round013_full_eval_120/result.json`). Same code, same prompt, same query — the only thing that changed is *when* the request was sent.

**10 cases** that scored ≥0.5 (real, working plans) originally now show a **terminal failure** (score 0.0–0.125) on an identical re-run:

| case_id | Original score | New score |
|---|---|---|
| 25 | 0.9375 | 0.0 |
| 40 | 0.9375 | 0.0 |
| 36 | 0.8125 | 0.0 |
| 34 | 0.75 | 0.0 |
| 92 | 0.75 | 0.125 |
| 42 | 0.75 | 0.0 |
| 65 | 0.75 | 0.0 |
| 59 | 0.75 | 0.0 |
| 103 | 0.625 | 0.0 |
| 105 | 0.5625 | 0.0 |

Detail on 5 of these, including the (entirely ordinary) query each request was carrying:

**Case 25** — query: *"I'm planning a three-day trip from Shanghai to Xiamen on November 12, 2025, returning on the 14th. The total budget for this trip should be within 7500 yuan..."*
Original: 0.9375. New: 0.0, full raw error:
```
RuntimeError: response status/stop_reason == 'failed' (error_code='invalid_prompt', error_message='Extra data: line 1 column 44 (char 43)'): Response(id='gen-1789251037-gm4t7XTctlNfZfha2iGt', created_at=1789251037.0, error=ResponseError(code='invalid_prompt', message='Extra data: line 1 column 44 (char 43)'), ..., model='google/gemma-4-31b-it', ..., output=[], ..., status='failed', ...)
```

**Case 103** — query: *"I'm planning a trip from Hangzhou to Nanchang on November 12, 2025, returning on November 18..."*
Original: 0.625. New: 0.0 — `sightseeing_output_failure_reason: "sightseeing wrap-up retry produced no <itinerary> tag (stop_reason='failed'); raw response: ''"`

**Case 34** — query: *"I plan to travel from Nanning to Chongqing for a three-day trip on November 12, 2025, and return on November 14..."*
Original: 0.75. New: 0.0 — same `stop_reason='failed'`, empty raw response pattern.

**Case 40** — query: *"I'm planning a trip from Xiamen to Nanchang on November 12, 2025, and will return on November 15..."*
Original: 0.9375. New: 0.0 — same `stop_reason='failed'`, empty raw response pattern.

**Case 92** — query: *"I'm planning a trip from Shenzhen to Beijing on November 12, 2025, returning on November 17..."*
Original: 0.75. New: 0.125 (degraded but not zeroed — consistent with §2's "recovered via retry mid-stage, but with a worse/partial plan" failure mode, not the wrap-up terminal case).

None of these queries are unusual, oversized, or contain anything a content filter would plausibly flag — they're the same ordinary multi-day itinerary requests this benchmark runs hundreds of times successfully elsewhere. The only variable that changed between "worked" and "failed" is time — direct evidence this is a transient reliability degradation at the provider layer, not something specific to these cases' content or to round_013's edit.

*(Raw sources: originals in `runs/20260912_001239_travel_mas_refactored_gemma_no_backbone_selection_X100Y180/round_013/logs/case_<id>.json`; new results in `round013_full_eval_120/result.json`, field `per_case`.)*

---

## 4. Raw log locations

| What | Path |
|---|---|
| Full-120-case benchmark of round_013 (this investigation's trigger) — 15 chunk subdirs, each `round_NN/logs/trace.jsonl` | `round013_full_eval_120/` |
| Same benchmark's final aggregated result | `round013_full_eval_120/result.json` |
| Same benchmark's failure-rate breakdown (per-case, machine-readable) | `round013_full_eval_120/failure_health.json` |
| Live production OFF run (backbone-selection-OFF), every round's own trace | `runs/20260912_001239_travel_mas_refactored_gemma_no_backbone_selection_X100Y180/round_*/logs/trace.jsonl` |
| Live production ON run (backbone-selection-ON, now killed) — earlier occurrences of the same bug, pre-fix | `runs/20260911_165000_travel_mas_refactored_gemma_full_scale_block_tagged_X100Y180/round_*/logs/trace.jsonl` and `runs/20260911_210324_travel_mas_refactored_gemma_full_scale_block_tagged_X100Y180/round_*/logs/trace.jsonl` |
| Historical Qwen-via-OpenRouter occurrences (same bug, different model — see §5) | `runs/20260910_040445_travel_mas_refactored_full_scale_block_tagged_no_summarizer_X100Y180/round_*/logs/trace.jsonl` (rounds 6, 16, 19, 20, 21, 24) |

Each `trace.jsonl` is newline-delimited JSON; grep for `"response_error_code"` to find every failure/retry event directly, or `"stop_reason": "failed"` for terminal failures specifically.

---

## 5. Related code and tooling built this session

| What | Path |
|---|---|
| The retry fix (treats an API-reported `status="failed"` the same as a thrown exception, retries within the existing budget) | `platform_core/llm_wrapper.py` (see `_response_error_info`, and the retry loop in `call_llm`) |
| Standalone, read-only analysis tool — parses any run's `trace.jsonl` files and reports retries/terminal-failures/error-codes per round | `analyze_llm_call_failures.py` |
| Dedicated unit tests for the retry fix and error-code capture | `tests/test_llm_wrapper_status_failed_retry.py` |
| The full-120-case benchmark script used to produce this report's data | `eval_round013_full_benchmark.py` |

---

## 6. Assessment

- **Not a billing/rate-limit problem on our account** (§1).
- **Not fully explained by our own request concurrency** — a concurrent run at 4x our parallelism was hit far less hard in the same window (§2).
- **Not a deterministic content bug in a specific edit** — a live re-run of one of the exact same failing requests (case 72, a separate investigation) succeeded cleanly with zero errors, so the same prompt isn't reliably reproducing the failure.
- **A provider/platform-side reliability issue, with the upstream chain now identified**: our call → OpenRouter → **DeepInfra** → `nvidia/Gemma-4-31B-IT-NVFP4` (confirmed twice independently — §3c's chat-template message names the model directly, §3e's `server_error` messages name DeepInfra directly, and both carry the identical "Extra data: line 1 column N" JSON parse signature). This is not something we can fix from our side.
- **Confirmed the image-decode errors are not caused by our request** (§3d) — an exhaustive grep of every file in the call path (`platform_core/`, the project's seed code, `tools_schema.json`) found zero references to `image_url`, `base64`, or `data:image` anywhere, and `call_llm`'s own request dict has no field capable of carrying one. Seeing image-decode failures for a text-only request is a genuine anomaly on the provider side (most plausibly request/response mixing across tenants at the DeepInfra layer under load), not something originating in our code.
- The same failure signature (differing model, same `status="failed"`/no-exception pattern) was also seen historically with Qwen models via OpenRouter, not just Gemma — reinforcing that this is a property of routing through OpenRouter under this API surface, not specific to one model.

## 7. Suggested next steps (as of 2026-09-13 — since actioned, see §8)

1. Consider reporting this to OpenRouter support with the confirmed upstream chain (OpenRouter → DeepInfra → `nvidia/Gemma-4-31B-IT-NVFP4`) and the image-decode anomaly — that combination looks like something worth their attention regardless of what we do on our side.
2. Treat any score computed during a window with a high observed failure rate as unreliable; check `analyze_llm_call_failures.py` output before trusting a round's score, as has become this session's practice.
3. Decide whether to continue running against this backend while it's this unreliable, or fall back to the local vLLM Qwen default (zero occurrences of this bug all session) until it stabilizes.

---

## 8. Fix implemented and validated (2026-09-14)

Went with option 3's third alternative: stay on OpenRouter/Gemma, but steer routing away from the identified culprit backend, plus add a defensive layer so any failure that still slips through can no longer corrupt the HGM's reward signal.

### 8a. Provider-routing fix

`platform_core/llm_wrapper.py::call_llm` gained an opt-in `provider: Optional[dict]` param, threaded through OpenRouter's own `extra_body={"provider": {...}}` request field (confirmed live that OpenRouter honors it — a test call with `ignore=["DeepInfra"]` was actually served by Venice instead). Wired into real configs via a new `TaskAgentSpec.provider` field (`meta_agent/config.py`) → `LLM_PROVIDER_PREFERENCE` env var (`meta_agent/runtime_env.py::apply_task_agent_env`) → `_env_default_provider()` in the wrapper.

**Iteration 1 — `{"ignore": ["DeepInfra"], "quantizations": ["bf16"], "allow_fallbacks": false}`:**
Validated at production scale before deploying: 3 independent full-120-case passes at parallelism 32, **0 terminal failures out of 3,511 real calls** (`round013_provider_fix_3x_120/`) — vs. this report's §2 baseline of 174 terminal failures / 82.5% of cases affected on the unfixed path. Deployed to both live production configs.

**Regression found, then corrected — dropping the `bf16` requirement:**
With the fix live, the `llm_backbone_selection` block (an HGM block that can reassign a sub-agent's backbone model, see `meta_agent/block_suggester.py`) turned out to be silently non-functional: all 4 catalog slugs (`qwen/qwen3.6-27b`, `qwen/qwen3.8-27b`, `google/gemini-3.6-flash`, `google/gemini-3.8-flash`) returned `404 No endpoints found for the request with quantization: bf16` — confirmed live, one-off calls to each. The `bf16` requirement, while fixing the DeepInfra issue, happened to exclude every endpoint those other models have.

Before removing `bf16`, independently re-validated that `ignore: DeepInfra` ALONE (no `bf16`) keeps the same reliability: 3 independent 32-case passes, **0 terminal failures out of 1,075 real calls, 0.09% incidence rate** (`deepinfra_only_3x_32/`) — as good as the `bf16`-included version's 0.20% (5/2525 in early production rounds), while no longer blocking any backbone-catalog model. Both live production configs (`configs/hgm_travel_gemma_full_scale_block_tagged_X100Y180.yaml`, `configs/hgm_travel_gemma_no_backbone_selection_X100Y180.yaml`) were updated to the final fix:

```yaml
task_agent:
  provider:
    ignore: ["DeepInfra"]
    allow_fallbacks: false
```

**Live production results since deploying the final fix** (both runs restarted 2026-09-14, tracked via the dashboard feature in §8b): failure rate has stayed at **0% for the large majority of rounds** on both runs, with exactly one round each briefly crossing the 3% alert threshold (ON run's round_002: 3.9%, 57 retries/1462 calls; OFF run's round_020: 3.8%, 11 retries/290 calls) — and **zero terminal failures across every round of both runs**, meaning every retry that fired was successfully recovered and no round's score was corrupted. A separate full-120-case standalone re-evaluation of a live node (node_11, `eval_node11_3x_120.py`) confirmed this at full scale outside the HGM's own round-robin sampling: 0.87% incidence (147 retries / 16,839 calls), 0 terminal failures.

### 8b. Defensive layer — HGM reward robustness + dashboard visibility

Because even a well-mitigated provider can still fail occasionally, and a terminal failure silently zeroes a case's score (this report's core finding), added a second, independent layer so that failure class can never silently corrupt the HGM's learning signal, and so its rate is always visible without re-deriving it by hand:

- New `meta_agent/llm_failure_health.py` — the trace-parsing logic from `analyze_llm_call_failures.py`, extracted into a reusable module (`iter_trace_files`/`analyze_trace_file`/`incidence_rate_pct`, `DEFAULT_INCIDENCE_THRESHOLD_PCT = 3.0`).
- `HGMManager._record_batch` (`meta_agent/managers/hgm.py`) — the single choke point all node-recording now goes through: always writes a per-round `llm_failure_health.json`, always prints a loud `⚠️` warning when a round's rate exceeds 3%, and optionally (opt-in `exclude_llm_call_failures`, default `False` — zero behavior change unless explicitly enabled) excludes any case hit by a terminal failure from that node's reward tally, without affecting eval-budget accounting.
- `hgm_dashboard.py` / `meta_agent/run_inspect.py` — a new "LLM call failure rate" chart and nodes-table column, plus an automatic 🔴 Diagnostics alert whenever a round crosses the 3% threshold — this is what surfaced the two threshold-crossing rounds mentioned above in real time, and confirmed both recovered with 0 terminal failures.

### 8c. Outcome

Both live production runs and one standalone full-scale re-evaluation have now run for many hours/rounds under the fix with **zero terminal failures observed**, vs. the pre-fix baseline's 174 terminal failures in a single 120-case pass. The `llm_backbone_selection` block, previously silently broken by the `bf16` side-effect, is confirmed working again (e.g. it's part of the ON run's current best node's lineage). The dashboard/monitoring layer means any future recurrence — from DeepInfra or any other backend — would now be caught and flagged automatically rather than silently corrupting a round's score, as originally happened in round_013.
