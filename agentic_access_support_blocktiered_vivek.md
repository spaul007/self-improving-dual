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

Running both together (production node-32-restart run, `configs/hgm_travel_deepseek27b_from_node32_agentic_X100Y300.yaml`) has shown zero malformed-JSON failures and zero `edit_failed` nodes across the first several EXPANDs — a sharp contrast to the same config's non-agentic predecessor (58% EXPAND failure rate, 45.9% specifically from malformed JSON). The tradeoff is wall-clock time per round (multiple LLM round-trips instead of one), not reliability.
