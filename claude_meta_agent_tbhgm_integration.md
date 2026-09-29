# Claude as meta-agent in Tier-Based, Block-Selecting HGM: integration plan

Status: design sketch, not implemented. Brainstorm only. Companion to `tier_based_hgm.md` (the
block/tier design this plugs into) and `astar_hgm.md`. Grounded in a standalone comparison
experiment (`experiment_harness_redesign.py` / `experiment_harness_staged.py`, this session) that
tested Claude, DeepSeek V4 Pro, Qwen-122B and Qwen-27B as meta-agents on `travel_mas_refactored`
under identical information/tool constraints. Claude reached composite 0.670 then 0.824 held-out
(the only agent to clear a codebase-wide, code-owned redesign); this doc is about carrying that
result — and the concrete lessons the whole comparison surfaced — into the real production loop.

## 0. What's verified, not assumed

- `call_llm` (`platform_core/llm_wrapper.py`) is hardcoded to the OpenAI Responses API shape
  (`client.responses.create`). It cannot reach Anthropic's native API (different endpoint/schema).
- **It can reach Claude via OpenRouter.** Live-verified this session: OpenRouter lists 32 Claude
  variants (`anthropic/claude-sonnet-5`, `anthropic/claude-opus-5`, `anthropic/claude-opus-5.5`,
  ...) and its `/v1/responses` endpoint returns a clean completion for `anthropic/claude-sonnet-5` —
  the exact same integration path already proven in this session for DeepSeek V4 Pro
  (`REDESIGN_META_MODEL`/`REDESIGN_META_BASE_URL` env vars in the standalone script;
  `AgentEditor.__init__`'s existing `model`/`base_url`/`reasoning_effort`/`max_output_tokens`
  constructor kwargs in production — no new plumbing needed for the model swap itself).
- `AgentEditor` already supports an agentic, multi-turn tool loop (`agentic_editing=True` →
  `_self_improve_agentic`), not just the single-shot `_self_improve`. Its tool set today
  (`_AGENTIC_TOOLS`): `read_file`, `write_file`, `run_code_validators`, `submit_self_improvement_summary`.
  No evaluation tool, no offline sandbox.
- `run_code_validators` → `self._run_validators(out_dir, base_dir)` → `for v in self.validators:
  v.validate(out_dir, base_dir)` — literally the same `editor_validators.py` classes and injection
  mechanism (`evaluator`/`benchmark_dir` auto-injected per `config.py::_build_with_injection`) used
  to build the standalone experiment's `check_workspace`. **Most of what `check_workspace` runs is
  already reachable through `run_code_validators` today — it's gated by which validators a
  project's YAML lists, not by missing code.** `travel_mas_refactored`'s current config only lists
  `- type: "syntax"`; the other 7 default validators plus `llm_backbone_config` are one YAML edit
  away, no code change.
- Two things are genuinely absent from `AgentEditor` today, confirmed by reading its tool schemas
  and dispatch: (a) a "does a plan actually get produced" check (`SmokeTestValidator` only checks
  for a crash, `case.error` — not `no_plan`, which was the single most common real-world failure
  mode this whole project encountered); (b) any in-session real-evaluation or offline-sandbox tool
  at all — `evaluate_variant` and `run_python` exist only in the standalone experiment's `Session`
  class, which is architecturally disconnected from `AgentEditor` (parallel code, not shared).

## 1. Two ways to put Claude in the loop — recommendation

| | A: Claude via `call_llm`/OpenRouter, inside `AgentEditor` | B: Claude Code (agentic CLI/SDK) as a custom editor |
|---|---|---|
| What it is | Point `editor: config: {model: "anthropic/claude-sonnet-5", base_url: "https://openrouter.ai/api/v1", ...}` at the existing `AgentEditor`/`agentic_editing=True` loop | A new `ClaudeCodeEditor` class that shells out to `claude` (Agent SDK/CLI) with the SAME tool surface as `redesign_cli.py`, driven programmatically per HGM round |
| Reuses existing machinery | Almost entirely — `AgentEditor`, its validators, its round/tree integration, its `EditResult` contract | Only the outer contract (`EditResult`); the tool loop itself is bespoke |
| Matches how Claude was actually validated this session | Partially — same model, but a stateless per-call loop, not the actual agentic session that produced 0.824 | Closely — this is literally how the 0.824 result was produced (via `redesign_cli.py`) |
| Engineering cost | Low: config change + the tool additions in §2 | High: a new editor implementation, a new subprocess/SDK integration, new logging/parity plumbing, new failure modes (process crashes, SDK auth) to handle |
| Fits one HGM round's granularity | Yes — one edit, one round, same shape as every other editor today | Awkward — Claude Code's own session model (many turns, its own context management) doesn't map cleanly onto "one round = one edit"; would need its own sub-budget and its own accept/reject inside one round |

**Recommendation: A, with the §2 tool additions**, not B. Option B duplicates a lot of what §2
adds to option A anyway (real eval feedback, an offline sandbox) while introducing a second,
parallel, harder-to-maintain editor implementation and new operational risk (an external CLI/SDK
process instead of a plain API call already unified with the rest of the framework). The
standalone experiment already demonstrated *that* Claude can do this well; option A puts Claude
inside the *existing* production surface everything else (validators, tree search, ratchets,
tier/block gating) already understands, at a fraction of the engineering cost. Revisit B only if A
is tried and the loss of "continuous agentic session" turns out to matter more in practice than
this analysis suggests.

## 2. Tool additions to `AgentEditor` (the concrete, scoped work)

Precisely itemized, not lumped together (per the discussion that got here):

1. **Config-only, zero code** (§0): add `signature`, `imports`, `schema_wrapper_consistency`,
   `mutable_tool_imports`, `mutable_tool_routing`, `immutable_files`, `load_test`,
   `llm_backbone_config` to `travel_mas_refactored`'s `validators:` YAML list. Worth doing
   regardless of the Claude question — closes a real, pre-existing gap (`SignatureValidator`
   currently only ever effectively checks the trivial `workflow.py` delegator's signature on this
   project's default config, per `tier_based_hgm.md`'s own "validator gap" section).
2. **Small new validator** (~30 lines, follows `SmokeTestValidator`'s exact existing pattern):
   `PlanProducedValidator` (or extend `SmokeTestValidator` with a `require_plan: bool` flag) —
   runs one real case (same isolated-scratch-dir pattern `SmokeTestValidator` already uses) and
   additionally checks `no_plan` (via the same `gold_label_sets()`-style logic used throughout the
   standalone experiment), not just `case.error`. Register it, add to `validators:` YAML.
3. **`evaluate_and_see` tool** (the substantial addition): new entry in `_AGENTIC_TOOLS`, new
   dispatch branch in `_self_improve_agentic`'s turn loop, logic ported from
   `Session.evaluate_variant()` — run the candidate against a fixed in-round sample (or, per the
   overfitting lesson in §3, ALL applicable cases when the budget allows), return per-case
   failures. Needs a call budget (`agentic_eval_rounds`, mirroring `stage_eval_rounds`) and the
   same "does not consume a call on an already-invalid workspace" guard `Session.evaluate_variant`
   already has.
4. **Offline sandbox tool** (`run_python` analogue): new schema entry + dispatch, logic ported
   from `Session.run_python()` — same allow-listed-imports/blocked-names/blocked-attrs sandbox
   (`python_problems()`), same `train_data`-shaped view over the round's own failure corpus. Zero
   marginal LLM cost, so no budget needed, same as today.
5. Wire both into `AgentEditor.__init__`'s existing injection pattern (`evaluator`, `benchmark_dir`
   already flow in via `_build_with_injection`, same as `SmokeTestValidator` gets them) — no new
   config-plumbing shape needed, just new constructor params following the existing convention.

## 3. This session's other lessons → where they plug in

Not just tools — several are pure prompt/process changes, generalizable to *any* editor LLM
(weak local model or Claude), not Claude-specific:

- **"If you don't want the LLM to call a tool, take it out of the schema; a prompt request isn't
  enforced."** Belongs in `AgentEditor`'s `_diagnosis_rules()`/`_AGENTIC_TOOLS`-adjacent guidance
  as a general code-writing rule the model applies to the AGENT IT IS EDITING (e.g., if the
  edit itself makes a tool-choice decision in code for a sub-agent, don't leave the tool available
  and just prompt the sub-agent not to call it) — this is a rule about how to fix the *target* MAS,
  not about `AgentEditor`'s own tool loop, so it goes in the same rules list as hard rule 9 (the
  existing "state if you followed the suggestion" rule) — call it hard rule 11.
- **"Verify your summary against your own code before submitting."** `AgentEditor` already has
  half of this (hard rule 9: state whether the edit follows the suggestion). Extend it to also
  require the model check its own `proposed_changes`/`rationale` claims against the diff it just
  wrote — same spirit, one more explicit ask, same enforcement point (the existing rule text,
  right before `submit_self_improvement_summary`).
- **"Test broadly; a small fixed sample can hide a regression; overfitting to it is easy."** Once
  `evaluate_and_see` exists (§2.3), this becomes a workflow instruction alongside it: prefer
  testing the full set of cases where a target check currently applies over a small fixed sample,
  and don't trust a small-sample "zero failures" result as proof of a fix. Directly motivated by a
  concrete failure this session: a fix that showed clean results on the same 8-case sample across
  4 evaluation calls was submitted as working, and a full 60-case check found it had introduced 2
  new regressions outside that sample, netting an overall regression on the targeted metric.
- **Unit/dimension-based curriculum.** This is a different axis from block selection (WHAT grader
  dimension/hard-constraint family to target, vs. WHERE in the code — `tier_based_hgm.md`'s
  `prompt`/`sub_agent`/`collaboration`/`verifiers`/`foundational_ability`). The two compose: a
  round can be steered on BOTH axes at once ("target the Cost Calculation Accuracy dimension" ×
  "via a foundational_ability-block fix"). `astar_hgm.md`'s own "Where it plugs into the code"
  table already names `curriculum.py`'s goals becoming "units (dimensions / hard families) ranked
  by oracle composite gain" — this is the existing hook; steering context should carry both a
  block AND a unit when both are configured, same way `describe_unit()` in the standalone
  experiment renders a unit's own failing-check breakdown for the editor to read.
- **Unit-score-based (not composite-based) accept/reject.** When a round's steering context names
  a specific unit, the ratchet/validator that decides whether to keep the child should check that
  unit's own pass-rate improved (plus the existing no-regression guards), not overall composite —
  composite blends every other unit and is a much noisier signal for a single-unit-targeted edit,
  confirmed directly by this session's staged-runner experiments. When no unit is targeted (a
  generic block-only round), composite remains the right signal — this is a targeted refinement,
  not a wholesale replacement of the existing ratchet.

## 4. Where Claude specifically plugs in

- `editor: config: {model: "anthropic/claude-sonnet-5" (or -opus-5), base_url:
  "https://openrouter.ai/api/v1", reasoning_effort: ..., max_output_tokens: ..., agentic_editing:
  true}` — task agent's own model/base_url are configured completely separately
  (`task_agent:` block) and are never touched by this; the hard local-only rule for the task agent
  stays exactly as strict as it's been all session.
- Same treatment optionally available for `block_suggester`/`failure_summarizer`/
  `behavior_summarizer` (`cfg.gatherer`/other roles) if their own quality turns out to matter —
  not required for a first pass, since the editor is the component actually writing code.
- `OPENAI_API_KEY` (the OpenRouter key) needs to reach whatever process runs the HGM loop, same
  `.env` + `set -a; source .env` pattern already used this session — never printed/echoed.

## 5. Rollout plan (mirrors how this session actually validated things)

1. **Config-only step first** (§2.1): enrich the validators YAML, confirm `run_code_validators`
   now reports on the fuller suite, at zero risk (this doesn't touch the model or the tool loop).
2. **Cheap, single-unit sanity check before a full run** — exactly the discipline used all session
   (`--force-unit`, one cheap unit, before committing to `--n-stages 6`): a `manager: config:
   max_rounds` small enough for 1-2 real EXPANDs, one block fixed (e.g. `foundational_ability`),
   Claude as editor, WITHOUT §2's new tools first — establish the "config-swap-only" baseline.
3. **Add `evaluate_and_see` + the offline sandbox, re-run the same sanity check** — isolate how
   much of the gain is "better model" vs. "better model + real feedback loop," the same
   before/after comparison structure this whole session has used throughout.
4. **Only then** widen to real `eval_budget`/`max_rounds`, multiple blocks, and the unit-based
   curriculum steering from §3.
5. Keep a full-composite, full-held-out measurement at the end of each stage, same held-out
   discipline used throughout this session (train-sample results overstate real generalization).

## 6. Risks

- **Cost.** Every EXPAND now potentially costs one Claude API call (editor) plus, if
  `evaluate_and_see` is used, real evaluator time on top — at real `eval_budget`/`max_rounds`
  scale (hundreds+) this is a materially different cost profile than the current local-model
  editor. Needs an explicit budget/cap, not an open-ended per-round allowance.
- **External dependency.** OpenRouter uptime/latency/key management becomes a load-bearing part of
  the production loop, not just a comparison experiment — the same `meta_call_failed`/timeout
  handling this session had to build for the standalone script (retry, don't silently stall a
  round) needs a production-quality equivalent in `AgentEditor`.
- **Information-parity enforcement must stay code-level, not prompt-level** — directly the lesson
  from §3's first bullet, applied reflexively to this integration itself: whatever Claude must not
  see (scorer source, eval-split data) must be enforced by what's actually readable/injected, not
  by asking it nicely, especially since a more capable, more agentic model is more likely to probe
  or route around a soft boundary than a weaker one would.
- **A capable model changes the failure profile, not just the success rate.** This session's own
  evidence (the hotel-selector regression that a confident, coherent-sounding submitted summary
  described inaccurately) shows a stronger model can still produce a subtly wrong result with a
  MORE convincing rationale — the "verify summary against code" and "unit-score gate" additions in
  §3 exist specifically because narrative quality and self-reported confidence are not reliable
  substitutes for the harness's own independent check, and that gets more (not less) important as
  the editor model gets more articulate.

## 6.5. Once §2's tools exist, is option A still meaningfully different from "real Claude Code"?

Revisiting §1's option A vs. B comparison after scoping §2 concretely: **yes, functionally very
similar for this task, and arguably a better fit, not just an equivalent one.** Once `AgentEditor`
has read/write/validate/`evaluate_and_see`/offline-sandbox/submit, that is the same *category* of
tool surface `redesign_cli.py` gave Claude this session — and what actually drove the 0.670→0.824
result wasn't anything specific to the Claude Code CLI/SDK itself, it was having real feedback
loops (test, see real failures, fix, re-test) instead of submitting blind. A narrow, purpose-built,
policy-enforced tool set (which `AgentEditor` already is, and which §2 only extends) is a more
natural fit for that than Claude Code's general-purpose `Bash`/`Read`/`Edit` — tighter, more
auditable, and it structurally enforces the "schema not prompt" principle (§3) rather than relying
on the model to self-restrict.

The one real, still-open difference is **session/turn budget per edit**: `AgentEditor`'s
`agentic_max_turns` defaults to 20 (its own existing comment already notes "typically 3-6 turns
even for a genuine 4-file fix" from real trials); this session's actual Claude redesign sessions
ran far longer (100+ turns). This isn't a tool gap — it's a deliberate HGM design choice: each
round stays small and bounded, and the *tree* accumulates progress across many rounds rather than
one long session doing everything. Whether that reaches the same place as one long session is
genuinely untested, but it's the same shape as this session's own staged-runner result (Claude's
separate hard-constraint follow-up session building on its own first pass, 0.670→0.824) — a
reasonable existence proof that incremental, code-continuity-based progress across sessions can
work, not just one continuous conversation. Worth treating `agentic_max_turns` (and how many
rounds a given lineage gets before the tree gives up on it) as a tunable to revisit empirically
once this is running for real, rather than assuming 20 is automatically enough.

## 7. Open questions

- Should `evaluate_and_see`'s sample selection be steering-context-aware (bias toward the targeted
  unit's currently-failing cases, per §3's curriculum hook) rather than a generic fixed sample?
- Does giving the editor a real in-session eval tool change `tier_based_hgm.md`'s block-Thompson-
  sampling reward attribution timing — today reward comes from the NEXT round's real EVALUATE
  step; would an editor that already self-validated in-session make that signal redundant, or just
  earlier/cheaper confirmation of the same thing?
- Worth a config flag to run BOTH a local-model editor and a Claude editor as sibling EXPANDs from
  the same parent, to get a live, continuously-running version of this session's comparison rather
  than a one-off experiment?
