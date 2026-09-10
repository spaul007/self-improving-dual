# EDIT_MEMORY.md — a worked example from one run

Rendered by `study/render_edit_memory_example.py` from `20260907_212018_travel_hgm_1000_qwen122b_gpt54_beliefs2stage` (node 30). Mode: delta-labelled, pre-v7 code (no `strategy_label` key; analysis v6 — records carry implementation verdicts but no `effect` lines). Records with judge effect lines: 0 of 32. Judge lines (`effect`, `regressions`, `targets`) appear only in judge-mode runs — re-run this script on the first judge-mode run to refresh the example. The layouts themselves are specified in `EDIT_MEMORY_SPEC.md`.

## §0 Run

## Best round (LCB-selected)
- Round: **030** (node 30)
- Train mean: **0.707** over 60 case(s)
- Optimization goal: Recover hallucinated `plan` tool calls into valid final `<plan>` outputs instead of tool errors or empty results.

models: task_agent=Qwen/Qwen3.5-122B-A10B, editor=gpt-5.4, edit_memory=gpt-5.4
edit_memory keys: steering_mode=belief, strategy_label=None, analysis_min_own_evals=None, judge_min_evidence=None, min_shared=8, verdict_threshold=0.02, max_strategies=30, max_subedits=3
beliefs keys: enabled=True, doc_char_cap=40000, optimize_enabled=True, optimize_every=8, optimize_min_scored=8, optimize_rollback_margin=0.02, instruction_char_cap=2500

## §1 The node record — `round_030/edit_memory.md`

```markdown
---
node: 30
parent: 9
depth: 3
lineage: 0 > 3 > 9 > 30
---

## Edit 1
- **name**: `add-plan-tool-shim`
- **category level 1 (strategy)**: `add-tool-compatibility-shim`
- **category level 2 (area)**: `finalization-tool-compatibility`
- **what**: Added a mutable `plan` tool shim that catches hallucinated `plan` calls and returns deterministic instructions to immediately emit exactly one final `<plan>...</plan>` block from already collected tool results without calling more tools.
- **why**: This targets runs that fail at the end because the model invents a nonexistent `plan` tool instead of directly finalizing the itinerary.

## Edit 2
- **name**: `log-plan-shim-activation`
- **category level 1 (strategy)**: `add-planner-telemetry`
- **category level 2 (area)**: `planning-observability`
- **what**: Logged a structured trace event whenever the hallucinated `plan` tool shim is triggered, including the intercepted argument names.
- **why**: This targets the inability to tell from runtime traces whether finalization recovery happened or which bogus `plan` call shape activated it.

## Outcome
- **performance**: child 0.7073 over 60 evaluated cases (vs parent on 60 shared: child 0.7073, parent 0.6396, Δ +0.0677)
- **generalization**: seen 0.7073/60 (Δ +0.0677)
- **new tools** (3 batches, 60 cases): `plan` 13 calls / 13 cases
- **new log point `decision_branch/hallucinated_plan_tool_intercepted`**: fired ≥13x (pass 13) (3 batches, 60 cases) -> scorer on those cases: 2 pass / 11 fail · SUSPECT VERIFIER

## Analysis
- **`add-plan-tool-shim (`plan` mutable tool)`** (other) — 13 calls / 13 cases (3 batches, 60 cases); No intrinsic checker verdict; each intercepted `plan` call returned the same finalize-now instruction string.
  - agreement: Sampled activated cases 86, 75, and 71 all scored fail; batch summary on the co-fired interception log over the same 13 activations is 2 pass / 11 fail, so interception often let the run continue to a final answer but did not reliably produce scorer-accepted plans.
  - likely cause: Diff +13..+20 and +25..+31: `run(**kwargs)` ignores arguments and always returns one fixed 'finalize now from existing tool results' message, so it only patches the nonexistent-tool error path.
  - likely cause: Case 75 dropped from 6 parent failed checks to 1 child fail, and case 46 from 5 to 2, consistent with the shim converting an abortive bogus-tool step into a completed final plan.
  - likely cause: Case 73 regressed from parent pass to child `attraction_visit_within_opening_hours`, and sampled activations 86/75/71 still failed; 11/13 activated cases failed overall, showing regeneration side-effects or unresolved content errors after recovery.
- **`log-plan-shim-activation (`decision_branch/hallucinated_plan_tool_intercepted`)`** (other) — fired ≥13x (pass 13) alongside the 13 `plan` calls; Always logged `verdict="pass"` when the shim ran.
  - agreement: Sampled firings on cases 86, 75, and 71 all correspond to scorer-fail outputs; batch summary for all firings is 2 pass / 11 fail, so this trace marks interception events rather than scorer-approved outputs.
  - likely cause: Diff +25..+30 logs `label="decision_branch"`, `name="hallucinated_plan_tool_intercepted"`, and `arg_names=sorted(kwargs.keys())`.
  - likely cause: Runtime samples for 86/75/71 show the event with `arg_names: []`, so the added telemetry does expose at least one bogus call shape.
  - likely cause: Only 12/400 runtime events are shown; beyond the empty-arg shape in 86/75/71, intercepted argument-shape coverage is unmeasured in the observed batches.
- **collateral**: commonsense:reasonable_transfer_time 28->33 fails (-5); commonsense:attraction_visit_within_opening_hours 7->11 fails (-4); commonsense:dining_within_service_hours 8->11 fails (-3); commonsense:cost_calculation_correctness 2->5 fails (-3); commonsense:essential_meal_coverage 7->9 fails (-2); commonsense:seamless_intercity_transfers 13->14 fails (-1); commonsense:validated_transportation 1->2 fails (-1); hard:restaurant_specific_cuisine_nearby 1->2 fails (-1); hard:train_cheapest_train_type 1->2 fails (-1); commonsense:no_time_overlaps 0->1 fails (-1); hard:train_cheapest_direct 0->1 fails (-1); commonsense:diverse_meal_options 18->12 fails (+6); commonsense:diverse_attraction_options 7->4 fails (+3); commonsense:validated_meals 5->2 fails (+3); commonsense:essential_attraction_coverage 24->22 fails (+2); hard:attraction_top_rated_must_visit 4->2 fails (+2); commonsense:reasonable_duration_at_attractions 4->3 fails (+1); hard:restaurant_must_eat_named 3->2 fails (+1); commonsense:validated_attractions 2->1 fails (+1); hard:flight_seat_status 2->1 fails (+1); hard:restaurant_specific_tag_nearby 2->1 fails (+1); commonsense:traceable_accommodation 1->0 fails (+1); hard:hotel_cheapest_brand 1->0 fails (+1); hard:restaurant_cheapest_nearby_attraction 1->0 fails (+1); hard:train_departure_time_range 1->0 fails (+1); hard:train_latest_arrival_direct 1->0 fails (+1)
- **implementation**: sound — Diff +25..+31 added a real mutable tool and trace log; the tool was called 13 times and the log fired ≥13x, with sampled events on 86/75/71 showing the wiring and captured `arg_names`, so this is implemented and exercised rather than dead c
- **implementation (edit 1)**: sound — Diff +25..+31 implements the shim and it was exercised 13 times; case 75 improving from 6 parent failed checks to 1 child fail is consistent with the claimed recovery path being live.
- **implementation (edit 2)**: sound — Diff +25..+30 logs the interception event with `arg_names`, and sampled runtime events for 86/75/71 show exactly that payload, matching the telemetry claim.
```

## §2 Its pre-registered prediction — `round_030/belief_prediction.json`

```json
{
  "version": 1,
  "node": 30,
  "parent": 9,
  "belief_version": 83,
  "instruction_version": 4,
  "tags": [
    {
      "edit": 1,
      "strategy": "add-tool-compatibility-shim",
      "area": "finalization-tool-compatibility",
      "fit": "exact"
    },
    {
      "edit": 2,
      "strategy": "add-planner-telemetry",
      "area": "planning-observability",
      "fit": "exact"
    }
  ],
  "coverable": true,
  "strategy": {
    "slug": "tool-compatibility-shim-helps",
    "p": 0.72,
    "scope": {
      "strategy": "add-tool-compatibility-shim",
      "area": null
    },
    "matched_edit": 1,
    "section": "### belief:tool-compatibility-shim-helps — hallucinated-tool compatibility shims have a clearly better-than-even chance to help when sound\n- kind: strategy\n- scope: strategy=add-tool-compatibility-shim\n- predict: p=0.72\n- evidence: the narrow finalize-now shim directly targets a severe missing-tool failure and now has a broader helped outcome plus a thinner same-direction slice [node 28: Δ+0.0586/16] [node 27: Δ+0.1250/3]; the remaining close sibling is still unmeasured [node 29: unmeasured]\n- next: count how often intercepted `plan` calls would otherwise end without an extractable `<plan>` block, and keep the shim only if those recovered cases stay tag-clean and tool-free afterward"
  },
  "implementation": {
    "slug": "tool-compatibility-shim-soundness",
    "p": 0.89,
    "scope": {
      "strategy": "add-tool-compatibility-shim",
      "area": null
    },
    "matched_edit": 1,
    "section": "### belief:tool-compatibility-shim-soundness — hallucinated-tool compatibility shims are very likely to be implemented soundly when the intercept contract stays narrow\n- kind: implementation\n- scope: strategy=add-tool-compatibility-shim\n- predict: p=0.89\n- evidence: the current family uses a narrow deterministic contract that only needs to catch a specific hallucinated `plan` call and return finalize-now guidance, and live same-direction evidence now exists in both observed siblings [node 27: Δ+0.1250/3] [node 28: Δ+0.0586/16]; one close sibling remains unmeasured [node 29: unmeasured]\n- next: force a hallucinated `plan` call in tests, confirm the intercept trace appears, and inspect kept outputs for a single extractable `<plan>` block with no fresh tool use after the shim"
  }
}
```

## §3 The planning pass's prediction — `round_030/edit_prediction.json`

```json
{
  "version": 1,
  "round_dir": "round_030",
  "belief_id": "tool-compatibility-shim-helps",
  "expected_direction": "up",
  "expected_delta": 0.06,
  "why": "This parent already shows `plan` tool usage with a 1.00 tool error rate and at least one zero-score no-plan failure, while the measured evidence for the narrow hallucinated-`plan` shim is the strongest currently available (nodes 27 and 28, with node 28 sound and positive on broader overlap). The mechanism is small, deterministic, and directly targets an observed scorer-visible failure mode.",
  "proposal_goals": [
    "Recover cases where the model hallucinates a `plan` tool call so they end with an extractable `<plan>` block instead of a tool error or empty output.",
    "Make the no-plan failure mode observable at the exact extraction boundary so the edit can be judged sound and any remaining empty-output cases are diagnosable."
  ],
  "query": {
    "nodes": [
      27,
      28,
      29
    ],
    "strategies": [
      "add-tool-compatibility-shim",
      "add-planner-telemetry"
    ],
    "areas": [
      "finalization-tool-compatibility",
      "empty-output-recovery",
      "planning-observability"
    ],
    "keywords": [
      "mutable plan tool shim",
      "finalize-now guidance",
      "intercepted plan call",
      "extractable <plan> block",
      "no-plan telemetry"
    ],
    "include_code": true
  }
}
```

## §4 The belief document at the end of the run (summary + first sections)

```markdown
## Summary
The narrow hallucinated-`plan` compatibility shim remains the clearest live quality lever: three sound finalize-now variants improved shared score by turning dead-end tool calls into scorable outputs [node 27: Δ+0.1563/16] [node 29: Δ+0.0937/16] [node 30: Δ+0.0677/60], while a broader sibling was slightly negative on broad overlap [node 28: Δ-0.0177/60]. Planner telemetry still looks more diagnostic than causal because the new intercept trace was wired cleanly yet its `pass` label did not track scorer success on firing cases [node 29: Δ+0.0937/16], and the wider telemetry family remains mixed [node 20: Δ-0.0567/32] [node 31: Δ-0.0938/16]. Intercity `Current City:` normalization still needs broader evidence beyond the thin conservative positive [node 31: Δ-0.0938/16] [node 32: Δ+0.0416/3].

### belief:call-memoization-helps — per-task tool-call memoization is unlikely to help when sound
- kind: strategy
- scope: strategy=add-call-memoization
- predict: p=0.16
- evidence: memoization stayed near neutral on broader overlap [node 1: Δ-0.0052/48] [node 3: Δ+0.0117/48], and the only clear upside slice is still too thin to move the prior much [node 12: Δ+0.1563/4]
- next: keep reuse limited to deterministic calls with canonical arguments, and add sampled fresh-versus-cached equivalence checks before expanding coverage
- track: n=1 · Brier 0.16 (0.25 = uninformative) · outcomes: 3 yes
### belief:call-memoization-soundness — per-task tool-call memoization is likely to be implemented soundly
- kind: implementation
- scope: strategy=add-call-memoization
- predict: p=0.78
- evidence: live runs showed miss, store, and hit behavior for the cache wrapper on repeated calls [node 1: Δ-0.0052/48] [node 3: Δ+0.0117/48], while the main remaining risk is key canonicalization rather than the mechanism failing to run [node 12: Δ+0.1563/4]
- next: fuzz cache-key edge cases such as reordered object keys, null omission, and string-versus-numeric fields before widening reuse
- track: n=1 · Brier 0.12 (0.25 = uninformative) · outcomes: 3 yes
### belief:decision-telemetry-helps — descriptive decision telemetry is still more diagnostic than quality-improving
- kind: strategy
- scope: strategy=add-decision-telemetry
- predict: p=0.30
- evidence: some telemetry-tagged bundles helped [node 5: Δ+0.1343/27] [node 7: Δ+0.0559/19], but other measured runs were neutral or harmful [node 1: Δ-0.0052/48] [node 3: Δ+0.0117/48] [node 6: Δ-0.1875/12], and the remaining upside slice is still thin [node 12: Δ+0.1563/4]
- next: log only branch facts that can explain a later keep, reject, fallback, or skip decision, and remove any field that could be read as a success verdict
- track: n=2 · Brier 0.26 (0.25 = uninformative) · outcomes: 2 yes, 4 no

(+17 more belief section(s) not shown)
```

## §4b Every belief's `- track:` line

| belief | track |
|---|---|
| `call-memoization-helps` | n=1 · Brier 0.16 (0.25 = uninformative) · outcomes: 3 yes |
| `call-memoization-soundness` | n=1 · Brier 0.12 (0.25 = uninformative) · outcomes: 3 yes |
| `decision-telemetry-helps` | n=2 · Brier 0.26 (0.25 = uninformative) · outcomes: 2 yes, 4 no |
| `decision-telemetry-soundness` | n=2 · Brier 0.47 (0.25 = uninformative) · outcomes: 2 no, 4 yes |
| `self-critique-loop-helps` | no scored predictions yet |
| `self-critique-loop-soundness` | n=2 · Brier 0.60 (0.25 = uninformative) · outcomes: 17 no, 15 no |
| `plan-validation-helps` | n=3 · Brier 0.22 (0.25 = uninformative) · outcomes: 6 no, 7 no, 11 no |
| `plan-validation-soundness` | n=5 · Brier 0.22 (0.25 = uninformative) · outcomes: 6 yes, 13 no, 5 no, 7 yes, 11 yes |
| `tighten-system-prompt-helps` | n=7 · Brier 0.18 (0.25 = uninformative) · outcomes: 16 no, 9 yes, 21 no, 24 no, 26 no |
| `tighten-system-prompt-soundness` | n=9 · Brier 0.22 (0.25 = uninformative) · outcomes: 14 no, 21 yes, 24 yes, 25 no, 26 yes |
| `planner-telemetry-helps` | n=3 · Brier 0.45 (0.25 = uninformative) · outcomes: 19 yes, 31 no, 27 yes |
| `planner-telemetry-soundness` | n=3 · Brier 0.24 (0.25 = uninformative) · outcomes: 19 yes, 27 no, 31 yes |
| `targeted-repair-guidance-helps` | no scored predictions yet |
| `targeted-repair-guidance-soundness` | no scored predictions yet |
| `bounded-fallback-pass-helps` | n=1 · Brier 0.31 (0.25 = uninformative) · outcomes: 20 no |
| `bounded-fallback-pass-soundness` | n=1 · Brier 0.09 (0.25 = uninformative) · outcomes: 20 yes |
| `tool-compatibility-shim-helps` | n=3 · Brier 0.33 (0.25 = uninformative) · outcomes: 28 yes, 30 yes, 29 yes |
| `tool-compatibility-shim-soundness` | n=3 · Brier 0.08 (0.25 = uninformative) · outcomes: 28 yes, 30 yes, 29 yes |
| `current-city-header-normalization-helps` | no scored predictions yet |
| `current-city-header-normalization-soundness` | no scored predictions yet |

## §5 The calibration report the maintainer last saw

```markdown
from `edit_memory_beliefs_prompts/update_0099.txt`:

## Calibration report
- 46 scored prediction(s) (strategy 20 / implementation 26); mean Brier 0.252 vs 0.25 uninformative; 0 of 46 were uncovered (scored at p=0.5)
- 2 prediction(s) skipped, not scored: the first node of a new strategy, which no belief could have covered
- guidance versions — v0: n=22, Brier 0.267 · v1: n=5, Brier 0.283 · v2: n=6, Brier 0.125 · v3: n=9, Brier 0.358 · v4: n=4, Brier 0.088 (current)

### Per belief
- belief:call-memoization-helps (strategy strategy=add-call-memoization, p=0.16): n=1 · Brier 0.16 · 3 yes · cited by 2 proposal(s)
- belief:call-memoization-soundness (implementation strategy=add-call-memoization, p=0.78): n=1 · Brier 0.12 · 3 yes
- belief:decision-telemetry-helps (strategy strategy=add-decision-telemetry, p=0.30): n=2 · Brier 0.26 · 2 yes, 4 no
- belief:decision-telemetry-soundness (implementation strategy=add-decision-telemetry, p=0.54): n=2 · Brier 0.47 · 2 no, 4 yes
- belief:self-critique-loop-helps (strategy strategy=add-self-critique-loop, p=0.10): no scored predictions · cited by 10 proposal(s)
- belief:self-critique-loop-soundness (implementation strategy=add-self-critique-loop, p=0.05): n=2 · Brier 0.60 · 17 no, 15 no
- belief:plan-validation-helps (strategy strategy=add-plan-validation, p=0.16): n=3 · Brier 0.22 · 6 no, 7 no, 11 no
- belief:plan-validation-soundness (implementation strategy=add-plan-validation, p=0.43): n=5 · Brier 0.22 · 6 yes, 13 no, 5 no, 7 yes, 11 yes
- belief:tighten-system-prompt-helps (strategy strategy=tighten-system-prompt, p=0.20): n=7 · Brier 0.18 · 8 no, 16 no, 9 yes, 21 no, 24 no, 26 no · cited by 13 proposal(s)
- belief:tighten-system-prompt-soundness (implementation strategy=tighten-system-prompt, p=0.54): n=9 · Brier 0.22 · 9 yes, 14 no, 21 yes, 24 yes, 25 no, 26 yes
- belief:planner-telemetry-helps (strategy strategy=add-planner-telemetry, p=0.36): n=3 · Brier 0.45 · 19 yes, 31 no, 27 yes
- belief:planner-telemetry-soundness (implementation strategy=add-planner-telemetry, p=0.56): n=3 · Brier 0.24 · 19 yes, 27 no, 31 yes
- belief:targeted-repair-guidance-helps (strategy strategy=inject-targeted-repair-guidance, p=0.10): no scored predictions
- belief:targeted-repair-guidance-soundness (implementation strategy=inject-targeted-repair-guidance, p=0.18): no scored predictions
- belief:bounded-fallback-pass-helps (strategy strategy=add-bounded-fallback-pass, p=0.15): n=1 · Brier 0.31 · 20 no · cited by 2 proposal(s)
- belief:bounded-fallback-pass-soundness (implementation strategy=add-bounded-fallback-pass, p=0.45): n=1 · Brier 0.09 · 20 yes
- belief:tool-compatibility-shim-helps (strategy strategy=add-tool-compatibility-shim, p=0.69): n=3 · Brier 0.33 · 28 yes, 30 yes, 29 yes · cited by 3 proposal(s)
- belief:tool-compatibility-shim-soundness (implementation strategy=add-tool-compatibility-shim, p=0.78): n=3 · Brier 0.08 · 28 yes, 30 yes, 29 yes
- belief:current-city-header-normalization-helps (strategy strategy=repair-generated-itinerary area=intercity-current-city-headers, p=0.24): no scored predictions
- belief:current-city-header-normalization-soundness (implementation strategy=repair-generated-itinerary area=intercity-current-city-headers, p=0.72): no scored predictions · cited by 1 proposal(s)

### Worst misses (by Brier)
- node 27 · belief:planner-telemetry-helps (strategy) p=0.19 → yes (Brier 0.66) · Δ+0.1563/16 · implementation sound — "The new `decision_branch/hallucinated_plan_tool_intercepted` log point fired ≥18x, and sampled cases 31/71/72 include th" · edit: "Added a mutable `plan` tool that catches hallucinated `plan` calls and returns a finalize-now instruction telling the model to emit the itinerary directly insid"
- node 2 · belief:decision-telemetry-soundness (implementation) p=0.80 → no (Brier 0.64) · Δ+0.0938/16 · implementation unsound — "The new telemetry does fire, but its `completed`/success-style signal is misleading as an outcome marker: `plan_repair_c" · edit: "Added a bounded post-draft review pass that takes an emitted plan, checks it against explicit itinerary logic rules, and can use the existing tool context plus "
- node 25 · belief:tighten-system-prompt-soundness (implementation) p=0.79 → no (Brier 0.62) · Δ+0.0241/13 · implementation unsound — "Edit 1's attraction-focused prompt change is real and helps (essential_attraction_coverage 4/8->1/8; case 48 repaired), " · edit: "Tightened the planner system prompt to make it internally classify each day, satisfy that day’s required meal and attraction skeleton before adding rest or buff"
- node 17 · belief:self-critique-loop-soundness (implementation) p=0.78 → no (Brier 0.61) · Δ-0.2125/10 · implementation unsound — "Observed raw review outputs violated the claimed structure preservation: 94 changed 7->6 days and 105 changed 21->7 befo" · edit: "Added a narrow no-tool post-repair review pass that asks the model to rewrite only later cross-day repeated attractions or restaurants using already-supported s"
- node 28 · belief:tool-compatibility-shim-helps (strategy) p=0.23 → yes (Brier 0.59) · Δ+0.0491/14 · implementation sound — "The shim existed and ran on 5 observed `plan` calls, returning the finalize-now message instead of a missing-tool failur" · edit: "Added a mutable `plan` tool shim that catches hallucinated `plan` calls and returns a deterministic instruction to finalize immediately in a single `<plan>...</"
- node 15 · belief:self-critique-loop-soundness (implementation) p=0.77 → no (Brier 0.59) · Δ-0.0885/12 · implementation unsound — "Diff 246-255 relies on instruction-following to preserve venues/transport/structure, but only day count is programmatica" · edit: "Added a single post-draft LLM review pass that runs only when a draft plan exists and may revise only travel_city timing continuity while preserving the itinera"
- node 14 · belief:tighten-system-prompt-soundness (implementation) p=0.76 → no (Brier 0.58) · Δ-0.3182/11 · implementation unsound — "Unsound for this sub-edit: although the prompt text was added at diff +176-183, the observed targets do not confirm enfo" · edit: "Strengthened the planner system prompt with a silent pre-output audit that enforces non-final return-to-hotel closure, bans final-day hotel endings, and require"
- node 27 · belief:planner-telemetry-soundness (implementation) p=0.76 → no (Brier 0.58) · Δ+0.1875/12 · implementation sound — "Diff lines 25-30 log `verdict="pass"` on every interception, but scorer agreement on fired cases was only 1 pass / 11 fa" · edit: "Added a mutable `plan` tool that catches hallucinated `plan` calls and returns a finalize-now instruction telling the model to emit the itinerary directly insid"

### Uncovered measured nodes (scored at p=0.5)
- (none)

### Not scored
- node 5 · implementation unsound — strategy belief not scored — "Validator logic clearly runs (logs on 10/17/110/119), but detected structure problems persist in final outputs on 10/119"
- node 13 · implementation unsound — strategy belief not scored — "Unsound: although the verifier ran, its clean pass branch had only 2 scorer pass / 24 fail and missed scorer transfer fa"
- node 14 · implementation unsound — strategy belief not scored — "Unsound for this sub-edit: although the prompt text was added at diff +176-183, the observed targets do not confirm enfo"
- node 15 · implementation unsound — strategy belief not scored — "Diff 246-255 relies on instruction-following to preserve venues/transport/structure, but only day count is programmatica"
- node 17 · implementation unsound — strategy belief not scored — "Observed raw review outputs violated the claimed structure preservation: 94 changed 7->6 days and 105 changed 21->7 befo"
- node 25 · implementation unsound — strategy belief not scored — "Edit 1's attraction-focused prompt change is real and helps (essential_attraction_coverage 4/8->1/8; case 48 repaired), "

### Open predictions (registered, not yet measurable)
- node 12 · belief:call-memoization-helps p=0.42 (strategy) · belief:call-memoization-soundness p=0.72 (implementation)
- node 18 · belief:tighten-system-prompt-helps p=0.47 (strategy) · belief:tighten-system-prompt-soundness p=0.85 (implementation)
- node 22 · belief:tighten-system-prompt-helps p=0.38 (strategy) · belief:tighten-system-prompt-soundness p=0.74 (implementation)
- node 23 · belief:tighten-system-prompt-helps p=0.40 (strategy) · belief:tighten-system-prompt-soundness p=0.70 (implementation)
- node 32 · belief:current-city-header-normalization-helps p=0.32 (strategy) · belief:current-city-header-normalization-soundness p=0.68 (implementation)

### Citation checks
- [SOFT] belief:planner-telemetry-soundness — quotes `[node 29: Δ+0.0625/3]` but the record shows Δ+0.0937 over 16 shared
- [SOFT] belief:tool-compatibility-shim-helps — quotes `[node 29: Δ+0.0625/3]` but the record shows Δ+0.0937 over 16 shared
- [SOFT] belief:tool-compatibility-shim-soundness — quotes `[node 29: Δ+0.0625/3]` but the record shows Δ+0.0937 over 16 shared
```

## §6 What the editor saw for this expand — `round_030/verbose/editor_attempt_1_user.txt`

```markdown

## Objective
Raise this agent's ABSOLUTE benchmark score. Seed 0.5979/60 · best so far 0.7109/16 (node 27) · this parent (node 9) 0.6396/60.

## Scope of this edit
Make ONE targeted, coherent change to this parent — one strategy in one area — small enough to apply correctly in a single pass, and instrument every new decision point with trace.log — the analysis that judges your edit reads those logs. Do not bundle independent mechanisms into one edit: each mechanism is judged on its own, and a broken one next to a working one costs the whole node its credit.

## Edits already applied along this lineage (root → parent)
  [depth 1] Reduce redundant deterministic tool executions to cut timeout risk and speed plan generation.
  [depth 2] Strengthen first-pass itinerary compliance with a final silent audit in the planner prompt, targeting transfer-time, diversity, coverage, and opening-hours errors without adding extra tool or LLM pass

## Edits already tried directly off this parent (node 9)
- node 25: neutral Δ-0.0157/16 · child 0.5781/16 · "Improve first-pass daily coverage so full sightseeing days reliably include enough attractions and meals without extra tool or LLM passes." · flags: implementation unsound
(Full records and implementation of any node are retrievable by node id.)

## Belief document
(the document itself — see §4 above)

## Planning-pass proposal (advisory — override it if the code says otherwise)
1. Recover cases where the model hallucinates a `plan` tool call so they end with an extractable `<plan>` block instead of a tool error or empty output. [add-tool-compatibility-shim / finalization-tool-compatibility]
   mechanism: Add the narrow `plan` compatibility shim used in successful siblings: expose/route a mutable `plan` tool that does not execute real planning, but deterministically returns finalize-now guidance telling the model to emit exactly one final `<plan>...</plan>` from already gathered data. Instrument every branch with factual `trace.log` events such as intercept, returned_guidance, and any unexpected argument shape.
2. Make the no-plan failure mode observable at the exact extraction boundary so the edit can be judged sound and any remaining empty-output cases are diagnosable. [add-planner-telemetry / planning-observability]
   mechanism: Add minimal planner-side telemetry around final extraction / budget exhaustion / no-tool terminal exits to log whether a `<plan>` block was present, without changing control flow beyond the shim itself.
prediction: belief:belief:tool-compatibility-shim-helps -> expected up (Δ ~0.06)

## Retrieved records and implementations (headers only)
- node 27 (explicit)
- node 28 (explicit)
- node 29 (explicit)
- node 10 (strategy:add-planner-telemetry)
```

## §7 What retrieval showed it — `round_030/retrieval_manifest.json`

```json
{
  "version": 2,
  "query": {
    "nodes": [
      27,
      28,
      29
    ],
    "strategies": [
      "add-tool-compatibility-shim",
      "add-planner-telemetry"
    ],
    "areas": [
      "finalization-tool-compatibility",
      "empty-output-recovery",
      "planning-observability"
    ],
    "keywords": [
      "mutable plan tool shim",
      "finalize-now guidance",
      "intercepted plan call",
      "extractable <plan> block",
      "no-plan telemetry"
    ],
    "include_code": true
  },
  "max_nodes": 4,
  "char_budget": 60000,
  "per_node": 15000,
  "total_chars": 20030,
  "selected": [
    {
      "node": 27,
      "why": "explicit",
      "chars": 3073,
      "record_chars": 1519,
      "code_chars": 1552,
      "code_source": "sources",
      "hunks_shown": 3,
      "hunks_omitted": 0,
      "defs_shown": 1,
      "defs_omitted": 0
    },
    {
      "node": 28,
      "why": "explicit",
      "chars": 6176,
      "record_chars": 4583,
      "code_chars": 1591,
      "code_source": "sources",
      "hunks_shown": 3,
      "hunks_omitted": 0,
      "defs_shown": 1,
      "defs_omitted": 0
    },
    {
      "node": 29,
      "why": "explicit",
      "chars": 2755,
      "record_chars": 1162,
      "code_chars": 1591,
      "code_source": "sources",
      "hunks_shown": 3,
      "hunks_omitted": 0,
      "defs_shown": 1,
      "defs_omitted": 0
    },
    {
      "node": 10,
      "why": "strategy:add-planner-telemetry",
      "chars": 8026,
      "record_chars": 6162,
      "code_chars": 1862,
      "code_source": "sources",
      "hunks_shown": 5,
      "hunks_omitted": 0,
      "defs_shown": 0,
      "defs_omitted": 0
    }
  ],
  "dropped": [
    {
      "node": 11,
      "why": "strategy:add-planner-telemetry",
      "reason": "over max_nodes"
    },
    {
      "node": 13,
      "why": "strategy:add-planner-telemetry",
      "reason": "over max_nodes"
    },
    {
      "node": 15,
      "why": "strategy:add-planner-telemetry",
      "reason": "over max_nodes"
    },
    {
      "node": 17,
      "why": "strategy:add-planner-telemetry",
      "reason": "over max_nodes"
    },
    {
      "node": 19,
      "why": "strategy:add-planner-telemetry",
      "reason": "over max_nodes"
    },
    {
      "node": 20,
      "why": "strategy:add-planner-telemetry",
      "reason": "over max_nodes"
    },
    {
      "node": 23,
      "why": "strategy:add-planner-telemetry",
      "reason": "over max_nodes"
    },
    {
      "node": 24,
      "why": "strategy:add-planner-telemetry",
      "reason": "over max_nodes"
    },
    {
      "node": 26,
      "why": "strategy:add-planner-telemetry",
      "reason": "over max_nodes"
    }
  ]
}
```

## §8 Per-strategy outcomes (rendered live from the records)

- `add-planner-telemetry` — 16 node(s) (10, 11, 13, 15, 17, 19, 20, 23, 24, 26, 27, 28, 29, 30, 31, 32) · judge: not judged yet · implementation: sound 14 / unsound 0 · score (context): paired Δ median -0.0123 — Add structured trace logging around planner-side guards, branches, or control decisions so their activation is visible during debugging.
- `tighten-system-prompt` — 13 node(s) (7, 8, 9, 10, 14, 16, 18, 21, 22, 23, 24, 25, 26) · judge: not judged yet · implementation: sound 8 / unsound 2 · score (context): paired Δ median -0.0157 — Revise instructions to make the model more explicit about required checks, output constraints, and planning heuristics before finalizing a plan.
- `add-decision-telemetry` — 8 node(s) (1, 2, 3, 4, 5, 6, 7, 12) · judge: not judged yet · implementation: sound 7 / unsound 0 · score (context): paired Δ median +0.0264 — Added structured trace logging for cache hits, misses, stores, and uncacheable argument cases during tool execution.
- `add-plan-validation` — 7 node(s) (4, 5, 6, 7, 11, 13, 17) · judge: not judged yet · implementation: sound 4 / unsound 3 · score (context): paired Δ median -0.0468 — Run deterministic checks over draft plans to detect rule violations and trigger targeted fixes before returning the final answer.
- `add-self-critique-loop` — 7 node(s) (2, 15, 16, 17, 21, 23, 24) · judge: not judged yet · implementation: sound 0 / unsound 6 · score (context): paired Δ median -0.0625 — Insert an extra reflection pass where the model reviews its own draft against known failure patterns and repairs them.
- `add-tool-compatibility-shim` — 4 node(s) (27, 28, 29, 30) · judge: not judged yet · implementation: sound 4 / unsound 0 · score (context): paired Δ median +0.0807 — Introduce a compatibility handler that intercepts hallucinated or legacy tool calls and redirects the model to the intended workflow step without another LLM pass.
- `inject-targeted-repair-guidance` — 4 node(s) (11, 13, 18, 22) · judge: not judged yet · implementation: sound 0 / unsound 2 · score (context): paired Δ median -0.0459 — Feed deterministic failure findings and explicit fix rules into an existing repair prompt so the model corrects specific known issues.
- `add-call-memoization` — 3 node(s) (1, 3, 12) · judge: not judged yet · implementation: sound 2 / unsound 0 · score (context): paired Δ median +0.0117 — Added a per-task cache in the tool wrapper that canonicalizes tool arguments and reuses stored results for repeated equivalent executions before falling back to
- `add-bounded-fallback-pass` — 2 node(s) (19, 20) · judge: not judged yet · implementation: sound 1 / unsound 1 · score (context): paired Δ median +0.0003 — Added a single tool-free fallback LLM pass that runs when the workflow ends without any extractable <plan> block and asks the model to turn already collected co
- `repair-generated-itinerary` — 2 node(s) (31, 32) · judge: not judged yet · implementation: sound 1 / unsound 0 · score (context): paired Δ median -0.0261 — Post-process model output to adjust times, activities, headers, or fields so the itinerary better satisfies evaluator-facing constraints.

## §9 Artefact map

run root: README_ISSUES.md, belief_instruction.md, belief_instruction_archive/, config.snapshot.yaml, edit_memory_beliefs.md, edit_memory_beliefs_archive/, edit_memory_beliefs_prompts/, edit_memory_beliefs_state.json, edit_memory_candidates.json, edit_memory_registry.json, run_summary.md

round_030/: belief_prediction.json, edit_analysis_prompt.txt, edit_code.md, edit_memory.md, edit_memory_prompt.txt, edit_memory_state.json, edit_prediction.json, edit_usage.json, eval_result.json, feedback.json, hgm_node.json, logs/, retrieval_manifest.json, strategy.json, task_agent/, verbose/
