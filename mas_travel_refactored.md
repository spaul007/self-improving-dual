# travel_mas_refactored — baseline performance and conversion-pipeline analysis

## What this project is

`projects/travel_mas_refactored/` is a structural refactor of `projects/travel_mas/`
(the 4-agent Flight → Train → Sightseeing → Accounting travel-planning MAS built
earlier this session). `projects/travel_mas/` itself is untouched — this is a new
sibling project, not a modification.

Two things changed, both purely structural (no behavior change from the original
`travel_mas`, confirmed via matched live smoke tests before/after each change):

1. **Standardized agent-to-agent collaboration interface.** Every stage function
   now has the signature `(task: Task, inbox: list[AgentMessage], ...) ->
   AgentMessage`, replacing four previously-inconsistent tuple/string return
   shapes. `AgentMessage(sender, content, ok, iterations, budget_exhausted)` is a
   frozen dataclass living in `seed/agents/immutable/message.py`, excluded from
   HGM's editable surface via `mutable_exclude` (matching `math_mas`/`shopping_mas`'s
   own `tools/immutable/` convention) so the contract itself can't be silently
   redefined by an edit. Upstream messages are looked up by sender name
   (`from_sender(inbox, "flight")`), not positional indexing, so each call site is
   self-explanatory without needing to trace back to the orchestrator.

   As a side effect, this surfaced (and fixed) a real, pre-existing framework gap:
   `meta_agent/config.py`'s `_read_tools_source()` — the mechanism that shows the
   editor read-only reference material for its `mutable_exclude`d files — had no
   glob pattern matching an `immutable/` convention, so `math_mas`'s own
   `tools/immutable/answer_extraction.py` was silently invisible to its own editor
   too. Fixed generically (`seed_dir/**/immutable/**/*.py`), benefiting both
   projects.

2. **Fully self-contained tools/data.** `projects/travel_mas_refactored/tools/`
   now holds its own real copy of the 9 tool implementations (previously a
   passthrough `import projects.travel.tools` hook). `data/database_en` is a
   symlink to `projects/travel/data/database_en` (avoids duplicating ~431MB of
   per-sample CSVs) rather than a live code/import dependency — confirmed
   `projects.travel.tools` is never imported at runtime.

Config: `configs/travel_mas_refactored_qwen35b_implicit.yaml` — `Qwen/Qwen3.5-35B-A3B`
on node-6, implicit mode (no `reasoning_effort`; `temperature=0.2`,
`max_output_tokens=16384`).

## Baseline performance (full 120-case benchmark, unedited seed)

No HGM optimization has been run against this project yet — these are the
unedited-seed numbers, the number any future optimization run needs to beat.

| Metric | Value |
|---|---|
| **Mean composite score** | **0.4135** |
| **Case accuracy (pass rate)** | **0/120 = 0%** |
| no_plan_rate | 23.3% (28/120) |
| Wall time | 9990s (~2h 47m), full 120 cases, parallelism 3 |

For reference, this lands close to (slightly above) the original `travel_mas`
seed's own full-120 baseline (0.3688) — expected, since this was a pure
structural refactor; the small delta is ordinary run-to-run LLM variance, not a
behavior change.

### Dimension breakdown (unweighted per-dimension means)

| Dimension | Mean |
|---|---:|
| Route Consistency | 0.609 |
| Sandbox Compliance | 0.348 |
| Itinerary Structure | 0.196 |
| **Time Feasibility** | **0.022** |
| (remaining dimensions not fully captured in this pass — see `top_failed_checks` below for the fuller picture) |

### Top failed checks (out of 120 cases)

| Check | Cases failed |
|---|---:|
| `commonsense:Time Feasibility:reasonable_transfer_time` | 89 |
| `commonsense:Itinerary Structure:traceable_accommodation` | 55 |
| `commonsense:Itinerary Structure:essential_meal_coverage` | 46 |
| `commonsense:Sandbox Compliance:validated_meals` | 42 |
| `commonsense:Sandbox Compliance:validated_attractions` | 39 |
| `commonsense:Business Hours:dining_within_service_hours` | 37 |
| `commonsense:Activity Diversity:diverse_meal_options` | 35 |
| `commonsense:Business Hours:attraction_visit_within_opening_hours` | 35 |
| `commonsense:Itinerary Structure:essential_attraction_coverage` | 33 |
| `commonsense:Route Consistency:seamless_intercity_transfers` | 33 |
| `commonsense:Duration Rationality:reasonable_duration_at_attractions` | 32 |
| `commonsense:Activity Diversity:diverse_attraction_options` | 26 |
| `commonsense:Itinerary Structure:ends_with_accommodation` | 25 |
| `hard:restaurant_specific_tag_nearby` | 9 |
| `commonsense:Business Hours:avoidance_of_closure_days` | 8 |

**`reasonable_transfer_time` (89/120, 74%) is the single dominant failure mode**,
consistent with `Time Feasibility`'s near-total collapse (mean 0.022) — the
Sightseeing agent is not reliably enforcing that scheduled gaps between
consecutive activities meet `query_road_route_info`'s reported travel duration,
despite this being explicit instruction text already in `DAY_STRUCTURE_RULES`.

## Conversion-pipeline failure analysis

Two specific questions asked and answered by direct analysis of all 120
`logs/case_*.json` files (not sampled — every completed case):

**Q1: What fraction of errors are due to the conversion model failing to parse
a *valid* raw plan into JSON** (as opposed to the agent itself never producing
a plan)?

**A: 0% (0/120 cases).** The plan→JSON conversion step (`_convert_plan_to_json`
in `adapter/scorer_impl.py`, currently routed at the same 35B/node-6 model) has
not failed once across the full benchmark to parse a genuinely-produced raw
plan. All 28 "no plan" failures (23.3%) are the agent itself never producing a
`<plan>` block — a distinct, and much larger, failure mode than conversion
error.

**Q2: Among successfully-converted cases, what fraction have a JSON field value
that doesn't trace back to the raw plan text** (i.e. the conversion step
introducing/hallucinating a value)?

**A: 0% (0/92 successful conversions; 0/15,708 individual name/price/time/route
fields checked).** Every leaf value in every `converted_plan` — hotel/restaurant/
attraction names, prices, time slots, route legs — was found verbatim
(normalized for currency symbols/commas) in the corresponding raw plan text.

**Conclusion: the text→JSON conversion pipeline is not a meaningful bottleneck
in this system, in either sense.** 100% of the measured failure is attributable
to the agent's own plan generation (no-plan production, and the specific
commonsense/hard-constraint dimensions above) — 0% to the conversion model.
This is a directly useful, evidence-grounded finding: any future optimization
effort here should not spend budget on conversion-pipeline robustness; it
should target the actual dominant failure (`reasonable_transfer_time` /
`Time Feasibility`, and the itinerary-structure/business-hours cluster below it).

## Comparative note: Travel-MAS-main-aounon

A separate, unrelated benchmark system at `/groups/AIC-MV/v.kulkarni1/Travel-MAS-main-aounon`
was investigated for architectural comparison (not adopted, not wired in):

- **No text→JSON conversion step exists there at all** — every agent emits
  structured JSON directly; the scored plan is assembled by deterministic code
  (`scheduler.compile_plan` + `budget_mod.attach_budget`), and a *deterministic
  inverse* (`render.py::render_markdown`) turns the already-final JSON into
  display markdown, never the other way around. Structurally immune to the
  failure mode this session confirmed we don't actually have either (see above).
- **Its `plan_repairer` directly imports the official scoring functions**
  (`tools/mutable/plan_checker.py:12-13`, `from eval.constraints_commonsense
  import eval_commonsense`) — i.e. it has full answer-key access, which is a
  design constraint we've explicitly ruled out for our own system (verifiers
  here must be learned from error logs/plans, never from reading
  `_eval/constraints_*.py`, which the editor doesn't have access to under
  `eval_visibility: "blackbox"`).
- **Most data retrieval there is deterministic code, not LLM tool-calling**
  (`tools/mutable/resolver.py`'s `resolve_transport`/`resolve_hotel`/etc. filter
  the sandbox DB directly; the LLM only reviews/judges a short pre-resolved
  candidate list). This part *is* legitimately transferable in principle — it
  doesn't touch grading logic, only how sandbox data is queried — and would
  directly target this session's independently-documented tool-calling
  fragility on the 35B endpoint (`qwen35bnotworking.md`).

See `tier_based_hgm.md` for the fuller design discussion this analysis fed
into (tier-gating EXPAND budget by failure leverage, and the currently-empty
`verifiers` block in this project's own block-bandit case study).

## Run artifacts

- Baseline eval: `runs/travel_mas_refactored_baseline_full120/`
  (`seed_full_benchmark/run_1/`, `summary.json`, per-case logs)
- No HGM optimization run has been performed against `travel_mas_refactored`
  yet — this document covers the unedited-seed baseline only.
