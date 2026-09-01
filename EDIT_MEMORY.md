# Edit memory — 20260615_095317_travel_hgm_8000

Tree-global record of every edit attempted in this run: what was changed, why, and what it did to the benchmark. Deduped into recurring **motifs**. This is not a behavior summary — it describes *actions and their payoff*, not runtime behavior, and it covers **all** branches, not one lineage.

## §0 Run

- project **travel** · manager **hgm** · editor **gpt-5.4** · task agent **gpt-5.4-mini**
- **76 nodes**, eval budget spent 4488
- seed (node 0) **0.7635** → best **node 47** at **0.9292**
- champion lineage: 0 → 3 → 11 → 23 → 26 → 36 → 37 → 46 → 47
- of 75 edits: **16 helped**, 31 hurt, 28 neutral/inconclusive

## §1 Motif ledger — what has been tried, and did it work

`Δ` is the mean score change measured **only on cases the parent and child both ran**, so it is not confounded by case sampling.

### `add-tool-backed-evidence-verifier` — 29× attempted
*re-queries the real tools and fails the draft when the plan disagrees with tool data*

- Δ median **-0.0260** · best +0.1031 · worst -0.1406
- verdicts: hurt 16, neutral 7, helped 6
- aimed at: intercity-transport ×14, restaurant ×13, transfer-time ×11, hotel ×11
- nodes: 13, 16, 20, 22, 29, 34, 38, 42, 43, 44, 46, 47, 48, 50, 51, 52, 56, 60, 61, 62, 65, 67, 68, 69, 70, 71, 73, 74, 75 · **on champion lineage: [46, 47]**
- net checks moved (summed over attempts, top-6 movers per attempt; the node count shows whether a figure is systematic or one bad edit): `commonsense:Time Feasibility:reasonable_transfer_time` +31 (25 nodes) · `commonsense:Itinerary Structure:traceable_accommodation` +15 (2 nodes) · `commonsense:Activity Diversity:diverse_attraction_options` +7 (19 nodes) · `hard:hotel_star_service_required` +7 (20 nodes) · `commonsense:Itinerary Structure:essential_meal_coverage` -65 (17 nodes) · `commonsense:Route Consistency:seamless_intercity_transfers` -135 (24 nodes) · `commonsense:Route Consistency:closed_loop_route_structure` -140 (4 nodes)
- best instance: node 47 (+0.1031); that edit's largest new symbol is `workflow._verify_restaurant_constraints` (see §4 / the node record for its code)
- **The run's dominant strategy and, on median, a losing one: 29 attempts, 16 hurt, median -0.026. Yet it produced the single best edit in the tree (node 47, +0.1031) and carries the champion lineage. Variance is the story, not the mean — the wins were narrow verifiers aimed at one failure family, the losses were broad suites re-querying everything (nodes 65, 71, 74 each added 400-800 lines and lost ground). Add one verifier, not a layer. Watch the side effect: seamless_intercity_transfers is net -135 across 24 of the 29 nodes, a systematic regression rather than a few bad edits — these verifiers keep pushing the model into repairs that drop the intercity leg.**

### `add-textual-plan-verifier` — 21× attempted
*parses the rendered plan and checks structural rules with no tool calls*

- Δ median **-0.0063** · best +0.1177 · worst -0.0990
- verdicts: neutral 11, hurt 7, helped 3
- aimed at: plan-structure ×10, attraction-diversity ×7, budget-cost ×6, intercity-transport ×5
- nodes: 1, 6, 12, 15, 18, 19, 21, 26, 27, 31, 35, 37, 41, 48, 49, 52, 54, 55, 58, 64, 72 · **on champion lineage: [26, 37]**
- net checks moved (summed over attempts, top-6 movers per attempt; the node count shows whether a figure is systematic or one bad edit): `commonsense:Activity Diversity:diverse_attraction_options` +50 (18 nodes) · `commonsense:Itinerary Structure:traceable_accommodation` +30 (7 nodes) · `commonsense:Cost Calculation Accuracy:cost_calculation_correctness` +25 (16 nodes) · `hard:flight_departure_time_range` +4 (2 nodes) · `commonsense:Itinerary Structure:essential_meal_coverage` -30 (13 nodes) · `commonsense:Route Consistency:seamless_intercity_transfers` -89 (18 nodes) · `commonsense:Route Consistency:closed_loop_route_structure` -120 (4 nodes)
- best instance: node 6 (+0.1177); that edit's largest new symbol is `workflow._describe_repair_reason` (see §4 / the node record for its code)
- **Roughly break-even and cheap: 21 attempts, median -0.006, best +0.1177 (node 6). Needs no tool calls, so it costs nothing at eval time. Its record is best when paired with a repair loop that consumes the issues it finds, rather than shipped alone.**

### `add-constraint-extractor` — 14× attempted
*parses the user request into structured hard constraints before or during auditing*

- Δ median **-0.0479** · best +0.0125 · worst -0.1406
- verdicts: hurt 9, neutral 5
- aimed at: restaurant ×9, hotel ×8, intercity-transport ×7, attraction-diversity ×2
- nodes: 34, 36, 42, 49, 55, 61, 64, 67, 70, 71, 72, 73, 74, 75 · **on champion lineage: [36]**
- net checks moved (summed over attempts, top-6 movers per attempt; the node count shows whether a figure is systematic or one bad edit): `hard:hotel_star_service_required` +11 (12 nodes) · `commonsense:Activity Diversity:diverse_attraction_options` +6 (8 nodes) · `commonsense:Cost Calculation Accuracy:cost_calculation_correctness` +6 (7 nodes) · `commonsense:Route Consistency:valid_trip_duration` +5 (4 nodes) · `commonsense:Itinerary Structure:essential_meal_coverage` -31 (9 nodes) · `commonsense:Route Consistency:seamless_intercity_transfers` -115 (9 nodes) · `commonsense:Route Consistency:closed_loop_route_structure` -118 (2 nodes)
- best instance: node 36 (+0.0125); that edit's largest new symbol is `workflow._build_query_risk_block` (see §4 / the node record for its code)
- **The clearest dead end in this run: 14 attempts, zero helped, median -0.048, worst -0.1406. Parsing the user's request into structured hard constraints by regex misfires often enough that the verifiers built on top reject good plans. Do not retry without changing the extraction mechanism itself.**

### `add-deterministic-postprocessor` — 10× attempted
*rewrites the plan in code without asking the model (headers, budget, buffers)*

- Δ median **-0.0161** · best +0.1177 · worst -0.0990
- verdicts: hurt 5, neutral 4, helped 1
- aimed at: plan-structure ×5, budget-cost ×5, meal-coverage ×3, transfer-time ×2
- nodes: 1, 6, 7, 18, 19, 27, 40, 53, 57, 59
- net checks moved (summed over attempts, top-6 movers per attempt; the node count shows whether a figure is systematic or one bad edit): `commonsense:Activity Diversity:diverse_attraction_options` +24 (7 nodes) · `commonsense:Itinerary Structure:traceable_accommodation` +21 (4 nodes) · `commonsense:Cost Calculation Accuracy:cost_calculation_correctness` +12 (9 nodes) · `hard:train_latest_arrival_direct` +9 (4 nodes) · `hard:hotel_star_service_required` -8 (6 nodes) · `commonsense:Route Consistency:closed_loop_route_structure` -36 (4 nodes) · `commonsense:Route Consistency:seamless_intercity_transfers` -39 (9 nodes)
- best instance: node 6 (+0.1177); that edit's largest new symbol is `workflow._describe_repair_reason` (see §4 / the node record for its code)
- **Mostly negative with one large exception: node 6's header/accommodation normalizer (+0.1177) is the second-best edit in the tree, while the other nine attempts sit at median -0.016. Rewriting the plan in code pays when the target is a rigid format rule the grader checks literally, and backfires once it touches itinerary content.**

### `add-selfcheck-repair-loop` — 9× attempted
*feeds detected issues back to the model for a bounded repair/audit turn*

- Δ median **+0.0073** · best +0.0896 · worst -0.0990
- verdicts: neutral 5, helped 3, hurt 1
- aimed at: plan-structure ×7, budget-cost ×3, transfer-time ×3, attraction-diversity ×3
- nodes: 1, 4, 8, 12, 13, 21, 23, 26, 31 · **on champion lineage: [23, 26]**
- net checks moved (summed over attempts, top-6 movers per attempt; the node count shows whether a figure is systematic or one bad edit): `commonsense:Activity Diversity:diverse_attraction_options` +31 (9 nodes) · `commonsense:Time Feasibility:reasonable_transfer_time` +25 (9 nodes) · `commonsense:Itinerary Structure:essential_attraction_coverage` +13 (5 nodes) · `commonsense:Sandbox Compliance:validated_transportation` +9 (6 nodes) · `commonsense:Itinerary Structure:traceable_accommodation` -33 (8 nodes) · `commonsense:Route Consistency:seamless_intercity_transfers` -33 (8 nodes) · `commonsense:Route Consistency:closed_loop_route_structure` -60 (1 node)
- best instance: node 23 (+0.0896); that edit's largest new symbol is `workflow._build_audit_prompt` (see §4 / the node record for its code)
- **One of only three motifs with a positive median (+0.0073 over 9 attempts) and just 1 hurt. Node 23 (+0.0896) introduced the bounded MAX_AUDIT_ROUNDS loop and sits on the champion lineage. The safest structural move in this run.**

### `harden-existing-tool` — 9× attempted
*fixes crashes, bad kwargs or brittle matching in a tool a previous edit added*

- Δ median **-0.0469** · best +0.0021 · worst -0.0760
- verdicts: hurt 5, neutral 4
- aimed at: intercity-transport ×5, budget-cost ×3, robustness ×2, restaurant ×2
- nodes: 11, 24, 28, 35, 39, 40, 43, 44, 53 · **on champion lineage: [11]**
- net checks moved (summed over attempts, top-6 movers per attempt; the node count shows whether a figure is systematic or one bad edit): `commonsense:Activity Diversity:diverse_attraction_options` +17 (3 nodes) · `commonsense:Route Consistency:valid_trip_duration` +6 (1 node) · `commonsense:Cost Calculation Accuracy:cost_calculation_correctness` +4 (9 nodes) · `hard:train_latest_arrival_direct` +4 (6 nodes) · `commonsense:Itinerary Structure:essential_meal_coverage` -19 (6 nodes) · `commonsense:Route Consistency:seamless_intercity_transfers` -86 (9 nodes) · `commonsense:Route Consistency:closed_loop_route_structure` -93 (2 nodes)
- best instance: node 39 (+0.0021); that edit's largest new symbol is `mutable_tools.select_intercity_option._normalize_sort_mode` (see §4 / the node record for its code)
- **Nine attempts, none helped, median -0.047. Going back to patch a tool an earlier edit added never recovered the ground that tool lost — the crashes fixed were real but were not what was costing score. Treat an underperforming mutable tool as sunk cost rather than a repair target.**

### `add-deterministic-selector-tool` — 8× attempted
*new mutable tool that filters and ranks candidates instead of letting the model choose*

- Δ median **-0.0177** · best +0.0208 · worst -0.0938
- verdicts: hurt 4, neutral 3, helped 1
- aimed at: intercity-transport ×6, hotel ×3, plan-structure ×2, restaurant ×2
- nodes: 2, 7, 28, 32, 33, 41, 45, 63
- net checks moved (summed over attempts, top-6 movers per attempt; the node count shows whether a figure is systematic or one bad edit): `commonsense:Itinerary Structure:traceable_accommodation` +17 (3 nodes) · `commonsense:Activity Diversity:diverse_attraction_options` +13 (7 nodes) · `hard:hotel_star_service_required` +6 (5 nodes) · `hard:train_latest_arrival_direct` +4 (1 node) · `commonsense:Itinerary Structure:essential_meal_coverage` -17 (2 nodes) · `commonsense:Route Consistency:closed_loop_route_structure` -60 (1 node) · `commonsense:Route Consistency:seamless_intercity_transfers` -65 (8 nodes)
- best instance: node 2 (+0.0208); that edit's largest new symbol is `mutable_tools.select_intercity_transport.run` (see §4 / the node record for its code)
- **Eight independent attempts (nodes 2, 7, 28, 32, 33, 41, 45, 63) at replacing the model's choice with coded filter/rank logic; only node 2 (+0.0208) helped, median -0.018. The idea is clearly attractive to editors and has repeatedly failed to pay off — strong dedup signal.**

### `add-evidence-recorder` — 6× attempted
*harvests tool outputs into an evidence store the verifiers later read*

- Δ median **-0.0140** · best +0.0375 · worst -0.0385
- verdicts: hurt 3, helped 2, neutral 1
- aimed at: restaurant ×4, transfer-time ×3, intercity-transport ×3, budget-cost ×2
- nodes: 29, 56, 65, 68, 69, 73
- net checks moved (summed over attempts, top-6 movers per attempt; the node count shows whether a figure is systematic or one bad edit): `commonsense:Activity Diversity:diverse_attraction_options` +3 (4 nodes) · `commonsense:Sandbox Compliance:validated_meals` +2 (4 nodes) · `commonsense:Time Feasibility:reasonable_transfer_time` +2 (6 nodes) · `hard:train_departure_time_range` +2 (1 node) · `commonsense:Itinerary Structure:essential_meal_coverage` -17 (4 nodes) · `commonsense:Route Consistency:closed_loop_route_structure` -27 (1 node) · `commonsense:Route Consistency:seamless_intercity_transfers` -29 (6 nodes)
- best instance: node 29 (+0.0375); that edit's largest new symbol is `workflow._verify_budget_summary` (see §4 / the node record for its code)
- **Harvesting tool outputs into an evidence store is the prerequisite for the evidence verifiers and inherits their variance: 6 attempts, median -0.014, best +0.0375 (node 29). Never tried on its own, so its independent effect is unmeasured.**

### `add-lookup-helper-tool` — 5× attempted
*new mutable tool that resolves or fetches authoritative facts*

- Δ median **+0.0250** · best +0.0396 · worst -0.0229
- verdicts: helped 3, neutral 1, hurt 1
- aimed at: transfer-time ×3, restaurant ×2
- nodes: 3, 5, 17, 30, 66 · **on champion lineage: [3]**
- net checks moved (summed over attempts, top-6 movers per attempt; the node count shows whether a figure is systematic or one bad edit): `commonsense:Time Feasibility:reasonable_transfer_time` +41 (5 nodes) · `commonsense:Itinerary Structure:traceable_accommodation` +7 (4 nodes) · `commonsense:Route Consistency:seamless_intercity_transfers` +2 (5 nodes) · `hard:restaurant_must_eat_named` +2 (1 node) · `commonsense:Sandbox Compliance:validated_meals` -2 (3 nodes) · `hard:train_cheapest_direct` -3 (1 node) · `commonsense:Itinerary Structure:essential_meal_coverage` -4 (1 node)
- best instance: node 30 (+0.0396); that edit's largest new symbol is `mutable_tools.schedule_named_transfer.run` (see §4 / the node record for its code)
- **The best-supported positive result in the run: 5 attempts, median +0.025, three helped and only one hurt. Giving the model a tool that resolves facts it was previously guessing beats checking its work afterwards. Node 30's schedule_named_transfer (+0.0396) and node 17's build_city_transfer (+0.0354) are the instances to copy.**

### `add-validator-tool` — 3× attempted
*new mutable tool whose job is to validate a drafted plan*

- Δ median **-0.0021** · best +0.0021 · worst -0.0375
- verdicts: neutral 2, hurt 1
- aimed at: plan-structure ×3, attraction-diversity ×1, transfer-time ×1
- nodes: 4, 8, 20
- net checks moved (summed over attempts, top-6 movers per attempt; the node count shows whether a figure is systematic or one bad edit): `commonsense:Time Feasibility:reasonable_transfer_time` +22 (3 nodes) · `commonsense:Activity Diversity:diverse_attraction_options` +16 (3 nodes) · `hard:restaurant_specific_tag_nearby` +3 (1 node) · `hard:train_latest_arrival_direct` +2 (2 nodes) · `commonsense:Itinerary Structure:essential_meal_coverage` -19 (1 node) · `commonsense:Route Consistency:closed_loop_route_structure` -20 (1 node) · `commonsense:Route Consistency:seamless_intercity_transfers` -23 (3 nodes)
- best instance: node 8 (+0.0021); that edit's largest new symbol is `mutable_tools.validate_travel_plan.run` (see §4 / the node record for its code)
- **Three attempts at moving plan validation into a mutable tool; all roughly neutral (median -0.002). Functionally the same as add-textual-plan-verifier with the code relocated out of workflow.py, and the relocation bought nothing.**

### `reduce-iteration-budget` — 2× attempted
*lowers MAX_ITERATIONS or audit rounds to curb timeouts*

- Δ median **+0.0052** · best +0.0823 · worst -0.0719
- verdicts: hurt 1, helped 1
- aimed at: restaurant ×2, budget-cost ×1, closure-hours ×1, transfer-time ×1
- nodes: 49, 60
- net checks moved (summed over attempts, top-6 movers per attempt; the node count shows whether a figure is systematic or one bad edit): `commonsense:Activity Diversity:diverse_attraction_options` +6 (2 nodes) · `hard:restaurant_must_eat_named` +4 (2 nodes) · `commonsense:Time Feasibility:reasonable_transfer_time` +2 (2 nodes) · `commonsense:Business Hours:avoidance_of_closure_days` +2 (1 node) · `hard:restaurant_closest_to_attraction` -2 (1 node) · `commonsense:Business Hours:dining_within_service_hours` -3 (2 nodes) · `commonsense:Itinerary Structure:essential_meal_coverage` -3 (2 nodes)
- best instance: node 60 (+0.0823); that edit's largest new symbol is `workflow._verify_anchor_transfer_links` (see §4 / the node record for its code)
- **Only ever bundled with other changes (nodes 49, 60), so its effect is not separable: node 60 helped (+0.0823), node 49 hurt (-0.0719). No usable prior.**

### `add-shortlist-generator-tool` — 2× attempted
*new mutable tool that returns a curated, de-duplicated candidate shortlist*

- Δ median **+0.0172** · best +0.0219 · worst +0.0125
- verdicts: neutral 1, helped 1
- aimed at: attraction-diversity ×2
- nodes: 9, 25
- net checks moved (summed over attempts, top-6 movers per attempt; the node count shows whether a figure is systematic or one bad edit): `commonsense:Itinerary Structure:traceable_accommodation` +12 (2 nodes) · `commonsense:Time Feasibility:reasonable_transfer_time` +8 (2 nodes) · `commonsense:Activity Diversity:diverse_attraction_options` +3 (2 nodes) · `commonsense:Sandbox Compliance:validated_transportation` +2 (1 node) · `commonsense:Itinerary Structure:essential_attraction_coverage` -3 (1 node) · `commonsense:Sandbox Compliance:validated_meals` -3 (1 node) · `hard:restaurant_closest_to_attraction` -3 (1 node)
- best instance: node 25 (+0.0219); that edit's largest new symbol is `mutable_tools.recommend_diverse_attractions.run` (see §4 / the node record for its code)
- **Two attempts (nodes 9, 25), both non-negative, both aimed at attraction diversity. Thin evidence, but the only motif in the run with no negative instance.**

### `tighten-system-prompt` — 2× attempted
*the edit's main lever is rewriting SYSTEM_PROMPT / audit-prompt rules*

- Δ median **+0.0047** · best +0.0125 · worst -0.0031
- verdicts: neutral 2
- aimed at: intercity-transport ×2, hotel ×1, plan-structure ×1
- nodes: 36, 37 · **on champion lineage: [36, 37]**
- net checks moved (summed over attempts, top-6 movers per attempt; the node count shows whether a figure is systematic or one bad edit): `hard:hotel_star_service_required` +4 (1 node) · `commonsense:Route Consistency:seamless_intercity_transfers` +3 (2 nodes) · `commonsense:Sandbox Compliance:validated_transportation` +2 (1 node) · `commonsense:Sandbox Compliance:validated_accommodation` +1 (1 node) · `hard:restaurant_closest_to_attraction` -1 (1 node) · `hard:flight_arrival_time_range` -2 (1 node) · `commonsense:Time Feasibility:reasonable_transfer_time` -4 (2 nodes)
- best instance: node 36 (+0.0125); that edit's largest new symbol is `workflow._build_query_risk_block` (see §4 / the node record for its code)
- **Counted only where prompt rewriting was the edit's main lever (nodes 36, 37); both neutral. Note 63 of 75 edits touched SYSTEM_PROMPT as a side effect, so this measures prompt-only edits, not prompt editing in general.**

### `add-tool-result-cache` — 2× attempted
*caches repeated tool lookups to cut latency and timeout risk*

- Δ median **-0.0036** · best +0.0083 · worst -0.0156
- verdicts: neutral 2
- aimed at: transfer-time ×1, intercity-transport ×1, hotel ×1, restaurant ×1
- nodes: 11, 32 · **on champion lineage: [11]**
- net checks moved (summed over attempts, top-6 movers per attempt; the node count shows whether a figure is systematic or one bad edit): `hard:train_latest_arrival_direct` +3 (1 node) · `hard:restaurant_must_eat_named` +2 (1 node) · `commonsense:Itinerary Structure:traceable_accommodation` +1 (1 node) · `commonsense:Activity Diversity:diverse_attraction_options` +1 (2 nodes) · `commonsense:Sandbox Compliance:validated_transportation` -4 (1 node) · `commonsense:Cost Calculation Accuracy:cost_calculation_correctness` -6 (2 nodes) · `commonsense:Route Consistency:seamless_intercity_transfers` -7 (2 nodes)
- best instance: node 32 (+0.0083); that edit's largest new symbol is `mutable_tools.select_intercity_option.run` (see §4 / the node record for its code)
- **Two attempts, both neutral. Added for latency and timeout headroom rather than score, and score is indeed unmoved — judge it on wall-time, not on this number.**

### `add-manifest-builder-tool` — 1× attempted
*new mutable tool that renders an exact, copy-ready itinerary fragment*

- Δ median **+0.0510** · best +0.0510 · worst +0.0510
- verdicts: helped 1
- aimed at: intercity-transport ×1
- nodes: 10
- net checks moved (summed over attempts, top-6 movers per attempt; the node count shows whether a figure is systematic or one bad edit): `commonsense:Route Consistency:seamless_intercity_transfers` +8 (1 node) · `commonsense:Cost Calculation Accuracy:cost_calculation_correctness` +3 (1 node) · `commonsense:Itinerary Structure:essential_attraction_coverage` +3 (1 node) · `commonsense:Time Feasibility:reasonable_transfer_time` +1 (1 node) · `commonsense:Sandbox Compliance:validated_meals` -2 (1 node) · `hard:restaurant_must_eat_named` -2 (1 node) · `commonsense:Itinerary Structure:traceable_accommodation` -3 (1 node)
- best instance: node 10 (+0.0510); that edit's largest new symbol is `mutable_tools.build_intercity_manifest.run` (see §4 / the node record for its code)
- **One attempt (node 10, +0.0510): a tool that renders an exact copy-ready intercity line instead of describing one. Promising but unreplicated, and the three later edits that modified it (28, 43, 44) all lost ground.**

### `add-arithmetic-tool` — 1× attempted
*new mutable tool that moves numeric computation out of the model*

- Δ median **-0.0010** · best -0.0010 · worst -0.0010
- verdicts: neutral 1
- aimed at: budget-cost ×1
- nodes: 14
- net checks moved (summed over attempts, top-6 movers per attempt; the node count shows whether a figure is systematic or one bad edit): `commonsense:Sandbox Compliance:validated_meals` +3 (1 node) · `commonsense:Cost Calculation Accuracy:cost_calculation_correctness` +2 (1 node) · `commonsense:Itinerary Structure:traceable_accommodation` -1 (1 node) · `hard:restaurant_specific_tag_nearby` -3 (1 node) · `commonsense:Itinerary Structure:traceable_accommodation` -1 (1 node) · `hard:restaurant_specific_tag_nearby` -3 (1 node) · `commonsense:Time Feasibility:reasonable_transfer_time` -6 (1 node)
- best instance: node 14 (-0.0010); that edit's largest new symbol is `mutable_tools.calculate_budget_summary.run` (see §4 / the node record for its code)
- **One attempt (node 14), neutral. Moving budget arithmetic into a tool changed nothing on its own; later branches attacked the same budget failures with deterministic postprocessors instead.**

### `add-conditional-prompt-injection` — 1× attempted
*injects an extra system note only when the task text matches a pattern*

- Δ median **-0.0229** · best -0.0229 · worst -0.0229
- verdicts: hurt 1
- aimed at: restaurant ×1
- nodes: 5
- net checks moved (summed over attempts, top-6 movers per attempt; the node count shows whether a figure is systematic or one bad edit): `commonsense:Sandbox Compliance:validated_transportation` +3 (1 node) · `commonsense:Itinerary Structure:traceable_accommodation` +3 (1 node) · `commonsense:Time Feasibility:reasonable_transfer_time` +2 (1 node) · `commonsense:Route Consistency:seamless_intercity_transfers` +2 (1 node) · `commonsense:Cost Calculation Accuracy:cost_calculation_correctness` -2 (1 node) · `commonsense:Itinerary Structure:essential_meal_coverage` -4 (1 node) · `commonsense:Sandbox Compliance:validated_meals` -5 (1 node)
- best instance: node 5 (-0.0229); that edit's largest new symbol is `mutable_tools.restaurant_constraint_helper.run` (see §4 / the node record for its code)
- **One attempt (node 5), hurt (-0.0229). The keyword-triggered system note fired on nearly every case, so it diluted the prompt rather than targeting the cases it was written for.**

## §2 What the best agent (node 47) is made of

Edits along the champion lineage, oldest first — this is the baseline a new edit would be stacking on top of.

- **node 0** — seed agent
- **node 3** (+0.0167) — Reduce transfer-time commonsense failures by giving the model a safer exact-route helper and stronger cluster-based scheduling gui
  - added tools `query_route_by_place_names`; edited SYSTEM_PROMPT; 5 new symbols · motifs: `add-lookup-helper-tool`
- **node 11** (-0.0156) — Harden named-place routing so hotel/attraction/restaurant transfers keep working when search_location lacks the exact POI.
  - 12 new symbols · motifs: `harden-existing-tool`, `add-tool-result-cache`
- **node 23** (+0.0896) — Raise pass rate by adding a targeted final audit pass for restaurant-constraint, transfer-duration, and budget-math failures.
  - edited SYSTEM_PROMPT; 3 new symbols · motifs: `add-selfcheck-repair-loop`
- **node 26** (+0.0250) — Raise success on itinerary-structure/route-consistency/time-feasibility failures with a targeted self-check + re-audit loop.
  - edited SYSTEM_PROMPT; 10 new symbols · motifs: `add-textual-plan-verifier`, `add-selfcheck-repair-loop`
- **node 36** (+0.0125) — Improve pass rate on hotel/train hard constraints and validation misses by making the final audit explicitly re-check request-deri
  - edited SYSTEM_PROMPT; 14 new symbols · motifs: `tighten-system-prompt`, `add-constraint-extractor`
- **node 37** (-0.0031) — Raise pass rate on remaining route-consistency and validation misses with a small audit/verifier upgrade focused on exact intercit
  - edited SYSTEM_PROMPT; 6 new symbols · motifs: `tighten-system-prompt`, `add-textual-plan-verifier`
- **node 46** (-0.0625) — Raise pass rate on transfer-feasibility and transport hard-constraint failures with evidence-based final verifiers.
  - 26 new symbols · motifs: `add-tool-backed-evidence-verifier`
- **node 47** (+0.1031) — Raise pass rate on remaining restaurant/train hard-constraint misses plus late-arrival meal coverage/time-feasibility misses.
  - edited SYSTEM_PROMPT; 15 new symbols · motifs: `add-tool-backed-evidence-verifier`

## §3 Worst track records — tried 3+ times, usually negative

Weak priors, not prohibitions: a motif here may still be the right move if the specific instance is better executed than its predecessors.

- `add-constraint-extractor` — 14 attempts (helped 0, hurt 9), median -0.0479, best +0.0125, on nodes 34, 36, 42, 49, 55, 61, 64, 67, 70, 71, 72, 73, 74, 75. The clearest dead end in this run: 14 attempts, zero helped, median -0.048, worst -0.1406. Parsing the user's request into structured hard constraints by regex misfires often enough that the verifiers built on top reject good plans. Do not retry without changing the extraction mechanism itself.
- `harden-existing-tool` — 9 attempts (helped 0, hurt 5), median -0.0469, best +0.0021, on nodes 11, 24, 28, 35, 39, 40, 43, 44, 53. Nine attempts, none helped, median -0.047. Going back to patch a tool an earlier edit added never recovered the ground that tool lost — the crashes fixed were real but were not what was costing score. Treat an underperforming mutable tool as sunk cost rather than a repair target.
- `add-deterministic-selector-tool` — 8 attempts (helped 1, hurt 4), median -0.0177, best +0.0208, on nodes 2, 7, 28, 32, 33, 41, 45, 63. Eight independent attempts (nodes 2, 7, 28, 32, 33, 41, 45, 63) at replacing the model's choice with coded filter/rank logic; only node 2 (+0.0208) helped, median -0.018. The idea is clearly attractive to editors and has repeatedly failed to pay off — strong dedup signal.
- `add-deterministic-postprocessor` — 10 attempts (helped 1, hurt 5), median -0.0161, best +0.1177, on nodes 1, 6, 7, 18, 19, 27, 40, 53, 57, 59. Mostly negative with one large exception: node 6's header/accommodation normalizer (+0.1177) is the second-best edit in the tree, while the other nine attempts sit at median -0.016. Rewriting the plan in code pays when the target is a rigid format rule the grader checks literally, and backfires once it touches itinerary content.
- `add-validator-tool` — 3 attempts (helped 0, hurt 1), median -0.0021, best +0.0021, on nodes 4, 8, 20. Three attempts at moving plan validation into a mutable tool; all roughly neutral (median -0.002). Functionally the same as add-textual-plan-verifier with the code relocated out of workflow.py, and the relocation bought nothing.

## §4 Edit log (all nodes, chronological)

| node | ← parent | Δ shared | verdict | changed | motifs | goal |
|---|---|---|---|---|---|---|
| 1 | 0 | -0.0990 | hurt | workflow.py | add-selfcheck-repair-loop, add-deterministic-postprocessor, add-textual-plan-verifier | Reduce itinerary commonsense failures by adding a post-plan audit/repair pass plus determi |
| 2 | 0 | +0.0208 | helped | select_intercity_transport.py, tools_schema.json, workflow.py | add-deterministic-selector-tool | Improve intercity transport selection so the agent more reliably satisfies train/flight ha |
| 3 ⭐ | 0 | +0.0167 | neutral | query_route_by_place_names.py, tools_schema.json, workflow.py | add-lookup-helper-tool | Reduce transfer-time commonsense failures by giving the model a safer exact-route helper a |
| 4 | 0 | -0.0021 | neutral | validate_itinerary_requirements.py, tools_schema.json, workflow.py | add-validator-tool, add-selfcheck-repair-loop | Reduce itinerary-structure failures by forcing a draft self-check for attraction coverage, |
| 5 | 0 | -0.0229 | hurt | restaurant_constraint_helper.py, tools_schema.json, workflow.py | add-lookup-helper-tool, add-conditional-prompt-injection | Improve satisfaction of restaurant-specific hard constraints without broad planner changes |
| 6 | 1 | +0.1177 | helped | workflow.py | add-deterministic-postprocessor, add-textual-plan-verifier | Fix systematic route/header and last-day accommodation errors with deterministic postproce |
| 7 | 3 | -0.0104 | neutral | query_route_by_place_names.py, select_train_option.py, tools_schema.json, workflow.py | add-deterministic-postprocessor, add-deterministic-selector-tool | Reduce accommodation-format misses and train/route selection errors while nudging the plan |
| 8 | 3 | +0.0021 | neutral | validate_travel_plan.py, tools_schema.json, workflow.py | add-validator-tool, add-selfcheck-repair-loop | Catch structure/diversity mistakes before return by auto-validating the drafted itinerary  |
| 9 | 3 | +0.0125 | neutral | recommend_diverse_attractions.py, tools_schema.json, workflow.py | add-shortlist-generator-tool | Reduce duplicate-attraction and low-diversity itinerary failures with a structured attract |
| 10 | 0 | +0.0510 | helped | build_intercity_manifest.py, tools_schema.json, workflow.py | add-manifest-builder-tool | Reduce route-consistency and train/flight hard-constraint failures by forcing exact interc |
| 11 ⭐ | 3 | -0.0156 | neutral | query_route_by_place_names.py | harden-existing-tool, add-tool-result-cache | Harden named-place routing so hotel/attraction/restaurant transfers keep working when sear |
| 12 | 2 | +0.0167 | neutral | workflow.py | add-selfcheck-repair-loop, add-textual-plan-verifier | Add a lightweight final-audit repair pass to improve budget accuracy, transfer consistency |
| 13 | 10 | +0.0073 | neutral | workflow.py | add-tool-backed-evidence-verifier, add-selfcheck-repair-loop | Reduce itinerary-structure and transfer-feasibility failures with an automatic final-plan  |
| 14 | 2 | -0.0010 | neutral | calculate_budget_summary.py, tools_schema.json, workflow.py | add-arithmetic-tool | Reduce budget and validation errors by forcing exact-price verification and tool-based bud |
| 15 | 13 | -0.0083 | neutral | workflow.py | add-textual-plan-verifier | Raise pass rate by catching duplicate POIs, missing transfer legs, and budget-summary/budg |
| 16 | 13 | +0.0490 | helped | workflow.py | add-tool-backed-evidence-verifier | Raise pass rate by catching anchor-to-anchor transfer-time mismatches and reinforcing exac |
| 17 | 10 | +0.0354 | helped | build_city_transfer.py, tools_schema.json, workflow.py | add-lookup-helper-tool | Reduce transfer-time mismatches and itinerary-structure misses by forcing exact city-trans |
| 18 | 6 | +0.0104 | neutral | workflow.py | add-textual-plan-verifier, add-deterministic-postprocessor | Raise itinerary commonsense pass rate by auditing and repairing missing transfers, duplica |
| 19 | 1 | +0.0073 | neutral | workflow.py | add-textual-plan-verifier, add-deterministic-postprocessor | Reduce remaining commonsense failures by auditing/repairing meal coverage, duplicate POIs, |
| 20 | 14 | -0.0375 | hurt | verify_itinerary_consistency.py, tools_schema.json, workflow.py | add-validator-tool, add-tool-backed-evidence-verifier | Reduce itinerary consistency failures by automatically verifying and repairing draft plans |
| 21 | 11 | +0.0354 | helped | workflow.py | add-textual-plan-verifier, add-selfcheck-repair-loop | Raise score by repairing common final-plan structural mistakes: duplicate attractions, wro |
| 22 | 13 | -0.0271 | hurt | workflow.py | add-tool-backed-evidence-verifier | Improve sandbox compliance and hard-constraint grounding by rejecting non-database attract |
| 23 ⭐ | 11 | +0.0896 | helped | workflow.py | add-selfcheck-repair-loop | Raise pass rate by adding a targeted final audit pass for restaurant-constraint, transfer- |
| 24 | 7 | -0.0760 | hurt | select_train_option.py, tools_schema.json, workflow.py | harden-existing-tool | Reduce route/time consistency failures and eliminate harmful train-selector misses while n |
| 25 | 11 | +0.0219 | helped | recommend_diverse_attractions.py, tools_schema.json, workflow.py | add-shortlist-generator-tool | Reduce attraction-duplication/diversity failures by giving the model a city-validated, de- |
| 26 ⭐ | 23 | +0.0250 | helped | workflow.py | add-textual-plan-verifier, add-selfcheck-repair-loop | Raise success on itinerary-structure/route-consistency/time-feasibility failures with a ta |
| 27 | 23 | -0.0719 | hurt | workflow.py | add-deterministic-postprocessor, add-textual-plan-verifier | Raise itinerary-structure and route-consistency scores with deterministic header/final-day |
| 28 | 17 | -0.0615 | hurt | build_intercity_manifest.py, select_best_transport.py, tools_schema.json, workflow.py | add-deterministic-selector-tool, harden-existing-tool | Improve transport hard-constraint selection while reducing exact-name sandbox failures and |
| 29 | 21 | +0.0375 | helped | workflow.py | add-evidence-recorder, add-tool-backed-evidence-verifier | Raise scores by catching tool-data mismatches before finalizing: route-duration drift, unv |
| 30 | 23 | +0.0396 | helped | schedule_named_transfer.py, tools_schema.json, workflow.py | add-lookup-helper-tool | Reduce transfer-time and route-consistency failures by giving the model a deterministic ci |
| 31 | 17 | -0.0094 | neutral | workflow.py | add-textual-plan-verifier, add-selfcheck-repair-loop | Reduce duplicate attraction/restaurant picks and thin non-transfer days by adding a determ |
| 32 | 16 | +0.0083 | neutral | select_hotel_option.py, select_intercity_option.py, select_restaurant_option.py, tool_wrapper.py, tools_schema.json, workflow.py | add-deterministic-selector-tool, add-tool-result-cache | Reduce hard-constraint selection misses and timeout risk by adding grounded selector tools |
| 33 | 17 | -0.0073 | neutral | select_exact_listing.py, tools_schema.json, workflow.py | add-deterministic-selector-tool | Reduce exact-name sandbox/hotel-service misses while nudging better day density and unique |
| 34 | 16 | +0.0062 | neutral | workflow.py | add-constraint-extractor, add-tool-backed-evidence-verifier | Reduce hotel/restaurant hard-constraint misses while trimming redundant validator route ch |
| 35 | 32 | -0.0052 | neutral | select_intercity_option.py, select_restaurant_option.py, tools_schema.json, workflow.py | harden-existing-tool, add-textual-plan-verifier | Raise pass rate by eliminating selector crashes and catching duplicate-POI / budget-summar |
| 36 ⭐ | 26 | +0.0125 | neutral | workflow.py | tighten-system-prompt, add-constraint-extractor | Improve pass rate on hotel/train hard constraints and validation misses by making the fina |
| 37 ⭐ | 36 | -0.0031 | neutral | workflow.py | tighten-system-prompt, add-textual-plan-verifier | Raise pass rate on remaining route-consistency and validation misses with a small audit/ve |
| 38 | 26 | +0.0260 | helped | workflow.py | add-tool-backed-evidence-verifier | Raise pass rate on validated accommodation/transportation and route-consistency failures b |
| 39 | 35 | +0.0021 | neutral | select_intercity_option.py, select_restaurant_option.py, workflow.py | harden-existing-tool | Eliminate selector crashes and tighten transport/restaurant constraint adherence plus repa |
| 40 | 35 | -0.0010 | neutral | tool_wrapper.py, workflow.py | add-deterministic-postprocessor, harden-existing-tool | Raise pass rate by auto-correcting budget summaries, catching transfer days missing interc |
| 41 | 30 | -0.0250 | hurt | select_hotel_candidate.py, tools_schema.json, workflow.py | add-deterministic-selector-tool, add-textual-plan-verifier | Reduce repeated structure/hotel-selection failures by adding a deterministic hotel selecto |
| 42 | 34 | -0.0687 | hurt | workflow.py | add-constraint-extractor, add-tool-backed-evidence-verifier | Raise pass rate by catching wrong explicit train/flight choices plus invalid/duplicate mea |
| 43 | 40 | -0.0552 | hurt | build_intercity_manifest.py, select_hotel_option.py, workflow.py | add-tool-backed-evidence-verifier, harden-existing-tool | Raise pass rate by fixing transfer-header route mismatches, closed-attraction scheduling,  |
| 44 | 40 | -0.0677 | hurt | build_intercity_manifest.py, select_intercity_option.py, workflow.py | add-tool-backed-evidence-verifier, harden-existing-tool | Catch constrained outbound-flight selection mistakes before submission and harden manufact |
| 45 | 33 | -0.0938 | hurt | build_intercity_manifest.py, select_transport_option.py, tools_schema.json, workflow.py | add-deterministic-selector-tool | Reduce wrong transport selection, route-header mismatches, and budget-summary arithmetic e |
| 46 ⭐ | 37 | -0.0625 | hurt | workflow.py | add-tool-backed-evidence-verifier | Raise pass rate on transfer-feasibility and transport hard-constraint failures with eviden |
| 47 ⭐ | 46 | +0.1031 | helped | workflow.py | add-tool-backed-evidence-verifier | Raise pass rate on remaining restaurant/train hard-constraint misses plus late-arrival mea |
| 48 | 38 | +0.0042 | neutral | workflow.py | add-tool-backed-evidence-verifier, add-textual-plan-verifier | Raise pass rate on budget accuracy and remaining hard-constraint misses with tool-backed f |
| 49 | 47 | -0.0719 | hurt | workflow.py | add-textual-plan-verifier, add-constraint-extractor, reduce-iteration-budget | Raise pass rate on remaining restaurant-name, budget-summary, and duplicate-attraction fai |
| 50 | 47 | -0.0865 | hurt | workflow.py | add-tool-backed-evidence-verifier | Raise pass rate on remaining route-consistency and transfer-feasibility failures with evid |
| 51 | 48 | -0.0375 | hurt | workflow.py | add-tool-backed-evidence-verifier | Raise pass rate on time-feasibility, closure-day, and meal-structure failures with tool-ba |
| 52 | 51 | +0.0167 | neutral | workflow.py | add-tool-backed-evidence-verifier, add-textual-plan-verifier | Reduce remaining nearby-restaurant hard-constraint misses and weak sightseeing-day coverag |
| 53 | 39 | -0.0469 | hurt | select_hotel_option.py, select_restaurant_option.py, workflow.py | add-deterministic-postprocessor, harden-existing-tool | Raise pass rate by eliminating budget-summary mismatches and improving hard restaurant/hot |
| 54 | 38 | -0.0354 | hurt | workflow.py | add-textual-plan-verifier | Reduce seamless-intercity-transfer failures by catching missing/misaligned intercity seque |
| 55 | 52 | -0.0813 | hurt | workflow.py | add-textual-plan-verifier, add-constraint-extractor | Catch remaining route-header, named-POI, and transfer-day meal misses before finalizing th |
| 56 | 41 | +0.0271 | helped | workflow.py | add-evidence-recorder, add-tool-backed-evidence-verifier | Raise pass rate by catching transport-price, hotel-constraint, named-restaurant, and suspi |
| 57 | 29 | -0.0333 | hurt | workflow.py | add-deterministic-postprocessor | Raise scores by deterministically fixing the biggest recurring plan-format failures before |
| 58 | 36 | -0.0177 | neutral | workflow.py | add-textual-plan-verifier | Raise pass rate on remaining route/time-feasibility and accommodation-traceability failure |
| 59 | 38 | -0.0219 | hurt | workflow.py | add-deterministic-postprocessor | Eliminate arithmetic-only itinerary failures by deterministically recomputing and normaliz |
| 60 | 50 | +0.0823 | helped | workflow.py | add-tool-backed-evidence-verifier, reduce-iteration-budget | Raise pass rate on named-restaurant, closure-day, transfer-link, duplicate-attraction, and |
| 61 | 60 | -0.0260 | hurt | workflow.py | add-constraint-extractor, add-tool-backed-evidence-verifier | Raise pass rate on remaining train hard-constraint, hotel-constraint, and budget failures  |
| 62 | 60 | -0.0677 | hurt | workflow.py | add-tool-backed-evidence-verifier | Catch remaining meal-price/business-hours and decimal fare mismatches before finalizing pl |
| 63 | 41 | -0.0250 | hurt | select_train_candidate.py, tools_schema.json, workflow.py | add-deterministic-selector-tool | Reduce train-selection and exact-train-price failures by adding a deterministic train choo |
| 64 | 61 | -0.0573 | hurt | workflow.py | add-textual-plan-verifier, add-constraint-extractor | Raise pass rate on attraction-coverage/diversity and hotel-brand hard-constraint misses wi |
| 65 | 58 | -0.0260 | hurt | workflow.py | add-evidence-recorder, add-tool-backed-evidence-verifier | Catch evidence-backed transport/route/hotel mismatches before finalizing plans. |
| 66 | 37 | +0.0250 | helped | query_restaurants_near_attraction.py, tools_schema.json, workflow.py | add-lookup-helper-tool | Raise pass rate on restaurant hard-constraint cases by giving the planner a dedicated attr |
| 67 | 66 | -0.1063 | hurt | workflow.py | add-tool-backed-evidence-verifier, add-constraint-extractor | Cut remaining transfer-consistency/time-feasibility and hotel-constraint failures with a t |
| 68 | 56 | -0.0021 | neutral | workflow.py | add-evidence-recorder, add-tool-backed-evidence-verifier | Raise pass rate by catching restaurant-anchor, closure-day, and budget-summary mistakes du |
| 69 | 68 | -0.0292 | hurt | workflow.py | add-evidence-recorder, add-tool-backed-evidence-verifier | Reduce route/time-feasibility and itinerary-structure failures with evidence-backed travel |
| 70 | 59 | -0.0812 | hurt | workflow.py | add-tool-backed-evidence-verifier, add-constraint-extractor | Raise pass rate on hotel/restaurant hard constraints and attraction-hours/cost leaks with  |
| 71 | 49 | -0.1406 | hurt | workflow.py | add-tool-backed-evidence-verifier, add-constraint-extractor | Reduce remaining hotel/meal/restaurant/train hard-constraint misses with targeted evidence |
| 72 | 68 | -0.0063 | neutral | workflow.py | add-textual-plan-verifier, add-constraint-extractor | Reduce remaining failures on trip-length/route-consistency and restaurant hard-constraint  |
| 73 | 56 | -0.0385 | hurt | workflow.py | add-evidence-recorder, add-tool-backed-evidence-verifier, add-constraint-extractor | Reduce train-constraint and named-restaurant misses with a valid, targeted verifier-backed |
| 74 | 54 | -0.0062 | neutral | workflow.py | add-tool-backed-evidence-verifier, add-constraint-extractor | Raise pass rate on hotel hard-constraints, train-selection constraints, transfer-time real |
| 75 | 48 | -0.0104 | neutral | workflow.py | add-tool-backed-evidence-verifier, add-constraint-extractor | Raise pass rate on restaurant hard-constraint misses by adding tool-backed restaurant-cons |

---

Full code for any node is at `/home/ubuntu/sudipta/agentic_reasoning/meta-agent-dev-v2-improved-feedback/runs/20260615_095317_travel_hgm_8000/round_NNN/task_agent/`; diffs are recomputable against the parent's snapshot.