# A*-HGM: goal-directed search (weighted A*) on top of HGM's bandit

Status: design sketch, not implemented. Brainstorm only. Companion to `tier_based_hgm.md` (which decides *which kind* of fix to spend budget on);
this note is about *which node to spend it on*.

Design decision: the search is **weighted A*** (`f = g + w * h`, `w > 1`), not plain A*. That removes the requirement that `h` be admissible: we do not
have to prove `h` never overestimates the cost-to-go, only make it a *useful* estimate. We give up the optimality guarantee that admissibility buys and
accept a bounded, empirically tuned trade of solution quality for far fewer node expansions, which is the right trade for a budgeted search like HGM.

## Motivation: a node's utility is not its progress

HGM ranks nodes by evaluation utility (own score for EVALUATE, Clade Metaproductivity for EXPAND). Utility measures where a node *is*, not how far
it is from a working solution. Two failure modes we saw directly in the travel_mas_refactored experiments:

| Observation | Utility says | Reality |
|---|---|---|
| Qwen-122B redesign run 1 (code-owned planner, no hard-constraint machinery): composite 0.363 vs baseline 0.501 | prune it | 5 of 9 dimensions already at 0.80-1.00 (baseline 0.09-0.68); the missing piece was one unit (hard constraints). Completed by a stronger agent, the same design reached 0.670 (+0.169, CI [+0.099, +0.236]). |
| Prompt-only variants of individual groups (e.g. diversity 0.40 -> 0.63, sandbox 0.58 -> 0.87 on group score) | keep, they "improve" | Combined agent: composite +0.002 vs the +0.17 the individual gains promised; the prompt edits interfered and pushed the no-plan rate 0.107 -> 0.156. |

The reward is also veto-shaped: each commonsense dimension counts only if *every* check in it passes, and the hard score only if *every* hard constraint
passes. Composite is sparse and non-additive, so a strategy that has solved five units but not the sixth looks worse than a strategy that has solved none.
HGM's Beta-sampling bandit has no notion of "unfinished but on the right path".

## Reframing: search over capability states, with a goal

The problem is a constraint-satisfaction problem over the grader's checks:

- **Abstract state** of a node: the vector `s(node)` of which *units* it satisfies. A unit is a dimension or a hard-constraint family (the same
  units a dimension-based curriculum in `meta_agent/curriculum.py` would use as goals): `s_u = 1` if the unit's pass rate is above a threshold with confidence,
  else 0 (or the continuous pass rate).
- **Goal state**: every unit satisfied, no-plan rate ~0, no crashes. This is a *specification*, not a gold plan; it needs no ground-truth answers,
  only the scorer's own per-check labels.
- **Cost** `g(node)`: budget spent to reach the node (edits + evaluations along its lineage), i.e. what HGM already accounts as `budget_spent`.
- **Needle in a haystack**: the set of agents satisfying all units is tiny relative to the space of agents; but the constraints are *decomposable*
  (units), which is exactly where A*-style search with a good heuristic is far more efficient than best-first on raw utility. Weighted A* commits harder
  to the heuristic's guidance, which is what a needle-in-a-haystack problem rewards.

## The heuristic, and why weighted A* removes the admissibility requirement

Plain A* is optimal only if `h` never overestimates the remaining cost (admissible) and is preferably consistent. Building such an `h` here is hard: repair
costs are stochastic, learned from history, and units interact (one edit can fix several units; prompt edits can undo each other). Weighted A* sidesteps
this. With `f = g + w * h` and `w > 1`:

- `h` only has to be an **informative estimate** of the remaining cost. Overestimates are tolerated; they simply make the search greedier toward nodes
  that look close to the goal, which is the behavior we want for needle-in-a-haystack problems.
- Weighted A* expands substantially fewer nodes than plain A* on hard problems, and it does not need re-expansion machinery to stay useful.
- **If** `h` happens to be admissible, the solution cost is at most `w` times optimal. Without admissibility we lose that formal bound and treat `w` as a
  tuning knob, calibrated empirically (see validation). This is an explicit, accepted trade.

Constructing the heuristic (no admissibility proof needed):

1. **Unit-independence estimate.** `h(n) = sum over unmet units of c(unit)`, where `c(unit)` is the expected edit-and-evaluation cost to repair that unit
   type, estimated from HGM's own history and from staged runs (median or a lower quantile, not a worst case). Start with a prior of one accepted stage per unit.
2. **Correct for edits that fix many units.** Fixing the no-plan rate, or a structural change, lifts several units at once. Cap the estimate with a
   maximum-units-per-edit factor `k` learned from history: `h(n) = min(sum c(unit), ceil(#unmet / k) * c_edit)`. Coupled units (Time Feasibility, Business
   Hours, Duration, Itinerary Structure) are treated as one cluster so their costs are not double-counted.
3. **Optimistic unit states under noise.** Estimate a unit's satisfaction with an upper confidence bound on its pass rate, so noise on small samples cannot
   inflate the distance. In the weighted setting this is a mild bias toward exploring uncertain-but-promising nodes, not a correctness requirement.
4. **Choosing `w`.** Start moderate (for example `w` between 1.5 and 3) and anneal it downward as budget is spent, as in anytime weighted A* (ARA*): large
   `w` early to find a working strategy quickly, smaller `w` later to refine among good nodes. Tune on replayed HGM logs.
5. **Tie-breaking and re-opening.** Break ties toward smaller `h` (closer to the goal). Allow a node to be re-opened when a cheaper lineage reaches its
   state, since the heuristic is not guaranteed consistent.

## Duplicate detection: merging lineages that reach the same state

A* keeps a closed list keyed by state. HGM currently keeps every lineage. In A*-HGM, nodes with the same abstract state signature `s(node)` compete on
`g` (cheaper wins) and robustness; the rest are pruned or demoted. This is a large efficiency gain when many edits reach an equivalent capability
state by different routes, and it directly targets HGM's duplicated effort on near-identical children.

## The hybrid: A*-HGM

HGM handles noisy evaluation well (Beta sampling, pseudo-count-normalized clade scores); A* handles structure. Combine them:

```
node value for EXPAND  = Thompson sample of clade score            # HGM: handles noisy, stochastic utility
                          + lambda * (h_prev - h_node)              # A*: progress toward the goal (units newly satisfied)
select by f = g + w * h_opt                                          # weighted A*: informed frontier ordering; w > 1, no admissibility needed
EVALUATE   spends samples on the units that decide h (uncertain unit pass rates), not uniformly on the node
```

- `argmax_expand` (in `meta_agent/managers/hgm_tree.py`) currently maximizes a sampled clade score; A*-HGM samples the same posterior but orders the
  frontier by f, so a low-utility node with a small h (few units left) is expanded before a higher-utility node with a large h.
- `_expandable()`'s `mean_utility > 0` gate becomes `h_opt(node) < h_root` (the node has made progress on the goal), since a node can be unfinished
  and low-scoring yet strictly closer.
- Alpha (EXPAND vs EVALUATE) can adapt to the *uncertainty in h*: high uncertainty in unit states means spend on EVALUATE.

## Completion rollouts (lookahead to sharpen h)

The staged runner (`experiment_harness_staged.py`) is a working prototype of a rollout: from a checkpoint it repeatedly picks the unit with the most
failures, runs a short focused session, and accepts or rolls back on a gate. Used as a lookahead, a node's estimated value is what a K-stage rollout reaches
(K small), which sharpens `h` and tells us whether a low-scoring strategy is worth finishing. Deterministic, code-owned pipelines make rollout
evaluation cheap and precise (near-zero run-to-run variance), so this is affordable for the strategies that matter.

## Where it plugs into the code

| Piece | Change |
|---|---|
| `HGMNode` (`hgm_tree.py`) | store the unit-state vector, its uncertainty, and `g` (cumulative cost along the lineage) |
| `HGMTree.argmax_expand` / `cmp` | order by f; keep CMP as the noise-robust value term |
| `hgm.py::_expandable` | progress gate `h_opt < h_parent` instead of `mean_utility > 0` |
| `curriculum.py` | goals become units (dimensions / hard families) ranked by oracle composite gain; resolution = unit pass rate + no composite regression; re-rank at accepted checkpoints |
| `project_metrics` (scorer adapter) | emit per-unit pass rates, oracle gain and coupling hints (which units' fixes interfere) |
| Tier gating (`tier_based_hgm.md`) | Tier 0/1 debt (no-plan, tool omission) is a unit with large `k` (fixing it lifts many units) |
| Ratchet + validators | accept a child only if composite improves and no-plan does not worsen; reuse the edit-policy guard |

## How to validate it cheaply (before touching the search)

1. **Retrospective predictive test.** On existing HGM runs, compute each node's unit-level potential and check whether early potential predicts the best
   descendant's utility better than early utility does (rank correlation, precision@k of "which nodes lead to the best descendant"). No search change needed.
2. **Replay.** Re-run the selection rule on logged trees: how many evaluations until the eventual best node would have been expanded under f vs under CMP?
3. **Controlled comparison** on travel_mas_refactored: A*-HGM vs HGM with matched budgets, measuring best-composite-at-budget and evaluations-to-target
   (e.g. composite 0.65).
4. **Ablations:** the value and schedule of `w` (1 = plain A*, fixed 1.5 / 3, annealed); unit-independence estimate with and without the max-units-per-edit cap;
   with and without duplicate detection; with and without completion rollouts. Also measure how often `h` overestimated in hindsight, to see whether
   the weighting is compensating for a biased estimate or for noise.

## Risks and open questions

- **No optimality guarantee.** Weighted A* with a possibly inadmissible `h` trades solution quality for speed. There is no formal bound; the check is empirical
  (replay and controlled comparison). Too large a `w` can chase a misleading heuristic and commit to a bad lineage; anneal `w` and keep some exploration.
- **A biased `h` is a systematic risk,** not just noise: if repair costs for a unit type are underestimated (say, prompt edits), the search will over-favor
  nodes that rely on that unit type. Calibrate `c(unit)` per strategy family and update it as the archive grows.
- **Shaping can be gamed.** Optimizing easy units while ignoring the veto-bound ones. Weights come from composite impact, and final selection and reporting
  always use true composite on held-out data.
- **Interaction between units.** Prompt edits interfered in our data; coupled dimensions (Time Feasibility, Business Hours, Duration, Itinerary Structure)
  should be one cluster in the pattern database.
- **Unit definition is task-specific.** The adapter must expose structure (dimensions, hard families) or derive it from scorer outputs; the HGM core
  must stay project-agnostic.
- **Small samples.** Unit pass rates on a few cases are noisy; use confidence bounds and the same held-out discipline as elsewhere. In our experiments
  the in-loop train samples (6-12 cases) overstated results relative to held-out cases.
- **Cost model.** What "one edit" costs differs across strategies (a prompt edit vs a new module); normalize by evaluations and LLM turns.

## Evidence from this project (for reference)

- Group-level strategy experiment (prompt / harness / code, 6 constraint groups, 20 held-out cases each): no strategy won everywhere; gains at the group
  level did not sum (combined agent composite 0.503 vs 0.501 baseline).
- Redesign attempts on 60 held-out cases (baseline composite 0.501): Qwen-122B run 1 0.363, run 2 0.095 (submitted a pipeline its own evaluations showed
  was far worse than baseline), Claude 0.670. The staged Qwen continuation and the Claude hard-constraint stage were still running when this note was written.
