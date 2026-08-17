# Tier-Based HGM: prioritizing fixes by leverage, not just utility

Status: design sketch, not implemented. Brainstorm only.

## Motivation

Today HGM's EXPAND/edit/evaluate/summarize loop treats every failing case as an
undifferentiated source of edit proposals, gated only by `mean_utility > 0`
(`hgm.py::_expandable()`). This mixes together failures with very different
leverage:

- Harness/plumbing bugs (nothing to do with the model) — e.g. the `plan` field
  head-truncation and the `mas_prompt_cfg.yaml` wrong-file edit found during
  the db_mas/math_mas integrations.
- General capability failures (instruction-following, malformed tool calls,
  missing `<answer>` tags) — model-general, high leverage, improve every case
  and every project at once.
- Task-specific strategy failures — domain reasoning, tool selection,
  decomposition. What the editor mostly targets today.
- Micro-tuning — wording/threshold nudges, diminishing returns, and the tier
  most vulnerable to editor-rationale hallucination (see math_mas: the editor
  twice asserted a specific false root cause for the same case, even after a
  "double-check yourself" prompt rule).

The core idea: classify failures into tiers by leverage, and don't let the
search spend EXPAND budget on a lower tier while a higher tier is still
unresolved above some severity threshold.

## Bottleneck taxonomy

| Tier | Description | Example | Fix path |
|---|---|---|---|
| 0 | Harness/plumbing bug | truncated context field, SIGKILL container leak on timeout, silent wrong-file edit | deterministic/auto-fix, never routed through the LLM editor as a "case failure" |
| 1 | General capability | malformed tool-call args, missing required output tags, instruction-following breakdown | editor edit, but to a shared/general prompt surface, not a task-specific one |
| 2 | Task-specific strategy | domain reasoning gaps, tool-selection heuristics, decomposition | editor edit to task-specific prompt/workflow, today's default path |
| 3 | Micro-tuning | wording tweaks, threshold nudges | lowest priority, most hallucination-prone, needs strongest evidence grounding |

## Detecting tier without trusting LLM self-report

The editor has already demonstrated confident, false root-cause claims (math_mas,
same case, two different false explanations across two runs). Tier
classification for 0/1 should therefore be grounded in deterministic signals,
not LLM narrative:

- Tier 0: parse errors, exceptions, truncation markers, timeout/SIGKILL events
  — detectable programmatically from logs/traces.
- Tier 1: schema validation failures on tool calls, regex checks for required
  output tags, format-conformance checks — deterministic where possible.
- Cross-project recurrence as corroborating evidence: if `failure_summarizer`
  (or a new `failure_categorizer`) sees the same failure fingerprint (e.g.
  "tool call missing required arg") across ≥2 unrelated projects (db_mas *and*
  math_mas), that's structural evidence of a Tier 0/1 harness-level issue, not
  a one-off quirk of a single task.
- Tier 2/3 are where LLM-based pattern-finding (the existing
  `failure_summarizer` "main patterns + hardest cases" digest) is actually
  appropriate, since determinism isn't feasible there.

## Severity scoring

`severity = fraction_of_cases_affected × utility_recoverable_if_fixed`,
weighted down by confidence in the diagnosis (avoid acting on an editor
rationale that hasn't been checked against the real per-case trajectory —
see the two verified-false editor claims on math_mas as the cautionary case).

## Gating algorithm (per round)

1. Categorize this round's failures into tiers 0-3 using the detectors above.
2. Compute severity per tier-bucket.
3. If any Tier 0 issue is open: auto-fix / flag, don't spend an editor call.
4. If Tier 1 severity is above threshold: bias/force the editor toward a
   general-surface fix (shared prompt/tool-calling wrapper) before allowing
   any Tier 2/3 proposal.
5. Only once Tier 0/1 severity drops below threshold does the search proceed
   to today's default behavior (Tier 2/3 task-specific edits).
6. Re-evaluate, recompute tiers/severity, iterate.

Concretely, this extends `_expandable()`'s `mean_utility > 0` gate with a
"bottleneck-debt" gate, so the search doesn't spend EXPAND budget optimizing
task-specific accuracy while (say) 30% of cases are failing on output format
alone — which would also produce misleading utility signals for every node
under that one.

## Main risk

The triage layer is itself new machinery that can hallucinate if implemented
as another LLM classifier end-to-end. The win only holds if Tier 0/1
detection stays mostly programmatic/deterministic, and the LLM is reserved
for Tier 2/3 pattern-finding, where determinism genuinely isn't feasible.

## Adaptive alpha: tuning explore/exploit by observed task difficulty

`alpha` (`hgm_tree.py::schedule_favors_expand`, `budget_spent**alpha >=
n_real_nodes - 1`) is the EXPAND-vs-EVALUATE schedule: higher alpha favors
widening the tree (EXPAND, structural exploration via new edits), lower
alpha favors spending budget re-evaluating existing nodes for confidence
(EVALUATE, exploitation). Today it's a single static constant per config,
solved up front from a *target* growth curve (`solve_alpha_from_xy`:
`alpha = ln(target_agents)/ln(eval_budget)`) — it encodes a budget-shape
prior, not anything observed about how the task is actually going.

**Idea: recompute alpha periodically from observed difficulty signals**
instead of holding it fixed for the whole run.

Candidate difficulty signals, all already available from existing tallies:

- Seed/root mean_utility (the free pre-loop evaluation) — a cheap prior on
  task difficulty before any edits happen.
- Rolling edit-acceptance / improvement rate: fraction of recent EXPAND
  events whose child actually beat its parent's mean_utility. A low or
  falling rate means structural search is hitting diminishing returns —
  favor EVALUATE (lower alpha) to firm up confidence on what's already been
  found rather than keep speculatively widening.
- Score variance per node relative to eval_budget spent on it: a noisy task
  needs more evaluations per node before a score can be trusted (the
  edit-rationale-hallucination precedent is a reminder that trusting an
  under-evaluated, lucky node is exactly the failure mode to avoid) — high
  variance should push alpha down (more EVALUATE) at the same total budget.

**Connection to the tier taxonomy above:** tier composition is itself a
difficulty signal. A round dominated by Tier 0/1 failures (harness bugs,
general capability) means task-specific EXPAND is premature regardless of
what the raw utility numbers say — no amount of structural search fixes a
malformed tool-call schema. So the gating rule from the tier section
(don't EXPAND task-specific edits while Tier 0/1 severity is above
threshold) and adaptive alpha are two instances of the same principle:
route search effort toward whatever actually has leverage right now, rather
than following a schedule fixed before the run started.

**Sketch:**

```
alpha_t = clip(alpha_base + f(recent_improvement_rate, seed_utility, tier_debt),
               alpha_min, alpha_max)
```

recomputed every round (or every K budget spent), with an EMA/smoothing
term so alpha doesn't oscillate round-to-round on noisy signals. `alpha_base`
stays the `solve_alpha_from_xy`-derived value — it's still a reasonable
prior for overall budget shape, adaptation should perturb it, not replace it
outright.

**Risks:**

- `schedule_favors_expand`'s guarantees (e.g. "a fresh tree evaluates before
  it widens further" at `budget_spent == 0`) were derived assuming alpha is
  fixed; need to re-check the degenerate cases (very small/large alpha,
  early-round noise) still behave sensibly under a moving alpha.
- A badly-tuned `f(...)` could make alpha collapse toward `alpha_min` early
  (e.g. from a rough seed evaluation with too little data) and get the
  search stuck over-evaluating a mediocre node instead of ever widening —
  needs a minimum-samples floor before difficulty signals are trusted at
  all, same spirit as the math_mas `train_size` fix (2 cases wasn't enough
  for `_expandable()`'s `mean_utility > 0` gate to ever trigger reliably).
- This adds a second adaptive control loop on top of the tier-gating one;
  worth prototyping independently before combining, so a bad interaction
  between the two isn't mistaken for a bug in either.

## Open questions / not yet decided

- Where does tier classification live: inside `failure_summarizer.py`, a new
  `meta_agent/failure_categorizer.py`, or a preprocessing step before
  `_expandable()`?
- What's the actual severity threshold, and is it a fixed constant or
  adaptive per project?
- Does a Tier 1 fix target a genuinely shared prompt surface across projects,
  or does "general" here still mean "general within one project's editor
  surface" (no cross-project shared prompt currently exists in the
  framework)?
- Auto-fix for Tier 0: safe to apply without human/editor review, or does it
  still need a lightweight approval step given the container-leak precedent?
