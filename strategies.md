# Strategies

Curated, editable guidance for `meta_agent/block_suggester.py`'s
improvement-proposal LLM call. This file is read fresh on every `suggest()`
call (not cached at process start), so edits here take effect on the very
next EXPAND that samples a block-suggester-backed strategy — no restart
needed.

This is reference material, not instructions the suggester is forced to
follow: it's shown as one more paragraph of context. Keep entries short,
general hints — not project-specific facts, not references to any
particular run's results.

Format: a `## General` section (always shown, every block), plus one
`## Block: <name>` section per canonical block (shown only when the
suggester is assigned that block; `mixed` sees every block's section at
once, not just its own). Block names must match
`block_suggester.py::_BLOCK_BODIES`'s keys exactly: `individual_subagent`,
`collaboration_workflow`, `foundation_capability`, `verifiers`, `mixed`.

## General

- Decompose a task into sub-tasks sized to what the backbone LLM can
  reliably do in one step, then stitch the sub-tasks together — in code
  where a step is deterministic or checkable, in another LLM call where it
  genuinely needs judgment. The lower the backbone's capability, the
  smaller and more deterministic each sub-task should be; a more capable
  backbone can support one larger call plus a lighter verifier instead of
  full decomposition. Most of the specific patterns below are instances of
  this one idea applied to a particular symptom.
- If an LLM's output for some step is too stochastic or unreliable, and the
  task could instead be done deterministically (a lookup, a computation, a
  tool call), consider writing code for it rather than relying on the
  model to get it right.
- To make an inherently stochastic step more reliable, consider retries,
  backoff, or sampling multiple times and picking the most consistent
  answer.
- When the model already knows a rule but doesn't reliably follow it,
  retrying with explicit feedback on exactly which constraint(s) failed
  typically helps more than only re-wording the original instruction.
- Some violations in an already-generated output can be corrected directly
  in code after the fact (no LLM call needed) rather than asking the model
  to redo it — e.g. rewriting a mismatched value once the correct one is
  known deterministically.
- Prefer the smallest change that addresses the diagnosed problem.
- Ground the diagnosis in something actually observed, not a plausible
  guess.
- Check for an information ceiling before proposing a fix: is the
  information needed to actually solve this available or derivable (from a
  tool result, the feedback shown to you, or the code itself)? If the fix
  would require information nothing currently available can provide,
  that's a sign the real fix lives elsewhere (e.g. a missing tool, or a
  different stage that has the information you're missing) — say so rather
  than proposing a change that can't actually be verified or grounded.
- All else equal, prefer a fix you can ground in information you actually
  have over one that depends on information you'd have to guess at.

### Reading the error-bucket prevalence table

When the feedback digest includes an "Error-bucket prevalence" section
(built by `meta_agent/error_bucket_analyzer.py`), it is your primary
evidence for BOTH what is failing and how capable this backbone actually
is — use it before proposing a fix, not just to pick which failing case to
cite.

**Step 1 — infer a capability tier from the table's overall shape** (not
any single bucket in isolation; look at which buckets dominate relative to
each other):

- **low**: `tool_omission` and/or `tool_calling_budget_exceeded` together
  account for a large share of failing cases (roughly half or more). The
  backbone is struggling with basic mechanics — whether to call a tool at
  all, whether to finish at all — before it even gets to using
  information correctly. Prompt-only fixes are unreliable at this tier
  even when explicit and repeated (confirmed directly: a much-strengthened
  "you MUST call this tool" instruction still left the omissions in
  place).
- **medium**: `tool_omission`/`tool_calling_budget_exceeded` are a small
  minority, but `constraint_misreading` and `apply_info_incorrectly`
  dominate instead. The backbone reliably executes the mechanical parts
  (calling tools, finishing) but drops or misapplies specifics under
  load — a real but more targeted gap than "low".
- **high**: the failing-case rate itself is already low, and what
  remains is mostly `apply_info_incorrectly`/`long_horizon_state_loss` on
  cases that plausibly needed real judgment — not the same failure
  showing up systematically on easy cases too.

If the evidence doesn't clearly fit one tier, say so and hedge rather than
forcing a label — this inference is meant to calibrate HOW aggressively
you intervene, not to be asserted as settled fact.

**Step 2 — look up the dominant bucket(s) at that tier** in the table
below, and let it decide the KIND of fix, not just its target:

| Error bucket | low capability | medium capability | high capability |
|---|---|---|---|
| **tool_omission** | Move the lookup into code entirely — harness calls the tool directly, pastes the result into the prompt. Removes the decision to skip it; don't just instruct harder. | A concrete instruction ("cite the tool call that produced this exact fact") plus a pre-output self-check can work — but verify on a sample. If omissions persist despite a specific instruction, that itself is evidence the tier is actually lower than assumed; fall back to the code-level fix. | A single clear non-fabrication instruction is usually enough. Add a cheap post-hoc verifier (does this fact trace to a real tool result?) as a safety net, not the primary fix. |
| **wrong_tool_or_arg** | Narrow the tool schema available at each step to only what's relevant right now (least-privilege per step) — fewer options, fewer ways to choose wrong. Don't rely on instructing careful selection. | Narrowing still helps, plus an explicit "use the exact value from tool X's result, never substitute" instruction and an argument-provenance check (does this argument trace to a prior real result?). | Argument-provenance verification alone (catch, don't prevent) is usually proportionate. |
| **constraint_misreading** | A mandatory extraction sub-step as its OWN call, before anything else, outputting a structured itemized requirement list — a separate, code-enforced checkpoint, not an instruction folded into a bigger task. | Fold the extraction into the same call as an explicit "restate every requirement first" instruction; verify on a sample that it restates ALL of them, not just the salient one or two. | A final verifier diffing output against the original request's explicit requirements is enough; the read step itself is usually reliable. |
| **wrong_tool_calling_order** | Enforce the sequence in code — a staged state machine or separate ordered sub-calls; a single open-ended tool loop won't reliably self-sequence. | An explicit ordering instruction ("do X before Y, never the reverse") plus a code-level check that flags an out-of-order call sequence. | Rare at this tier; a cheap post-hoc order-consistency check is enough. |
| **long_horizon_state_loss** | Decompose into one call per repeated unit, re-injecting everything decided so far as explicit input to each call — never rely on recall from earlier in the same generation. | A single generation across the whole horizon may hold up; add an explicit running-state reminder in the prompt plus a post-hoc consistency verifier that triggers a targeted patch (not a full regen) on drift. | Usually consistent; a lightweight verifier catching the rare drift is more proportionate than decomposing. |
| **apply_info_incorrectly** | This is a composition-load problem, not an information problem — shrink the scope of what one call must reconcile at once (fewer facts/units per call). Giving an already-overloaded call MORE resolved facts does not fix it. | Same as low: shrink scope per call before adding more input. | Never fully disappears even here — keep a narrow, targeted verifier on exactly this signature (does output stay consistent with facts it was already given?); cheaper than trying to prevent every instance through prompting. |
| **tool_calling_budget_exceeded** | Prefer many small, bounded calls over one large open-ended one — a capability-limited backbone is more likely to finish several small tasks than self-terminate one long one within budget. | A generous budget plus a forced wrap-up fallback call (no further tool access, explicitly asked to close out) is usually enough. | Rare; the same forced wrap-up fallback as a safety net is proportionate. |

A case can carry more than one bucket label — when it does, treat the
combination as the actual diagnosis (e.g. `tool_omission` +
`long_horizon_state_loss` together suggests a low-capability backbone
that also needs per-unit decomposition, not two unrelated problems to fix
independently).

## Block: verifiers

- A check is only useful if something acts on the result. Decide what
  happens on failure — block, patch, or retry — not just log it.
- "Block the result entirely" is the weakest of those three options, not
  a neutral default — blocking an output that would otherwise have
  gotten partial credit turns that partial credit into zero, which can
  make the outcome WORSE than not having the check at all. Only choose
  block-with-no-retry when you can say why patching the specific problem
  or giving the stage one retry to fix it itself isn't feasible (e.g. the
  failure is a genuine hard-safety/correctness gate, not just "this one
  field is wrong"). Default to patch or retry; justify block explicitly
  when you pick it.
- Check against what will actually be evaluated, not a generic sanity
  check.
- Consider checking an intermediate output as well as the final one, so a
  problem can be caught and fixed earlier and more cheaply.
- For a check that only ever fails in one narrow, well-understood way (a
  single inconsistent value against otherwise-correct output), consider
  patching just that piece deterministically rather than regenerating the
  whole output from scratch — a full retry throws away everything that
  was already right along with the one thing that wasn't.

## Block: individual_subagent

- If a role's output is unreliable due to stochasticity, consider a retry
  or a self-consistency check (ask more than once, compare answers).
- If a role is doing something that could be computed or looked up
  deterministically, consider moving that into a tool/helper instead of
  relying on the model to reason its way there.
- Rule out a control-flow bug (a broken retry, lost context, a shadowed
  variable) before assuming the problem is the prompt or the model's
  reasoning.
- If a role occasionally selects the wrong tool, or invents a
  plausible-looking argument instead of using a value from a real prior
  result, consider narrowing which tools/options that role can even reach
  at a given step (scope it to only what's relevant right now) rather
  than trusting broader instructions to produce the right choice among
  many alternatives.
- If a role's output shows it misread or dropped a stated requirement,
  consider adding an explicit sub-step that extracts the raw input into a
  structured, itemized list of requirements before anything else happens —
  this makes a misreading visible and checkable on its own, instead of
  being silently baked into later output that's harder to trace back to
  the misreading.
- If a role is already given every fact it needs (from a tool, from
  context) but still combines or applies them incorrectly, the problem is
  usually how much it's being asked to reconcile in one call, not what
  information it has. Shrinking the scope of what a single call must
  handle at once (fewer items/facts per call) is more likely to help than
  giving it still more resolved information.

## Block: collaboration_workflow

- Consider whether a step is missing that reviews the combined result and
  can send work back to an earlier stage when something is wrong.
- Prefer fixing information at its source (the sending stage) over having
  a downstream stage compensate for it.
- Be explicit about which stage sends and which stage receives whatever
  you're changing.
- If a role's output degrades or grows inconsistent the further into a
  long, repeated structure it gets (more items, more turns, more units),
  consider decomposing that structure into one call per repeated unit,
  with whatever's already been decided fed back explicitly as input to
  the next call — don't rely on the model to recall it unprompted from
  earlier in its own context.

## Block: foundation_capability

- To make a stochastic call (an LLM or tool call) more reliable, consider
  retries, backoff, and timeouts.
- A retry loop needs both a per-attempt limit and an overall limit, not
  just one — otherwise many bounded attempts can still add up to an
  unbounded wait.
- Consider whether an existing budget/capacity knob is simply set too
  tight before adding new logic — retry counts, tool-calling iteration
  caps, and max output tokens are all easy to under-provision, and a
  capacity limit hit mid-task can silently discard otherwise-good
  progress rather than failing loudly.
- If a role's own tool-calling via the LLM is unreliable (skipped calls,
  wrong arguments, inconsistent invocation), consider having the code
  call the tool directly and deterministically instead, then pass its
  result into the prompt as context — this removes the LLM from a step
  it doesn't actually need to perform itself.
- Before concluding that unreliable tool-calling needs that code-level
  fix, check whether a stronger instruction resolves it first — a
  concrete statement of the exact rule plus a pre-output self-check can be
  enough on a more capable backbone, and is cheaper than moving the step
  into code. Verify it actually worked on a sample rather than assuming
  the stronger wording landed; a backbone that keeps skipping the same
  call despite an explicit, specific instruction not to is the signal
  that it's time for the code-level fix instead.
- Match how much a single call is asked to do to the backbone's
  demonstrated reliable range: a capability-limited backbone is more
  likely to reliably finish several small, bounded calls than one large,
  open-ended one within any given budget, even if the total amount of
  work is the same either way.
- Be cautious with shared/foundational changes — they affect every role at
  once.
