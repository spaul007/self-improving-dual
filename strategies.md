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
suggester is assigned that block). Block names must match
`block_suggester.py::_BLOCK_BODIES`'s keys exactly: `individual_subagent`,
`collaboration_workflow`, `foundation_capability`, `verifiers`.

## General

- If an LLM's output for some step is too stochastic or unreliable, and the
  task could instead be done deterministically (a lookup, a computation, a
  tool call), consider writing code for it rather than relying on the
  model to get it right.
- To make an inherently stochastic step more reliable, consider retries,
  backoff, or sampling multiple times and picking the most consistent
  answer.
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

## Block: verifiers

- A check is only useful if something acts on the result. Decide what
  happens on failure — block, patch, or retry — not just log it.
- Check against what will actually be evaluated, not a generic sanity
  check.
- Consider checking an intermediate output as well as the final one, so a
  problem can be caught and fixed earlier and more cheaply.

## Block: individual_subagent

- If a role's output is unreliable due to stochasticity, consider a retry
  or a self-consistency check (ask more than once, compare answers).
- If a role is doing something that could be computed or looked up
  deterministically, consider moving that into a tool/helper instead of
  relying on the model to reason its way there.
- Rule out a control-flow bug (a broken retry, lost context, a shadowed
  variable) before assuming the problem is the prompt or the model's
  reasoning.

## Block: collaboration_workflow

- Consider whether a step is missing that reviews the combined result and
  can send work back to an earlier stage when something is wrong.
- Prefer fixing information at its source (the sending stage) over having
  a downstream stage compensate for it.
- Be explicit about which stage sends and which stage receives whatever
  you're changing.

## Block: foundation_capability

- To make a stochastic call (an LLM or tool call) more reliable, consider
  retries, backoff, and timeouts.
- A retry loop needs both a per-attempt limit and an overall limit, not
  just one — otherwise many bounded attempts can still add up to an
  unbounded wait.
- Be cautious with shared/foundational changes — they affect every role at
  once.
