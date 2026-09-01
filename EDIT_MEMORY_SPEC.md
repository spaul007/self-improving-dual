# Spec — how the edit memory is generated

What one edit record contains, where each field comes from, and how the final
memory is assembled. (`PLAN.md` covers the migration roadmap; this file is the
generation procedure itself.)

---

## 0. The unit

The unit is the **atomic edit**, not the node. A parent→child generation is one
`submit_self_improvement` call, but it typically ships several unrelated changes
— empirically **3.8 per generation** (range 1–6) over 75 generations. Storing
one record per node conflates them: a node that adds a hotel selector *and*
rewrites the prompt gets one score and one description covering both.

A generation therefore produces **N edit records + 1 observation record per
evaluation batch**. The score is not stored on the edit (§6).

---

## 1. Decomposition — generation → atomic edits

Run against the parent and child `task_agent/` snapshots. Deterministic.

1. Diff the mutable surface: `workflow.py`, `tool_wrapper.py`,
   `tools_schema.json`, `mutable_tools/*.py`.
2. Collect top-level symbols added / modified / removed on each side
   (functions, classes, UPPER_CASE module constants).
3. Cluster:
   - each changed `mutable_tools/<X>.py` → **one edit** (clean file boundary);
   - added symbols in `workflow.py` → **connected components of the call
     graph** (a verifier plus its private helpers is one edit);
   - a modified `SYSTEM_PROMPT` → **always its own** `prompt_rule` edit;
   - modified symbols attach to the component they reference most; unattached
     ones form a `modified_function` edit.
4. **Exclude pre-existing hubs from the graph** — `run_task`,
   `_build_audit_prompt`, `_check_plan_issues`. They reference everything, and
   including them collapses an entire generation into one component. This is
   the single detail that makes the clustering work.
5. Cap at 4 code clusters per generation, merging the smallest tail; prompt and
   orphan clusters are appended beyond the cap.

Each cluster's **entry point** is its `run` function if present, else its
largest function.

---

## 2. What each edit record stores

Six blocks. Cost is measured, not estimated (`o200k_base`).

### A · Identity / lineage — *deterministic, free*

| field | why it is stored |
|---|---|
| `edit_id` `run:node:index` | stable, sortable, joins to observations |
| `fingerprint` | hash(kind, normalized entry name, consumes) — detects **the same edit re-invented on another branch**; the selector tool was independently reinvented 8 times here |
| `run_id`, `node_id`, `parent_id`, `depth` | tree position |
| `cosubmitted_with` | the other edits in this generation — required to interpret a shared score |
| `created_at_snapshot`, `created_at_budget` | true creation order; makes category assignment replayable online |

### B · Intent — *LLM, one call per generation*

| field | why |
|---|---|
| `goal` | the editor's `optimization_goal`, **verbatim** — never paraphrase the primary source |
| `what` | one sentence: what this edit mechanically does |
| `why` | **the failure THIS edit targets** |
| `targets_checks` | benchmark checks it claims to move |

`why` must be edit-specific. Copying the node rationale onto every edit in a
generation is the failure mode to avoid — it made all 72 multi-edit generations
read identically and destroyed the field's value.

### C · Implementation — *deterministic + LLM*

| field | source | why |
|---|---|---|
| `kind` | det. | `new_tool` / `new_function` / `modified_function` / `modified_tool` / `prompt_rule` / `config` |
| `entry_point` | det. | the joinable anchor |
| `interface` | det. | signature with defaults and return type — **the contract; the body is re-derivable from it, the reverse is not** |
| `returns` | det. | payload keys the component promises |
| `consumes` | det. | tools it calls; `<dynamic>` when the name is computed |
| `integration` | det. | `invoked_by` **code vs model**, `callers`, `phase` |
| `mechanism` | LLM | 2–4 lines: the algorithm, not the interface |
| `seed_hunk` | det. | ≤14 lines of the **decision core** — the branch that fires, not entry boilerplate |

`integration.invoked_by` is the field most worth adding and easiest to miss. A
tool merely *named in the prompt* and one *called deterministically* behave
completely differently, and only the second is guaranteed to run. Recovering it
requires parsing, not text search: `workflow.py` mentions tool names inside its
`SYSTEM_PROMPT` string, so grep reports "used" for components never invoked.

### D · Signature — *deterministic*

`files`, `symbols_added/modified`, `tools_added`, `lines`, and `weight` — this
edit's share of the generation's inserted code. `weight` is the prior on credit
when one score covers several edits.

### E · Source link — *deterministic*

`path`, `parent_path`, `diff_ref`. **Full code is never stored**; a single
lineage diff is 116 KB and a generation's diff up to 31 KB. The record carries
the pointer and the reader fetches on demand.

### F · Category — *LLM, against a live registry (§5)*

`technique_id`, `domain_id`, `assign_confidence`, `proposed_new`.

### Not stored

- **the score** — see §6;
- **full source or full diff** — pointer only;
- plumbing symbols (`_to_float`, `_load_json`, regex constants). Of 29 symbols
  in one representative tool, ~22 were boilerplate every tool reinvents.

**Cost:** ~289 tokens/edit with the snippet, ~168 without.

---

## 3. Field provenance

Deterministic wherever possible, because those fields cannot hallucinate and
cost nothing: identity, signature, interface, returns, consumes, integration,
seed hunk, source link, weight.

LLM only where judgment is required: `what`, `why`, `mechanism`, and the
technique/domain assignment. Structure is free and exact; meaning is not.

---

## 4. The per-generation LLM call

**One call per generation**, fired after the edit passes validation and before
the child is evaluated.

*Inputs* — the diff; the editor's `optimization_goal` / `proposed_changes` /
`rationale`; the deterministic cluster list with each cluster's entry point,
interface, consumes and seed hunk; the current category registry (id, name,
definition, one exemplar each).

*Output* — per cluster: `what`, `why`, `mechanism`, `targets_checks`, plus
either an existing `technique_id`/`domain_id` with a confidence, or a proposal
for a new one.

*Contract* — the call is given the cluster boundaries and must not re-cut them;
it must ground every statement in the supplied diff and hunks; it must not
propose a new category when an existing definition fits.

Doing this at generation time is what makes `why` genuinely per-edit: the
editor's reasoning is available first-hand, rather than being reconstructed from
its prose afterwards.

---

## 5. Category assignment

Two **crossed** axes, not one list: `technique` (how it was built) and `domain`
(what it was aimed at). Crossed rather than flat because a technique's effect
flips sign across domains — the same technique measured **+0.016** on
sandbox-compliance and **−0.063** on flight — so a single pooled category
shrinks both toward a value wrong for both. Crossed also keeps the parameter
count down: 16×13 is 208 flat buckets but 29 crossed.

Protocol, applied in true creation order:

1. Match an existing entry → return its id + confidence.
2. No fit → propose `{name, definition}`; accept **only if** the axis is under
   cap (20 techniques / 15 domains).
3. At cap → force the best match and record the low confidence.
4. When low-confidence assignments accumulate in one bucket, run a merge/split
   pass.

**Invariant:** no domain may be a subset of a single technique. If it is, the
two are collinear and unidentifiable in the effect model.

Empirically the vocabulary saturates fast — every technique appeared within the
first 100 edits; the remaining 188 introduced none. Caps guard against
cross-run drift, not within-run explosion.

---

## 6. Observations — where the score lives

The edit record holds **no score**. `β_m = μ + a_technique + b_domain + e_m` is
refit as evidence arrives and every refit changes β for every edit, including
ones written 60 nodes earlier; storing it inline would rewrite the whole store
each evaluation.

One record per **evaluation batch**, not per node (4 batches × 16 cases here):

```
obs_id · node_id · parent_id · batch_index · n_cases
batch_mean · parent_cum_mean_at_time · delta_vs_parent · per_check_delta
edit_ids   ← the multi-hot design-matrix row
```

`edit_ids` is stored **undivided**. Do not pre-split credit across
co-submitted edits: it is precisely edit A appearing with B in one generation
and with C in another that lets a joint fit separate them.

---

## 7. Assembling the final edit memory

Four artifacts, split by write frequency:

| file | mutability | written |
|---|---|---|
| `edits.jsonl` | append-only, immutable | per generation |
| `observations.jsonl` | append-only, immutable | per evaluation batch |
| `categories.json` | small, mutable | on registry change |
| `scores.json` | **derived, disposable** | after each refit |

`scores.json` must be reconstructible from the other three; nothing else may be.

**The rendered memory** is a *query*, not a dump. Measured budgets:

| view | tokens | per edit |
|---|---|---|
| full, with snippets | 83,116 | 289 |
| without snippets | 48,522 | 168 |
| index, 1 line/edit | ~11,000 | 38 |

Assemble in three parts:

1. **Ledger** — per technique/domain cell: attempts, β with uncertainty,
   verdict tally, best instance. The action–value table; the highest-signal
   section.
2. **Index** — one line per edit: id, lineage, technique/domain, weight,
   invoked_by, entry point, one-line `what`.
3. **Detail on demand** — full record including the seed hunk, fetched for
   specific edits. Snippets are 42% of the file, so they are the right thing to
   fetch rather than carry.

Render markdown, not JSON: same records, **57% fewer tokens**, because `run_id`
and the path convention are stated once, `cosubmitted_with` becomes implicit
under a generation heading, and keys/braces/quotes/escapes disappear.

---

## 8. Invariants

1. Edits and observations are append-only; a written record is never edited.
2. `scores.json` is a pure function of the other three files.
3. Every field is either deterministically extracted or explicitly marked as
   model-written; a field that could not be grounded says so
   (`why_source: node-level-fallback`) rather than passing off a weaker source.
4. No domain is a subset of a single technique.
5. Category assignment is replayable from `created_at_snapshot` — no field may
   depend on evidence that did not exist when the edit was written.
