# The Confounded Memo

Why `behavior_memory.md` keeps filing sound strategies under "didn't help" —
and what's actually behind the crash rate it blames them for.

- **Run:** `hgm_travel_full_scale_block_tagged_X100Y180`
- **Project:** `travel_mas_refactored`
- **Scope:** 61 `behavior_memory.md` files inspected, rounds 1–72

## Summary

| | |
|---|---|
| Memos citing a high crash rate as the round's dominant outcome | **54 / 61** |
| Memos explicitly blaming network / LLM-endpoint instability, not the code change | **12 / 61** |
| Hand-verified "didn't help" nodes traced to a real implementation bug, not a bad idea | **2 / 2** |

The suspicion: a poor strategy and a poorly-implemented strategy produce the
identical symptom — a case that crashes or scores zero — and
`meta_agent/behavior_summarizer.py` has no way to tell them apart. It only
ever sees a diff, a score delta, and instrumentation counts, never a
traceback or a validator log. Two nodes were pulled from among the memos
that read as clean, confident regressions, and checked against their actual
code. Both turned out to be bugs, not bad ideas.

## Mechanism 1 — a one-line contract mismatch, filed as "the edit introduced instability"

**Node 19** (parent: node 12). Goal: *"Fix train terminology errors and
strengthen itinerary generation to reduce empty outputs."*

> The `terminology_violation` check in `mas_workflow.py` did not prevent
> crashes. The edit introduced instability, causing 68% of cases to fail
> before evaluation (Traceback/plan conversion errors).
> — `behavior_memory.md`, node 19

The diff, in `agents/sightseeing.py`:

```python
if not is_valid:
    return AgentMessage(
        sender="sightseeing", content="", ok=False,
        iterations=iters, budget_exhausted=exhausted,
        metadata={"terminology_violation": True, "terminology_error": error_msg}
    )
```

Against the frozen contract, unchanged this round, in
`agents/immutable/message.py`:

```python
@dataclass(frozen=True)
class AgentMessage:
    sender: str
    content: str
    ok: bool = True
    iterations: int = 0
    budget_exhausted: bool = False
    # no `metadata` field, ever
```

**Root cause:** `AgentMessage` is a frozen dataclass with a fixed field set.
Constructing it with `metadata=` raises `TypeError` on the spot — every
single time the new check fires. The terminology idea was never exercised;
it crashed on contact with its own return statement. The memo's "the edit
introduced instability" is accurate as far as it goes, but stops short of
the one fact that matters to the next editor: this is a fixable one-line
bug, not evidence against verifying terminology at all.

## Mechanism 2 — a working verifier whose all-or-nothing gate manufactures its own crash rate

**Node 40** (parent: node 11). Goal: *"Add itinerary factual accuracy
verifier to catch transfer time, price, and business hour errors."*

> `verify_itinerary_facts` blocked 20 cases where it fired… The new
> verifier prevents progression on broken itineraries, but the high crash
> rate (31/32) dominates the failure.
> — `behavior_memory.md`, node 40

The diff, in `mas_workflow.py`:

```python
facts_ok, fact_failures = verify_itinerary_facts(sightseeing_msg.content, task.description)
if not facts_ok:
    metadata["itinerary_facts_failed"] = True
    metadata["fact_failures"] = fact_failures
    return AgentOutput(result="", metadata=metadata)   # zero score — same bucket as a crash
```

And `verify_itinerary_facts()` itself, self-documented as incomplete:

```python
# For now, just flag if accommodation is not '-' on any day
# More sophisticated logic would track which is the last day
...
# Skip this check for now, too complex to determine last day
```

**Root cause:** the checker is a fragile regex parser that admits its own
incompleteness in-line — a quality problem, not a design flaw by itself.
The actual damage is the gate around it: on *any* flagged issue, the whole
itinerary is discarded for a hard zero, far harsher than the real scorer,
which gives partial credit per failed constraint. 20 of the round's 31
"crashes" are this gate choosing to zero out an itinerary that may
otherwise have scored fine. The memo's "the high crash rate dominates"
reads as if something external swamped the round; the verifier caused most
of it, on purpose.

## Mechanism 3 — a third confound, at larger scale than either bug

Across the 61 memos, the crash rate itself is frequently the story — and
the summarizer is often the one telling you it can't trust it.

- Node 10: *"`httpx` instability: 100% crash rate… masks all agent logic."*
- Node 3: *"the httpx Traceback (transport layer) suggests network/LLM
  service instability, not agent code issues."*

12 of 61 memos name this explicitly. The summarizer is often right about
this and says so — but the verdict still lands in the same "What didn't
help" section as a genuine implementation failure, indistinguishable to a
reader skimming the memo for what to avoid next.

## Why the summarizer can't fix this on its own

`behavior_summarizer.py` is handed a diff, a score delta, and
instrumentation counts — never a traceback, a validator log, or the
distinction between "this code raised an exception" and "this code ran and
produced a bad result." A crashing metadata bug, an over-eager gate, and a
flaky endpoint all present the same way at its input: a case that didn't
score. It is, in several memos, visibly careful about this — correctly
refusing to call check-level noise "progress," or blaming agent logic
outright for an httpx traceback. But carefulness about *uncertainty* isn't
the same capability as attribution: nothing currently tells it "this
specific exception traces to a nonexistent dataclass field," so a
perfectly reasonable idea — verify terminology, verify facts before
returning — reads in the memo as tried, and found wanting.

The two confirmed cases here were reachable with an ordinary diff and a
read of the frozen contract in `agents/immutable/message.py`. That's
within reach of a cheap, mechanical check run before the memo is written —
not a rewrite of the summarizer's judgment, just giving it (or a gate ahead
of it) the one fact it's currently missing: did this fail because of what
it decided, or because of how it was built.

---
Sampled from 73 rounds · 2 nodes hand-verified against source (node 19, node 40)
