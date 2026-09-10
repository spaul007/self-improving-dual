# Retrieval cap probe — 20260908_225039_gpt54_beliefs2stage

Every expand's own recorded query, replayed against the world as it stood just before that expand, sweeping the node cap and the char budget.

## Fidelity gate

Replay at the run's own settings (`max_nodes=4`, `char_budget=60000`) reproduces the recorded manifest for **27/29** expands.

Mismatches (as-of reconstruction vs what the run recorded):
- node 4: run [2] vs replay [2, 3, 1] — replay admits [3, 1]
- node 5: run [4, 2] vs replay [4, 2, 3, 1] — replay admits [3, 1]

**Cause, and why it is benign.** Every mismatch is in the *keyword* channel. `resolve_query` keyword-scans each node's `edit_memory.md` body, and records are **refreshed in place** as evaluations arrive — their Outcome/Analysis text today is richer than it was at the moment of the expand, and no history is kept (`study/asof.py` documents this). So a stale record can match a keyword it did not contain then.

The bias has a known sign: replay admits **more** nodes than the run did, never fewer. Every 'the cap hid something' number below is therefore an **upper bound** on what better retrieval could have delivered.

## What a bigger cap admits

| cap | budget | mean nodes | explicit | strategy | area | keyword | defs omitted |
|---|---|---|---|---|---|---|---|
| 4 | 60,000 | 3.6 | 73 | 23 | 3 | 6 | 66 |
| 8 | 60,000 | 6.4 | 73 | 77 | 25 | 11 | 446 |
| 12 | 60,000 | 8.6 | 73 | 97 | 37 | 41 | 666 |
| ∞ | 60,000 | 9.8 | 73 | 101 | 40 | 70 | 812 |
| ∞ | 240,000 | 9.8 | 73 | 101 | 40 | 70 | 205 |

> **The cap and the budget fight each other.** `resolve_query` sets `per_node = char_budget // len(selected)` (`edit_archive.py:167`), and each node's code budget is what is left after its record text. So admitting more nodes under a fixed budget starves every one of them: `defs omitted` climbs from 66 to 812 across the sweep. Raising the node cap alone does not buy more memory — it buys more record headers and less code.

## Payoff — does the extra memory contain what the child actually edited?

`covered` = definitions this node's own diff touched that appear in the retrieval block. This is the only measure of the cap costing something real.

| cap | budget | defs shown | edited defs covered | coverage | retrieval chars |
|---|---|---|---|---|---|
| 4 | 60,000 | 340 | 61/135 | 45% | 1,128,238 |
| 8 | 60,000 | 148 | 50/135 | 37% | 1,553,755 |
| 12 | 60,000 | 130 | 42/135 | 31% | 1,975,044 |
| ∞ | 60,000 | 128 | 41/135 | 30% | 2,258,328 |
| ∞ | 240,000 | 764 | 63/135 | 47% | 3,449,185 |

- coverage at the run's settings: **61/135** (45%)
- best achievable in this sweep: **63/135** (47%)
- headroom from lifting the caps alone: **2 definitions**

> Coverage well short of 100% at an unlimited cap means the definitions the editor edits are mostly not in ANY sibling's implementation — they are in the parent's own sources, which the editor always sees in full. Retrieval cannot be the binding constraint on those edits.

## Per expand

| node | nodes then | cap 4 | cap ∞ | new nodes admitted | edited covered 4 → ∞ |
|---|---|---|---|---|---|
| 1 | 1 | 0 | 0 | — | 0 → 0 |
| 2 | 2 | 1 | 1 | — | 0 → 0 |
| 3 | 3 | 2 | 2 | — | 1 → 1 |
| 4 | 4 | 3 | 3 | — | 2 → 2 |
| 5 | 5 | 4 | 4 | — | 6 → 6 |
| 6 | 6 | 3 | 3 | — | 11 → 11 |
| 7 | 7 | 4 | 6 | [3, 1] | 6 → 3 |
| 8 | 8 | 4 | 7 | [1, 3, 5] | 1 → 0 |
| 9 | 9 | 4 | 6 | [5, 6] | 1 → 1 |
| 10 | 10 | 4 | 7 | [4, 5, 6] | 2 → 1 |
| 11 | 11 | 4 | 10 | [10, 4, 5, 6, 3, 1] | 2 → 0 |
| 12 | 12 | 4 | 5 | [2] | 1 → 1 |
| 13 | 13 | 4 | 11 | [2, 4, 5, 6, 11, 3, 1] | 2 → 0 |
| 14 | 14 | 4 | 12 | [9, 13, 5, 6, 4, 2, 3, 1] | 1 → 0 |
| 15 | 15 | 4 | 13 | [4, 5, 6, 7, 8, 14, 9, 3, 1] | 2 → 0 |
| 16 | 16 | 4 | 14 | [7, 9, 2, 13, 6, 4, 11, 5, 3, 1] | 1 → 0 |
| 17 | 17 | 4 | 15 | [7, 8, 9, 14, 13, 6, 4, 11, 5, 3, 1] | 2 → 0 |
| 18 | 18 | 4 | 13 | [8, 9, 14, 2, 17, 13, 4, 5, 6] | 2 → 0 |
| 19 | 19 | 4 | 17 | [7, 8, 14, 18, 2, 17, 13, 6, 4, 11, 5, 3, 1] | 1 → 0 |
| 20 | 20 | 4 | 6 | [13, 17] | 3 → 3 |
| 21 | 21 | 4 | 18 | [8, 16, 18, 2, 4, 5, 6, 13, 20, 19, 17, 11, 3, 1] | 1 → 1 |
| 22 | 22 | 4 | 10 | [8, 14, 18, 21, 13, 15] | 2 → 2 |
| 23 | 23 | 4 | 14 | [7, 8, 10, 14, 16, 18, 11, 6, 5, 4] | 2 → 1 |
| 24 | 24 | 4 | 19 | [8, 9, 10, 14, 16, 18, 21, 22, 13, 15, 19, 20, 6, 5, 4] | 2 → 2 |
| 25 | 25 | 4 | 12 | [8, 9, 14, 16, 18, 22, 23, 4] | 1 → 1 |
| 26 | 26 | 4 | 17 | [8, 10, 14, 16, 18, 21, 22, 24, 11, 17, 6, 5, 4] | 2 → 1 |
| 27 | 27 | 4 | 15 | [15, 16, 17, 21, 24, 11, 6, 9, 7, 5, 4] | 0 → 0 |
| 28 | 28 | 4 | 13 | [11, 13, 15, 17, 23, 24, 26, 1, 12] | 2 → 2 |
| 29 | 29 | 4 | 11 | [10, 23, 24, 26, 1, 17, 12] | 2 → 2 |
