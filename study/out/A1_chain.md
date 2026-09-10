# A1 chain audit — 20260908_225039_gpt54_beliefs2stage

29 expand events with a parent.

## L4 — did the planning pass name real registry ids?

- invented ids in **1/29** expands
  - node 1: strategies ['postplan_self_check', 'tool_result_cache', 'tool_error_recovery'] areas ['workflow', 'tool_wrapper']

## L5a — retrieval selection: which channel survives the node cap?

| channel | selected | dropped `over max_nodes` | survival |
|---|---|---|---|
| explicit | 73 | 1 | 99% |
| strategy | 23 | 78 | 23% |
| area | 3 | 37 | 8% |
| keyword | 2 | 41 | 5% |
| **all** | **101** | **157** | **39%** |

> Explicit ids (nodes the planning pass already named) take precedence in `edit_archive.resolve_query`. Associative matches — the part that could surface something the planner did NOT think of — were selected 28 times and dropped 156 times (15% survival).

## L5b / L6 — was the code shown, and did the editor use it?

`shown` = top-level definitions rendered in the retrieval block. `omitted` = named on an `omitted (budget):` line, i.e. announced and withheld. `edited` = definitions this node's own diff touched.

| node | shown | omitted | edited | edited∩shown | edited∩omitted | novel |
|---|---|---|---|---|---|---|
| 1 | 0 | 0 | 2 | 0 | 0 | 2 |
| 2 | 1 | 0 | 10 | 0 | 0 | 10 |
| 3 | 10 | 0 | 2 | 1 | 0 | 1 |
| 4 | 9 | 0 | 10 | 2 | 0 | 8 |
| 5 | 17 | 0 | 9 | 6 | 0 | 3 |
| 6 | 20 | 0 | 15 | 12 | 0 | 3 |
| 7 | 18 | 6 | 9 | 6 | 1 | 3 |
| 8 | 18 | 8 | 1 | 1 | 0 | 0 |
| 9 | 16 | 4 | 1 | 1 | 0 | 0 |
| 10 | 15 | 1 | 3 | 2 | 0 | 1 |
| 11 | 15 | 1 | 7 | 2 | 0 | 5 |
| 12 | 13 | 7 | 2 | 1 | 0 | 1 |
| 13 | 10 | 1 | 20 | 2 | 0 | 18 |
| 14 | 15 | 1 | 1 | 1 | 0 | 0 |
| 15 | 30 | 2 | 6 | 2 | 0 | 4 |
| 16 | 6 | 0 | 1 | 1 | 0 | 0 |
| 17 | 13 | 0 | 8 | 2 | 0 | 6 |
| 18 | 14 | 1 | 2 | 2 | 0 | 0 |
| 19 | 6 | 0 | 4 | 1 | 0 | 3 |
| 20 | 15 | 0 | 3 | 3 | 0 | 0 |
| 21 | 10 | 1 | 1 | 1 | 0 | 0 |
| 22 | 10 | 1 | 2 | 2 | 0 | 0 |
| 23 | 9 | 1 | 2 | 2 | 0 | 0 |
| 24 | 20 | 2 | 2 | 2 | 0 | 0 |
| 25 | 10 | 1 | 1 | 1 | 0 | 0 |
| 26 | 10 | 1 | 2 | 2 | 0 | 0 |
| 27 | 10 | 1 | 3 | 0 | 0 | 3 |
| 28 | 6 | 0 | 3 | 2 | 0 | 1 |
| 29 | 5 | 0 | 3 | 2 | 0 | 1 |

- definitions edited across all expands: **135**
  - also shown in retrieved memory: **62** (46%)
  - named on an `omitted (budget):` line — announced but withheld: **1** (1%)
  - not in the retrieval block at all: **73** (54%)

> **Read this carefully.** "not in the retrieval block" does NOT mean the editor was flying blind: `AgentEditor._format_current_sources` always shows the parent's mutable files in full. Retrieval only adds OTHER nodes' implementations. So this row means the edit landed on code the editor could already see, and memory contributed nothing to that particular definition.

> Only **1** of 135 edited definitions had been withheld by the char budget. So the `omitted (budget):` truncation — frequent as it is — mostly withheld code the editor was not going to touch. The char budget is **not** the load-bearing failure; the node cap and the explicit-echo behaviour in L5a are.

### Definitions the editor edited that memory had withheld

These are the cleanest evidence that the char budget cost something: the editor was told the definition existed, was not shown it, and edited it anyway.

- **node 7**: `workflow.py :: _repair_plan`
