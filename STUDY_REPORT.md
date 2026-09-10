# Edit-memory study — findings

Plan: `~/.claude/plans/squishy-growing-hummingbird.md`.
Scripts: `study/`. Generated reports: `study/out/`.

**Source run.** `runs/20260907_212018_travel_hgm_1000_qwen122b_gpt54_beliefs2stage`
(task agent local Qwen3.5-122B medium, meta agents GPT-5.4 medium, belief steering +
two-stage editor). The run finished 2026-09-09 (33 nodes; best node 30 at
0.707/60; launched via `run_seeded.py`). Nothing in this study touched it. All analysis runs against a read-only snapshot,
`edit-memory-sep6/study_snapshots/20260908_225039_gpt54_beliefs2stage`, taken
2026-09-08 22:50 UTC (15:50 PT) at 29 evaluated nodes / 896 evals.

Status: **free tier complete (A0, A1, C Tier 1)**. A2, B and C Tier 2 not yet run.

---

## Headline

1. **Retrieval is mostly an echo.** Associative matches — the only channel that can
   surface something the planning pass did not already think of — survive the node
   cap **15%** of the time. Explicit node ids survive **99%**.
2. **Truncation is real but not load-bearing.** 41 definitions were announced to the
   editor and withheld, across 17 of 29 expands. Only **1 of 135** definitions the
   editor went on to edit had been withheld. The char budget is not the problem.
3. **The score cannot referee any of this.** The run's best-minus-seed gap of +0.113
   has **p = 0.62** against a null in which every node is exactly as good as the seed.
   40% of the observed spread between node means is real; 60% is sampling error.

---

## A0 — Truncation audit (`study/out/A0_truncation_gpt54.md`)

| stage | knob | bound on | lost | status |
|---|---|---|---|---|
| tagger diff | `diff_char_cap` 6000 | 8/29 (28%) | 69,194 chars middle-elided | **live** |
| code record | `code_diff_char_cap` 20000 | 1/29 | 2,332 chars | live (fallback only) |
| judge | `diff_char_cap` (old path) | 8/29 (28%) | 69,194 chars | **fixed in working tree** |
| retrieval nodes | `max_retrieved_nodes` 4 | 23/29 (79%) | **156 node-drops** | **live** |
| retrieval chars | `retrieval_char_budget` 60000 | 17/29 (59%) | **41 defs omitted** | **live** |
| belief doc | `doc_char_cap` 40000 | 0/82 | — | not binding |
| task agent | ctx 262144 / out 65536 | — | 3 `incomplete` in 8674 calls | negligible |

Notes:

- **The belief document was never truncated.** Docs run 3.3k–16.4k against a 40000 cap,
  and `belief_contract` rejects an over-cap doc with a retry rather than cutting it. The
  16k cap that forced belief retirement was the earlier DeepSeek run, not this one.
- **The judge truncation is already fixed.** All 23 analysis prompts carry the
  `# Code diff vs parent` header — the pre-2026-09-08 path that fed the judge a
  `truncate_middle` diff at 6000 chars instead of the 20000-char implementation view.
  Re-rendering all 8 affected nodes through the current
  `edit_code.render_implementation_view` gives `defs_omitted: 0` for every one, so the
  working-tree change genuinely resolves it. The run predates the change.
- **The tagger truncation is still live** (`edit_memory.py:578`). For 8 of 29 nodes the
  record that describes the edit was written from a middle-elided diff. Those records
  then feed the judge, the beliefs and retrieval.
- `retrieval_char_budget` is divided by the number of selected nodes
  (`edit_archive.py:167`), so the 4-node cap is what squeezes each node to ~15k.
- One definition dominates the omissions: **`workflow.py :: _build_day_audit_report`
  (8,027 chars) was named-but-absent in 10 separate expands** (nodes 9–26).

Correlational only (small counts, not a result): nodes truncated at the tagger/code
stage were later judged implementation-unsound 6/8 (75%) vs 6/15 (40%) for the rest.

---

## A1 — Chain audit (`study/out/A1_chain.md`)

### L4 — does the planner name real ids? Yes.

Invented ids in 1 of 29 expands (node 1, the very first). The
`## Registry ids for the memory query` block is doing its job.

### L5a — which retrieval channel survives the cap?

| channel | selected | dropped `over max_nodes` | survival |
|---|---|---|---|
| explicit | 73 | 1 | **99%** |
| strategy | 23 | 78 | 23% |
| area | 3 | 37 | 8% |
| keyword | 2 | 41 | 5% |
| **all** | **101** | **157** | 39% |

Explicit ids take precedence in `edit_archive.resolve_query`, and the cap is 4. So when
the planning pass names 3–4 nodes — which it did in 24 of 29 expands — the associative
channels get whatever is left, usually nothing. **Associative survival is 15%.**

This is the structural finding: retrieval largely returns what the planner already asked
for by id. It is not, in practice, a search over memory.

### L5b / L6 — was the code shown, and did the editor use it?

Of **135** definitions edited across all expands:

- **62 (46%)** were also shown in retrieved memory
- **1 (1%)** had been named on an `omitted (budget):` line and withheld
- **73 (54%)** were not in the retrieval block at all

The 54% is not "the editor was blind": `AgentEditor._format_current_sources` always
shows the parent's mutable files in full. It means the edit landed on code the editor
could already see, and memory contributed nothing to that definition.

**This is what demotes the truncation story.** The char budget withheld plenty, but
almost never something the editor then needed. Priority moves to the node cap and the
explicit-echo behaviour.

---

## C Tier 1 — what the measurement can detect (`study/out/C1_noise.md`)

### Noise floor

Five repeat evaluations of the **same seed code** on the **same 120 cases**
(`../self-improving-dual/runs/eval_20260828_1502*_seed`):
0.6089 / 0.4896 / 0.5870 / 0.5724 / 0.5401 — run-to-run sd **0.046**.

- between-case variance (real difficulty) 0.0393; within-case (same code, same case,
  different run) 0.0505 → **ICC 0.437**
- 0 of 120 cases are deterministic across repeats

| n cases | SE of one mean | SE of paired Δ | SE of unpaired Δ |
|---|---|---|---|
| 16 | ±0.0749 | ±0.0795 | ±0.1060 |
| 32 | ±0.0530 | ±0.0562 | ±0.0749 |
| 60 | ±0.0387 | ±0.0410 | ±0.0547 |
| 120 | ±0.0274 | ±0.0290 | ±0.0387 |

Cases needed for 80% power at α=0.05: Δ=0.02 → **1,984 paired**; Δ=0.03 → 882;
Δ=0.05 → 318; Δ=0.10 → 80. `edit_outcome.NEUTRAL_BAND` is ±0.02.

### Discriminability and the winner's curse

- 29 evaluated nodes; observed spread of node means sd 0.0794
- spread expected from sampling error alone sd 0.0613
- **residual real signal sd 0.0505 → 40% of the observed spread**
- best node 27 = 0.7109 over **16** evals; seed = 0.5979 over 60 → **+0.113**

Null model — every node given identical true quality, drawn with its own real `n_evals`
and the measured noise, take the max over the tree:

- null median gap **+0.129**, p90 +0.200, p95 +0.222
- **P(null gap ≥ observed) = 0.62** over 20,000 trials

The observed improvement is *below the median* of what the selection rule manufactures
from noise alone. Note the current best is a 16-eval node that overtook a 60-eval node —
the winner's curse in action, live.

### The published arm comparison

| arm | best node | best mean | n | SE |
|---|---|---|---|---|
| control (no edit memory) | 33 | 0.7823 | 60 | ±0.0387 |
| edit memory | 8 | 0.7635 | 60 | ±0.0387 |
| OR no-editmem | 31 | 0.7135 | 60 | ±0.0387 |
| OR editmem | 27 | 0.6885 | 60 | ±0.0387 |
| belief + GPT-5.4 | 27 | 0.7109 | 16 | ±0.0749 |

Widest gap 0.0938 ± 0.0547 = **1.7σ**, before any winner's-curse correction, and every
entry is itself a max over ~30 nodes. **No arm difference in this table is
distinguishable from noise.** That is the direct answer to "it is hard to justify that
edit memory is helping": on this measurement, it is equally hard to justify that
anything is helping, including the control.

---

## What this changes about the rest of the study

- **Probe C (truncation causality) drops in priority.** L6 already shows the char budget
  withheld a definition the editor needed exactly once in 135.
- **Probe B (cap stress) rises.** The 15% associative survival is the mechanism most
  likely to matter, and it is the one Probe B isolates.
- **Probe D (oracle ceiling) is unchanged and still decisive** — it is the only test that
  separates "retrieval never delivered it" from "the meta agent could not use it".
- **Part B's outcome metrics must stay process-side.** C Tier 1 confirms score cannot
  referee edit quality at any batch size this search uses.

## Not yet run

A2 planted-bug probes (~$20) · Part B counterfactual ablation (~$51, gated on the
byte-diff fidelity check) · C Tier 2 cross-agent evaluation (≤$60, gated on a 5-case
cost pilot).

## Reproduce

```bash
cd /groups/AIC-MV/sudipta.paul/code/rsi/edit-memory-sep6/self-improving-dual
D=$(cat study/out/SNAPSHOT_PATH.txt)
B=/groups/AIC-MV/sudipta.paul/code/rsi/self-improving-dual/runs
PYTHONPATH=. python3 study/audit_truncation.py --run "$D" --out study/out/A0_truncation_gpt54.md
PYTHONPATH=. python3 study/audit_chain.py      --run "$D" --out study/out/A1_chain.md
PYTHONPATH=. python3 study/noise.py \
  --repeats $B/eval_20260828_1502{13,18,23,29,35}_seed \
  --run "$D" --out study/out/C1_noise.md
```
