# A0 truncation audit — 20260908_225039_gpt54_beliefs2stage

30 nodes, 29 with an edit record.

## Caps this run ran with

| knob | value |
|---|---|
| `diff_char_cap` | 6000 |
| `code_diff_char_cap` | 20000 |
| `analysis_code_char_budget` | (default) |
| `steering_token_budget` | 48000 |
| `retrieval_char_budget` | 60000 |
| `max_retrieved_nodes` | 4 |
| `doc_char_cap` | 40000 |
| `evidence_char_budget` | (default) |
| `instruction_char_cap` | 2500 |

## Did each cap bind?

`status` distinguishes a cap that is still live in the current tree from one
this run hit but the working tree has since changed.

| stage | knob | nodes where it bound | what was lost | status |
|---|---|---|---|---|
| tagger diff | `diff_char_cap` | 8/29 (28%) | 69,194 chars middle-elided | **live** (`edit_memory.py:578`) |
| code record | `code_diff_char_cap` | 1/29 (3%) | 2,332 chars | live (retrieval fallback only) |
| judge | `diff_char_cap` (old path) | 8/29 (28%) | 69,194 chars | **fixed in working tree** |
| retrieval nodes | `max_retrieved_nodes` | 23/29 (79%) | **156 node-drops** | **live** |
| retrieval chars | `retrieval_char_budget` | 17/29 (59%) | **41 defs omitted** | **live** |
| editor prompt (rendered) | — | 17/29 (59%) | 41 named-but-absent units | **live** |
| belief doc | `doc_char_cap` | 0/82 within 2% of cap | sizes 3,348–16,407, cap 40000 | not binding (reject+retry, never silent) |

**Judge attribution.** All 23 analysis prompts carry the `# Code diff vs parent` header (8 of them long enough to be cut), i.e. the pre-2026-09-08 path that fed the judge a `truncate_middle` diff at `diff_char_cap` (6000), not the 20000-char implementation view. The run predates that change (`edit_memory.py` mtime is after the run's `config.snapshot.yaml`), so this is a truncation the run suffered and the current tree does not. See the verification section below.

## Retrieval: selection channel and the cap

- selected **101** node-slots total: **73 explicit** (named by the planning pass), **28 associative** (strategy/area/keyword)
- dropped **156** `over max_nodes`
- explicit share of the returned slots: **72%**

| node | query n/s/a/k | selected | explicit | assoc | dropped | defs omitted |
|---|---|---|---|---|---|---|
| 1 | 1/3/2/5 | 0 | 0 | 0 | 0 | 0 |
| 2 | 1/2/2/8 | 1 | 1 | 0 | 0 | 0 |
| 3 | 2/2/2/7 | 2 | 2 | 0 | 0 | 0 |
| 4 | 1/1/2/7 | 1 | 1 | 0 | 0 | 0 |
| 5 | 1/2/2/8 | 2 | 1 | 1 | 0 | 0 |
| 6 | 3/2/2/9 | 3 | 3 | 0 | 0 | 0 |
| 7 | 3/2/2/6 | 4 | 3 | 1 | 1 | 7 |
| 8 | 3/3/2/6 | 4 | 3 | 1 | 3 | 8 |
| 9 | 2/2/2/5 | 4 | 2 | 2 | 2 | 4 |
| 10 | 3/1/2/7 | 4 | 3 | 1 | 3 | 1 |
| 11 | 2/2/3/9 | 4 | 2 | 2 | 5 | 1 |
| 12 | 1/1/2/6 | 4 | 1 | 3 | 1 | 7 |
| 13 | 4/1/3/5 | 4 | 4 | 0 | 5 | 1 |
| 14 | 2/2/2/6 | 4 | 2 | 2 | 5 | 1 |
| 15 | 2/3/3/6 | 4 | 2 | 2 | 7 | 2 |
| 16 | 4/2/2/6 | 4 | 4 | 0 | 7 | 0 |
| 17 | 2/2/1/6 | 4 | 2 | 2 | 8 | 0 |
| 18 | 3/2/3/7 | 4 | 3 | 1 | 9 | 1 |
| 19 | 4/2/2/7 | 4 | 4 | 0 | 10 | 0 |
| 20 | 2/2/2/6 | 4 | 2 | 2 | 2 | 0 |
| 21 | 3/1/2/6 | 4 | 3 | 1 | 11 | 1 |
| 22 | 3/1/2/5 | 4 | 3 | 1 | 6 | 1 |
| 23 | 4/1/2/7 | 4 | 4 | 0 | 9 | 1 |
| 24 | 3/2/2/5 | 4 | 3 | 1 | 14 | 2 |
| 25 | 3/1/2/6 | 4 | 3 | 1 | 9 | 1 |
| 26 | 3/1/2/5 | 4 | 3 | 1 | 12 | 1 |
| 27 | 3/2/2/6 | 4 | 3 | 1 | 11 | 1 |
| 28 | 3/3/2/6 | 4 | 3 | 1 | 9 | 0 |
| 29 | 3/1/3/6 | 4 | 3 | 1 | 7 | 0 |

## Headroom — what budget would have removed the omissions

- 41 omitted units state their size: median 2,088 chars, max 8,027, total 139,867
- per-node char allowance actually granted: median 15,000 (budget 60000 ÷ selected nodes, `edit_archive.py:167`)
- so the per-node squeeze is a direct consequence of the node cap: naming 4 nodes divides the same budget 4 ways.

## Compounding (correlational — see study/probe_bug.py for the causal test)

Nodes truncated at the **tagger or code-record** stage: [2, 4, 5, 6, 7, 11, 13, 17]

| downstream flag | truncated early | not truncated early |
|---|---|---|
| implementation judged unsound | 6/8 (75%) | 6/15 (40%) |
| tag forced by the registry cap | 0/8 (0%) | 0/21 (0%) |
| suspect verifier flagged | 3/8 (38%) | 7/21 (33%) |

These counts are small; treat them as a pointer, not a result.

## Does the working-tree fix actually resolve the judge truncation?

Re-render each affected node through the CURRENT `edit_code.render_implementation_view` at the budget the current tree would use. `defs_omitted: 0` means the judge would now see the whole edit.

| node | old: chars elided | new: view chars | defs omitted | hunks omitted |
|---|---|---|---|---|
| 2 | 3,894 | 6,581 | 0 | 0 |
| 4 | 12,193 | 16,276 | 0 | 0 |
| 5 | 9,031 | 13,894 | 0 | 0 |
| 6 | 16,332 | 19,454 | 0 | 0 |
| 7 | 10,273 | 15,229 | 0 | 0 |
| 11 | 2,112 | 7,352 | 0 | 0 |
| 13 | 12,318 | 16,967 | 0 | 0 |
| 17 | 3,041 | 9,119 | 0 | 0 |

## Named-but-absent units, per node

Each entry was announced to the editor by name and then not shown.

- **node 7** (7): workflow.py :: _log_plan_validation (full source 439 chars); workflow.py :: _select_preferred_plan (full source 1971 chars); workflow.py :: _log_structure_validation (full source 1043 chars); workflow.py :: _format_structure_report (full source 571 chars); workflow.py :: _log_structure_validation (full source 1080 chars); workflow.py :: _select_repaired_plan (full source 2088 chars); workflow.py :: _repair_plan (full source 2267 chars)
- **node 8** (8): workflow.py :: _build_day_audit_report (full source 8027 chars); workflow.py :: _format_structure_report (full source 571 chars); workflow.py :: _log_structure_validation (full source 1080 chars); workflow.py :: _select_repaired_plan (full source 2088 chars); workflow.py :: _repair_plan (full source 2267 chars); workflow.py :: _format_validation_report (full source 592 chars); workflow.py :: _log_plan_validation (full source 439 chars); workflow.py :: _select_preferred_plan (full source 1971 chars)
- **node 9** (4): workflow.py :: _build_day_audit_report (full source 8027 chars); workflow.py :: _format_validation_report (full source 592 chars); workflow.py :: _log_plan_validation (full source 439 chars); workflow.py :: _select_preferred_plan (full source 1971 chars)
- **node 10** (1): workflow.py :: _build_day_audit_report (full source 8027 chars)
- **node 11** (1): workflow.py :: _build_day_audit_report (full source 8027 chars)
- **node 12** (7): workflow.py :: _format_structure_report (full source 571 chars); workflow.py :: _log_structure_validation (full source 1080 chars); workflow.py :: _select_repaired_plan (full source 2088 chars); workflow.py :: _repair_plan (full source 2267 chars); workflow.py :: _format_validation_report (full source 592 chars); workflow.py :: _log_plan_validation (full source 439 chars); workflow.py :: _select_preferred_plan (full source 1971 chars)
- **node 13** (1): workflow.py :: _build_day_audit_report (full source 8027 chars)
- **node 14** (1): workflow.py :: _build_day_audit_report (full source 8027 chars)
- **node 15** (2): workflow.py :: _run_repair_loop (full source 1133 chars); workflow.py :: _maybe_repair_transfer_timing (full source 2401 chars)
- **node 18** (1): workflow.py :: _build_day_audit_report (full source 8027 chars)
- **node 21** (1): workflow.py :: _build_day_audit_report (full source 8027 chars)
- **node 22** (1): workflow.py :: _build_day_audit_report (full source 8027 chars)
- **node 23** (1): workflow.py :: _review_duplicate_diversity (full source 3693 chars)
- **node 24** (2): workflow.py :: _review_duplicate_diversity (full source 3693 chars); workflow.py :: _build_day_audit_report (full source 8027 chars)
- **node 25** (1): workflow.py :: _build_day_audit_report (full source 8027 chars)
- **node 26** (1): workflow.py :: _build_day_audit_report (full source 8027 chars)
- **node 27** (1): workflow.py :: _repair_plan (full source 2146 chars)
