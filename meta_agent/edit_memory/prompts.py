"""Every text the edit-memory layer sends to a model.

Four roles, two of them agentic:

* the **memory curator** — a tool-use session over the last m nodes; writes
  ``curation.md`` (Z);
* the **memory generator** — one call over (Z, B_j, I_k); writes B_{j+1};
* the **instruction curator** — a tool-use session over the nodes expanded
  with memory; writes ``q.md`` (Q);
* the **instruction updater** — one call over (Q, I_k); writes the new
  addendum I_{k+1}.

Two rules run through all of them: the memory ranks edits by *judged
usefulness from evidence* and never states a predicted score; and the
curators read raw artifacts themselves — nothing is pre-digested.
"""
from __future__ import annotations

from typing import Any, Optional

from ..agentic.session import SessionMessages

# ---------------------------------------------------------------------- #
# Output files and required structure
# ---------------------------------------------------------------------- #

CURATION_FILE = "curation.md"
Q_FILE = "q.md"

# Per-node sections the memory curator must write (checked by the submit
# handler; the heading carries the node id so the check is exact).
NODE_SECTION_HEADING = "## Node {node_id}"
NODE_SUBSECTIONS = (
    "What changed",
    "Intent",
    "Editor process",
    "What the evaluation shows",
    "Shortcomings",
    "Usefulness verdict",
)
CURATION_CROSS_HEADING = "## Across the window"
CURATION_GRADIENT_HEADING = "## Gradient w.r.t. the current edit memory"

# Sections of the edit memory itself (validated by generator.validate_memory).
MEMORY_SECTIONS = (
    "## 1. Ranked edits",
    "## 2. Usefulness",
    "## 3. Strategy vs implementation",
    "## 4. Guidance for the next editor",
)

# Sections of the instruction curator's output.
Q_SECTIONS = (
    "## Edit memory usage by editors",
    "## Effect on task agents",
    "## Missing information",
    "## Representation issues",
    "## Proposed instruction changes",
)

# Sentinel that opens the fixed core of the generator instruction; the
# addendum may never contain it (validate_addendum).
CORE_SENTINEL = "=== FIXED CORE INSTRUCTION (not editable) ==="

# ---------------------------------------------------------------------- #
# Shared evidence legend
# ---------------------------------------------------------------------- #

ROUND_LEGEND_LINES = (
    "  strategy.json              the editor's own summary of the edit (goal, changes, rationale)",
    "  hgm_node.json              tree stats: parent_id, n_evals, mean_utility, memory_arm, memory_version",
    "  task_agent/                the node's code (workflow.py, tool_wrapper.py, tools_schema.json, mutable_tools/)",
    "  agentic/transcript.jsonl   the editor session that produced this node: every model turn and tool call",
    "  agentic/session.json       session summary (end_reason, n_llm_calls, changed_files, memory_path)",
    "  feedback.json              evaluation digest: failure_report (categories, representative failures,",
    "                             hardest cases), project_metrics, tool_error_rate, runtime_exceptions",
    "  eval_result.json           per-case score / passed / details incl. failed_checks (cumulative)",
    "  logs/case_<id>.json        one result per evaluated case: query, the agent's plan, what failed",
    "  logs/trace.jsonl           llm/tool trace of the LAST evaluation batch only (the evaluator",
    "                             truncates it per batch); grep for \"mutable_log\" (the agent's own",
    "                             trace.log verdicts) and \"error\" events rather than cat — it is large",
)


def _legend() -> str:
    return "\n".join(ROUND_LEGEND_LINES)


# ---------------------------------------------------------------------- #
# Tool descriptions for the curators (the editor keeps its own texts)
# ---------------------------------------------------------------------- #

def curator_bash_tool_info(*, bash_timeout_s: float, max_output_chars: int,
                           root_vars: tuple[str, ...]) -> dict[str, Any]:
    roots = ", ".join(root_vars[:-1]) + " and " + root_vars[-1]
    return {
        "name": "bash",
        "description": (
            "Run a bash command in a fresh, sandboxed shell.\n"
            "* No internet access. Only the paths listed in the task message "
            "exist; everything else is absent or read-only. The roots "
            + roots + " are environment variables here.\n"
            "* Each call starts a NEW shell in $WORK_DIR — cwd, variables and "
            "background processes do NOT persist between calls (chain with && or ;).\n"
            "* Good for: `diff -u $PARENT_1/task_agent/workflow.py $NODE_1/task_agent/workflow.py`, "
            "`python3 -c` one-liners over the JSON artifacts, "
            "`grep '\"kind\": \"mutable_log\"' $NODE_1/logs/trace.jsonl | head`, "
            "`python3 -m json.tool`. The task agent cannot be run here (no model access).\n"
            f"* Avoid commands with very large output; output is truncated to "
            f"{max_output_chars} chars (head and tail kept). Commands are killed "
            f"after {bash_timeout_s:g}s.\n"
            "* Everything is read-only except $WORK_DIR: write your output there "
            "with the `editor` tool (or shell redirection)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The bash command to run."},
            },
            "required": ["command"],
        },
    }


def curator_editor_tool_info(*, max_view_chars: int, output_file: str) -> dict[str, Any]:
    return {
        "name": "editor",
        "description": (
            "View files and write your output document.\n"
            "* `view` a file (line-numbered; optional `view_range: [start, end]`; "
            f"truncated to {max_view_chars} chars) or list a directory.\n"
            f"* `create` writes a whole file — use it for $WORK_DIR/{output_file} "
            "and for notes under $WORK_DIR; `str_replace` / `insert` / "
            "`replace_lines` edit it in place.\n"
            "* Paths: absolute, `$VAR/...` with the roots from the task message, "
            "or relative to $WORK_DIR. Writes anywhere outside $WORK_DIR are refused."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string",
                            "enum": ["view", "create", "str_replace", "insert", "replace_lines"]},
                "path": {"type": "string",
                         "description": "Path: absolute, $VAR/... (roots from the task message), or relative to $WORK_DIR."},
                "file_text": {"type": "string", "description": "create: the full file content."},
                "old_str": {"type": "string", "description": "str_replace: exact text to replace (must occur once)."},
                "new_str": {"type": "string", "description": "str_replace / insert: replacement or inserted text."},
                "insert_line": {"type": "integer", "description": "insert: insert after this 1-based line (0 = top)."},
                "view_range": {"type": "array", "items": {"type": "integer"},
                               "description": "view: [start, end] 1-based inclusive; end -1 = EOF."},
                "start_line": {"type": "integer", "description": "replace_lines: first line to replace (1-based)."},
                "end_line": {"type": "integer", "description": "replace_lines: last line to replace (inclusive)."},
            },
            "required": ["command", "path"],
        },
    }


SUBMIT_CURATION_NAME = "submit_curation"
SUBMIT_CURATION_TOOL: dict[str, Any] = {
    "name": SUBMIT_CURATION_NAME,
    "description": (
        "Finish the session. Your document is already on disk in $WORK_DIR — "
        "this call submits only a one-paragraph summary. The document is "
        "checked on this call; if sections are missing you are told which and "
        "may fix the file and call this again."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "summary": {"type": "string",
                        "description": "One paragraph: the main findings of your document."},
        },
        "required": ["summary"],
    },
}


def curator_messages(output_file: str) -> SessionMessages:
    return SessionMessages(
        nudge=(f"You stopped without calling {SUBMIT_CURATION_NAME}. If $WORK_DIR/"
               f"{output_file} is complete, call {SUBMIT_CURATION_NAME} now; otherwise "
               "continue with the bash/editor tools."),
        halfway=("Budget check: you have used {used} of {total} model calls and have not "
                 f"written $WORK_DIR/{output_file} yet. Start writing it now with the editor "
                 "tool (create), then refine; reading more than you can write up is wasted budget."),
        final_stretch=("Only {left} model calls remain (used {used} of {total}). Finish "
                       f"$WORK_DIR/{output_file} in the next call(s), then call "
                       f"{SUBMIT_CURATION_NAME}. A session that ends without a submitted "
                       "document is wasted."),
        wrap_up=("The session budget is exhausted ({reason}). Call "
                 f"{SUBMIT_CURATION_NAME} now with a summary of what your document says. "
                 "Do not call other tools."),
    )


# ---------------------------------------------------------------------- #
# Memory curator
# ---------------------------------------------------------------------- #

MEMORY_CURATOR_SYSTEM = (
    "You are the curator of the edit memory of a self-evolving agent. A "
    "meta-agent (the editor) keeps producing new versions of a task agent's "
    "code, each version is evaluated on benchmark cases, and every few "
    "versions you review the newest ones and write what was actually tried, "
    "what the evidence shows about whether it helped, and where it fell "
    "short. You have three tools: `bash`, `editor`, and "
    f"`{SUBMIT_CURATION_NAME}`.\n"
    "Judge from evidence, not from scores: read the code diff against the "
    "parent, the editor's own transcript, the per-case results and what the "
    "agent logged at runtime. A node's mean score is context; the failure "
    "categories, the failed checks, whether a new mechanism fired at all, "
    "and whether the same cases pass or fail as in the parent are the "
    "evidence. Never write a predicted score, an expected gain, or any "
    "number that forecasts future evaluation — describe what happened and "
    "why.\n"
    "Distinguish strategy from implementation: an idea can be right and its "
    "implementation broken (the mechanism never fired, fired on the wrong "
    "cases, crashed, was too coarse), or an implementation can be fine and "
    "the idea itself not address what fails. Say which.\n"
    "Environment:\n"
    "  - Every bash call is a fresh shell in $WORK_DIR with no network; only "
    "the roots listed in the task message exist. The benchmark's cases, the "
    "scoring code and the database are not available.\n"
    "  - $WORK_DIR is the only writable place. Write your document with the "
    "`editor` tool (`create`, then `str_replace`/`insert` to refine).\n"
    "  - Budget discipline: the task message states your call budget. Spend "
    "at most half of it reading (bash can `cat`/`sed -n`/`diff` several files "
    "in one call), then write. A complete document that is submitted beats "
    "a perfect reading that never becomes one.\n\n"
    f"Finishing: when $WORK_DIR/{CURATION_FILE} is complete, call "
    f"`{SUBMIT_CURATION_NAME}` with a one-paragraph summary. You must end the "
    "session by submitting; do not stop with a plain message."
)


def render_memory_curation_instruction(
    *,
    nodes: list[dict[str, Any]],
    previous_memory_exists: bool,
    addendum: str,
    max_llm_calls: int,
    timeout_s: float,
    max_attempts: int,
) -> str:
    """``nodes``: dicts with node_id, parent_id, memory_arm, memory_version,
    n_evals, mean_utility, edit_failed (failed nodes are listed but have no
    $NODE_i root)."""
    live = [n for n in nodes if not n.get("edit_failed")]
    failed = [n for n in nodes if n.get("edit_failed")]
    rows = ["  var        node  parent  arm      memory  evals  mean score (context only)"]
    for i, n in enumerate(live, start=1):
        rows.append(
            f"  $NODE_{i:<5} {n['node_id']:>4}  {n['parent_id']:>6}  "
            f"{n['memory_arm']:<8} {str(n.get('memory_version') if n.get('memory_version') is not None else '-'):<7} "
            f"{n['n_evals']:>5}  {n['mean_utility']:.3f}   (parent: $PARENT_{i})"
        )
    failed_lines = ""
    if failed:
        failed_lines = ("\nEdits that failed validation in this window (no code to review; "
                        "not counted, listed for completeness): "
                        + ", ".join(f"node {n['node_id']}" for n in failed) + "\n")
    per_node = "\n".join(f"  - {sub}" for sub in NODE_SUBSECTIONS)
    memory_note = (
        "$MEMORY_DIR/edit_memory.md is the current edit memory (read it first: your "
        "gradient section is written against it)"
        if previous_memory_exists else
        "There is no edit memory yet — this is the first window; the gradient "
        "section then lists what the first memory must contain"
    )
    addendum_block = ""
    if addendum.strip():
        addendum_block = (
            "\n## Current guidance for what the memory should contain (the "
            "generator's instruction addendum)\n" + addendum.strip() + "\n"
        )
    return (
        "# Task\n"
        f"Review the {len(live)} newest versions of the task agent (one node each) "
        f"and write $WORK_DIR/{CURATION_FILE}: what each edit changed, what the "
        "evidence shows about whether it helped, and where it fell short. The "
        "generator of the edit memory will read your document and nothing else "
        "from these nodes, so be concrete and cite node ids and file paths.\n\n"
        "## Nodes under review\n" + "\n".join(rows) + "\n" + failed_lines +
        "\n## Roots (environment variables in every bash call; the editor tool "
        "accepts the same $VAR form)\n"
        "  WORK_DIR     your output directory (the only writable path)\n"
        f"  MEMORY_DIR   the run's edit-memory directory: {memory_note}; "
        "instruction.md and window_*/curation.md are earlier material\n"
        "  NODE_i       round dir of the i-th node under review; PARENT_i its parent's\n"
        "  REPO_DIR     the repository (platform_core/, projects/<p>/tools/, db_schema.md)\n\n"
        "## What every node / parent dir contains\n" + _legend() + "\n\n"
        "A node's code diff is `diff -u $PARENT_i/task_agent/workflow.py "
        "$NODE_i/task_agent/workflow.py` (same for tool_wrapper.py, "
        "tools_schema.json, mutable_tools/*.py). Cases the node and its parent "
        "both ran are the pairs in their logs/case_<id>.json with the same id.\n"
        + addendum_block +
        f"\n## Required structure of $WORK_DIR/{CURATION_FILE}\n"
        f"For every node under review, a section headed `{NODE_SECTION_HEADING.format(node_id='<id>')}` "
        "with these subsections (as `### <name>`):\n" + per_node + "\n"
        "  'What the evaluation shows': which failure categories / failed checks moved "
        "versus the parent on shared cases, whether the node's new mechanism fired "
        "(mutable_log events, tool errors), regressions. 'Shortcomings': split into "
        "*strategy* (the idea) and *implementation* (how it was built). 'Usefulness "
        "verdict': one of helped / unclear / hurt, with the evidence cited.\n"
        f"Then `{CURATION_CROSS_HEADING}`: recurring patterns, contradictions between "
        "nodes, what remains untried.\n"
        f"Then `{CURATION_GRADIENT_HEADING}`: for each claim or ranked entry of the "
        "current edit memory, one of confirmed / contradicted / too coarse / missing, "
        "with node ids as evidence; plus claims the memory should contain but does not.\n"
        "Rules: cite node ids and file paths; no numeric score predictions or "
        "expected gains; do not rank by mean score.\n\n"
        "## Procedure\n"
        "  1. Read $MEMORY_DIR/edit_memory.md (if present) and each node's strategy.json and hgm_node.json.\n"
        "  2. For each node: diff its code against its parent; skim agentic/transcript.jsonl "
        "for what the editor read and decided; compare feedback.json / eval_result.json / "
        "logs/case_*.json with the parent's on shared cases; grep trace.jsonl for mutable_log and error events.\n"
        f"  3. Write $WORK_DIR/{CURATION_FILE} with the required structure.\n"
        f"  4. Call {SUBMIT_CURATION_NAME}.\n\n"
        "## Budget\n"
        f"  - at most {max_llm_calls} model calls and {timeout_s:g}s wall-clock for this session\n"
        f"  - at most {max_attempts} submission attempts\n"
    )


# ---------------------------------------------------------------------- #
# Memory generator (one call)
# ---------------------------------------------------------------------- #

EDIT_MEMORY_CORE_INSTRUCTION = (
    f"{CORE_SENTINEL}\n"
    "You maintain the edit memory of a self-evolving agent: a document that a "
    "meta-agent (the editor) reads before it produces the next version of a "
    "task agent's code. You are given the current memory (possibly empty), a "
    "curator's review of the newest nodes (their edits, evidence and a "
    "gradient section stating which memory claims were confirmed, "
    "contradicted, too coarse or missing), and this instruction. Write the "
    "new memory as one complete document with exactly these sections:\n\n"
    f"{MEMORY_SECTIONS[0]}\n"
    "Every edit seen so far in this run, ranked by judged usefulness (most "
    "useful first). One entry per edit: node id(s), the mechanism in one "
    "line, the verdict (helped / unclear / hurt) and the evidence in a few "
    "words. Merge the new nodes into the existing ranking; keep entries "
    "from the previous memory unless the review contradicts them.\n\n"
    f"{MEMORY_SECTIONS[1]}\n"
    "Why the top-ranked edits helped and what evidence supports it; which "
    "edits are unclear and what evidence would settle them.\n\n"
    f"{MEMORY_SECTIONS[2]}\n"
    "Ideas that were good but implemented poorly (and how the implementation "
    "fell short); implementations that were fine but the idea did not "
    "address what fails; open shortcomings worth fixing next.\n\n"
    f"{MEMORY_SECTIONS[3]}\n"
    "What to build on, what to avoid, what is untested — written for an "
    "editor that will read the cited nodes' code itself.\n\n"
    "Rules: change the previous memory minimally — answer the gradient, do "
    "not rewrite what still holds. Cite node ids everywhere. Never state a "
    "predicted score, an expected gain, or any number that forecasts a "
    "future evaluation: rank by judged usefulness from evidence. Stay within "
    "{max_chars} characters. Output the document only, no preamble.\n"
    "=== END OF FIXED CORE ==="
)


def render_memory_generation_messages(
    *,
    previous_memory: str,
    curation: str,
    addendum: str,
    window_meta: dict[str, Any],
    max_chars: int,
    rejection: Optional[list[str]] = None,
) -> list[dict[str, str]]:
    system = EDIT_MEMORY_CORE_INSTRUCTION.format(max_chars=max_chars)
    if addendum.strip():
        system += ("\n\n=== ADDENDUM (learned guidance for this run) ===\n"
                   + addendum.strip())
    nodes = ", ".join(
        f"node {n['node_id']} (arm {n['memory_arm']}"
        + (f", memory v{n['memory_version']}" if n.get("memory_version") is not None else "")
        + ")"
        for n in window_meta.get("nodes", [])
    )
    user = (
        f"# Window {window_meta.get('window_index')}: {nodes}\n\n"
        "# Current edit memory\n"
        + (previous_memory.strip() if previous_memory.strip() else "(none yet — this is the first memory)")
        + "\n\n# Curator's review of this window\n" + curation.strip() + "\n"
    )
    if rejection:
        user += ("\n# Your previous draft was rejected\n"
                 + "\n".join(f"- {r}" for r in rejection)
                 + "\nWrite the complete document again, fixing these.\n")
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


# ---------------------------------------------------------------------- #
# Instruction curator
# ---------------------------------------------------------------------- #

INSTRUCTION_CURATOR_SYSTEM = (
    "You audit how the edit memory of a self-evolving agent is being used, in "
    "order to improve the instruction under which that memory is written. A "
    "meta-agent (the editor) produced new versions of a task agent's code "
    "while reading the memory; each version was evaluated on benchmark "
    f"cases. You have three tools: `bash`, `editor`, and `{SUBMIT_CURATION_NAME}`.\n"
    "Three questions, answered from evidence: (a) what did editors look for "
    "in the memory and not find, or misread, or ignore — what information is "
    "missing from it; (b) did the memory-guided edits actually fire and help "
    "when the task agent ran — are they too coarse, is the task agent unable "
    "to use them, does the memory need finer-grained or more concrete "
    "entries; (c) would a different representation of the memory (structure, "
    "granularity, length, what it cites) serve the editor better.\n"
    "The major instruction is fixed; you propose changes to its addendum "
    "only, and only where the evidence calls for them. Never propose that "
    "the memory predict scores.\n"
    "Environment:\n"
    "  - Every bash call is a fresh shell in $WORK_DIR with no network; only "
    "the roots listed in the task message exist.\n"
    "  - $WORK_DIR is the only writable place. Write your document with the "
    "`editor` tool.\n"
    "  - Budget discipline: spend at most half of the call budget reading, "
    "then write.\n\n"
    f"Finishing: when $WORK_DIR/{Q_FILE} is complete, call "
    f"`{SUBMIT_CURATION_NAME}` with a one-paragraph summary. You must end the "
    "session by submitting; do not stop with a plain message."
)


def render_instruction_curation_instruction(
    *,
    nodes: list[dict[str, Any]],
    addendum: str,
    max_llm_calls: int,
    timeout_s: float,
    max_attempts: int,
) -> str:
    rows = ["  var        node  parent  memory  evals  mean score (context only)"]
    for i, n in enumerate(nodes, start=1):
        rows.append(
            f"  $NODE_{i:<5} {n['node_id']:>4}  {n['parent_id']:>6}  "
            f"v{n.get('memory_version')!s:<6} {n['n_evals']:>5}  {n['mean_utility']:.3f}   (parent: $PARENT_{i})"
        )
    q_sections = "\n".join(f"  {h}" for h in Q_SECTIONS)
    return (
        "# Task\n"
        f"Audit the {len(nodes)} nodes below — every one was produced by an editor "
        "that had the edit memory ($MEMORY_DIR/edit_memory_v<version>.md, the "
        "version in the table) in its workspace — and write "
        f"$WORK_DIR/{Q_FILE}: how the memory was used, whether the edits it "
        "guided worked at task-agent runtime, what is missing from it, and what "
        "should change in the instruction that generates it.\n\n"
        "## Nodes expanded with the memory since the last instruction update\n"
        + "\n".join(rows) + "\n\n"
        "## Roots (environment variables in every bash call; the editor tool "
        "accepts the same $VAR form)\n"
        "  WORK_DIR     your output directory (the only writable path)\n"
        "  MEMORY_DIR   the run's edit-memory directory: edit_memory_v*.md (the memory "
        "versions), instruction.md (the CURRENT addendum you are auditing), "
        "instruction_v*.md, window_*/curation.md (the curators' reviews)\n"
        "  NODE_i       round dir of the i-th node; PARENT_i its parent's\n"
        "  REPO_DIR     the repository (platform_core/, projects/<p>/tools/, db_schema.md)\n\n"
        "## What every node / parent dir contains\n" + _legend() + "\n\n"
        "Evidence roles:\n"
        "  - the editors' trajectories: $NODE_i/agentic/transcript.jsonl — did the "
        "session read $EDIT_MEMORY_FILE, which parts did it act on, what did it "
        "search for and not find, did it ignore or misread the memory;\n"
        "  - the task agents' trajectories: $NODE_i/logs/case_<id>.json split by "
        "`passed` (positive vs negative cases), and the mutable_log verdicts in "
        "$NODE_i/logs/trace.jsonl — did the memory-guided mechanisms fire, help, "
        "or never trigger;\n"
        "  - the memory versions and the current addendum in $MEMORY_DIR.\n\n"
        "## Current instruction addendum\n"
        + (addendum.strip() if addendum.strip() else "(empty — the generator runs on its fixed core only)")
        + f"\n\n## Required structure of $WORK_DIR/{Q_FILE}\n" + q_sections + "\n"
        "Under 'Proposed instruction changes': for each bullet of the current addendum "
        "keep / change (how) / drop, plus bullets to add — concrete and minimal, each "
        "tied to the evidence above. No proposal may ask the memory to predict scores.\n\n"
        "## Procedure\n"
        "  1. Read $MEMORY_DIR/instruction.md and the memory version(s) the nodes saw.\n"
        "  2. For each node: grep its transcript for EDIT_MEMORY_FILE and the memory's "
        "wording; read strategy.json; compare its cases with the parent's; grep trace.jsonl "
        "for mutable_log events.\n"
        f"  3. Write $WORK_DIR/{Q_FILE} with the required structure.\n"
        f"  4. Call {SUBMIT_CURATION_NAME}.\n\n"
        "## Budget\n"
        f"  - at most {max_llm_calls} model calls and {timeout_s:g}s wall-clock for this session\n"
        f"  - at most {max_attempts} submission attempts\n"
    )


# ---------------------------------------------------------------------- #
# Instruction updater (one call)
# ---------------------------------------------------------------------- #

INSTRUCTION_UPDATE_SYSTEM = (
    "You maintain the addendum of the instruction under which the edit memory "
    "of a self-evolving agent is generated. The fixed core of that "
    "instruction is shown for context and cannot be changed. You are given "
    "the current addendum and an auditor's report (how the memory was used, "
    "its effect on the task agents, what is missing, representation issues, "
    "proposed changes). Output the NEW addendum only: a short list of "
    "concrete guidance bullets for the memory generator — what the memory "
    "must additionally contain, at what granularity, in what form. Change "
    "the current addendum minimally: keep what the report confirms, apply "
    "the changes it justifies, drop what it refutes. Never ask the memory to "
    "predict scores. Stay within {max_chars} characters. Do not repeat the "
    "fixed core; do not include the sentinel line. Output the addendum only, "
    "no preamble."
)


def render_instruction_update_messages(
    *,
    addendum: str,
    q: str,
    previous_q: list[str],
    max_chars: int,
    rejection: Optional[list[str]] = None,
) -> list[dict[str, str]]:
    system = INSTRUCTION_UPDATE_SYSTEM.format(max_chars=max_chars)
    user = (
        "# Fixed core (context only, immutable)\n"
        + EDIT_MEMORY_CORE_INSTRUCTION.format(max_chars="N") + "\n\n"
        "# Current addendum\n"
        + (addendum.strip() if addendum.strip() else "(empty)")
        + "\n\n# Auditor's report\n" + q.strip() + "\n"
    )
    if previous_q:
        user += "\n# Earlier reports (older first; for momentum, not to be re-applied)\n"
        for i, pq in enumerate(previous_q, start=1):
            user += f"\n## Earlier report {i}\n{pq.strip()}\n"
    if rejection:
        user += ("\n# Your previous draft was rejected\n"
                 + "\n".join(f"- {r}" for r in rejection)
                 + "\nWrite the addendum again, fixing these.\n")
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]
