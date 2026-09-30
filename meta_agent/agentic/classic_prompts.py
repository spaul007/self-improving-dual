"""Single-agent (include-list) prompt fragments of the agentic editor.

Verbatim from the sep18 source (``edit-memory-sep18-agentic`` branch
``agentic-clean`` @ b08cb8b, ``meta_agent/agent_editor.py:78-143``), where the
agentic editor was built for the single-agent ``projects/travel`` layout
(``workflow.py`` / ``tool_wrapper.py`` / ``tools_schema.json`` /
``mutable_tools/``). Kept byte-identical so the ported editor reproduces
that repo's prompts exactly on that layout (``tests/test_agentic_golden.py``).
Exclude-list (multi-agent) projects use ``agent_editor.exclude_surface_and_core_rules``
instead -- see ``session.agentic_system_prompt``.
"""
from __future__ import annotations

from ..editor_validators import MUTABLE_FILES

# Prompt fragments the agentic editor's system prompt is assembled from. One
# home so tests/test_editor_import_contract.py — which pins that the prompt
# and the validators name the same platform paths — covers them.
EDITOR_MUTABLE_SURFACE = (
    "You may only modify these "
    "files in the task_agent workspace:\n"
    f"  - {', '.join(sorted(MUTABLE_FILES))}\n"
    f"  - any *.py file under mutable_tools/\n\n"
)

EDITOR_HARD_RULES = (
    "Hard rules:\n"
    "  1. workflow.py MUST define "
    "`def run_task(task: Task) -> AgentOutput`. The single arg must "
    "be named `task` (the validator enforces this).\n"
    "  2. workflow.py may import only: "
    "platform_core.llm_wrapper.call_llm, platform_core.runner "
    "(Task, AgentOutput), platform_core.trace, tool_wrapper, plus "
    "stdlib. It must NOT import platform_core.tools — workflow.py "
    "reaches tools through tool_wrapper.\n"
    "  3. tool_wrapper.py may import only: platform_core.tools, "
    "platform_core.trace, mutable_tools.*, plus stdlib.\n"
    "  4. Mutable tools (mutable_tools/*.py) may import only: "
    "platform_core.tools, platform_core.trace, sibling "
    "mutable_tools.*, plus stdlib.\n"
    "  5. Reach capabilities only via `platform_core.tools`: "
    "`call_tool(name, **kwargs)` for immutable tools, and "
    "`call_mutable_tool(name, **kwargs)` for `mutable_tools/*`. Those "
    "live in tool_wrapper.py and mutable_tools/*.py; from workflow.py, "
    "go through tool_wrapper. Never "
    "invoke a mutable tool's `run()` directly — routing through "
    "`call_mutable_tool` keeps its calls recorded in the trace.\n"
    "  6. tools_schema.json: every entry's `name` must be backed by "
    "either an immutable tool OR a `mutable_tools/<name>.py` file. "
    "No collisions between immutable and mutable names.\n"
    "  7. Instrument your edits. When you "
    "add a verifier, helper, or decision branch in workflow.py or a "
    "mutable_tools/*.py file, call "
    "`platform_core.trace.log(label='your_label', verdict='pass'|'fail'|'skip', "
    "name='specific_check_name', **context)` at the decision point. "
    "Conventions: pick a stable `label` (e.g. 'verifier_fired', "
    "'decision_branch'); use `name` to disambiguate within a label; "
    "set `verdict` to `pass`, `fail`, or `skip`. After evaluation these "
    "logs are cross-tabulated against case outcomes, so it becomes possible "
    "to tell which of your additions fired, which helped, and which did not. "
    "Skip instrumentation only for trivial edits (renames, docstrings).\n\n"
)


def editor_import_forms(
    wrapper_ref: str = "The current tool_wrapper.py shown below",
) -> str:
    """The IMPORTS — EXACT FORMS block. ``wrapper_ref`` names where the
    reader will see tool_wrapper.py's wrapper-only spelling: inlined below
    (single-shot editors) or on disk (agentic editor)."""
    return (
        "IMPORTS — EXACT FORMS. The validator rejects the *bare* package "
        "`platform_core`, so the statement form matters, not just the "
        "module you mean:\n"
        "  OK    from platform_core.trace import log\n"
        "  OK    import platform_core.trace\n"
        "  BAD   from platform_core import trace     <- imports bare "
        "'platform_core'; rejected\n"
        "  BAD   import platform_core                <- same\n"
        f"{wrapper_ref} contains "
        "`from platform_core import tools`. That spelling is accepted "
        "ONLY in tool_wrapper.py. Do not copy it into workflow.py or a "
        "mutable tool — use the dotted-submodule form there.\n\n"
    )
