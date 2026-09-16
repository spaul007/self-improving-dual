"""Editor base: the mutable-surface contract, the shared prompt fragments and
the workspace / validator plumbing every editor kind builds on.

The only concrete editor is ``meta_agent.agent_editor_agentic.AgenticEditor``
(registered as ``editor: agentic``): it subclasses ``AgentEditor`` and runs a
tool-use session (bash + editor + validate + submit) in a copy of the parent
node's ``task_agent/``. ``apply`` is the contract the managers call:
``EditResult.success`` says whether a validated edit exists in ``out_dir``,
and ``EditResult.strategy`` carries the editor's own summary of the edit,
which the gatherer persists as ``strategy.json``.
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Protocol

from .editor_validators import MUTABLE_DIRS, MUTABLE_FILES
from .models import AgentFeedback, EditResult, EvolutionStrategy


class Validator(Protocol):
    def validate(self, out_dir: Path, base_dir: Path) -> list[str]: ...


# Allowed values for an ``EvolutionStrategy.target_files`` entry. Mirrors the
# Literal in models.py. Lives here (the single home) so the coercion helpers
# below — and the managers, via import — share one definition.
_ALLOWED_TARGET_FILES = ("workflow.py", "tool_wrapper.py", "tools_schema.json")


def _coerce_target_files(value: Any) -> list[str]:
    """Coerce a model's ``target_files``-shaped value into a clean list[str].

    Models without strict schema enforcement (notably local vLLM-hosted
    open-weights models — caught gpt-oss-120b returning the bare string
    ``"workflow.py"`` on 2026-05-12) sometimes return a single string
    instead of a list. Unknown values are dropped; empty/None falls back to
    ``["workflow.py"]`` so a strategy summary always validates.
    """
    if value is None or value == "":
        return ["workflow.py"]
    if isinstance(value, str):
        candidates = [value]
    elif isinstance(value, (list, tuple)):
        candidates = [str(v) for v in value if v]
    else:
        return ["workflow.py"]
    cleaned = [c for c in candidates if c in _ALLOWED_TARGET_FILES]
    return cleaned or ["workflow.py"]


def _coerce_str(value: Any) -> str:
    """Coerce a response field into a string — guards against a model
    returning the wrong scalar type (number/bool/None) for a text field."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def fallback_strategy() -> EvolutionStrategy:
    """A constant placeholder ``EvolutionStrategy`` — used by managers when
    an ``EditResult`` carries no summary (defensive; ``apply`` always sets
    one, so this normally never fires)."""
    return EvolutionStrategy(
        target_files=["workflow.py"],
        optimization_goal="(editor produced no strategy summary)",
        proposed_changes="",
        rationale="",
    )


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


class AgentEditor:
    """Base class: holds the injected LLM caller, validators and the static
    project context, and provides the workspace copy + validator run every
    editor needs. Subclasses implement :meth:`apply`."""

    MUTABLE_FILES = MUTABLE_FILES
    MUTABLE_DIRS = MUTABLE_DIRS

    def __init__(
        self,
        llm_caller: Callable[..., object],
        validators: Iterable[Validator],
        *,
        max_attempts: int = 2,
        model: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        base_url: Optional[str] = None,
        # Static project context injected by build_components (read from the
        # project folder by convention). scorer_source is injected only when
        # eval_visibility == "whitebox". The agentic editor accepts these for
        # the shared injection contract but reads tools and schema from disk.
        tools_source: Optional[str] = None,
        db_schema: Optional[str] = None,
        scorer_source: Optional[str] = None,
    ) -> None:
        self.llm = llm_caller
        self.validators = list(validators)
        self.max_attempts = max_attempts
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.base_url = base_url
        self.tools_source = tools_source
        self.db_schema = db_schema
        self.scorer_source = scorer_source

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def apply(
        self,
        feedback: Optional[AgentFeedback],
        base_dir: Path,
        out_dir: Path,
        *,
        context: Optional[str] = None,
        memory_path: Optional[Path] = None,
    ) -> EditResult:
        """Produce one self-improvement of the agent in ``base_dir`` into
        ``out_dir/task_agent``. ``context`` is optional manager-supplied
        steering text (lineage, scores); ``memory_path`` is the run's edit
        memory file when this expansion is on the with-memory arm. Returns
        ``EditResult``; ``.strategy`` carries the editor's emitted summary."""
        raise NotImplementedError("use a registered editor (editor.type: agentic)")

    # ------------------------------------------------------------------ #
    # Internals shared by editors
    # ------------------------------------------------------------------ #

    def _copy_workspace(self, base_dir: Path, out_dir: Path) -> None:
        src = base_dir / "task_agent"
        dst = out_dir / "task_agent"
        if dst.exists():
            shutil.rmtree(dst)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dst)

    def _is_path_allowed(self, rel_path: str) -> bool:
        if rel_path in MUTABLE_FILES:
            return True
        parts = Path(rel_path).parts
        if len(parts) == 2 and parts[0] in MUTABLE_DIRS and parts[1].endswith(".py"):
            return True
        return False

    def _run_validators(self, out_dir: Path, base_dir: Path) -> list[str]:
        errors: list[str] = []
        for validator in self.validators:
            errors.extend(validator.validate(out_dir, base_dir))
        return errors
