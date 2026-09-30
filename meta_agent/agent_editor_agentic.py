"""Agentic editor — the meta-agent as an HGM-seed-style coding agent.

Instead of one LLM call that must emit full replacement files, the model
runs a tool-use session (``bash`` in a bubblewrap sandbox, a targeted-edit
``editor``, ``validate``) directly on the round's ``task_agent/`` copy, and
ends by calling ``submit_self_improvement`` with a summary. There is one
instruction prompt: roots as ``$VAR`` paths, workspace map, procedure,
budget. No feedback digest, no steering context — the agent reads the
parent node's evidence from disk itself. ``read_scope`` decides how far it
may look: ``"run"`` exposes every node of the run, ``"parent"`` only the
parent node and its own workspace (see ``meta_agent/agentic/policy.py``).

Ported from the sep18 line (``agentic-clean`` @ b08cb8b), where it was built
for the single-agent ``projects/travel`` layout. Two additions for block HGM:

- **Exclude-list surface** (``mutable_exclude``, multi-agent projects such as
  ``travel_mas_refactored``): everything under ``task_agent/`` is writable
  except the excluded paths, which the sandbox re-binds read-only; a
  built-in scope check rejects any change to them (bash writes bypass the
  editor tool). With ``mutable_exclude`` unset the prompts and behavior are
  sep18's, byte for byte.
- **Assignment** (``steering="assignment"``): the manager passes the selected
  block (and, when those axes are on, implementation strategy / curriculum
  focus / an advisory suggestion) as an ``ExpandAssignment``; the editor
  renders it as ``## Selected block for this EXPAND`` in the instruction.

Contract to the managers: ``apply(feedback, base_dir, out_dir, ...)`` →
``EditResult`` with the final code under ``out_dir/task_agent``. Session
artifacts land in ``out_dir/agentic/``.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence, Union

from . import verbose_log
from .agent_editor import AgentEditor, Validator
from .agentic.policy import (
    ALWAYS_IGNORE_DIRS,
    MEMORY_DIR_NAME,
    READ_SCOPE_RUN,
    READ_SCOPES,
    REPO_ROOT,
    build_policy,
    find_run_root,
)
from .agentic.sandbox import Sandbox
from .agentic.session import (
    SESSION_NAME,
    TRANSCRIPT_NAME,
    AgenticSession,
    SessionConfig,
    Transcript,
    agentic_system_prompt,
    editor_submit_spec,
    render_instruction,
    strategies_note,
)
from .agentic.tools import (
    BashTool,
    EditorTool,
    ToolSet,
    ValidateTool,
    bash_tool_info,
    editor_tool_info,
    validate_tool_info,
)
from .assignment import ExpandAssignment
from .edit_diff import changed_agent_files, changed_mutable_files
from .editor_validators import is_excluded
from .models import AgentFeedback, EditResult
from .registry import get as registry_get, register

STEERING_MODES = ("assignment", "none")
PROMPT_SYSTEM_FILE = "prompt_system.txt"
PROMPT_INSTRUCTION_FILE = "prompt_instruction.txt"
STRATEGIES_LABEL = "curated fix patterns (reference material, not rules)"


@register("editor", "agentic")
class AgenticEditor(AgentEditor):
    """``AgentEditor`` whose self-improvement step is a tool-use session.

    The ctor re-declares every injectable kwarg explicitly:
    ``config._build_with_injection`` injects by signature name.
    ``tools_source`` / ``db_schema`` / ``scorer_source`` are accepted for
    compatibility but never inlined — the agent reads tools and schema from
    disk, and the scorer is never exposed.
    """

    def __init__(
        self,
        llm_caller: Callable[..., object],
        validators: Iterable[Validator],
        *,
        max_attempts: int = 3,
        model: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        base_url: Optional[str] = None,
        tools_source: Optional[str] = None,
        db_schema: Optional[str] = None,
        scorer_source: Optional[str] = None,
        mutable_exclude: Optional[list[str]] = None,
        project_root: Optional[Union[str, Path]] = None,
        max_llm_calls: int = 40,
        timeout_s: float = 1800.0,
        bash_timeout_s: float = 120.0,
        max_tool_output_chars: int = 20000,
        max_view_chars: int = 40000,
        sandbox: str = "auto",
        transcript_result_chars: int = 4000,
        include_manager_context: bool = False,
        api_key_env: Optional[str] = None,
        llm_timeout_s: Optional[float] = None,
        # Merged into every request body of this editor's calls, e.g.
        # OpenRouter's provider pin {"provider": {"order": ["Baidu"],
        # "allow_fallbacks": false}}.
        extra_body: Optional[dict[str, Any]] = None,
        # Per-call output cap. None = call_llm's default, which in this repo
        # falls back to the task agent's exported LLM_MAX_OUTPUT_TOKENS.
        max_output_tokens: Optional[int] = None,
        # How much of the run the meta-agent may read (bash + editor tool):
        # "run" — the whole run directory, every node's code and evidence;
        # "parent" — only $PARENT_DIR and its own $NODE_DIR (no $RUN_DIR root).
        read_scope: str = "run",
        # "assignment": the manager passes an ExpandAssignment (selected block
        # etc.) instead of its prose steering context. "none": no steering.
        steering: str = "assignment",
        # Validators the `validate` tool skips (still run on submit): slow
        # ones such as smoke_test, which runs a real benchmark case.
        validate_exclude: Sequence[str] = ("smoke_test",),
        # Repo-relative paths: strategies.md (readable reference, pointed at
        # by the assignment) and a short orientation text for multi-agent
        # projects, inlined in the system prompt.
        strategies_path: Optional[str] = None,
        project_overview_path: Optional[str] = None,
    ) -> None:
        super().__init__(
            llm_caller, validators, max_attempts=max_attempts, model=model,
            reasoning_effort=reasoning_effort, base_url=base_url,
            tools_source=tools_source, db_schema=db_schema,
            scorer_source=scorer_source, mutable_exclude=mutable_exclude,
            max_output_tokens=max_output_tokens,
        )
        if read_scope not in READ_SCOPES:
            raise ValueError(f"editor read_scope must be one of "
                             f"{sorted(READ_SCOPES)}, got {read_scope!r}")
        if steering not in STEERING_MODES:
            raise ValueError(f"editor steering must be one of {STEERING_MODES}, "
                             f"got {steering!r}")
        self.read_scope = read_scope
        self.steering = steering
        self.project_root = Path(project_root) if project_root else None
        self.max_llm_calls = int(max_llm_calls)
        self.timeout_s = float(timeout_s)
        self.bash_timeout_s = float(bash_timeout_s)
        self.max_tool_output_chars = int(max_tool_output_chars)
        self.max_view_chars = int(max_view_chars)
        self.sandbox = sandbox
        self.transcript_result_chars = int(transcript_result_chars)
        self.include_manager_context = bool(include_manager_context)
        # Second-provider support: read the key for THIS editor's calls from a
        # different env var (e.g. OpenRouter_API_KEY from api.sh) while the
        # task agent keeps the global OPENAI_API_KEY, and cap each request so
        # one stalled call cannot eat the whole session budget.
        self.api_key_env = api_key_env or None
        self.llm_timeout_s = float(llm_timeout_s) if llm_timeout_s else None
        self.extra_body = dict(extra_body) if extra_body else None
        self.validate_exclude = tuple(validate_exclude or ())
        excluded_classes = tuple(registry_get("validator", n) for n in self.validate_exclude)
        self._quick_validators = [v for v in self.validators
                                  if not isinstance(v, excluded_classes)]
        # Only name skipped validators that are actually configured.
        self.validate_skipped = tuple(
            n for n, cls in zip(self.validate_exclude, excluded_classes)
            if any(isinstance(v, cls) for v in self.validators)
        )
        self.hardcode_rule = any(
            getattr(v, "rejects_hardcoded_answers", False) for v in self.validators
        )
        self.strategies_path = (REPO_ROOT / strategies_path) if strategies_path else None
        if self.strategies_path is not None and not self.strategies_path.is_file():
            raise FileNotFoundError(f"editor strategies_path not found: {self.strategies_path}")
        self.project_overview: Optional[str] = None
        if project_overview_path:
            path = REPO_ROOT / project_overview_path
            if not path.is_file():
                raise FileNotFoundError(f"editor project_overview_path not found: {path}")
            self.project_overview = path.read_text(encoding="utf-8")

    @property
    def exclude_mode(self) -> bool:
        return self.mutable_exclude is not None

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
        # Accepted for manager compatibility; an assignment carries any
        # suggestion itself.
        has_suggestion: bool = False,
        memory_path: Optional[Path] = None,
        assignment: Optional[ExpandAssignment] = None,
    ) -> EditResult:
        base_dir, out_dir = Path(base_dir), Path(out_dir)
        memory_path = Path(memory_path) if memory_path else None
        self._copy_workspace(base_dir, out_dir)
        if not self.exclude_mode:
            (out_dir / "task_agent" / "mutable_tools").mkdir(exist_ok=True)
        agentic_dir = out_dir / "agentic"
        if agentic_dir.exists():
            shutil.rmtree(agentic_dir)
        (agentic_dir / "scratch").mkdir(parents=True)

        # The run's edit_memory/ (if any) is always masked; on the with-memory
        # arm the one memory file comes back as $EDIT_MEMORY_FILE and the whole
        # run is readable regardless of read_scope, so the editor can open the
        # code of every node the memory cites. The configured scope therefore
        # governs the without-memory arm only.
        run_root = find_run_root(out_dir)
        memory_dir = (run_root / MEMORY_DIR_NAME) if run_root is not None else None
        if memory_path is not None and memory_dir is None:
            memory_dir = memory_path.parent
        read_scope = READ_SCOPE_RUN if memory_path is not None else self.read_scope
        extra_reference = (
            ((STRATEGIES_LABEL, self.strategies_path),) if self.strategies_path else ()
        )
        policy = build_policy(
            out_dir=out_dir, base_dir=base_dir, repo_root=REPO_ROOT,
            project_root=self.project_root,
            project_name=(self.project_root.name if self.project_root
                          else os.environ.get("META_AGENT_PROJECT", "")),
            read_scope=read_scope,
            memory_dir=memory_dir if (memory_dir and memory_dir.exists()) or memory_path else None,
            memory_file=memory_path,
            mutable_exclude=self.mutable_exclude,
            evidence_hints=assignment is not None,
            extra_reference=extra_reference,
        )
        strategies_ref = policy.var_path(self.strategies_path) if self.strategies_path else None
        sandbox = Sandbox(policy, mode=self.sandbox, bash_timeout_s=self.bash_timeout_s)
        # Decide the confinement up front: mode="bwrap" fails fast here when
        # bubblewrap is unusable, mode="auto" prints its fallback warning once.
        sandbox_mode = sandbox.effective_mode
        toolset = ToolSet([
            (bash_tool_info(bash_timeout_s=self.bash_timeout_s,
                            max_output_chars=self.max_tool_output_chars,
                            root_vars=policy.root_vars(),
                            exclude_mode=self.exclude_mode),
             BashTool(sandbox, max_output_chars=self.max_tool_output_chars)),
            (editor_tool_info(max_view_chars=self.max_view_chars,
                              exclude_mode=self.exclude_mode),
             EditorTool(policy, max_view_chars=self.max_view_chars)),
            (validate_tool_info(self.validate_skipped),
             ValidateTool(lambda: self._run_quick_validators(out_dir, base_dir))),
        ])
        cfg = SessionConfig(
            max_llm_calls=self.max_llm_calls, timeout_s=self.timeout_s,
            max_attempts=self.max_attempts,
            transcript_result_chars=self.transcript_result_chars,
            model=self.model, reasoning_effort=self.reasoning_effort,
            base_url=self.base_url, api_key_env=self.api_key_env,
            llm_timeout_s=self.llm_timeout_s, extra_body=self.extra_body,
            max_output_tokens=self.max_output_tokens,
        )
        session = AgenticSession(
            self.llm, toolset,
            submit=editor_submit_spec(
                run_validators=lambda: self._run_validators(out_dir, base_dir),
                changed_files=lambda: self._changed_files(out_dir, base_dir),
                exclude_mode=self.exclude_mode,
            ),
            cfg=cfg,
            transcript=Transcript(agentic_dir / TRANSCRIPT_NAME),
        )
        system_prompt = agentic_system_prompt(
            read_scope,
            mutable_exclude=self.mutable_exclude,
            project_overview=self.project_overview,
            validate_skipped=self.validate_skipped,
            hardcode_rule=self.hardcode_rule,
            strategies_ref=strategies_ref,
        )
        strategies_line = (
            strategies_note(self.strategies_path, assignment.block, strategies_ref)
            if assignment is not None and self.strategies_path is not None else None
        )
        instruction = render_instruction(
            policy, max_llm_calls=self.max_llm_calls, timeout_s=self.timeout_s,
            max_attempts=self.max_attempts,
            manager_context=context if self.include_manager_context else None,
            assignment=assignment,
            strategies_line=strategies_line,
        )
        (agentic_dir / PROMPT_SYSTEM_FILE).write_text(system_prompt, encoding="utf-8")
        (agentic_dir / PROMPT_INSTRUCTION_FILE).write_text(instruction, encoding="utf-8")
        if verbose_log.is_enabled():
            verbose_log.write_text(out_dir, "editor_agentic_system.txt", system_prompt)
            verbose_log.write_text(out_dir, "editor_agentic_instruction.txt", instruction)

        result = session.run(system_prompt, instruction)

        summary = result.to_dict()
        summary["sandbox_mode"] = sandbox_mode
        summary["read_scope"] = read_scope
        summary["memory_path"] = str(memory_path) if memory_path else None
        summary["roots"] = {k: str(v) for k, v in policy.roots().items()}
        if assignment is not None:
            summary["assignment_block"] = assignment.block
            summary["assignment_implementation_strategy"] = assignment.implementation_strategy
        (agentic_dir / SESSION_NAME).write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        if verbose_log.is_enabled():
            verbose_log.write_json(out_dir, "editor_agentic_messages.json", session.messages)
        return EditResult(
            success=result.success, errors=list(result.errors),
            edited_files=list(result.changed_files), strategy=result.strategy,
        )

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _changed_files(self, out_dir: Path, base_dir: Path) -> list[str]:
        if self.exclude_mode:
            return changed_agent_files(base_dir, out_dir, mutable_exclude=self.mutable_exclude)
        return changed_mutable_files(base_dir, out_dir)

    def _run_validators(self, out_dir: Path, base_dir: Path) -> list[str]:
        """Submit / fallback: the scope check (exclude mode) plus every
        configured validator."""
        return self._scope_errors(out_dir, base_dir) + super()._run_validators(out_dir, base_dir)

    def _run_quick_validators(self, out_dir: Path, base_dir: Path) -> list[str]:
        """The ``validate`` tool: as ``_run_validators`` minus
        ``validate_exclude`` (slow validators that run only on submit)."""
        errors = self._scope_errors(out_dir, base_dir)
        for validator in self._quick_validators:
            errors.extend(validator.validate(out_dir, base_dir))
        return errors

    def _scope_errors(self, out_dir: Path, base_dir: Path) -> list[str]:
        """Exclude mode only: every file added, modified or deleted under an
        excluded path, and every new symlink anywhere in task_agent/ (the
        task agent later runs unsandboxed). bash can write the whole tree, so
        this is what enforces the surface, whether or not the
        ``immutable_files`` validator is configured. Single-agent mode keeps
        sep18's behavior (the configured validators alone)."""
        if not self.exclude_mode:
            return []
        new_root, old_root = Path(out_dir) / "task_agent", Path(base_dir) / "task_agent"
        new, old = _tree(new_root), _tree(old_root)
        errors: list[str] = []
        for rel in sorted(set(new) | set(old)):
            n, o = new.get(rel), old.get(rel)
            if n is not None and n.is_symlink() and not (o is not None and o.is_symlink()):
                errors.append(f"scope violation: {rel} is a new symlink; symlinks may not be "
                              "added to the agent")
                continue
            if not is_excluded(rel, list(self.mutable_exclude or ())):
                continue
            if n is None or o is None:
                what = "deleted" if n is None else "created"
            elif _bytes(n) != _bytes(o):
                what = "modified"
            else:
                continue
            errors.append(f"scope violation: {rel} was {what} but is excluded from editing "
                          f"({', '.join(sorted(self.mutable_exclude or ()))}); restore it")
        return errors


def _tree(root: Path) -> dict[str, Path]:
    """Every file and symlink under ``root`` (relative POSIX path -> path),
    skipping ``__pycache__`` / ``results`` output."""
    if not root.exists():
        return {}
    out: dict[str, Path] = {}
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        rel_dir = Path(dirpath).relative_to(root)
        if set(rel_dir.parts) & ALWAYS_IGNORE_DIRS:
            dirnames[:] = []
            continue
        for name in list(dirnames):
            p = Path(dirpath) / name
            if p.is_symlink():
                out[(rel_dir / name).as_posix()] = p
        for name in filenames:
            p = Path(dirpath) / name
            out[(rel_dir / name).as_posix()] = p
    return out


def _bytes(p: Path) -> bytes:
    try:
        if p.is_symlink():
            return b"symlink:" + os.readlink(p).encode()
        return p.read_bytes()
    except OSError:
        return b""
