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

Contract to the managers is unchanged: ``apply(feedback, base_dir, out_dir,
context=...)`` → ``EditResult`` with the final code under
``out_dir/task_agent``. Session artifacts land in ``out_dir/agentic/``.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Union

from . import verbose_log
from .agent_editor import AgentEditor, Validator
from .agentic.policy import (
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
)
from .agentic.tools import (
    VALIDATE_TOOL,
    BashTool,
    EditorTool,
    ToolSet,
    ValidateTool,
    bash_tool_info,
    editor_tool_info,
)
from .edit_diff import changed_mutable_files
from .models import AgentFeedback, EditResult
from .registry import register


@register("editor", "agentic")
class AgenticEditor(AgentEditor):
    """``AgentEditor`` whose self-improvement step is a tool-use session.

    The ctor re-declares every injectable kwarg explicitly (see
    ``TwoStageEditor``): ``config._build_with_injection`` injects by
    signature name. ``tools_source`` / ``db_schema`` / ``scorer_source`` are
    accepted for compatibility but never inlined — the agent reads tools and
    schema from disk, and the scorer is never exposed.
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
        # How much of the run the meta-agent may read (bash + editor tool):
        # "run" — the whole run directory, every node's code and evidence;
        # "parent" — only $PARENT_DIR and its own $NODE_DIR (no $RUN_DIR root).
        read_scope: str = "run",
    ) -> None:
        super().__init__(
            llm_caller, validators, max_attempts=max_attempts, model=model,
            reasoning_effort=reasoning_effort, base_url=base_url,
            tools_source=tools_source, db_schema=db_schema,
            scorer_source=scorer_source,
        )
        if read_scope not in READ_SCOPES:
            raise ValueError(f"editor read_scope must be one of "
                             f"{sorted(READ_SCOPES)}, got {read_scope!r}")
        self.read_scope = read_scope
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
        base_dir, out_dir = Path(base_dir), Path(out_dir)
        memory_path = Path(memory_path) if memory_path else None
        self._copy_workspace(base_dir, out_dir)
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
        policy = build_policy(
            out_dir=out_dir, base_dir=base_dir, repo_root=REPO_ROOT,
            project_root=self.project_root,
            project_name=(self.project_root.name if self.project_root
                          else os.environ.get("META_AGENT_PROJECT", "")),
            read_scope=read_scope,
            memory_dir=memory_dir if (memory_dir and memory_dir.exists()) or memory_path else None,
            memory_file=memory_path,
        )
        sandbox = Sandbox(policy, mode=self.sandbox, bash_timeout_s=self.bash_timeout_s)
        # Decide the confinement up front: mode="bwrap" fails fast here when
        # bubblewrap is unusable, mode="auto" prints its fallback warning once.
        sandbox_mode = sandbox.effective_mode
        toolset = ToolSet([
            (bash_tool_info(bash_timeout_s=self.bash_timeout_s,
                            max_output_chars=self.max_tool_output_chars,
                            root_vars=policy.root_vars()),
             BashTool(sandbox, max_output_chars=self.max_tool_output_chars)),
            (editor_tool_info(max_view_chars=self.max_view_chars),
             EditorTool(policy, max_view_chars=self.max_view_chars)),
            (VALIDATE_TOOL,
             ValidateTool(lambda: self._run_validators(out_dir, base_dir))),
        ])
        cfg = SessionConfig(
            max_llm_calls=self.max_llm_calls, timeout_s=self.timeout_s,
            max_attempts=self.max_attempts,
            transcript_result_chars=self.transcript_result_chars,
            model=self.model, reasoning_effort=self.reasoning_effort,
            base_url=self.base_url, api_key_env=self.api_key_env,
            llm_timeout_s=self.llm_timeout_s, extra_body=self.extra_body,
        )
        session = AgenticSession(
            self.llm, toolset,
            submit=editor_submit_spec(
                run_validators=lambda: self._run_validators(out_dir, base_dir),
                changed_files=lambda: self._changed_files(out_dir, base_dir),
            ),
            cfg=cfg,
            transcript=Transcript(agentic_dir / TRANSCRIPT_NAME),
        )
        system_prompt = agentic_system_prompt(read_scope)
        instruction = render_instruction(
            policy, max_llm_calls=self.max_llm_calls, timeout_s=self.timeout_s,
            max_attempts=self.max_attempts,
            manager_context=context if self.include_manager_context else None,
        )
        if verbose_log.is_enabled():
            verbose_log.write_text(out_dir, "editor_agentic_system.txt", system_prompt)
            verbose_log.write_text(out_dir, "editor_agentic_instruction.txt", instruction)

        result = session.run(system_prompt, instruction)

        summary = result.to_dict()
        summary["sandbox_mode"] = sandbox_mode
        summary["read_scope"] = read_scope
        summary["memory_path"] = str(memory_path) if memory_path else None
        summary["roots"] = {k: str(v) for k, v in policy.roots().items()}
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
        return changed_mutable_files(base_dir, out_dir)
