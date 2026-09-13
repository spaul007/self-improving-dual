"""Agentic editor — the meta-agent as an HGM-seed-style coding agent.

Instead of one LLM call that must emit full replacement files, the model
runs a tool-use session (``bash`` in a bubblewrap sandbox, a targeted-edit
``editor``, ``validate``) directly on the round's ``task_agent/`` copy, and
ends by calling ``submit_self_improvement`` with a summary. There is one
instruction prompt: paths and budget. No feedback digest, no steering
context, no retrieval stage — the agent reads the parent node's evidence and
the run's edit memory / diffs / beliefs / category registry from disk itself
(see ``meta_agent/agentic/policy.py`` for exactly what is visible).

Contract to the managers is unchanged: ``apply(feedback, base_dir, out_dir,
context=...)`` → ``EditResult`` with the final code under
``out_dir/task_agent``. Session artifacts land in ``out_dir/agentic/``.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Callable, Iterable, Optional, Union

from . import verbose_log
from .agent_editor import AgentEditor, Validator
from .agentic.policy import REPO_ROOT, build_policy
from .agentic.sandbox import Sandbox
from .agentic.session import (
    AGENTIC_SYSTEM_PROMPT,
    SESSION_NAME,
    TRANSCRIPT_NAME,
    AgenticSession,
    SessionConfig,
    Transcript,
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
from .edit_beliefs import write_prediction
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
    ) -> None:
        super().__init__(
            llm_caller, validators, max_attempts=max_attempts, model=model,
            reasoning_effort=reasoning_effort, base_url=base_url,
            tools_source=tools_source, db_schema=db_schema,
            scorer_source=scorer_source,
        )
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
        # task agent / edit memory keep the global OPENAI_API_KEY, and cap each
        # request so one stalled call cannot eat the whole session budget.
        self.api_key_env = api_key_env or None
        self.llm_timeout_s = float(llm_timeout_s) if llm_timeout_s else None

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
    ) -> EditResult:
        base_dir, out_dir = Path(base_dir), Path(out_dir)
        self._copy_workspace(base_dir, out_dir)
        (out_dir / "task_agent" / "mutable_tools").mkdir(exist_ok=True)
        agentic_dir = out_dir / "agentic"
        if agentic_dir.exists():
            shutil.rmtree(agentic_dir)
        (agentic_dir / "scratch").mkdir(parents=True)

        policy = build_policy(
            out_dir=out_dir, base_dir=base_dir, repo_root=REPO_ROOT,
            project_root=self.project_root,
            project_name=(self.project_root.name if self.project_root
                          else os.environ.get("META_AGENT_PROJECT", "")),
        )
        sandbox = Sandbox(policy, mode=self.sandbox, bash_timeout_s=self.bash_timeout_s)
        # Decide the confinement up front: mode="bwrap" fails fast here when
        # bubblewrap is unusable, mode="auto" prints its fallback warning once.
        sandbox_mode = sandbox.effective_mode
        toolset = ToolSet([
            (bash_tool_info(bash_timeout_s=self.bash_timeout_s,
                            max_output_chars=self.max_tool_output_chars),
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
            llm_timeout_s=self.llm_timeout_s,
        )
        session = AgenticSession(
            self.llm, toolset,
            run_validators=lambda: self._run_validators(out_dir, base_dir),
            changed_files=lambda: self._changed_files(out_dir, base_dir),
            write_prediction=lambda pred: write_prediction(out_dir, pred),
            cfg=cfg,
            transcript=Transcript(agentic_dir / TRANSCRIPT_NAME),
        )
        instruction = render_instruction(
            policy, max_llm_calls=self.max_llm_calls, timeout_s=self.timeout_s,
            max_attempts=self.max_attempts,
            manager_context=context if self.include_manager_context else None,
        )
        if verbose_log.is_enabled():
            verbose_log.write_text(out_dir, "editor_agentic_system.txt", AGENTIC_SYSTEM_PROMPT)
            verbose_log.write_text(out_dir, "editor_agentic_instruction.txt", instruction)

        result = session.run(AGENTIC_SYSTEM_PROMPT, instruction)

        summary = result.to_dict()
        summary["sandbox_mode"] = sandbox_mode
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
