"""Environment-variable exports shared by every entry point.

Both ``main_loop.py`` (the optimization loop) and ``evaluate.py`` (the
standalone full-benchmark evaluator) need to push the same set of
environment variables before instantiating the framework so that the
evaluator's child subprocesses inherit them. Keeping the helpers here
means the two entry points stay in lockstep.

The framework only knows three categories of env vars:

- **Task-agent settings** — ``LLM_MODEL`` / ``LLM_REASONING_EFFORT`` /
  ``LLM_BASE_URL`` picked up by the seed's ``call_llm`` invocations.
  ``LLM_BASE_URL`` lets the task-agent target a non-OpenAI endpoint
  (e.g. a locally-hosted vLLM server). When it's set, the wrapper
  also tolerates a missing ``OPENAI_API_KEY``; we still export
  ``OPENAI_API_KEY=EMPTY`` here as a courtesy so subprocesses don't
  have to know about the relaxation.
- **Project selector** — ``META_AGENT_PROJECT`` so the immutable-tool
  registry loads the right ``projects.<name>.tools`` package (or, when
  ``tool_source_dirs`` is set, ``META_AGENT_TOOL_SOURCE_DIRS`` scans those
  folders instead — see ``config.FrameworkConfig.tool_source_dirs``).
- **Free-form user env** — whatever the YAML's ``env:`` block declares.
  Project-specific config (database paths, API keys, etc.) flows
  through here. Project tools that want a default should compute it
  themselves, not fall back to a framework helper.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from . import config as cfg_mod


def apply_task_agent_env(spec: cfg_mod.TaskAgentSpec) -> None:
    """Push task-agent model/reasoning/base-url settings into env vars so
    the seed workflow (which doesn't see the YAML) can use them via the
    LLM wrapper. The evaluator subprocess inherits this environment."""
    if spec.model:
        os.environ["LLM_MODEL"] = spec.model
    if spec.reasoning_effort:
        os.environ["LLM_REASONING_EFFORT"] = spec.reasoning_effort
    if spec.temperature is not None:
        os.environ["LLM_TEMPERATURE"] = str(spec.temperature)
    if spec.max_output_tokens is not None:
        os.environ["LLM_MAX_OUTPUT_TOKENS"] = str(spec.max_output_tokens)
    if spec.base_url:
        os.environ["LLM_BASE_URL"] = spec.base_url
        # Local OpenAI-compatible servers ignore auth, but the SDK
        # constructor requires *some* string. Set a placeholder only when
        # the parent doesn't already have a real key — never overwrite.
        os.environ.setdefault("OPENAI_API_KEY", "EMPTY")


def apply_project_tools(
    project_name: str,
    tool_source_dirs: Optional[list[str]] = None,
    seed_dir: Optional[Path] = None,
) -> None:
    """Export ``META_AGENT_PROJECT`` (and, when set,
    ``META_AGENT_TOOL_SOURCE_DIRS``) so the evaluator's child subprocesses
    populate the immutable-tool registry the same way. Eagerly populate the
    parent's registry too — surfaces a real bug in an existing tool module
    at startup rather than at first child invocation.

    ``tool_source_dirs`` (relative to ``seed_dir``) overrides the default
    ``projects.<project_name>.tools`` convention — see
    ``config.FrameworkConfig.tool_source_dirs``. Resolved to absolute paths
    before export so they stay valid regardless of the child subprocess's
    cwd. A project with neither pattern is valid too — ``load_project``
    treats that as "no immutable tools to register," not an error.
    """
    os.environ["META_AGENT_PROJECT"] = project_name
    resolved: list[str] = []
    if tool_source_dirs and seed_dir is not None:
        resolved = [str((seed_dir / d).resolve()) for d in tool_source_dirs]
    if resolved:
        os.environ["META_AGENT_TOOL_SOURCE_DIRS"] = os.pathsep.join(resolved)
        os.environ["META_AGENT_TOOL_SOURCE_ROOT"] = str(seed_dir)
    else:
        os.environ.pop("META_AGENT_TOOL_SOURCE_DIRS", None)
        os.environ.pop("META_AGENT_TOOL_SOURCE_ROOT", None)
    from platform_core.tools import load_project

    load_project(
        project_name,
        source_dirs=[Path(p) for p in resolved] if resolved else None,
        sys_path_root=seed_dir if resolved else None,
    )


def apply_user_env(env: dict[str, str]) -> None:
    """Export each entry of the YAML's ``env:`` block to the parent's
    environment so that evaluator child subprocesses inherit it. Project
    tools can read anything they need from ``os.environ``; the framework
    doesn't interpret the keys.
    """
    for key, value in env.items():
        os.environ[str(key)] = str(value)


def apply_verbose(verbose: bool) -> None:
    """Export ``META_AGENT_VERBOSE`` when verbose logging is enabled.
    The flag is read by ``meta_agent.verbose_log.is_enabled()`` in the
    main process and inherited by the evaluator's child subprocesses,
    where ``platform_core.llm_wrapper`` checks it to decide whether to
    emit full-prompt trace events."""
    if verbose:
        os.environ["META_AGENT_VERBOSE"] = "1"
    else:
        os.environ.pop("META_AGENT_VERBOSE", None)


def apply_all(cfg: cfg_mod.FrameworkConfig) -> None:
    """One-call helper: apply every env block declared on ``cfg``."""
    apply_task_agent_env(cfg.task_agent)
    apply_project_tools(
        cfg.project, cfg.tool_source_dirs, cfg_mod.resolve_seed_dir(cfg)
    )
    apply_user_env(cfg.env)
    apply_verbose(cfg.verbose)
