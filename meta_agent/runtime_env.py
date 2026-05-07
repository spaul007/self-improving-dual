"""Environment-variable exports shared by every entry point.

Both ``main_loop.py`` (the optimization loop) and ``evaluate.py`` (the
standalone full-benchmark evaluator) need to push the same set of
environment variables before instantiating the framework so that the
evaluator's child subprocesses inherit them. Keeping the helpers here
means the two entry points stay in lockstep.

The framework only knows three categories of env vars:

- **Task-agent settings** — ``LLM_MODEL`` / ``LLM_REASONING_EFFORT``
  picked up by the seed's ``call_llm`` invocations.
- **Project selector** — ``META_AGENT_PROJECT`` so the immutable-tool
  registry loads the right ``projects.<name>.tools`` package.
- **Free-form user env** — whatever the YAML's ``env:`` block declares.
  Project-specific config (database paths, API keys, etc.) flows
  through here. Project tools that want a default should compute it
  themselves, not fall back to a framework helper.
"""
from __future__ import annotations

import os

from . import config as cfg_mod


def apply_task_agent_env(spec: cfg_mod.TaskAgentSpec) -> None:
    """Push task-agent model/reasoning settings into env vars so the seed
    workflow (which doesn't see the YAML) can use them via the LLM wrapper.
    The evaluator subprocess inherits this environment."""
    if spec.model:
        os.environ["LLM_MODEL"] = spec.model
    if spec.reasoning_effort:
        os.environ["LLM_REASONING_EFFORT"] = spec.reasoning_effort


def apply_project_tools(project_name: str) -> None:
    """Export ``META_AGENT_PROJECT`` so the evaluator's child subprocesses
    populate the immutable-tool registry from this project. Eagerly populate
    the parent's registry too — surfaces a misnamed project at startup
    rather than at first child invocation."""
    os.environ["META_AGENT_PROJECT"] = project_name
    from platform_core.tools import load_project

    load_project(project_name)


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
    apply_project_tools(cfg.project)
    apply_user_env(cfg.env)
    apply_verbose(cfg.verbose)
