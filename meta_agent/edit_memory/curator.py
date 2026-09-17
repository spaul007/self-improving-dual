"""The two agentic curators: one ``AgenticSession`` each, over a curator
policy, ending with ``submit_curation``.

``run_curator`` is generic: the caller names the output file, the required
structure check and the prompts; the memory curator and the instruction
curator differ only in those. Artifacts land under ``<workspace>/agentic/``
exactly like an editor session's.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from .. import verbose_log
from ..agentic.sandbox import Sandbox
from ..agentic.session import (
    SESSION_NAME,
    TRANSCRIPT_NAME,
    AgenticSession,
    SessionConfig,
    SubmitSpec,
    Transcript,
)
from ..agentic.tools import BashTool, EditorTool, ToolSet
from . import prompts as P


@dataclass
class CuratorConfig:
    max_llm_calls: int = 60
    timeout_s: float = 2400.0
    max_attempts: int = 2
    bash_timeout_s: float = 120.0
    max_tool_output_chars: int = 20000
    max_view_chars: int = 40000
    sandbox: str = "auto"
    transcript_result_chars: int = 4000


@dataclass
class CuratorResult:
    success: bool
    end_reason: str
    errors: list[str]
    output_path: Path
    summary: str
    session: dict[str, Any]


def curator_submit_spec(
    *, output_path: Path, validate: Callable[[str], list[str]], output_file: str,
    salvage: Optional[Callable[[str], tuple[str, list[str]]]] = None,
) -> tuple[SubmitSpec, dict[str, Any]]:
    """The document check on ``submit_curation``. A failing check is quoted
    back to the model, which may fix the file and submit again — but the
    document is never rejected (2026-09-17 policy): on the last allowed
    attempt, and at wrap-up when the budget is gone, a non-empty file is
    accepted with the remaining findings kept in the returned ``state``
    dict (``summary``, ``fallback_errors``). Only a missing/empty file
    fails the session. When ``salvage`` is given it runs on that final
    acceptance: missing sections/subsections are inserted with a placeholder
    line (``salvaged`` in the state lists them) so the document is
    structurally complete and the gaps are explicit."""
    state: dict[str, Any] = {"summary": "", "fallback_errors": [], "salvaged": []}

    def _accept_final(text: str, errors: list[str]) -> list[str]:
        """Salvage the on-disk document if the check still fails; returns
        the findings that remain after salvage."""
        if errors and salvage is not None:
            fixed, inserted = salvage(text)
            if inserted:
                output_path.write_text(fixed, encoding="utf-8")
                state["salvaged"] = inserted
                errors = validate(fixed)
        state["fallback_errors"] = errors
        return errors

    def _read() -> str:
        try:
            return output_path.read_text(encoding="utf-8")
        except OSError:
            return ""

    def handle(session: AgenticSession, args: Any) -> tuple[str, bool, bool]:
        session.validation_rounds += 1
        k, n = session.validation_rounds, session.cfg.max_attempts
        args = args if isinstance(args, dict) else {}
        text = _read()
        errors = [f"$WORK_DIR/{output_file} does not exist or is empty"] if not text.strip() \
            else validate(text)
        session.transcript.write("validation", round=k, changed_files=changed_files(), errors=errors)
        if errors and not text.strip():
            session.last_errors = errors
            exhausted = k >= n
            return (f"Document check failed (attempt {k}/{n}):\n  - {errors[0]}"
                    + ("\nNo attempts left; the session ends." if exhausted else
                       f"\nWrite the document and call {P.SUBMIT_CURATION_NAME} again."),
                    False, exhausted)
        if errors and k < n:
            session.last_errors = errors
            bullets = "\n".join(f"  - {e}" for e in errors)
            return (f"Document check failed (attempt {k}/{n}):\n{bullets}\n"
                    f"Fix the document and call {P.SUBMIT_CURATION_NAME} again "
                    f"(on the last attempt it is accepted as is).", False, False)
        # Passed, or last attempt: accept; salvage the structure if needed.
        errors = _accept_final(text, errors)
        state["summary"] = str(args.get("summary") or "")
        if state["salvaged"]:
            msg = ("Submission accepted; these missing parts were inserted as placeholders: "
                   + ", ".join(state["salvaged"]))
        elif errors:
            msg = "Submission accepted (with unresolved check findings recorded)."
        else:
            msg = "Submission accepted."
        return msg, True, False

    def fallback(session: AgenticSession, reason: str) -> tuple[bool, list[str]]:
        text = _read()
        if not text.strip():
            return False, [f"no {output_file} written"]
        _accept_final(text, validate(text))
        state["summary"] = f"(no summary submitted; ended by {reason})"
        # Accept a non-empty but incomplete document; the caller sees the
        # structure errors in the result and decides.
        return True, []

    def progress() -> bool:
        return bool(_read().strip())

    def changed_files() -> list[str]:
        return [output_file] if output_path.exists() else []

    spec = SubmitSpec(
        tool=P.SUBMIT_CURATION_TOOL, name=P.SUBMIT_CURATION_NAME, handle=handle,
        fallback=fallback, progress=progress, changed_files=changed_files,
        messages=P.curator_messages(output_file),
    )
    return spec, state


def run_curator(
    llm: Callable[..., Any],
    *,
    policy,
    system_prompt: str,
    instruction: str,
    output_file: str,
    validate: Callable[[str], list[str]],
    cfg: CuratorConfig,
    llm_kwargs: dict[str, Any],
    salvage: Optional[Callable[[str], tuple[str, list[str]]]] = None,
) -> CuratorResult:
    """One curator session over ``policy`` (built by
    ``policy.build_curator_policy``). ``llm_kwargs`` carries model /
    reasoning_effort / base_url / api_key_env / llm_timeout_s."""
    workspace: Path = policy.out_dir
    agentic_dir = workspace / "agentic"
    (agentic_dir / "scratch").mkdir(parents=True, exist_ok=True)
    output_path = workspace / output_file

    sandbox = Sandbox(policy, mode=cfg.sandbox, bash_timeout_s=cfg.bash_timeout_s)
    sandbox_mode = sandbox.effective_mode
    toolset = ToolSet([
        (P.curator_bash_tool_info(bash_timeout_s=cfg.bash_timeout_s,
                                  max_output_chars=cfg.max_tool_output_chars,
                                  root_vars=policy.root_vars()),
         BashTool(sandbox, max_output_chars=cfg.max_tool_output_chars)),
        (P.curator_editor_tool_info(max_view_chars=cfg.max_view_chars, output_file=output_file),
         EditorTool(policy, max_view_chars=cfg.max_view_chars)),
    ], submit_name=P.SUBMIT_CURATION_NAME)
    submit, state = curator_submit_spec(output_path=output_path, validate=validate,
                                        output_file=output_file, salvage=salvage)
    session_cfg = SessionConfig(
        max_llm_calls=cfg.max_llm_calls, timeout_s=cfg.timeout_s, max_attempts=cfg.max_attempts,
        transcript_result_chars=cfg.transcript_result_chars,
        model=llm_kwargs.get("model"), reasoning_effort=llm_kwargs.get("reasoning_effort"),
        base_url=llm_kwargs.get("base_url"), api_key_env=llm_kwargs.get("api_key_env"),
        llm_timeout_s=llm_kwargs.get("llm_timeout_s"), extra_body=llm_kwargs.get("extra_body"),
    )
    session = AgenticSession(llm, toolset, submit=submit, cfg=session_cfg,
                             transcript=Transcript(agentic_dir / TRANSCRIPT_NAME))
    if verbose_log.is_enabled():
        verbose_log.write_text(workspace, "curator_system.txt", system_prompt)
        verbose_log.write_text(workspace, "curator_instruction.txt", instruction)
    result = session.run(system_prompt, instruction)
    summary = result.to_dict()
    summary["sandbox_mode"] = sandbox_mode
    summary["output_file"] = output_file
    summary["summary"] = state["summary"]
    summary["fallback_errors"] = state["fallback_errors"]
    summary["salvaged"] = state["salvaged"]
    summary["roots"] = {k: str(v) for k, v in policy.roots().items()}
    (agentic_dir / SESSION_NAME).write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    if verbose_log.is_enabled():
        verbose_log.write_json(workspace, "curator_messages.json", session.messages)
    return CuratorResult(
        success=result.success, end_reason=result.end_reason,
        errors=list(result.errors) + list(state["fallback_errors"]),
        output_path=output_path, summary=state["summary"], session=summary,
    )
