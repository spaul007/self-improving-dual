"""LLM-driven code editor — applies an EvolutionStrategy to a round folder.

The editor copies the base round into the out dir, prompts an LLM with the
current source of the mutable files plus the strategy + feedback, parses a
structured "edits" payload from the response, writes the new files, and runs
validators. Failed validation surfaces as ``EditResult.success=False`` with
a list of error strings — the manager decides what to do with that.
"""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Protocol

from . import verbose_log
from .editor_validators import MUTABLE_DIRS, MUTABLE_FILES
from .feedback_gatherer import render_metrics
from .models import AgentFeedback, EditResult, EvolutionStrategy
from .registry import register


class Validator(Protocol):
    def validate(self, out_dir: Path, base_dir: Path) -> list[str]: ...


# Tool schema for the LLM's structured edit proposal. Lifted out of
# ``_propose_edits`` so it's reviewable in isolation and reusable from
# tests / docs.
APPLY_EDITS_TOOL: dict[str, Any] = {
    "name": "apply_edits",
    "description": (
        "Apply a set of file edits to the task agent workspace. Each "
        "entry contains the relative path and the full replacement "
        "content for that file."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "files": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"},
                    },
                    "required": ["path", "content"],
                },
            }
        },
        "required": ["files"],
    },
}


@register("editor", "default")
class AgentEditor:
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
    ) -> None:
        self.llm = llm_caller
        self.validators = list(validators)
        self.max_attempts = max_attempts
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.base_url = base_url

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def apply(
        self,
        strategy: EvolutionStrategy,
        feedback: Optional[AgentFeedback],
        base_dir: Path,
        out_dir: Path,
    ) -> EditResult:
        self._copy_workspace(base_dir, out_dir)

        attempt_errors: list[str] = []
        for attempt in range(1, self.max_attempts + 1):
            edits = self._propose_edits(
                out_dir=out_dir,
                strategy=strategy,
                feedback=feedback,
                prior_errors=attempt_errors,
                attempt=attempt,
            )
            if not edits.get("files"):
                attempt_errors = ["editor returned no file edits"]
                continue

            written, write_errors = self._write_edits(out_dir, edits["files"])
            if write_errors:
                attempt_errors = write_errors
                continue

            errors = self._run_validators(out_dir, base_dir)
            if not errors:
                return EditResult(success=True, edited_files=written)
            attempt_errors = errors
            # Reset the workspace and try again with the validator feedback.
            self._copy_workspace(base_dir, out_dir)

        return EditResult(success=False, errors=attempt_errors)

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _copy_workspace(self, base_dir: Path, out_dir: Path) -> None:
        src = base_dir / "task_agent"
        dst = out_dir / "task_agent"
        if dst.exists():
            shutil.rmtree(dst)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dst)

    def _propose_edits(
        self,
        *,
        out_dir: Path,
        strategy: EvolutionStrategy,
        feedback: Optional[AgentFeedback],
        prior_errors: list[str],
        attempt: int = 1,
    ) -> dict:
        agent_dir = out_dir / "task_agent"
        current = self._read_mutable_sources(agent_dir)

        system = (
            "You are a code editor for a self-evolving agent. You may only "
            "modify these files in the task_agent workspace:\n"
            f"  - {', '.join(sorted(MUTABLE_FILES))}\n"
            f"  - any *.py file under mutable_tools/\n\n"
            "Hard rules:\n"
            "  1. workflow.py MUST define "
            "`def run_task(task: Task) -> AgentOutput`. The single arg must "
            "be named `task` (the validator enforces this).\n"
            "  2. workflow.py may import only: "
            "platform_core.llm_wrapper.call_llm, platform_core.runner "
            "(Task, AgentOutput), tool_wrapper, plus stdlib.\n"
            "  3. tool_wrapper.py may import only: platform_core.tools, "
            "mutable_tools.*, plus stdlib.\n"
            "  4. Mutable tools (mutable_tools/*.py) may import only: "
            "platform_core.tools, sibling mutable_tools.*, plus stdlib.\n"
            "  5. Reach immutable capabilities only via "
            "`platform_core.tools.call_tool(name, **kwargs)`.\n"
            "  6. tools_schema.json: every entry's `name` must be backed by "
            "either an immutable tool OR a `mutable_tools/<name>.py` file. "
            "No collisions between immutable and mutable names.\n\n"
            "Respond by calling the `apply_edits` tool with a list of files. "
            "Each file is the FULL replacement content — do not produce diffs. "
            "Omit files you do not change."
        )

        user_parts: list[str] = []
        user_parts.append(f"## Optimization goal\n{strategy.optimization_goal}\n")
        if strategy.proposed_changes:
            user_parts.append(f"## Proposed changes\n{strategy.proposed_changes}\n")
        if strategy.rationale:
            user_parts.append(f"## Rationale\n{strategy.rationale}\n")
        if feedback is not None:
            user_parts.append(self._format_feedback(feedback))
        user_parts.append(self._format_current_sources(current))
        if prior_errors:
            joined = "\n".join(f"  - {e}" for e in prior_errors)
            user_parts.append(
                "## Previous attempt failed validation. Fix these errors:\n"
                f"{joined}\n"
            )

        llm_kwargs: dict[str, Any] = {
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": "\n".join(user_parts)},
            ],
            "tools": [APPLY_EDITS_TOOL],
        }
        if self.model:
            llm_kwargs["model"] = self.model
        if self.reasoning_effort:
            llm_kwargs["reasoning_effort"] = self.reasoning_effort
        else:
            llm_kwargs["temperature"] = 0.2
        if self.base_url:
            llm_kwargs["base_url"] = self.base_url
        response = self.llm(**llm_kwargs)

        if verbose_log.is_enabled():
            user_body = "\n".join(user_parts)
            verbose_log.write_text(
                out_dir, f"editor_attempt_{attempt}_system.txt", system
            )
            verbose_log.write_text(
                out_dir, f"editor_attempt_{attempt}_user.txt", user_body
            )
            verbose_log.write_json(
                out_dir,
                f"editor_attempt_{attempt}_response.json",
                {
                    "content": getattr(response, "content", None),
                    "tool_calls": [
                        {"name": c.name, "arguments": c.arguments}
                        for c in (getattr(response, "tool_calls", []) or [])
                    ],
                },
            )

        for call in getattr(response, "tool_calls", []) or []:
            if call.name == "apply_edits":
                return call.arguments
        # Fallback: try to parse a fenced JSON block from text.
        text = getattr(response, "content", None) or ""
        match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
        return {"files": []}

    def _read_mutable_sources(self, agent_dir: Path) -> dict[str, str]:
        sources: dict[str, str] = {}
        for fname in sorted(MUTABLE_FILES):
            path = agent_dir / fname
            if path.exists():
                sources[fname] = path.read_text(encoding="utf-8")
        mutable_dir = agent_dir / "mutable_tools"
        if mutable_dir.exists():
            for py in sorted(mutable_dir.glob("*.py")):
                if py.name == "__init__.py":
                    continue
                rel = f"mutable_tools/{py.name}"
                sources[rel] = py.read_text(encoding="utf-8")
        return sources

    def _format_current_sources(self, sources: dict[str, str]) -> str:
        if not sources:
            return "## Current sources\n(empty)\n"
        parts = ["## Current sources"]
        for path, body in sources.items():
            parts.append(f"### {path}\n```\n{body}\n```")
        return "\n".join(parts) + "\n"

    def _format_feedback(self, feedback: AgentFeedback) -> str:
        ev = feedback.eval_result
        lines = [
            "## Last round's feedback",
            f"score={ev.score:.3f}  passed={ev.passed}  failed={ev.failed}  "
            f"crashed={ev.crashed}",
            f"llm_calls={feedback.llm_calls}",
            f"tool_usage={feedback.tool_usage}",
        ]
        if feedback.tool_error_rate:
            ranked = sorted(
                feedback.tool_error_rate.items(), key=lambda kv: -kv[1]
            )
            err_lines = [f"{n}={r:.2f}" for n, r in ranked[:5] if r > 0]
            if err_lines:
                lines.append("tool error rates: " + ", ".join(err_lines))
        if feedback.project_metrics:
            lines.append("project metrics:")
            lines.extend(render_metrics(feedback.project_metrics, cap=5, indent="  "))
        if feedback.runtime_exceptions:
            lines.append("runtime_exceptions:")
            for exc in feedback.runtime_exceptions[:5]:
                lines.append(f"  - {exc}")
        if feedback.edit_errors:
            lines.append("edit_errors (previous round did not run — these are validator complaints):")
            for err in feedback.edit_errors[:5]:
                lines.append(f"  - {err}")
        if feedback.log_excerpt:
            lines.append("log excerpt:")
            lines.append(feedback.log_excerpt[:2000])
        return "\n".join(lines) + "\n"

    def _write_edits(
        self, out_dir: Path, files: list[dict]
    ) -> tuple[list[str], list[str]]:
        agent_dir = out_dir / "task_agent"
        written: list[str] = []
        errors: list[str] = []
        for entry in files:
            path = (entry.get("path") or "").lstrip("/")
            content = entry.get("content")
            if not path or content is None:
                errors.append(f"malformed edit entry: {entry!r}")
                continue
            if not self._is_path_allowed(path):
                errors.append(
                    f"forbidden edit path: {path!r} "
                    f"(must be one of {sorted(MUTABLE_FILES)} or under mutable_tools/)"
                )
                continue
            target = agent_dir / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            written.append(path)
        return written, errors

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
