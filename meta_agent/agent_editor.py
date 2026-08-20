"""LLM-driven self-improvement step — mutates a task agent in one call.

The editor copies the base round into the out dir, then makes a SINGLE LLM
call that diagnoses what to change (from the current source + the previous
round's feedback + an optional manager-supplied steering ``context``) and
emits both a short strategy summary and the full file edits. It writes the
new files and runs validators; failed validation surfaces as
``EditResult.success=False`` with a list of error strings — the manager
decides what to do with that.

This replaced an older two-call design (a separate manager "strategy
proposal" call feeding the editor). Collapsing to one call removes a lossy
text hand-off: the same LLM context that diagnoses also writes the code.
``EvolutionStrategy`` is now an *output* (carried on ``EditResult.strategy``
for logging), not an input.
"""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Protocol

from . import source_context, verbose_log
from .editor_validators import MUTABLE_DIRS, MUTABLE_FILES, is_excluded
from .failure_report import render_failure_report
from .feedback_gatherer import render_metrics
from .models import AgentFeedback, EditResult, EvolutionStrategy
from .registry import register


class Validator(Protocol):
    def validate(self, out_dir: Path, base_dir: Path) -> list[str]: ...


# Allowed values for an ``EvolutionStrategy.target_files`` entry. Mirrors the
# Literal in models.py. Lives here (the single home) so the coercion helpers
# below — and the managers, via import — share one definition.
_ALLOWED_TARGET_FILES = ("workflow.py", "tool_wrapper.py", "tools_schema.json")


def _coerce_target_files(
    value: Any, valid: Optional[Callable[[str], bool]] = None
) -> list[str]:
    """Coerce a model's ``target_files``-shaped value into a clean list[str].

    Models without strict schema enforcement (notably local vLLM-hosted
    open-weights models — caught gpt-oss-120b returning the bare string
    ``"workflow.py"`` on 2026-05-12) sometimes return a single string
    instead of a list. Unknown values are dropped; empty/None falls back to
    ``["workflow.py"]`` so a strategy summary always validates.

    ``valid`` overrides what counts as a "known" path. Default (``None``)
    preserves the exact legacy behavior — membership in
    ``_ALLOWED_TARGET_FILES`` — for projects using the include-list mutable
    surface. Projects using an exclude-list surface (see
    ``config.FrameworkConfig.mutable_exclude``) pass a predicate instead,
    since their editable paths aren't a fixed 3-name set.
    """
    if value is None or value == "":
        return ["workflow.py"]
    if isinstance(value, str):
        candidates = [value]
    elif isinstance(value, (list, tuple)):
        candidates = [str(v) for v in value if v]
    else:
        return ["workflow.py"]
    check = valid if valid is not None else (lambda c: c in _ALLOWED_TARGET_FILES)
    cleaned = [c for c in candidates if check(c)]
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


# Tool schema for the single self-improvement call. The model states what it
# is doing (optimization_goal / proposed_changes / rationale) AND does it
# (files) in one structured call.
SELF_IMPROVEMENT_TOOL: dict[str, Any] = {
    "name": "submit_self_improvement",
    "description": (
        "Diagnose the task agent from its current code and last round's "
        "feedback, then submit a targeted self-improvement: a short "
        "strategy summary plus the full file edits that implement it."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "optimization_goal": {"type": "string"},
            "proposed_changes": {"type": "string"},
            "rationale": {"type": "string"},
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
            },
        },
        "required": ["optimization_goal", "proposed_changes", "files"],
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
        # Static project context injected by build_components (read from the
        # project folder by convention). tools_source + db_schema are shown
        # in BOTH modes (the agent's own tooling/data shape, not ground
        # truth); scorer_source is shown only when eval_visibility == "whitebox".
        tools_source: Optional[str] = None,
        db_schema: Optional[str] = None,
        scorer_source: Optional[str] = None,
        # `None` (default) = legacy include-list mode (MUTABLE_FILES/
        # MUTABLE_DIRS). A list = exclude-list mode: everything under the
        # seed dir is editable except these paths. See
        # config.FrameworkConfig.mutable_exclude.
        mutable_exclude: Optional[list[str]] = None,
        # Caps runaway generation (e.g. a reasoning-model repetition loop) --
        # 16384 matches the task_agent default. None disables the cap.
        max_output_tokens: Optional[int] = 16384,
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
        self.mutable_exclude = mutable_exclude
        self.max_output_tokens = max_output_tokens

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
        """Produce one self-improvement of the agent in ``base_dir``.

        Copies ``base_dir/task_agent`` into ``out_dir``, then runs the
        single-call self-improvement LLM step (up to ``max_attempts`` times,
        re-diagnosing against validator errors on each retry). ``context`` is
        optional manager-supplied steering text (history, lineage, scores) —
        the editor reads the actual code itself, so ``context`` carries only
        cheap signal, never source.

        Returns ``EditResult``; ``.strategy`` carries the editor's emitted
        summary (the last attempt's, on failure).
        """
        self._copy_workspace(base_dir, out_dir)

        attempt_errors: list[str] = []
        last_strategy: Optional[EvolutionStrategy] = None
        for attempt in range(1, self.max_attempts + 1):
            strategy, files = self._self_improve(
                out_dir=out_dir,
                feedback=feedback,
                context=context,
                prior_errors=attempt_errors,
                attempt=attempt,
            )
            last_strategy = strategy
            if not files:
                attempt_errors = ["editor returned no file edits"]
                continue

            written, write_errors = self._write_edits(out_dir, files)
            if write_errors:
                attempt_errors = write_errors
                continue

            errors = self._run_validators(out_dir, base_dir)
            if not errors:
                return EditResult(
                    success=True, edited_files=written, strategy=strategy
                )
            attempt_errors = errors
            # Reset the workspace and try again with the validator feedback.
            self._copy_workspace(base_dir, out_dir)

        return EditResult(
            success=False, errors=attempt_errors, strategy=last_strategy
        )

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

    def _self_improve(
        self,
        *,
        out_dir: Path,
        feedback: Optional[AgentFeedback],
        context: Optional[str],
        prior_errors: list[str],
        attempt: int = 1,
    ) -> tuple[EvolutionStrategy, list[dict]]:
        """One self-improvement LLM call: diagnose + edit.

        Builds the prompt from the hard rules, the optional steering
        ``context``, the previous round's ``feedback`` digest, the agent's
        current mutable sources, and any ``prior_errors`` from a failed
        validation attempt. Returns ``(EvolutionStrategy, files)`` where
        ``files`` is the raw ``[{path, content}]`` payload to write.
        """
        agent_dir = out_dir / "task_agent"
        current = self._read_mutable_sources(agent_dir)

        if self.mutable_exclude is not None:
            excl = ", ".join(sorted(self.mutable_exclude)) or "(nothing)"
            system = (
                "You are the self-improvement module of a self-evolving agent. "
                "Diagnose what to change from the feedback and the current code, "
                "then call `submit_self_improvement`.\n"
                "First understand the task: read the agent's own code, and "
                "(when provided below) the tool implementations, database "
                "schema, and evaluation scoring code — together they show what "
                "the system does, what the data looks like, and how output is "
                "graded. Target edits at the failures that most affect the score.\n"
                "You may edit ANY file in the task_agent workspace EXCEPT:\n"
                f"  - {excl}\n"
                "This includes prompt/config text AND the actual orchestration "
                "code (workflow logic, tool implementations, retry/decision "
                "branches, control flow) — not just one or the other. If the "
                "failure pattern points to a structural or logical problem "
                "(e.g. how evidence is gathered, when a retry fires, how a "
                "decision is made), change the code that implements it; don't "
                "default to a prompt-only wording tweak just because it is the "
                "easiest edit to make. Pick whichever kind of change actually "
                "fixes the diagnosed failure, or both together.\n\n"
                "Hard rules:\n"
                "  1. The workspace MUST keep exposing "
                "`def run_task(task: Task) -> AgentOutput` from its top-level "
                "`workflow.py` (the single arg must be named `task` — the "
                "validator enforces this); it may delegate to any other file "
                "you're allowed to edit.\n\n"
                "If this workspace uses, or you introduce, a `tool_wrapper.py` + "
                "`tools_schema.json` + `mutable_tools/` pattern (the framework's "
                "generic tool-calling convention), these additional rules apply "
                "to that pattern specifically — irrelevant otherwise:\n"
                "  2. workflow.py may import only: "
                "platform_core.llm_wrapper.call_llm, platform_core.runner "
                "(Task, AgentOutput), tool_wrapper, plus stdlib.\n"
                "  3. tool_wrapper.py may import only: platform_core.tools, "
                "mutable_tools.*, plus stdlib.\n"
                "  4. Mutable tools (mutable_tools/*.py) may import only: "
                "platform_core.tools, sibling mutable_tools.*, plus stdlib.\n"
                "  5. Reach capabilities only via `platform_core.tools`: "
                "`call_tool(name, **kwargs)` for immutable tools, and "
                "`call_mutable_tool(name, **kwargs)` for `mutable_tools/*`. Never "
                "invoke a mutable tool's `run()` directly — routing through "
                "`call_mutable_tool` keeps its calls recorded in the trace.\n"
                "  6. tools_schema.json: every entry's `name` must be backed by "
                "either an immutable tool OR a `mutable_tools/<name>.py` file. "
                "No collisions between immutable and mutable names.\n\n"
                "  7. Instrument your edits for the behavior summarizer. When "
                "you add a verifier, helper, or decision branch, call "
                "`platform_core.trace.log(label='your_label', "
                "verdict='pass'|'fail'|'skip', name='specific_check_name', "
                "**context)` at the decision point. Conventions: pick a stable "
                "`label` (e.g. 'verifier_fired', 'decision_branch'); use `name` "
                "to disambiguate within a label; set `verdict` to `pass`, "
                "`fail`, or `skip`. The summarizer cross-tabs these logs "
                "against case outcomes so the next editor knows which of your "
                "additions helped, which didn't, and why. Skip instrumentation "
                "only for trivial edits (renames, docstrings).\n\n"
                "  8. Double-check your own rationale before submitting. Every "
                "factual claim in `rationale`/`proposed_changes` (e.g. \"the tag "
                "is malformed\", \"case X failed because of Y\") must be verified "
                "against the actual current source shown above or the "
                "feedback/metrics below — re-read the specific line or field "
                "you are citing and confirm it really says what you're about "
                "to claim. Do not invent a plausible-sounding diagnosis you "
                "have not checked. If a claim doesn't hold up on re-reading, "
                "drop it or soften it (e.g. \"possibly\", \"this may "
                "contribute\") rather than stating it as settled fact — a "
                "correct edit with an honest, hedged rationale is better than "
                "a confident but unverified one.\n\n"
                "  9. If the steering context above includes a specific "
                "suggestion or recommendation for what to change, state in "
                "your `rationale` whether your edit follows it. If your "
                "edit departs from it in any way — a different target, a "
                "different mechanism, or scope beyond what it proposed — "
                "say so explicitly: what you changed instead and why. "
                "Silently doing something else is not acceptable; an "
                "honest \"I deviated from the suggestion because X\" is.\n\n"
                "Call `submit_self_improvement` with a one-line optimization_goal, "
                "a proposed_changes summary, a rationale, and the `files` payload. "
                "Each file is the FULL replacement content — do not produce diffs. "
                "Omit files you do not change."
            )
        else:
            system = (
                "You are the self-improvement module of a self-evolving agent. "
                "Diagnose what to change from the feedback and the current code, "
                "then call `submit_self_improvement`.\n"
                "First understand the task: read the agent's system prompt in "
                "workflow.py, and (when provided below) the tool implementations, "
                "database schema, and evaluation scoring code — together they show "
                "what each tool does, what the data looks like, and how output is "
                "graded. Target edits at the failures that most affect the score.\n"
                "You may only modify these "
                "files in the task_agent workspace:\n"
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
                "  5. Reach capabilities only via `platform_core.tools`: "
                "`call_tool(name, **kwargs)` for immutable tools, and "
                "`call_mutable_tool(name, **kwargs)` for `mutable_tools/*`. Never "
                "invoke a mutable tool's `run()` directly — routing through "
                "`call_mutable_tool` keeps its calls recorded in the trace.\n"
                "  6. tools_schema.json: every entry's `name` must be backed by "
                "either an immutable tool OR a `mutable_tools/<name>.py` file. "
                "No collisions between immutable and mutable names.\n"
                "  7. Instrument your edits for the behavior summarizer. When you "
                "add a verifier, helper, or decision branch in workflow.py or a "
                "mutable_tools/*.py file, call "
                "`platform_core.trace.log(label='your_label', verdict='pass'|'fail'|'skip', "
                "name='specific_check_name', **context)` at the decision point. "
                "Conventions: pick a stable `label` (e.g. 'verifier_fired', "
                "'decision_branch'); use `name` to disambiguate within a label; "
                "set `verdict` to `pass`, `fail`, or `skip`. The summarizer "
                "cross-tabs these logs against case outcomes so the next editor "
                "knows which of your additions helped, which didn't, and why. "
                "Skip instrumentation only for trivial edits (renames, docstrings).\n\n"
                "  8. Double-check your own rationale before submitting. Every "
                "factual claim in `rationale`/`proposed_changes` (e.g. \"the tag "
                "is malformed\", \"case X failed because of Y\") must be verified "
                "against the actual current source shown above or the "
                "feedback/metrics below — re-read the specific line or field "
                "you are citing and confirm it really says what you're about "
                "to claim. Do not invent a plausible-sounding diagnosis you "
                "have not checked. If a claim doesn't hold up on re-reading, "
                "drop it or soften it (e.g. \"possibly\", \"this may "
                "contribute\") rather than stating it as settled fact — a "
                "correct edit with an honest, hedged rationale is better than "
                "a confident but unverified one.\n\n"
                "  9. If the steering context above includes a specific "
                "suggestion or recommendation for what to change, state in "
                "your `rationale` whether your edit follows it. If your "
                "edit departs from it in any way — a different target, a "
                "different mechanism, or scope beyond what it proposed — "
                "say so explicitly: what you changed instead and why. "
                "Silently doing something else is not acceptable; an "
                "honest \"I deviated from the suggestion because X\" is.\n\n"
                "Call `submit_self_improvement` with a one-line optimization_goal, "
                "a proposed_changes summary, a rationale, and the `files` payload. "
                "Each file is the FULL replacement content — do not produce diffs. "
                "Omit files you do not change."
            )

        user_parts: list[str] = []
        if context:
            user_parts.append(f"## Steering context\n{context}\n")
        if feedback is not None:
            user_parts.append(self._format_feedback(feedback))
        user_parts.extend(self._format_project_context())
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
            "tools": [SELF_IMPROVEMENT_TOOL],
        }
        if self.model:
            llm_kwargs["model"] = self.model
        if self.reasoning_effort:
            llm_kwargs["reasoning_effort"] = self.reasoning_effort
        else:
            llm_kwargs["temperature"] = 0.2
        if self.base_url:
            llm_kwargs["base_url"] = self.base_url
        if self.max_output_tokens is not None:
            llm_kwargs["max_output_tokens"] = self.max_output_tokens
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
            if call.name == "submit_self_improvement":
                return self._parse_self_improvement(call.arguments)

        # Fallback: the model didn't tool-call. Try to recover a files
        # payload from a fenced JSON block; warn so the failure is visible.
        text = getattr(response, "content", None) or ""
        files: list[dict] = []
        match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            try:
                files = json.loads(match.group(1)).get("files") or []
            except (json.JSONDecodeError, AttributeError):
                files = []
        print(
            "[editor] warning: model did not call submit_self_improvement; "
            f"recovered {len(files)} file(s) from fenced JSON",
            flush=True,
        )
        fallback = EvolutionStrategy(
            target_files=["workflow.py"],
            optimization_goal="(editor produced no structured proposal)",
            proposed_changes=text[:500],
            rationale="",
        )
        return fallback, files

    def _parse_self_improvement(
        self, args: dict[str, Any]
    ) -> tuple[EvolutionStrategy, list[dict]]:
        """Split a ``submit_self_improvement`` tool call into a validated
        ``EvolutionStrategy`` summary and the raw ``files`` payload.
        ``target_files`` is derived from the emitted file paths."""
        files = args.get("files") or []
        edited = [
            f.get("path", "") for f in files if isinstance(f, dict)
        ]
        if self.mutable_exclude is not None:
            valid = lambda p: not is_excluded(p, self.mutable_exclude)  # noqa: E731
        else:
            valid = None
            edited = [p for p in edited if p in _ALLOWED_TARGET_FILES]
        strategy = EvolutionStrategy(
            target_files=_coerce_target_files(edited, valid=valid),
            optimization_goal=_coerce_str(args.get("optimization_goal")),
            proposed_changes=_coerce_str(args.get("proposed_changes")),
            rationale=_coerce_str(args.get("rationale")),
        )
        return strategy, files

    # Noise directories skipped regardless of mode -- generated/scratch
    # output (e.g. db-mas's own `results/raw/*.json` writes) and Python's
    # own cache, never source the editor should read or be judged against.
    _ALWAYS_IGNORE_DIRS = {"__pycache__", "results"}

    def _read_mutable_sources(self, agent_dir: Path) -> dict[str, str]:
        return source_context.read_mutable_sources(
            agent_dir, mutable_exclude=self.mutable_exclude
        )

    def _format_project_context(self) -> list[str]:
        return source_context.format_project_context(
            tools_source=self.tools_source,
            db_schema=self.db_schema,
            scorer_source=self.scorer_source,
        )

    def _format_current_sources(self, sources: dict[str, str]) -> str:
        return source_context.format_current_sources(sources)

    def _format_feedback(self, feedback: AgentFeedback) -> str:
        """Render the previous round's ``AgentFeedback`` into a compact
        prompt section — score, tool usage/errors, project metrics,
        exceptions, validator complaints, and a trace excerpt."""
        ev = feedback.eval_result
        lines = [
            "## Last round's feedback",
            f"score={ev.score:.3f}  passed={ev.passed}  failed={ev.failed}  "
            f"crashed={ev.crashed}",
        ]
        # Data-driven scope note: the trace-derived stats below cover only the
        # cases present in the parsed trace (typically the latest evaluation
        # batch), which can be fewer than the cumulative evaluated set behind
        # the score / project metrics / failure analysis. Stated as actual
        # counts (not a hardcoded "last batch" claim) so it stays correct
        # regardless of how the trace was produced.
        n_eval = len(ev.per_case)
        if feedback.trace_n_cases and n_eval and feedback.trace_n_cases < n_eval:
            lines.append(
                f"(scope: tool_usage / tool error rates / llm_calls / log excerpt "
                f"below are over {feedback.trace_n_cases} traced case(s); score, "
                f"project metrics, and failure analysis cover {n_eval} evaluated "
                f"case(s))"
            )
        lines += [
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
        rendered = "\n".join(lines) + "\n"
        # Example-driven failure analysis (query → plan → what failed +
        # hardest cases) replaces the old generic trace tail, which bloated
        # the prompt with low-signal events. Error events still surface above
        # via runtime_exceptions.
        report = render_failure_report(feedback.failure_report)
        if report:
            rendered += "\n" + report
        return rendered

    def _write_edits(
        self, out_dir: Path, files: list[dict]
    ) -> tuple[list[str], list[str]]:
        """Write each ``{path, content}`` entry into ``out_dir/task_agent``.
        Returns ``(written_paths, errors)`` — an entry whose path is outside
        the mutable surface is rejected into ``errors``, not written."""
        agent_dir = out_dir / "task_agent"
        written: list[str] = []
        errors: list[str] = []
        for entry in files:
            # A non-compliant model can return a bare string / junk instead
            # of a {path, content} object — skip it rather than crashing.
            if not isinstance(entry, dict):
                errors.append(f"malformed edit entry (not an object): {entry!r}")
                continue
            path = (entry.get("path") or "").lstrip("/")
            content = entry.get("content")
            if not path or content is None:
                errors.append(f"malformed edit entry: {entry!r}")
                continue
            if not self._is_path_allowed(path):
                if self.mutable_exclude is not None:
                    errors.append(
                        f"forbidden edit path: {path!r} "
                        f"(excluded: {sorted(self.mutable_exclude)})"
                    )
                else:
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
        parts = Path(rel_path).parts
        if ".." in parts or Path(rel_path).is_absolute():
            return False
        if set(parts) & self._ALWAYS_IGNORE_DIRS:
            # Generated/scratch output (e.g. db-mas's own results/raw/*.json
            # writes) and __pycache__ -- never a legitimate edit target,
            # mode-independent. Matches what _read_mutable_sources already
            # never shows the editor in the first place.
            return False
        if self.mutable_exclude is not None:
            return not is_excluded(Path(rel_path).as_posix(), self.mutable_exclude)
        if rel_path in MUTABLE_FILES:
            return True
        if len(parts) == 2 and parts[0] in MUTABLE_DIRS and parts[1].endswith(".py"):
            return True
        return False

    def _run_validators(self, out_dir: Path, base_dir: Path) -> list[str]:
        errors: list[str] = []
        for validator in self.validators:
            errors.extend(validator.validate(out_dir, base_dir))
        return errors
