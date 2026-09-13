"""The agentic editor's tool-use loop, prompts and per-session artifacts.

A port of HGM's ``llm_withtools.chat_with_agent_openai`` over
``platform_core.llm_wrapper.call_llm`` (Responses API items echoed back
verbatim, ``function_call_output`` per tool call) with its known quirks
fixed: every tool call in a response is processed (not just the first), the
final assistant text is kept, termination is explicit (``end_reason``) rather
than exception-driven, and the session ends on a structured
``submit_self_improvement`` call whose validation result is fed back to the
model instead of resetting the workspace.

Artifacts under ``<out_dir>/agentic/``: ``transcript.jsonl`` (one JSON event
per LLM call / tool call / validation, appended and flushed as they happen so
a crashed session is still readable) and ``session.json`` (summary).
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from ..agent_editor import (
    EDITOR_HARD_RULES,
    EDITOR_MUTABLE_SURFACE,
    _ALLOWED_TARGET_FILES,
    _coerce_str,
    _coerce_target_files,
    editor_import_forms,
)
from ..models import EvolutionStrategy
from .policy import BELIEFS_FILE, PathPolicy
from .tools import SUBMIT_TOOL, SUBMIT_TOOL_NAME, ToolSet

TRANSCRIPT_NAME = "transcript.jsonl"
SESSION_NAME = "session.json"

NUDGE_MESSAGE = (
    "You stopped without calling submit_self_improvement. If your edits are "
    "complete, call submit_self_improvement now (summary only); otherwise "
    "continue with the editor/bash tools."
)
HALFWAY_MESSAGE = (
    "Budget check: you have used {used} of {total} model calls and have not "
    "edited anything yet. Decide on ONE change now and make it with the "
    "editor tool; you can keep refining it afterwards. Reading more than "
    "you can act on is wasted budget."
)
FINAL_STRETCH_MESSAGE = (
    "Only {left} model calls remain (used {used} of {total}). Finish your edit "
    "with the editor tool in the next call(s), run `validate` once, then call "
    "submit_self_improvement. A session that ends without an accepted "
    "submission is a FAILED node."
)
WRAP_UP_MESSAGE = (
    "The session budget is exhausted ({reason}). Call submit_self_improvement "
    "now with a summary of what you changed. Do not call other tools."
)

AGENTIC_SYSTEM_PROMPT = (
    "You are the self-improvement module of a self-evolving agent, working as "
    "an autonomous coding agent. You have four tools: `bash`, `editor`, "
    "`validate`, and `submit_self_improvement`. Diagnose what to change from "
    "the evidence in the run directory and the current code, make targeted "
    "edits, verify them, then submit.\n"
    "First understand the task: read the agent's system prompt in "
    "workflow.py, the tool implementations, and the database schema with the "
    "editor/bash tools — nothing is inlined in this conversation. The parent "
    "node's evaluation evidence (feedback.json, logs/case_*.json) holds the "
    "concrete failures to target. Aim your edit at the failures that most "
    "affect the score.\n"
    + EDITOR_MUTABLE_SURFACE
    + EDITOR_HARD_RULES
    + editor_import_forms("The seed's tool_wrapper.py (on disk)")
    + "Environment:\n"
    "  - Every bash call is a fresh shell with no network; only the paths "
    "listed in the task message exist. The benchmark's cases, the scoring "
    "code and the database are not available — do not look for them, and do "
    "not try to run the task agent on cases (there is no model access inside "
    "the sandbox).\n"
    "  - Paths: the task message defines the roots RUN_DIR, NODE_DIR, "
    "PARENT_DIR and REPO_DIR. They are environment variables in every bash "
    "call; the editor tool accepts the same `$VAR/...` form, absolute paths, "
    "or paths relative to the task_agent directory.\n"
    "  - Verify with `validate` (the same validators that gate your "
    "submission), `python3 -c \"import workflow\"`, "
    "`python3 -m json.tool tools_schema.json`, and small throwaway scripts "
    "in the scratch directory (yours alone — nothing there ships with the "
    "agent or is validated).\n"
    "  - Edit with the `editor` tool's `str_replace` / `insert` commands "
    "(they are commands of `editor`, not tools of their own); when a "
    "`str_replace` fails because the exact text cannot be reproduced, use "
    "`replace_lines` with line numbers from a fresh `view` instead of "
    "retrying. Never re-create or rewrite a whole file; `create` is only "
    "for a new mutable_tools/<name>.py or a scratch file.\n"
    "  - Budget discipline: the task message states your call budget. Spend "
    "at most the first third of it reading (bash can `cat`/`sed -n` several "
    "files in one call; put independent commands in one call), then EDIT. A "
    "modest, well-verified change that is actually submitted beats a perfect "
    "diagnosis that never becomes an edit. You will get reminders at a third, "
    "at the halfway point and near the end.\n\n"
    "Finishing: when your edits are complete and `validate` passes, call "
    "`submit_self_improvement` with a one-line optimization_goal, a "
    "proposed_changes summary and a rationale (a summary only — your edits "
    "are already on disk). The validators run on submit; on errors your "
    "workspace is kept — fix them and submit again (limited attempts). You "
    "must end the session by submitting; do not stop with a plain message."
)

STEP_PARENT = (
    "Read the parent's hgm_node.json, strategy.json and feedback.json; open "
    "the failing logs/case_*.json it names. Your edits should be motivated by "
    "these failures."
)
STEP_MEMORY = (
    "Before deciding what to change, check $RUN_DIR/edit_memory_beliefs.md{note}. "
    "It is the accumulated understanding of every previous edit: how each "
    "strategy worked out, whether something like your idea was already tried, "
    "and when it failed whether the idea or its implementation was at fault. "
    "Use it as guidance. Since it refers to where those strategies were used, "
    "you can also check relevant node's edit_memory.md, edit_code.md, "
    "task_agent/ and results to see what was actually done and how it scored, "
    "and do better this time. You do not need to repeat the dominant strategy, "
    "you can also diversify and try different kind of edits time to time."
)
STEP_VIEW = (
    "View workflow.py (and tool_wrapper.py / tools_schema.json as needed); "
    "check the immutable tools' source and db_schema.md for any tool you touch."
)
STEP_EDIT = (
    "Make targeted str_replace edits; instrument new decision points with "
    "platform_core.trace.log."
)
STEP_SUBMIT = "Run validate (and python3 -c \"import workflow\"); fix errors; submit."


def render_instruction(
    policy: PathPolicy,
    *,
    max_llm_calls: int,
    timeout_s: float,
    max_attempts: int,
    memory: Optional[bool] = None,
    manager_context: Optional[str] = None,
) -> str:
    """The single task message: roots, workspace map, procedure, budget.
    Paths are ``$VAR`` forms only — no experiment path, no inlined source,
    no feedback digest. ``memory`` (default: the run has edit-memory files)
    adds the memory block to the map and the belief-guidance step; the
    no-memory arm never mentions memory or beliefs."""
    if memory is None:
        memory = policy.memory_enabled()
    steps = [STEP_PARENT]
    if memory:
        beliefs_exist = (policy.roots()["RUN_DIR"] / BELIEFS_FILE).exists()
        note = "" if beliefs_exist else " (not written yet at this round — skip this step)"
        steps.append(STEP_MEMORY.format(note=note))
    steps += [STEP_VIEW, STEP_EDIT, STEP_SUBMIT]
    procedure = "\n".join(f"  {i}. {text}" for i, text in enumerate(steps, 1))
    parts = [
        "# Task\n"
        "Improve the task agent's harness — its prompts, control flow, "
        "verification and repair logic, and tools — so that it scores higher "
        "on the benchmark it is evaluated on. The agent under edit is "
        "$NODE_DIR/task_agent, a copy of its parent node $PARENT_DIR. You "
        "decide what to change based on the evidence in the run directory.\n",
        policy.describe(memory=memory),
        f"## Procedure\n{procedure}\n",
        "## Budget\n"
        f"  - at most {max_llm_calls} model calls and {timeout_s:g}s "
        "wall-clock for this session\n"
        f"  - at most {max_attempts} submission attempts (validation rounds)\n",
    ]
    if manager_context:
        parts.append(f"## Steering context\n{manager_context}\n")
    return "\n".join(parts)


# ---------------------------------------------------------------------- #
# Responses-API item helpers (mirrors projects/travel/seed/workflow.py)
# ---------------------------------------------------------------------- #

def _item_type(item: Any) -> str:
    t = getattr(item, "type", None)
    if t is None and isinstance(item, dict):
        t = item.get("type")
    return t or ""


def strip_reasoning(raw_output: Any) -> list:
    """Drop ``reasoning`` items before echoing prior output back into the
    next call's input — only when ``META_AGENT_STRIP_REASONING=1`` (local
    vLLM models misread echoed fragments); OpenAI reasoning models need
    them echoed for cross-turn continuity."""
    items = list(raw_output or [])
    if os.environ.get("META_AGENT_STRIP_REASONING") != "1":
        return items
    return [item for item in items if _item_type(item) != "reasoning"]


def budget_thresholds(total: int) -> tuple[int, int, int]:
    """``(early, halfway, final_stretch)`` call counts at which reminders
    fire. Early/halfway only fire while nothing has been edited yet; final
    stretch always, leaving ~a sixth of the budget (2..8 calls) for
    edit + validate + submit. All 0 (never) for tiny budgets."""
    if total < 6:
        return 0, 0, 0
    reserve = min(8, max(2, total // 6))
    return total // 3, total // 2, total - reserve


def usage_of(response: Any) -> dict[str, int]:
    raw = getattr(response, "raw", None)
    usage = getattr(raw, "usage", None) if raw is not None else None
    if usage is None:
        return {}
    details = getattr(usage, "output_tokens_details", None)
    out = {
        "input_tokens": getattr(usage, "input_tokens", None),
        "output_tokens": getattr(usage, "output_tokens", None),
        "reasoning_tokens": getattr(details, "reasoning_tokens", None) if details else None,
    }
    return {k: int(v) for k, v in out.items() if isinstance(v, (int, float))}


# ---------------------------------------------------------------------- #
# Session
# ---------------------------------------------------------------------- #

@dataclass
class SessionConfig:
    max_llm_calls: int = 40
    timeout_s: float = 1800.0
    max_attempts: int = 3
    transcript_result_chars: int = 4000
    model: Optional[str] = None
    reasoning_effort: Optional[str] = None
    base_url: Optional[str] = None
    api_key_env: Optional[str] = None      # take the key from this env var
    llm_timeout_s: Optional[float] = None  # per-request client timeout


@dataclass
class SessionResult:
    success: bool
    errors: list[str]
    strategy: Optional[EvolutionStrategy]
    end_reason: str
    n_llm_calls: int
    validation_rounds: int
    changed_files: list[str]
    n_tool_calls: dict[str, int] = field(default_factory=dict)
    usage: dict[str, int] = field(default_factory=dict)
    elapsed_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "end_reason": self.end_reason,
            "errors": self.errors,
            "strategy": self.strategy.model_dump() if self.strategy else None,
            "n_llm_calls": self.n_llm_calls,
            "n_tool_calls": self.n_tool_calls,
            "validation_rounds": self.validation_rounds,
            "changed_files": self.changed_files,
            "usage": self.usage,
            "elapsed_s": round(self.elapsed_s, 3),
        }


class Transcript:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")

    def write(self, kind: str, **fields: Any) -> None:
        rec = {"t": round(time.time(), 3), "kind": kind, **fields}
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
            fh.flush()


class AgenticSession:
    def __init__(
        self,
        llm: Callable[..., Any],
        toolset: ToolSet,
        *,
        run_validators: Callable[[], list[str]],
        changed_files: Callable[[], list[str]],
        cfg: SessionConfig,
        transcript: Transcript,
    ) -> None:
        self.llm = llm
        self.toolset = toolset
        self.run_validators = run_validators
        self.changed_files = changed_files
        self.cfg = cfg
        self.transcript = transcript
        # Mutable session state
        self.messages: list[Any] = []
        self.n_llm_calls = 0
        self.n_tool_calls: dict[str, int] = {}
        self.usage: dict[str, int] = {}
        self.validation_rounds = 0
        self.last_errors: list[str] = []
        self.strategy: Optional[EvolutionStrategy] = None
        self.last_text = ""
        self.llm_error: Optional[str] = None
        self._started = 0.0

    # ------------------------------------------------------------------ #
    # Main loop
    # ------------------------------------------------------------------ #

    def run(self, system: str, instruction: str) -> SessionResult:
        self._started = time.time()
        self.messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": instruction},
        ]
        tools_all = self.toolset.infos() + [SUBMIT_TOOL]
        nudged = False
        end_reason = "max_llm_calls"

        for i in range(self.cfg.max_llm_calls):
            if time.time() - self._started > 0.9 * self.cfg.timeout_s:
                end_reason = "timeout"
                break
            response = self._call(self.messages, tools_all)
            if response is None:
                end_reason = "llm_error"
                break
            tool_calls = list(getattr(response, "tool_calls", None) or [])
            self._echo(response, tool_calls)

            if not tool_calls:
                if not nudged:
                    nudged = True
                    self.messages.append({"role": "user", "content": NUDGE_MESSAGE})
                    self.transcript.write("nudge", i=i)
                    continue
                end_reason = "no_tool_calls"
                break

            submitted, exhausted = self._process_tool_calls(tool_calls, i)
            if submitted:
                end_reason = "submitted"
                break
            if exhausted:
                end_reason = "max_attempts"
                break
            self._maybe_remind(used=i + 1)

        return self._finish(end_reason)

    def _maybe_remind(self, *, used: int) -> None:
        """Append a budget reminder as a user turn at the halfway point and
        at the start of the final stretch (``budget_thresholds``). The model
        otherwise tends to read until the budget is gone and only discover
        at wrap-up that it never edited."""
        total = self.cfg.max_llm_calls
        early, half, final = budget_thresholds(total)
        if used in (early, half) and used != final and not self._changed_since_start():
            msg = HALFWAY_MESSAGE.format(used=used, total=total)
        elif used == final:
            msg = FINAL_STRETCH_MESSAGE.format(left=total - used, used=used, total=total)
        else:
            return
        self.messages.append({"role": "user", "content": msg})
        self.transcript.write("budget_reminder", used=used, total=total)

    def _changed_since_start(self) -> bool:
        try:
            return bool(self.changed_files())
        except Exception:  # noqa: BLE001
            return False

    def _process_tool_calls(self, tool_calls: list, i: int) -> tuple[bool, bool]:
        submitted = exhausted = False
        for tc in tool_calls:
            name = getattr(tc, "name", "") or ""
            args = getattr(tc, "arguments", None)
            t0 = time.time()
            if submitted:
                output = "Error: submission already accepted; call ignored"
            elif name == SUBMIT_TOOL_NAME:
                output, accepted, exhausted = self._handle_submit(args)
                submitted = accepted
            else:
                output = self.toolset.call(name, args)
            self.n_tool_calls[name] = self.n_tool_calls.get(name, 0) + 1
            self.messages.append({
                "type": "function_call_output",
                "call_id": getattr(tc, "id", None),
                "output": output,
            })
            self.transcript.write(
                "tool_call", i=i, call_id=getattr(tc, "id", None), name=name,
                input=args, result=output[: self.cfg.transcript_result_chars],
                result_chars=len(output), elapsed_s=round(time.time() - t0, 3),
            )
            if exhausted:
                break
        return submitted, exhausted

    # ------------------------------------------------------------------ #
    # LLM call + echo
    # ------------------------------------------------------------------ #

    def _call(self, messages: list, tools: list) -> Any:
        kw: dict[str, Any] = {"messages": messages, "tools": tools}
        if self.cfg.model:
            kw["model"] = self.cfg.model
        if self.cfg.reasoning_effort:
            kw["reasoning_effort"] = self.cfg.reasoning_effort
        else:
            kw["temperature"] = 0.2
        if self.cfg.base_url:
            kw["base_url"] = self.cfg.base_url
        if self.cfg.api_key_env:
            kw["api_key_env"] = self.cfg.api_key_env
        if self.cfg.llm_timeout_s:
            kw["timeout_s"] = self.cfg.llm_timeout_s
        i = self.n_llm_calls
        self.transcript.write("llm_call", i=i, n_messages=len(messages),
                              tools=[t["name"] for t in tools])
        t0 = time.time()
        try:
            response = self.llm(**kw)
        except Exception as exc:  # noqa: BLE001 - call_llm already retried
            self.llm_error = f"{type(exc).__name__}: {exc}"
            self.transcript.write("llm_error", i=i, error=self.llm_error[:2000])
            return None
        self.n_llm_calls += 1
        text = getattr(response, "content", None) or ""
        if text:
            self.last_text = text
        usage = usage_of(response)
        for k, v in usage.items():
            self.usage[k] = self.usage.get(k, 0) + v
        self.transcript.write(
            "llm_response", i=i, elapsed_s=round(time.time() - t0, 3),
            content=text[:2000],
            tool_calls=[{"id": getattr(c, "id", None), "name": getattr(c, "name", None)}
                        for c in (getattr(response, "tool_calls", None) or [])],
            usage=usage, stop_reason=getattr(response, "stop_reason", None),
        )
        return response

    def _echo(self, response: Any, tool_calls: list) -> None:
        """Append the assistant turn to the history. Prefer the raw
        Responses-API items (required for reasoning models); synthesize
        equivalent items when the caller gave none (stubs, other providers)."""
        raw = getattr(response, "raw", None)
        items = getattr(raw, "output", None) if raw is not None else None
        if items:
            self.messages.extend(strip_reasoning(items))
            return
        text = getattr(response, "content", None) or ""
        if text:
            self.messages.append({"role": "assistant", "content": text})
        for tc in tool_calls:
            args = getattr(tc, "arguments", None)
            self.messages.append({
                "type": "function_call",
                "call_id": getattr(tc, "id", None),
                "name": getattr(tc, "name", None),
                "arguments": json.dumps(args if isinstance(args, dict) else {}),
            })

    # ------------------------------------------------------------------ #
    # Submission
    # ------------------------------------------------------------------ #

    def _handle_submit(self, args: Any) -> tuple[str, bool, bool]:
        """Returns ``(output, accepted, exhausted)``."""
        self.validation_rounds += 1
        k, n = self.validation_rounds, self.cfg.max_attempts
        args = args if isinstance(args, dict) else {}
        changed = self.changed_files()
        if not changed:
            errors = ["no changes detected — the mutable files are identical to "
                      "the parent's; make an edit before submitting"]
        else:
            errors = self.run_validators()
        self.transcript.write("validation", round=k, changed_files=changed, errors=errors)
        if errors:
            self.last_errors = errors
            exhausted = k >= n
            bullets = "\n".join(f"  - {e}" for e in errors)
            tail = ("\nNo attempts left; the session ends." if exhausted else
                    "\nThe workspace is kept as-is. Fix these and call "
                    "submit_self_improvement again.")
            return f"Validation failed (attempt {k}/{n}):\n{bullets}{tail}", False, exhausted
        self.strategy = EvolutionStrategy(
            target_files=_coerce_target_files(
                [p for p in changed if p in _ALLOWED_TARGET_FILES]
            ),
            optimization_goal=_coerce_str(args.get("optimization_goal")),
            proposed_changes=_coerce_str(args.get("proposed_changes")),
            rationale=_coerce_str(args.get("rationale")),
        )
        return "Submission accepted.", True, False

    # ------------------------------------------------------------------ #
    # Termination
    # ------------------------------------------------------------------ #

    def _finish(self, end_reason: str) -> SessionResult:
        if end_reason == "submitted":
            result = self._result(True, [], end_reason)
        elif end_reason == "llm_error":
            result = self._result(False, [f"LLM call failed: {self.llm_error}"], end_reason)
        elif end_reason == "max_attempts":
            result = self._result(False, list(self.last_errors), end_reason)
        else:
            result = self._wrap_up(end_reason)
        self.transcript.write("end", reason=result.end_reason, success=result.success,
                              n_llm_calls=self.n_llm_calls,
                              validation_rounds=self.validation_rounds)
        return result

    def _wrap_up(self, reason: str) -> SessionResult:
        """Budget ran out (or the model stopped talking) without an accepted
        submission: give it one submit-only call for the summary; failing
        that, accept the edit only if it is non-empty and validates."""
        if self.validation_rounds < self.cfg.max_attempts:
            self.messages.append(
                {"role": "user", "content": WRAP_UP_MESSAGE.format(reason=reason)}
            )
            self.transcript.write("wrap_up", reason=reason)
            response = self._call(self.messages, [SUBMIT_TOOL])
            if response is not None:
                tool_calls = list(getattr(response, "tool_calls", None) or [])
                self._echo(response, tool_calls)
                for tc in tool_calls:
                    if getattr(tc, "name", None) != SUBMIT_TOOL_NAME:
                        continue
                    output, accepted, _ = self._handle_submit(getattr(tc, "arguments", None))
                    self.messages.append({
                        "type": "function_call_output",
                        "call_id": getattr(tc, "id", None), "output": output,
                    })
                    self.transcript.write("tool_call", i=self.n_llm_calls - 1,
                                          call_id=getattr(tc, "id", None),
                                          name=SUBMIT_TOOL_NAME,
                                          input=getattr(tc, "arguments", None),
                                          result=output)
                    if accepted:
                        return self._result(True, [], f"submitted_after_{reason}")
                    break
        changed = self.changed_files()
        errors = self.run_validators() if changed else ["no changes made"]
        if changed and not errors:
            self.strategy = EvolutionStrategy(
                target_files=_coerce_target_files(
                    [p for p in changed if p in _ALLOWED_TARGET_FILES]
                ),
                optimization_goal=(
                    f"(agentic editor: no summary submitted; ended by {reason})"
                ),
                proposed_changes=self.last_text[:500],
                rationale="",
            )
            return self._result(True, [], reason)
        return self._result(
            False,
            [f"agentic session ended without an accepted submission ({reason})", *errors],
            reason,
        )

    def _result(self, success: bool, errors: list[str], end_reason: str) -> SessionResult:
        return SessionResult(
            success=success, errors=errors, strategy=self.strategy,
            end_reason=end_reason, n_llm_calls=self.n_llm_calls,
            validation_rounds=self.validation_rounds,
            changed_files=self.changed_files(),
            n_tool_calls=dict(self.n_tool_calls), usage=dict(self.usage),
            elapsed_s=time.time() - self._started,
        )
