"""Task-agent runner: the cross-process contract between the evaluator and a
task agent.

This module owns the call shape ``def run_task(task: Task) -> AgentOutput``.
It exposes:

- :class:`Task`         — what the evaluator hands to the agent per case
- :class:`AgentOutput`  — what the agent hands back
- a CLI (``python -m platform_core.runner``) used by the evaluator's
  subprocess and by humans for standalone debugging

The evaluator subprocess invokes the runner in **stdin mode**: it writes a
single JSON Task to stdin and reads a JSON envelope from stdout. The same
module is runnable as a CLI (``--benchmark``/``--case-id`` or
``--task-file``) for debugging a single case without spinning up the full
evaluator.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import traceback
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


# --------------------------------------------------------------------------- #
# Public types
# --------------------------------------------------------------------------- #


@dataclass
class Task:
    """What the evaluator hands to the agent for one case.

    ``description`` is the prose prompt the agent feeds to the LLM.
    ``case_id`` uniquely identifies the case within a benchmark.
    ``context`` is per-case structured data the agent may inspect (budgets,
    hints, sample ids, anything not part of the prose). Defaults to ``{}``.
    Ground-truth fields used only by the scorer (e.g. ``expected``) live on
    the case dict, not on the Task — by construction the agent never sees
    them.
    """

    description: str
    case_id: str = ""
    context: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Task":
        return cls(
            description=str(payload.get("description", "")),
            case_id=str(payload.get("case_id", "")),
            context=dict(payload.get("context") or {}),
        )


@dataclass
class AgentOutput:
    """What the agent hands back to the evaluator.

    ``result`` is the primary payload the scorer consumes (typically a
    string for free-text benchmarks; may be a dict for structured ones).
    ``metadata`` is free-form annotations the agent attaches for
    observability (iteration counts, fallback flags, etc.). The scorer is
    free to ignore it.
    """

    result: Any
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"result": self.result, "metadata": dict(self.metadata)}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AgentOutput":
        return cls(
            result=payload.get("result"),
            metadata=dict(payload.get("metadata") or {}),
        )


# --------------------------------------------------------------------------- #
# Invocation
# --------------------------------------------------------------------------- #


def _invoke_workflow(task: Task) -> AgentOutput:
    """Import the seed's ``workflow`` module from ``sys.path`` and call its
    ``run_task``. If the workflow returns a bare value (string or dict)
    rather than an ``AgentOutput``, wrap it so the contract is uniform.

    Uses duck-typing rather than ``isinstance`` because invoking this module
    as ``python -m platform_core.runner`` loads it under both ``__main__``
    and ``platform_core.runner``, giving two distinct ``AgentOutput`` class
    objects. The workflow imports the latter; we run as the former.

    Wraps the call in ``platform_core.trace.case_scope`` so every event
    emitted during this case (``tool_call``, ``tool_result``, ``llm_call``,
    ``mutable_log``, ``error``) automatically carries the case_id. The
    BehaviorSummarizer uses that tag to cross-tab verifier/branch firings
    against per-case pass/fail outcomes.
    """
    from . import trace

    workflow = importlib.import_module("workflow")
    fn = getattr(workflow, "run_task", None)
    if fn is None:
        raise RuntimeError("workflow.py does not define run_task(task)")
    with trace.case_scope(task.case_id):
        out = fn(task)
    if hasattr(out, "result") and hasattr(out, "metadata"):
        return AgentOutput(
            result=out.result,
            metadata=dict(out.metadata or {}),
        )
    return AgentOutput(result=out)


def _load_case(benchmark_dir: Path, case_id: str) -> dict[str, Any]:
    cases_path = benchmark_dir / "cases.jsonl"
    if not cases_path.exists():
        raise FileNotFoundError(f"cases.jsonl not found in {benchmark_dir}")
    for line in cases_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        case = json.loads(line)
        if str(case.get("id") or case.get("case_id")) == str(case_id):
            return case
    raise KeyError(f"case_id {case_id!r} not found in {cases_path}")


def _task_from_case(case: dict[str, Any]) -> Task:
    """Map a benchmark case dict to a Task. The case's ``input`` becomes the
    description; the case's optional ``context`` field is forwarded as-is.
    Ground-truth fields (``expected``, ``meta_info``, etc.) are left on the
    case dict and never reach the agent."""
    return Task(
        description=str(case.get("input", "")),
        case_id=str(case.get("id") or case.get("case_id") or ""),
        context=dict(case.get("context") or {}),
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _stdin_mode() -> int:
    """Subprocess mode: read a Task JSON from stdin, run, write an envelope
    JSON to stdout. Exit 0 either way; ``ok`` in the envelope tells the
    caller whether the agent succeeded."""
    raw = sys.stdin.read()
    try:
        task = Task.from_dict(json.loads(raw))
    except Exception:
        sys.stdout.write(
            json.dumps(
                {"ok": False, "error": "bad task payload: " + traceback.format_exc()}
            )
        )
        return 0
    try:
        out = _invoke_workflow(task)
    except Exception:
        sys.stdout.write(
            json.dumps({"ok": False, "error": traceback.format_exc()})
        )
        return 0
    sys.stdout.write(json.dumps({"ok": True, "output": out.to_dict()}))
    return 0


def _invoke_workflow_with_communication_trace(
    task: Task, case: dict[str, Any] | None, args: argparse.Namespace
) -> AgentOutput:
    """Like ``_invoke_workflow``, but also captures a per-task communication trace
    (prompts, tool calls, inter-agent hand-offs) via
    ``platform_core.communication_instrumentation``, writing it to
    ``args.export_communication`` once the workflow returns.

    Purely additive and fully reversible: installs the ``--patch-agent-run``/
    ``--patch-transform`` targets only for the duration of this one call and restores
    the originals afterward (via each patch's ``uninstall()``), so nothing in the
    target project's own files is ever touched. ``ground_truth`` is read from the raw
    benchmark ``case`` dict's ``meta_info`` (available here, before ``_task_from_case``
    stripped it) -- never from ``task`` itself, preserving the existing invariant that
    ground truth never reaches agent-facing code.
    """
    from . import communication_instrumentation as citrace

    ground_truth = case.get("meta_info") if case else None
    recorder = citrace.CommunicationRecorder(
        task_id=task.case_id,
        communication_id=args.communication_id or "",
        task_prompt=task.description,
        ground_truth=ground_truth,
    )
    with citrace.recording_scope(recorder):
        uninstalls = [
            citrace.patch_agent_method(target, context_param=args.context_param)
            for target in (args.patch_agent_run or [])
        ]
        uninstalls += [
            citrace.patch_transform_function(target)
            for target in (args.patch_transform or [])
        ]
        try:
            out = _invoke_workflow(task)
        finally:
            for undo in uninstalls:
                undo()
    args.export_communication.write_text(
        json.dumps(recorder.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return out


def _standalone_mode(args: argparse.Namespace) -> int:
    agent_dir = args.agent_dir.resolve()
    if not agent_dir.exists():
        print(f"agent dir not found: {agent_dir}", file=sys.stderr)
        return 2
    if str(agent_dir) not in sys.path:
        sys.path.insert(0, str(agent_dir))

    case: dict[str, Any] | None = None
    if args.task_file:
        task = Task.from_dict(json.loads(args.task_file.read_text(encoding="utf-8")))
    elif args.benchmark and args.case_id:
        case = _load_case(args.benchmark.resolve(), args.case_id)
        # Apply per-case env overrides (whatever keys the case's ``env`` block
        # carries, e.g. a project's ``*_SAMPLE_ID``) before the workflow
        # imports run, mirroring what SubprocessEvaluator does.
        for k, v in (case.get("env") or {}).items():
            os.environ[str(k)] = str(v)
        task = _task_from_case(case)
    else:
        print(
            "standalone mode requires either --task-file or "
            "(--benchmark and --case-id)",
            file=sys.stderr,
        )
        return 2

    try:
        if args.export_communication:
            out = _invoke_workflow_with_communication_trace(task, case, args)
        else:
            out = _invoke_workflow(task)
    except Exception:
        print(traceback.format_exc(), file=sys.stderr)
        return 1
    print(json.dumps(out.to_dict(), indent=2, ensure_ascii=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m platform_core.runner",
        description=(
            "Run a task agent against a single Task. Default (no flags): read "
            "the Task as JSON from stdin and emit a one-line envelope JSON to "
            "stdout — this is the mode the evaluator subprocess uses. With "
            "--agent-dir set, run in standalone mode against a benchmark or a "
            "task file."
        ),
    )
    p.add_argument(
        "--agent-dir",
        type=Path,
        default=None,
        help="Directory containing workflow.py / tool_wrapper.py / tools_schema.json. "
        "When omitted, stdin mode is used and the agent must already be on sys.path "
        "(the evaluator sets cwd=agent_dir for this).",
    )
    p.add_argument(
        "--benchmark",
        type=Path,
        default=None,
        help="Benchmark directory containing cases.jsonl (standalone mode).",
    )
    p.add_argument(
        "--case-id",
        type=str,
        default=None,
        help="Case id to run from --benchmark/cases.jsonl (standalone mode).",
    )
    p.add_argument(
        "--task-file",
        type=Path,
        default=None,
        help="Path to a JSON file containing a Task object (standalone mode).",
    )
    p.add_argument(
        "--export-communication",
        type=Path,
        default=None,
        help="Write a per-task communication trace (prompts, tool calls, inter-agent "
        "hand-offs) as JSON to this path, without modifying the target project's own "
        "files. Requires at least one --patch-agent-run; see "
        "platform_core.communication_instrumentation for the underlying mechanism.",
    )
    p.add_argument(
        "--patch-agent-run",
        action="append",
        default=None,
        metavar="MODULE:CLASS.METHOD",
        help="Monkey-patch this class method (e.g. 'agents.base:BaseAgent.arun') to "
        "record its agent's prompt/tool-calls/hand-offs for --export-communication. "
        "Repeatable -- pass one per distinct agent-invocation shape the project has.",
    )
    p.add_argument(
        "--patch-transform",
        action="append",
        default=None,
        metavar="MODULE:FUNCTION",
        help="Monkey-patch this standalone function (e.g. "
        "'tools.mutable.compress:compress') to record it as a tool call attributed "
        "to whichever agent's own output it transforms. Repeatable.",
    )
    p.add_argument(
        "--communication-id",
        type=str,
        default=None,
        help="Override the exported trace's communication_id (default: "
        "'<case_id>_thread_1').",
    )
    p.add_argument(
        "--context-param",
        type=str,
        default="context",
        help="Name of the argument each --patch-agent-run target receives the prior "
        "agent's hand-off content through (default: 'context').",
    )
    args = p.parse_args(argv)

    if args.agent_dir is None:
        return _stdin_mode()
    return _standalone_mode(args)


if __name__ == "__main__":
    sys.exit(main())
