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
    """
    workflow = importlib.import_module("workflow")
    fn = getattr(workflow, "run_task", None)
    if fn is None:
        raise RuntimeError("workflow.py does not define run_task(task)")
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


def _standalone_mode(args: argparse.Namespace) -> int:
    agent_dir = args.agent_dir.resolve()
    if not agent_dir.exists():
        print(f"agent dir not found: {agent_dir}", file=sys.stderr)
        return 2
    if str(agent_dir) not in sys.path:
        sys.path.insert(0, str(agent_dir))

    if args.task_file:
        task = Task.from_dict(json.loads(args.task_file.read_text(encoding="utf-8")))
    elif args.benchmark and args.case_id:
        case = _load_case(args.benchmark.resolve(), args.case_id)
        # Apply per-case env overrides (e.g. TRAVEL_SAMPLE_ID) before the
        # workflow imports run, mirroring what SubprocessEvaluator does.
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
    args = p.parse_args(argv)

    if args.agent_dir is None:
        return _stdin_mode()
    return _standalone_mode(args)


if __name__ == "__main__":
    sys.exit(main())
