from __future__ import annotations

import json
import os
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator, Optional

TRACE_PATH_ENV = "META_AGENT_TRACE_PATH"

# A per-evaluation-run isolated scratch directory. The evaluator creates a
# fresh dir per ``run()`` and exports its path under this env var to every
# case subprocess, analogous to TRACE_PATH_ENV. Projects that keep mutable
# on-disk state (e.g. the shopping cart) write it under here so concurrent
# evaluations of the same case never collide. Generic — no project knows
# about any other project; consumers read this var by its string name.
SCRATCH_DIR_ENV = "META_AGENT_SCRATCH_DIR"

_lock = threading.Lock()

# Ambient case_id for the currently-running case. Set by ``case_scope`` (which
# the runner wraps around each ``workflow.run_task`` call) so every event
# emitted during a case — ``tool_call``, ``tool_result``, ``llm_call``,
# ``mutable_log``, ``error`` — automatically carries the originating case_id.
# Without this the BehaviorSummarizer's per-case cross-tab can't correlate
# verifier/branch firings with which case passed or failed.
_case_id_ctx: ContextVar[Optional[str]] = ContextVar(
    "platform_core_trace_case_id", default=None
)


def _resolve_path() -> str | None:
    return os.environ.get(TRACE_PATH_ENV)


@contextmanager
def case_scope(case_id: str) -> Iterator[None]:
    """Tag every trace event emitted within this context with ``case_id``.

    Used by ``platform_core.runner._invoke_workflow`` to wrap each case's
    ``workflow.run_task(task)`` call. Per-task scoping is via
    ``contextvars.ContextVar``, which is per-thread / per-async-task — so
    parallel cases (each in its own subprocess, but also any in-process
    concurrency the agent might use) don't cross-contaminate.

    An empty / falsy ``case_id`` (e.g. when running outside a case scope)
    is treated as "no ambient case_id" — events get emitted without one,
    same as before this primitive existed.

    Events whose payload already specifies an explicit ``case_id`` keep
    their own — the ambient value is only used when no explicit one is set.
    """
    token = _case_id_ctx.set(case_id or None)
    try:
        yield
    finally:
        _case_id_ctx.reset(token)


def emit(kind: str, payload: dict[str, Any]) -> None:
    """Append one JSONL TraceEvent line to the file named by META_AGENT_TRACE_PATH.

    Silently no-ops when the env var is unset (e.g. unit tests not driven by the
    evaluator). Concurrency-safe within a single process via a module lock.

    If a ``case_scope`` is active and the payload does not already specify a
    ``case_id``, the ambient case_id is merged into the payload so the
    BehaviorSummarizer's per-case cross-tab works without callers having to
    remember to tag every event manually.
    """
    path = _resolve_path()
    if not path:
        return
    case_id = _case_id_ctx.get()
    if case_id and "case_id" not in payload:
        payload = {**payload, "case_id": case_id}
    event = {
        "timestamp": time.time(),
        "kind": kind,
        "payload": payload,
    }
    # ``default=str`` so Pydantic ``BaseModel`` instances (e.g. Responses-API
    # output items echoed back into the conversation by reasoning-model
    # agents) serialize as their ``repr`` rather than raising — they're
    # informational here, not part of the consumed schema.
    line = json.dumps(event, ensure_ascii=False, default=str)
    with _lock:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")


def log(label: str, **kwargs: Any) -> None:
    """Structured logging channel for mutable code (workflow.py / mutable_tools).

    Emits a ``mutable_log`` trace event with ``{"label": label, **kwargs}`` as
    its payload. The BehaviorSummarizer reads these events to correlate the
    edit's custom logic (verifiers, decision branches, helper functions) with
    per-case outcomes, then produces a "what helped / didn't help" memo for
    the next editor call.

    Example::

        from platform_core.trace import log

        def verify_budget_total(plan, max_budget):
            total = sum(item["cost"] for item in plan["items"])
            if total > max_budget:
                log("verifier_fired",
                    name="budget_total",
                    verdict="fail",
                    reason="exceeds limit",
                    total=total,
                    max=max_budget,
                )
                return False
            log("verifier_fired", name="budget_total", verdict="pass", total=total)
            return True

    Conventions (not enforced, but the summarizer keys on them):
        - ``label``: stable identifier; the summarizer groups events by label.
        - ``name``: when ``label`` is generic (e.g. ``"verifier_fired"``), use
          ``name`` to disambiguate the specific verifier/branch.
        - ``verdict``: ``"pass"`` | ``"fail"`` | ``"skip"`` — drives the
          cross-tab against case outcomes.
        - Anything else: free-form keyword args; included in the audit dump
          but not used for cross-tab grouping.

    Silently no-ops when ``META_AGENT_TRACE_PATH`` is unset.
    """
    emit("mutable_log", {"label": label, **kwargs})
