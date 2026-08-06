"""Capture per-task-instance agent communication (prompts, tool calls, inter-agent
hand-offs) from an UNMODIFIED MAS project, by monkey-patching a small number of
already-existing chokepoints at runtime.

Nothing is added to any vendored project's own files. A `CommunicationRecorder` is
activated for the current task via `recording_scope` (a `ContextVar`, same idiom as
`platform_core.trace.case_scope`), then `patch_agent_method`/`patch_transform_function`
temporarily replace a named class method / module function with a wrapper that reads
data the ORIGINAL, unmodified code already computes (a returned `prompt`/`tool_calls`
attribute) and infers inter-agent hand-offs by matching text: whichever agent's own
previously-returned output shows up verbatim (exactly or as a substring) inside a
later call's "context" argument is recorded as that call's sender.

Typical usage (see `platform_core/runner.py`'s `--patch-agent-run`/`--patch-transform`
CLI flags, which drive this module for a single case run):

    from platform_core.communication_instrumentation import (
        CommunicationRecorder, recording_scope, patch_agent_method,
    )

    recorder = CommunicationRecorder(task_id="1", task_prompt=case["input"],
                                      ground_truth=case.get("meta_info"))
    with recording_scope(recorder):
        undo = patch_agent_method("agents.base:BaseAgent.arun")
        try:
            asyncio.run(mas_workflow.run_task(item))
        finally:
            undo()
    recorder.to_dict()   # -> the fixed communication-trace JSON schema

Every helper here silently no-ops when no `recording_scope` is active, so leaving a
patch installed (or calling the module-level convenience functions directly) has zero
effect outside of an explicit recording session.
"""
from __future__ import annotations

import functools
import importlib
import inspect
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Callable, Iterator, Optional

_active: ContextVar[Optional["CommunicationRecorder"]] = ContextVar(
    "platform_core_communication_recorder", default=None
)


class CommunicationRecorder:
    """Accumulates one task instance's communication trace, thread/async-task safe.

    Guarded by a `threading.Lock` (mirrors `platform_core.trace`'s module-level
    `_lock`) since the target projects' agents may run concurrently
    (`asyncio.gather`), and a future Docker-`db_mas`-style project may use
    `ThreadPoolExecutor` (see `platform_core/runner.py`'s CLI help for that caveat).
    """

    def __init__(
        self,
        task_id: str,
        communication_id: str = "",
        task_prompt: str = "",
        ground_truth: Any = None,
    ) -> None:
        self.task_id = task_id
        self.communication_id = communication_id or f"{task_id}_thread_1"
        self.task_prompt = task_prompt
        self.ground_truth = ground_truth
        self._lock = threading.Lock()
        # agent_id -> {"prompt": str, "tool_calls": list[dict]}, insertion-ordered.
        self._agents: dict[str, dict[str, Any]] = {}
        self._tool_call_counters: dict[str, int] = {}
        self._communications: list[dict[str, Any]] = []
        self._comm_ordinal = 0
        # agent_id -> list of texts that agent has produced (raw/short/answer/compressed
        # outputs), used only to detect hand-offs -- never part of the output schema.
        self._outputs: dict[str, list[str]] = {}

    def _get_agent_locked(self, agent_id: str) -> dict[str, Any]:
        return self._agents.setdefault(agent_id, {"prompt": "", "tool_calls": []})

    def set_agent_prompt(self, agent_id: str, prompt: str) -> None:
        if not agent_id:
            return
        with self._lock:
            self._get_agent_locked(agent_id)["prompt"] = prompt

    def record_tool_call(
        self, agent_id: str, tool_name: str, arguments: Any, result: Any
    ) -> None:
        if not agent_id:
            return
        with self._lock:
            ordinal = self._tool_call_counters.get(agent_id, 0) + 1
            self._tool_call_counters[agent_id] = ordinal
            self._get_agent_locked(agent_id)["tool_calls"].append(
                {
                    "ordinal": ordinal,
                    "tool_name": tool_name,
                    "arguments": arguments,
                    "result": result,
                }
            )

    def record_message(self, sender: str, receiver: str, content: str) -> None:
        if not sender or not receiver:
            return
        with self._lock:
            self._comm_ordinal += 1
            self._communications.append(
                {
                    "ordinal": self._comm_ordinal,
                    "sender": sender,
                    "receiver": receiver,
                    "content": content,
                }
            )

    def register_output(self, agent_id: str, text: str) -> None:
        """Record `text` as something `agent_id` produced, for later hand-off matching.

        Idempotent per `(agent_id, text)`: some projects' own `AgentOutput` shape has
        two attributes that end up holding the exact same string for a given call
        (e.g. `db_mas_snapshot`'s `answer = (raw or "").strip()`, which equals `raw`
        whenever there's no surrounding whitespace to strip) -- registering the same
        text twice under different attribute names must not create two independent
        "producers" of it, or a later hand-off/transform match would double-count.
        """
        if not agent_id or not text:
            return
        with self._lock:
            bucket = self._outputs.setdefault(agent_id, [])
            if text not in bucket:
                bucket.append(text)

    def find_producers(
        self, text: str, exact_only: bool = False
    ) -> list[tuple[str, str]]:
        """Which already-registered agent outputs appear in `text`?

        Returns at most one `(agent_id, matched_text)` pair per producing agent,
        ordered by where `matched_text` first appears in `text` (so a joined/
        concatenated hand-off like db_mas_snapshot's investigator briefings comes
        back in the same left-to-right order it was assembled in). `exact_only=True`
        is for 1:1 transforms (e.g. `compress(raw)`, where `raw` IS one agent's own
        full output, not a fragment of a larger joined string) — substring matching
        there would risk false positives on short/coincidentally-repeated text.

        Deliberately capped at one match per agent: some projects register more than
        one distinct text per agent per call (e.g. both `raw` and a later `short`
        summary), and if more than one of those texts legitimately appears in `text`,
        that's still just one hand-off from that agent, not two.
        """
        if not text:
            return []
        with self._lock:
            outputs_snapshot = {k: list(v) for k, v in self._outputs.items()}
        matches: list[tuple[str, str]] = []
        for agent_id, texts in outputs_snapshot.items():
            for t in texts:
                if not t:
                    continue
                if t == text or (not exact_only and t in text):
                    matches.append((agent_id, t))
        matches.sort(key=lambda m: text.find(m[1]))
        seen_agents: set[str] = set()
        deduped: list[tuple[str, str]] = []
        for agent_id, matched_text in matches:
            if agent_id in seen_agents:
                continue
            seen_agents.add(agent_id)
            deduped.append((agent_id, matched_text))
        return deduped

    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            agents = [
                {
                    "agent_id": agent_id,
                    "prompt": rec["prompt"],
                    "tool_calls": list(rec["tool_calls"]),
                }
                for agent_id, rec in self._agents.items()
            ]
            communications = list(self._communications)
        return {
            "task_id": self.task_id,
            "communication_id": self.communication_id,
            "task_prompt": self.task_prompt,
            "ground_truth": self.ground_truth,
            "agents": agents,
            "communications": communications,
        }


@contextmanager
def recording_scope(recorder: CommunicationRecorder) -> Iterator[CommunicationRecorder]:
    """Make `recorder` the active recorder for every patched call inside this scope."""
    token = _active.set(recorder)
    try:
        yield recorder
    finally:
        _active.reset(token)


def current_recorder() -> Optional[CommunicationRecorder]:
    return _active.get()


# --------------------------------------------------------------------------- #
# Runtime patching
# --------------------------------------------------------------------------- #


def _resolve(target: str) -> tuple[Any, str, Any]:
    """Resolve `"module.path:Qual.if.ied.name"` to `(owner, attr_name, original)`.

    `owner` is whatever holds the attribute (a class for `"mod:Class.method"`, the
    module itself for `"mod:function"`) — patching reassigns `attr_name` on `owner`.
    """
    module_path, sep, qualname = target.partition(":")
    if not sep or not qualname:
        raise ValueError(
            f"target must look like 'module.path:Qualified.Name', got {target!r}"
        )
    module = importlib.import_module(module_path)
    parts = qualname.split(".")
    owner: Any = module
    for part in parts[:-1]:
        owner = getattr(owner, part)
    attr_name = parts[-1]
    original = getattr(owner, attr_name)
    return owner, attr_name, original


def _bind_args(func: Callable[..., Any], *args: Any, **kwargs: Any) -> dict[str, Any]:
    try:
        bound = inspect.signature(func).bind_partial(*args, **kwargs)
        return dict(bound.arguments)
    except TypeError:
        return {}


def _record_from_agent_call(
    recorder: CommunicationRecorder,
    self_obj: Any,
    agent_id_attr: str,
    result: Any,
    context_value: Any,
) -> None:
    agent_id = (
        getattr(self_obj, agent_id_attr, None)
        or getattr(self_obj, "name", None)
        or type(self_obj).__name__
    )

    prompt = getattr(result, "prompt", None)
    if isinstance(prompt, str) and prompt:
        recorder.set_agent_prompt(agent_id, prompt)

    tool_calls = getattr(result, "tool_calls", None)
    if tool_calls is None:
        tool_calls = getattr(result, "trace", None)
    for tc in tool_calls or []:
        name = tc.get("name") or tc.get("tool") or tc.get("tool_name")
        arguments = tc.get("arguments") if "arguments" in tc else tc.get("args")
        recorder.record_tool_call(agent_id, name, arguments, tc.get("result"))

    if isinstance(context_value, str) and context_value:
        for producer_id, matched_text in recorder.find_producers(context_value):
            recorder.record_message(producer_id, agent_id, matched_text)

    for attr in ("raw", "short", "answer"):
        text = getattr(result, attr, None)
        if isinstance(text, str) and text:
            recorder.register_output(agent_id, text)


def patch_agent_method(
    target: str,
    agent_id_attr: str = "name",
    context_param: str = "context",
) -> Callable[[], None]:
    """Monkey-patch `target` (e.g. `"agents.base:BaseAgent.arun"`) so every call,
    while a recorder is active, also records the calling agent's prompt, tool calls,
    and (if `context_param` is a non-empty string argument) any prior agent whose own
    output appears in it, as `communications` entries. Works for both `async def` and
    plain `def` methods. No-ops (delegates straight to the original) when no recorder
    is active, so this is safe to install even outside an explicit recording session.

    Returns an `uninstall()` callable that restores the original method.
    """
    owner, attr_name, original = _resolve(target)

    if inspect.iscoroutinefunction(original):

        @functools.wraps(original)
        async def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            recorder = _active.get()
            if recorder is None:
                return await original(self, *args, **kwargs)
            bound = _bind_args(original, self, *args, **kwargs)
            context_value = bound.get(context_param)
            result = await original(self, *args, **kwargs)
            _record_from_agent_call(recorder, self, agent_id_attr, result, context_value)
            return result

    else:

        @functools.wraps(original)
        def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:  # type: ignore[misc]
            recorder = _active.get()
            if recorder is None:
                return original(self, *args, **kwargs)
            bound = _bind_args(original, self, *args, **kwargs)
            context_value = bound.get(context_param)
            result = original(self, *args, **kwargs)
            _record_from_agent_call(recorder, self, agent_id_attr, result, context_value)
            return result

    setattr(owner, attr_name, wrapper)

    def uninstall() -> None:
        setattr(owner, attr_name, original)

    return uninstall


def _record_transform_call(
    recorder: CommunicationRecorder,
    tool_name: str,
    input_param: str,
    input_value: Any,
    result: Any,
) -> None:
    if not isinstance(input_value, str) or not input_value:
        return
    for agent_id, _matched in recorder.find_producers(input_value, exact_only=True):
        recorder.record_tool_call(agent_id, tool_name, {input_param: input_value}, result)
        if isinstance(result, str) and result:
            recorder.register_output(agent_id, result)


def patch_transform_function(
    target: str,
    input_param: str = "raw",
) -> Callable[[], None]:
    """Monkey-patch a standalone function (e.g. `"tools.mutable.compress:compress"`)
    that transforms one agent's own output before a hand-off. Attributes the call, as
    a tool call, to whichever agent's previously-registered output exactly equals the
    `input_param` argument (a plain string match, not the substring match
    `patch_agent_method` uses for joined/fan-in hand-offs — a transform's input is
    always one specific agent's own full text, so an exact match is both correct and
    safer against accidental false positives). No-ops when no recorder is active or no
    producer match is found. Returns an `uninstall()` callable.
    """
    owner, attr_name, original = _resolve(target)
    tool_name = target.rsplit(":", 1)[-1].rsplit(".", 1)[-1]

    if inspect.iscoroutinefunction(original):

        @functools.wraps(original)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            recorder = _active.get()
            if recorder is None:
                return await original(*args, **kwargs)
            bound = _bind_args(original, *args, **kwargs)
            input_value = bound.get(input_param)
            result = await original(*args, **kwargs)
            _record_transform_call(recorder, tool_name, input_param, input_value, result)
            return result

    else:

        @functools.wraps(original)
        def wrapper(*args: Any, **kwargs: Any) -> Any:  # type: ignore[misc]
            recorder = _active.get()
            if recorder is None:
                return original(*args, **kwargs)
            bound = _bind_args(original, *args, **kwargs)
            input_value = bound.get(input_param)
            result = original(*args, **kwargs)
            _record_transform_call(recorder, tool_name, input_param, input_value, result)
            return result

    setattr(owner, attr_name, wrapper)

    def uninstall() -> None:
        setattr(owner, attr_name, original)

    return uninstall
