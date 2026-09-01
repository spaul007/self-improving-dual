"""Sole LLM entry point exposed to task agents and meta-agent components.

Backed by the OpenAI Responses API (``client.responses.create``). Accepts tool
schemas in any of three shapes — Responses-API, Chat-Completions, or Anthropic
— and normalises to Responses-API shape internally.

Defaults come from environment variables so the same task-agent code runs in
the meta-agent's parent process and in the evaluator's subprocesses without
needing to thread config through:

    LLM_MODEL              default model name
    LLM_REASONING_EFFORT   "low" | "medium" | "high" (reasoning models only)
    LLM_BASE_URL           OpenAI-compatible endpoint to target instead of
                           the SDK default (e.g. a locally hosted vLLM
                           Responses-API server). When set, the
                           ``OPENAI_API_KEY`` check is relaxed because
                           local servers ignore auth — any non-empty
                           string ("EMPTY" by convention) is accepted.
    LLM_TEMPERATURE        sampling temperature used when the caller does
                           not pass ``temperature=`` explicitly. Unlike the
                           vars above it is NOT exported globally by
                           runtime_env — the evaluator sets it on each case
                           subprocess's env only (see
                           ``SubprocessEvaluator._child_env``), so task-agent
                           calls get it while meta-agent calls in the parent
                           process do not. With reasoning effort active it is
                           sent alongside ``reasoning`` — except for OpenAI
                           first-party reasoning models (gpt-5*, o-series),
                           which reject explicit temperature and keep the
                           omit behaviour.

Trace events ("llm_call", "llm_response") are emitted via
:mod:`platform_core.trace` whenever ``META_AGENT_TRACE_PATH`` is set.

Required env var: ``OPENAI_API_KEY`` (or ``LLM_BASE_URL`` set, in which
case any value of ``OPENAI_API_KEY`` is accepted).
"""
from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from . import trace

DEFAULT_MODEL_FALLBACK = "gpt-5.4-mini"
# None = let the API choose. The reference agent codebase omits
# max_output_tokens entirely from its responses.create call; mirror that
# so reasoning-heavy cases don't hit a self-imposed cap.
DEFAULT_MAX_OUTPUT_TOKENS: Optional[int] = None

# Legacy default for the non-reasoning branch when neither the caller nor
# the LLM_TEMPERATURE env var supplies a value.
DEFAULT_TEMPERATURE = 1.0

# API-error retry policy for `client.responses.create`. Mirrors the reference
# agent's `call_llm` (max_retries=30, backoff=1.5s) so transient
# network/server errors don't kill a case.
DEFAULT_API_MAX_RETRIES = 30
DEFAULT_API_BACKOFF_S = 1.5

# Per-request wall clock. The SDK's own default is 600s, which is not enough
# headroom for a reasoning meta-agent: one measured editor call on
# deepseek-v4-pro took 456s and emitted 27,480 reasoning tokens, leaving ~24%
# margin. A legitimate call that overran would be retried by the loop below —
# up to 30 times, each one billed — so too tight a timeout is far more
# expensive than too loose a one.
#
# 3600s is ~8x the measured call. The margin is deliberate: reasoning output
# scales with prompt size, and the edit-memory arm's prompt grows all run as
# the steering block fills, so the slowest call comes last. This is an upper
# bound, not a wait — fast endpoints are unaffected. Override with
# LLM_TIMEOUT_S.
DEFAULT_API_TIMEOUT_S = 3600.0


def _env_default_model() -> str:
    return os.environ.get("LLM_MODEL", DEFAULT_MODEL_FALLBACK)


def _env_default_reasoning_effort() -> Optional[str]:
    val = os.environ.get("LLM_REASONING_EFFORT")
    return val if val else None


def _env_default_base_url() -> Optional[str]:
    val = os.environ.get("LLM_BASE_URL")
    return val if val else None


def _env_default_temperature() -> Optional[float]:
    """Sampling temperature from ``LLM_TEMPERATURE``. Falls back to ``None``
    (= unset) on a missing or unparseable value — degrading to the legacy
    behaviour, never to an accidental greedy/zero."""
    raw = os.environ.get("LLM_TEMPERATURE")
    if not raw:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


# OpenAI first-party reasoning families that 400 on an explicit temperature
# when reasoning is active. Matched as exact name or "<family>-"/"<family>."
# prefix after lowercasing and stripping an optional "openai/" route prefix —
# so "gpt-oss-120b" (which accepts temperature) inherently does not match
# "gpt-5", and non-OpenAI routes like "z-ai/..." are never touched.
# Extend this tuple when new OpenAI reasoning families ship (gpt-6, o5, ...).
_TEMPERATURE_REJECTING_FAMILIES = ("gpt-5", "o1", "o3", "o4")


def _rejects_temperature_with_reasoning(model: str) -> bool:
    name = (model or "").lower()
    if name.startswith("openai/"):
        name = name[len("openai/"):]
    return any(
        name == fam or name.startswith(fam + "-") or name.startswith(fam + ".")
        for fam in _TEMPERATURE_REJECTING_FAMILIES
    )


def _env_default_timeout_s() -> float:
    """Per-request timeout, from ``LLM_TIMEOUT_S``. Falls back to the default
    on anything unparseable or non-positive rather than disabling the timeout —
    a missing timeout is what leaves a hung request blocking forever."""
    raw = os.environ.get("LLM_TIMEOUT_S")
    if not raw:
        return DEFAULT_API_TIMEOUT_S
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return DEFAULT_API_TIMEOUT_S
    return val if val > 0 else DEFAULT_API_TIMEOUT_S


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMResponse:
    content: Optional[str]
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: Optional[str] = None
    raw: Any = None


# ---------------------------------------------------------------------------
# Schema / message normalisation
# ---------------------------------------------------------------------------


def _normalise_tool_schema(tool: dict[str, Any]) -> dict[str, Any]:
    """Return Responses-API tool shape:
        {"type": "function", "name", "description", "parameters"}
    """
    if (
        tool.get("type") == "function"
        and "name" in tool
        and "parameters" in tool
        and "function" not in tool
    ):
        # Already Responses-API shape.
        return {
            "type": "function",
            "name": tool["name"],
            "description": tool.get("description", ""),
            "parameters": tool["parameters"],
        }
    if tool.get("type") == "function" and isinstance(tool.get("function"), dict):
        # Chat-Completions shape.
        fn = tool["function"]
        return {
            "type": "function",
            "name": fn["name"],
            "description": fn.get("description", ""),
            "parameters": fn.get("parameters", {"type": "object", "properties": {}}),
        }
    if "input_schema" in tool and "name" in tool:
        # Anthropic shape.
        return {
            "type": "function",
            "name": tool["name"],
            "description": tool.get("description", ""),
            "parameters": tool["input_schema"],
        }
    if "name" in tool and "parameters" in tool:
        # Bare {name, description, parameters}.
        return {
            "type": "function",
            "name": tool["name"],
            "description": tool.get("description", ""),
            "parameters": tool["parameters"],
        }
    raise ValueError(f"Unrecognised tool schema shape: keys={sorted(tool)}")


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------


def _reasoning_item_text(item: Any) -> str:
    """Concatenate the visible text from a single ``reasoning`` output
    item. vLLM and the OpenAI SDK both expose reasoning content as a
    list of parts; depending on the model server the part list may live
    under ``.content`` or ``.summary`` and each part may carry the text
    as ``.text``, the bare string itself, or a ``str()``-able object."""
    parts = getattr(item, "content", None)
    if not parts:
        parts = getattr(item, "summary", None) or []
    chunks: list[str] = []
    for part in parts:
        text = getattr(part, "text", None)
        if isinstance(text, str) and text:
            chunks.append(text)
        elif isinstance(part, str):
            chunks.append(part)
        elif part:
            chunks.append(str(part))
    return "\n".join(c for c in chunks if c)


def _extract_output(response: Any) -> tuple[Optional[str], list[ToolCall]]:
    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    output = getattr(response, "output", None) or []

    for item in output:
        item_type = getattr(item, "type", None)
        if item_type == "message":
            for part in getattr(item, "content", None) or []:
                part_type = getattr(part, "type", None)
                if part_type in ("output_text", "text"):
                    text_parts.append(getattr(part, "text", "") or "")
        elif item_type == "function_call":
            raw_args = getattr(item, "arguments", "") or ""
            try:
                args = json.loads(raw_args) if raw_args else {}
            except json.JSONDecodeError:
                args = {"_raw_arguments": raw_args}
            call_id = (
                getattr(item, "call_id", None)
                or getattr(item, "id", None)
                or uuid.uuid4().hex[:12]
            )
            tool_calls.append(
                ToolCall(
                    id=call_id,
                    name=getattr(item, "name", "") or "",
                    arguments=args if isinstance(args, dict) else {},
                )
            )
        elif item_type == "reasoning":
            # Captured separately and used as a fallback below. OpenAI's
            # reasoning models always emit a real ``message`` item after
            # the reasoning, so this fallback is a no-op for them. But
            # vLLM-hosted open-weights models (notably Qwen3.5 with
            # `--reasoning-parser qwen3`) sometimes put the model's
            # entire final answer — including the seed's expected
            # `<plan>...</plan>` block — into the reasoning item, with
            # the message item left empty. Without this fallback the
            # wrapper returned `content=None` and the seed extracted an
            # empty plan, scoring 0 across the board (caught live on
            # 2026-05-12).
            text = _reasoning_item_text(item)
            if text:
                reasoning_parts.append(text)

    fallback_text = getattr(response, "output_text", None)
    if not text_parts and isinstance(fallback_text, str) and fallback_text:
        text_parts.append(fallback_text)

    # Use message text when present; fall through to reasoning when
    # message text is empty or whitespace-only. Keeps the OpenAI path
    # unchanged (which always has real message text) and rescues the
    # vLLM/Qwen path where the answer lands in the reasoning item.
    joined_message = "\n".join(p for p in text_parts if p)
    if joined_message.strip():
        content: Optional[str] = joined_message
    elif reasoning_parts:
        content = "\n".join(reasoning_parts)
    elif text_parts:
        # Whitespace-only message and no reasoning — preserve whatever
        # whitespace we saw rather than silently dropping it.
        content = joined_message
    else:
        content = None
    return content, tool_calls


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def call_llm(
    messages: list[dict[str, Any]],
    *,
    tools: Optional[list[dict[str, Any]]] = None,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    reasoning_effort: Optional[str] = None,
    base_url: Optional[str] = None,
    max_output_tokens: Optional[int] = DEFAULT_MAX_OUTPUT_TOKENS,
    **kwargs: Any,
) -> LLMResponse:
    """Make one Responses-API round-trip and return a normalised response.

    Required env var: ``OPENAI_API_KEY`` (relaxed when ``base_url`` /
    ``LLM_BASE_URL`` is set — local servers ignore auth, so any
    non-empty string is accepted).

    ``model``, ``reasoning_effort``, and ``base_url`` fall back to the
    ``LLM_MODEL`` / ``LLM_REASONING_EFFORT`` / ``LLM_BASE_URL`` environment
    variables, which lets the meta-agent set them once for the whole run
    and have them propagate into every evaluator subprocess without
    threading them through the seed code.

    ``temperature`` resolution: explicit argument > ``LLM_TEMPERATURE`` env
    var > legacy default (1.0 without reasoning effort; omitted with it).
    With reasoning effort active, only an env-sourced temperature is sent
    alongside ``reasoning`` (and only for models that tolerate the
    combination — see ``_rejects_temperature_with_reasoning``); an explicit
    argument keeps the historical omit behaviour so meta-agent call sites
    that pass ``temperature=`` are byte-identical whether or not a global
    reasoning effort is configured. ``LLM_TEMPERATURE`` is set per evaluator
    child process, never globally (see module docstring).
    """
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "openai SDK is required. Install with: pip install 'openai>=1.50'"
        ) from exc

    resolved_model = model or _env_default_model()
    resolved_effort = reasoning_effort or _env_default_reasoning_effort()
    resolved_base_url = base_url or _env_default_base_url()

    # The temperature actually sent, or None when omitted (see docstring
    # for the resolution rules). Computed up front so the llm_call trace
    # event can record it.
    env_temperature = _env_default_temperature()
    if resolved_effort:
        if (
            temperature is None
            and env_temperature is not None
            and not _rejects_temperature_with_reasoning(resolved_model)
        ):
            resolved_temperature: Optional[float] = env_temperature
        else:
            resolved_temperature = None
    else:
        if temperature is not None:
            resolved_temperature = temperature
        elif env_temperature is not None:
            resolved_temperature = env_temperature
        else:
            resolved_temperature = DEFAULT_TEMPERATURE

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        if resolved_base_url:
            # Local OpenAI-compatible servers (vLLM, etc.) ignore the key,
            # but the SDK constructor still requires *some* string.
            api_key = "EMPTY"
        else:
            raise RuntimeError("OPENAI_API_KEY environment variable is required")

    norm_tools = [_normalise_tool_schema(t) for t in (tools or [])]

    call_id = uuid.uuid4().hex[:12]
    trace.emit(
        "llm_call",
        {
            "id": call_id,
            "model": resolved_model,
            "reasoning_effort": resolved_effort,
            "base_url": resolved_base_url,
            "temperature": resolved_temperature,
            "tool_names": [t["name"] for t in norm_tools],
            "num_messages": len(messages),
        },
    )
    if os.environ.get("META_AGENT_VERBOSE") == "1":
        trace.emit(
            "llm_call_full",
            {
                "id": call_id,
                "messages": messages,
                "tools": norm_tools,
            },
        )

    client_kwargs: dict[str, Any] = {
        "api_key": api_key,
        "timeout": _env_default_timeout_s(),
        # The retry loop below is the single source of retry truth. The SDK
        # defaults to 2 of its own retries *per* iteration of that loop, so
        # leaving it on multiplies the effective attempt count (30 -> 90) and
        # with it the worst-case wall clock and the bill.
        "max_retries": 0,
    }
    if resolved_base_url:
        client_kwargs["base_url"] = resolved_base_url
    client = OpenAI(**client_kwargs)
    request: dict[str, Any] = {
        "model": resolved_model,
        "input": messages,
    }
    # Omit the key entirely when max_output_tokens is None — callers pass
    # None to run uncapped (model's full output budget), matching agents
    # whose reference does not send max_output_tokens at all.
    if max_output_tokens is not None:
        request["max_output_tokens"] = max_output_tokens
    if norm_tools:
        request["tools"] = norm_tools
    if resolved_effort:
        request["reasoning"] = {"effort": resolved_effort}
    # resolved_temperature is None when the temperature must be omitted:
    # OpenAI first-party reasoning models reject the combination, and an
    # explicit temperature arg keeps the historical drop-with-effort
    # behaviour (meta-agent call sites). Open-weights reasoners (DeepSeek,
    # GLM, Qwen, gpt-oss) accept reasoning + temperature together.
    if resolved_temperature is not None:
        request["temperature"] = resolved_temperature

    started = time.time()
    last_err: Optional[Exception] = None
    response = None
    for attempt in range(DEFAULT_API_MAX_RETRIES):
        try:
            response = client.responses.create(**request)
            break
        except Exception as exc:  # noqa: BLE001 - retry on any transient error
            last_err = exc
            if attempt == DEFAULT_API_MAX_RETRIES - 1:
                raise
            trace.emit(
                "llm_call_retry",
                {
                    "id": call_id,
                    "attempt": attempt + 1,
                    "max_retries": DEFAULT_API_MAX_RETRIES,
                    "error": repr(exc)[:500],
                },
            )
            time.sleep(DEFAULT_API_BACKOFF_S)
    elapsed = time.time() - started

    content, tool_calls = _extract_output(response)
    stop_reason = getattr(response, "status", None) or getattr(
        response, "stop_reason", None
    )

    usage = getattr(response, "usage", None)
    trace.emit(
        "llm_response",
        {
            "id": call_id,
            "elapsed_s": elapsed,
            "stop_reason": stop_reason,
            "input_tokens": getattr(usage, "input_tokens", None) if usage else None,
            "output_tokens": getattr(usage, "output_tokens", None) if usage else None,
            "reasoning_tokens": (
                getattr(getattr(usage, "output_tokens_details", None), "reasoning_tokens", None)
                if usage
                else None
            ),
            "num_tool_calls": len(tool_calls),
            "content_preview": (content or "")[:200],
        },
    )
    if os.environ.get("META_AGENT_VERBOSE") == "1":
        trace.emit(
            "llm_response_full",
            {
                "id": call_id,
                "content": content,
                "tool_calls": [
                    {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                    for tc in tool_calls
                ],
            },
        )

    return LLMResponse(
        content=content,
        tool_calls=tool_calls,
        stop_reason=stop_reason,
        raw=response,
    )
