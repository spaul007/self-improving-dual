"""Sole LLM entry point exposed to task agents and meta-agent components.

Backed by the OpenAI Responses API (``client.responses.create``). Accepts tool
schemas in any of three shapes — Responses-API, Chat-Completions, or Anthropic
— and normalises to Responses-API shape internally.

Defaults come from environment variables so the same task-agent code runs in
the meta-agent's parent process and in the evaluator's subprocesses without
needing to thread config through:

    LLM_MODEL              default model name
    LLM_REASONING_EFFORT   "low" | "medium" | "high" (reasoning models only)
    LLM_TEMPERATURE        sampling temperature, only applied when
                           LLM_REASONING_EFFORT is unset (reasoning models
                           reject explicit temperature). Defaults to 1.0.
    LLM_MAX_OUTPUT_TOKENS  cap on completion tokens. Unset -> no cap is
                           sent at all (model's/server's own default).
    LLM_BASE_URL           OpenAI-compatible endpoint to target instead of
                           the SDK default (e.g. a locally hosted vLLM
                           Responses-API server). When set, the
                           ``OPENAI_API_KEY`` check is relaxed because
                           local servers ignore auth — any non-empty
                           string ("EMPTY" by convention) is accepted.

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

# API-error retry policy for `client.responses.create`. Mirrors the reference
# agent's `call_llm` (max_retries=30, backoff=1.5s) so transient
# network/server errors don't kill a case.
DEFAULT_API_MAX_RETRIES = 30
DEFAULT_API_BACKOFF_S = 1.5
# Per-attempt timeout -- bounds how long a single hung/degenerate attempt
# can block before the retry loop's exception handling gets a chance to
# run. Smaller than DEFAULT_OVERALL_TIMEOUT_S on purpose, so more than one
# retry can actually fit inside the overall ceiling below.
#
# 300s, not 60s: 60s turned out too tight for real task-agent generations
# under load -- confirmed live (2026-08-31, node-6, reduced to 4 GPUs from
# 8, i.e. half the compute of when these calls were last tuned): a normal
# multi-turn call sequence had several calls complete in 4-24s, then one
# later call hit APITimeoutError twice in a row and the case crashed with
# an uncaught exception (0 score, no plan) -- the same per-attempt-too-
# tight failure class already found and fixed in scorer_impl.py's
# CONVERT_PER_ATTEMPT_TIMEOUT_S, just discovered here later since this
# file's overall ceiling (below) masked it until the hardware got slower.
# 300s x 2 attempts fits the 600s overall ceiling below.
DEFAULT_REQUEST_TIMEOUT_S = 300.0
# Hard wall-clock ceiling on the WHOLE retry loop, independent of
# DEFAULT_API_MAX_RETRIES or DEFAULT_REQUEST_TIMEOUT_S -- 30 retries x a
# large per-attempt timeout can still legitimately sum to hours even when
# every individual timeout fires correctly (same class of bug fixed in
# projects/travel_mas_refactored/adapter/scorer_impl.py's
# CONVERT_OVERALL_TIMEOUT_S; confirmed live there: a real run stalled 6.6+
# hours in an analogous loop despite its own per-attempt timeout).
#
# 600s, not 300s: doubled alongside DEFAULT_REQUEST_TIMEOUT_S above so two
# full-length attempts still fit inside the ceiling under the current
# halved-compute reality (node-6 at 4 GPUs, not 8) -- a 300s/300s pairing
# would only ever allow a single attempt.
DEFAULT_OVERALL_TIMEOUT_S = 600.0


def _env_default_model() -> str:
    return os.environ.get("LLM_MODEL", DEFAULT_MODEL_FALLBACK)


def _env_default_reasoning_effort() -> Optional[str]:
    val = os.environ.get("LLM_REASONING_EFFORT")
    return val if val else None


def _env_default_base_url() -> Optional[str]:
    val = os.environ.get("LLM_BASE_URL")
    return val if val else None


def _env_default_temperature() -> float:
    val = os.environ.get("LLM_TEMPERATURE")
    if val is None or val == "":
        return 1.0
    return float(val)


def _env_default_max_output_tokens() -> Optional[int]:
    val = os.environ.get("LLM_MAX_OUTPUT_TOKENS")
    if val is None or val == "":
        return None
    return int(val)


def _env_default_provider() -> Optional[dict[str, Any]]:
    """OpenRouter-specific provider-routing preference (``order``/``ignore``/
    ``quantizations``/``allow_fallbacks``, forwarded verbatim as the
    request's ``provider`` field) -- e.g. ``{"ignore": ["DeepInfra"],
    "quantizations": ["bf16"]}`` to steer away from a specific provider or
    quantization level. Confirmed live (2026-09-13) against the real
    OpenRouter API + its /generation stats endpoint: this field is
    genuinely honored (a test call with ignore=["DeepInfra"] was served by
    Venice instead). Local/non-OpenRouter endpoints simply ignore an
    unrecognized ``provider`` field in extra_body, so this is safe to set
    unconditionally once desired -- but defaults to None (todays exact
    behavior) until a caller or the LLM_PROVIDER_PREFERENCE env var
    (a JSON object, same shape) opts in."""
    val = os.environ.get("LLM_PROVIDER_PREFERENCE")
    if not val:
        return None
    try:
        parsed = json.loads(val)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _env_default_request_timeout() -> float:
    """Per-attempt HTTP timeout for the OpenAI client (seconds). Without an
    explicit value, the SDK's own default (~600s) applies -- combined with
    DEFAULT_API_MAX_RETRIES=30, a run of slow/degenerate attempts (e.g. a
    local model generating a long repetitive completion against a large
    max_output_tokens budget) can legitimately stack up to hours before any
    exception is ever raised to trigger the retry-count logic. Confirmed
    live: one case stalled a real hgm_dual run for 3+ hours this way.
    DEFAULT_REQUEST_TIMEOUT_S below caps each individual attempt -- but
    that alone still isn't sufficient (30 retries x even a bounded
    per-attempt timeout can still sum to hours); call_llm's retry loop also
    enforces DEFAULT_OVERALL_TIMEOUT_S as a hard ceiling on the whole loop."""
    val = os.environ.get("LLM_REQUEST_TIMEOUT_S")
    if val is None or val == "":
        return DEFAULT_REQUEST_TIMEOUT_S
    return float(val)


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


def _response_error_info(response: Any) -> tuple[Optional[str], Optional[str]]:
    """Extract (error_code, error_message) from a Responses-API response's
    own ``error`` field (a ``ResponseError`` with ``code``/``message`` --
    populated by the API itself when ``status == "failed"``, distinct
    from a thrown Python exception). Returns (None, None) when absent --
    e.g. a normal successful response, or a provider that doesn't
    populate this field even on failure."""
    error = getattr(response, "error", None)
    if error is None:
        return None, None
    return getattr(error, "code", None), getattr(error, "message", None)


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
    provider: Optional[dict[str, Any]] = None,
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
    resolved_temperature = (
        temperature if temperature is not None else _env_default_temperature()
    )
    resolved_max_output_tokens = (
        max_output_tokens if max_output_tokens is not None
        else _env_default_max_output_tokens()
    )
    resolved_provider = provider if provider is not None else _env_default_provider()

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
            "provider": resolved_provider,
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
        "timeout": _env_default_request_timeout(),
        # The SDK's own default (2) retries silently *inside* a single
        # call, so one manual attempt in the retry loop below could cost
        # up to 3x the configured timeout before ever raising -- confirmed
        # live as part of the same 2026-08-31 investigation that found
        # DEFAULT_REQUEST_TIMEOUT_S=60s too tight (see its comment above).
        # The retry loop below already provides its own retry/backoff,
        # deliberately timed against DEFAULT_OVERALL_TIMEOUT_S.
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
    if resolved_max_output_tokens is not None:
        request["max_output_tokens"] = resolved_max_output_tokens
    if norm_tools:
        request["tools"] = norm_tools
    if resolved_effort:
        # Reasoning models reject explicit temperature; let them default.
        request["reasoning"] = {"effort": resolved_effort}
    else:
        request["temperature"] = resolved_temperature
    if resolved_provider:
        # OpenRouter-specific extension, not part of the OpenAI Responses
        # API schema -- forwarded via extra_body, which the SDK passes
        # through verbatim in the JSON body. Any other OpenAI-compatible
        # endpoint (local vLLM, etc.) simply ignores an unrecognized field.
        request["extra_body"] = {"provider": resolved_provider}

    started = time.time()
    last_err: Optional[Exception] = None
    response = None
    for attempt in range(DEFAULT_API_MAX_RETRIES):
        elapsed_so_far = time.time() - started
        if elapsed_so_far >= DEFAULT_OVERALL_TIMEOUT_S:
            # Hard ceiling this loop exists to guarantee: give up after
            # ~DEFAULT_OVERALL_TIMEOUT_S total, no matter how many of
            # DEFAULT_API_MAX_RETRIES attempts have actually run or how
            # long any single per-attempt timeout takes to fire.
            raise TimeoutError(
                f"call_llm timed out after {elapsed_so_far:.0f}s "
                f"({attempt} attempt(s) made); last error: {last_err!r}"
            ) from last_err
        try:
            response = client.responses.create(**request)
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
            continue
        # A call that returns without raising can still report its own
        # generation as failed (status/stop_reason == "failed", content
        # empty) -- confirmed live (2026-09-11) to happen intermittently
        # with OpenRouter-proxied models under the Responses API (seen with
        # both google/gemma-4-31b-it and, in an earlier run, qwen models
        # accessed the same way -- never with a local vLLM endpoint).
        # Unlike a thrown exception, this was previously accepted as-is and
        # handed downstream unchanged, masquerading as the task agent's own
        # output-quality problem (e.g. sightseeing's "no <itinerary> tag"
        # failure, traced to a 43.75%-vs-1.7% no_plan_rate spike in one
        # HGM round) instead of a transient provider issue. Retry it the
        # same way, against the same budget, unless this was the last
        # attempt -- in which case fall through unchanged, exactly as
        # today, so exhausting retries never raises where it didn't before.
        resp_status = getattr(response, "status", None) or getattr(
            response, "stop_reason", None
        )
        if resp_status == "failed" and attempt < DEFAULT_API_MAX_RETRIES - 1:
            error_code, error_message = _response_error_info(response)
            last_err = RuntimeError(
                f"response status/stop_reason == 'failed' "
                f"(error_code={error_code!r}, error_message={error_message!r}): {response!r}"
            )
            trace.emit(
                "llm_call_retry",
                {
                    "id": call_id,
                    "attempt": attempt + 1,
                    "max_retries": DEFAULT_API_MAX_RETRIES,
                    "error": "response status/stop_reason == 'failed' (no exception raised)",
                    "response_error_code": error_code,
                    "response_error_message": error_message,
                },
            )
            time.sleep(DEFAULT_API_BACKOFF_S)
            continue
        break
    elapsed = time.time() - started

    content, tool_calls = _extract_output(response)
    stop_reason = getattr(response, "status", None) or getattr(
        response, "stop_reason", None
    )
    response_error_code, response_error_message = _response_error_info(response)

    usage = getattr(response, "usage", None)
    trace.emit(
        "llm_response",
        {
            "id": call_id,
            "elapsed_s": elapsed,
            "stop_reason": stop_reason,
            # Populated by the API itself only when stop_reason=="failed"
            # (see _response_error_info) -- None the rest of the time.
            "response_error_code": response_error_code,
            "response_error_message": response_error_message,
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
