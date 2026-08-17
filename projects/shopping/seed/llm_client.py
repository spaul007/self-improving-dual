"""Chat-Completions-API LLM client for the shopping single-agent seed.

`platform_core.llm_wrapper.call_llm` (Responses API) was found to make
this vLLM deployment send array-typed tool-call arguments back
double-encoded as JSON strings (e.g. `{"product_ids": "[\\"a\\", \\"b\\"]"}`
instead of a native array) -- every tool taking an array parameter
(`get_product_details`, `filter_by_brand`, ...) then silently matched
nothing, since the tool code does `for pid in product_ids` expecting a
list and got individual characters of a string instead. Confirmed live:
100% of a full 120-case baseline scored 0.0 this way, despite the model
choosing correct tool names/call sequences throughout.

`projects/shopping_mas` (the 4-agent vendor MAS, same model/endpoint)
never hit this -- it goes through the Chat Completions API
(`client.chat.completions.create`) via its own vendored `llm_client.py`,
not the Responses API. This module ports that same API choice to the
single-agent seed, as a project-local drop-in replacement for
`platform_core.llm_wrapper.call_llm` -- same public interface
(`call_llm(messages, *, tools=None, ...) -> LLMResponse` with
`.content`/`.tool_calls`/`.stop_reason`/`.raw`), so `workflow.py` only
needs its import line changed plus its Responses-API-specific
message-building (`_append_raw_output`/`_strip_reasoning`, which have no
Chat-Completions equivalent) swapped for the Chat-Completions turn
format. Reads the same `LLM_MODEL`/`LLM_BASE_URL`/`LLM_TEMPERATURE`/
`LLM_MAX_OUTPUT_TOKENS` env vars as `platform_core.llm_wrapper`, so the
existing `task_agent:` config block needs no changes.

Deliberately does not offer a `reasoning_effort`/`enable_thinking` knob:
this project's confirmed-working setting is implicit mode (thinking
determined entirely by the model's own chat-template default, never
toggled explicitly) -- see `qwen35bnotworking.md` and this project's own
`configs/hgm_dual_shopping.yaml` comments for why explicit reasoning
control breaks tool-calling on this endpoint.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

DEFAULT_MODEL_FALLBACK = "gpt-5.4-mini"
DEFAULT_MAX_TOKENS: Optional[int] = None

DEFAULT_API_MAX_RETRIES = 30
DEFAULT_API_BACKOFF_S = 1.5
DEFAULT_REQUEST_TIMEOUT_S = 300.0


def _env_default_model() -> str:
    return os.environ.get("LLM_MODEL", DEFAULT_MODEL_FALLBACK)


def _env_default_base_url() -> Optional[str]:
    val = os.environ.get("LLM_BASE_URL")
    return val if val else None


def _env_default_temperature() -> float:
    val = os.environ.get("LLM_TEMPERATURE")
    if val is None or val == "":
        return 1.0
    return float(val)


def _env_default_max_tokens() -> Optional[int]:
    val = os.environ.get("LLM_MAX_OUTPUT_TOKENS")
    if val is None or val == "":
        return None
    return int(val)


def _env_default_request_timeout() -> float:
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


def _to_chat_completions_tool_schema(tool: dict[str, Any]) -> dict[str, Any]:
    """Normalise any of the shapes this codebase's tool schemas show up in
    (bare {name,description,input_schema/parameters}, Chat-Completions
    {type:function,function:{...}}, Responses-API {type:function,name,...})
    into Chat-Completions' `{"type":"function","function":{...}}` shape."""
    if tool.get("type") == "function" and isinstance(tool.get("function"), dict):
        return tool  # already Chat-Completions shape
    if tool.get("type") == "function" and "name" in tool and "parameters" in tool:
        return {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool["parameters"],
            },
        }
    if "input_schema" in tool and "name" in tool:
        return {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool["input_schema"],
            },
        }
    if "name" in tool and "parameters" in tool:
        return {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool["parameters"],
            },
        }
    raise ValueError(f"Unrecognised tool schema shape: keys={sorted(tool)}")


def call_llm(
    messages: list[dict[str, Any]],
    *,
    tools: Optional[list[dict[str, Any]]] = None,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    base_url: Optional[str] = None,
    max_output_tokens: Optional[int] = DEFAULT_MAX_TOKENS,
    **kwargs: Any,
) -> LLMResponse:
    """One Chat-Completions round-trip, normalised to the same
    `LLMResponse` shape `platform_core.llm_wrapper.call_llm` returns."""
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "openai SDK is required. Install with: pip install 'openai>=1.50'"
        ) from exc

    resolved_model = model or _env_default_model()
    resolved_base_url = base_url or _env_default_base_url()
    resolved_temperature = (
        temperature if temperature is not None else _env_default_temperature()
    )
    resolved_max_tokens = (
        max_output_tokens if max_output_tokens is not None
        else _env_default_max_tokens()
    )

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        if resolved_base_url:
            api_key = "EMPTY"
        else:
            raise RuntimeError("OPENAI_API_KEY environment variable is required")

    norm_tools = [_to_chat_completions_tool_schema(t) for t in (tools or [])]

    client_kwargs: dict[str, Any] = {
        "api_key": api_key,
        "timeout": _env_default_request_timeout(),
    }
    if resolved_base_url:
        client_kwargs["base_url"] = resolved_base_url
    client = OpenAI(**client_kwargs)

    request: dict[str, Any] = {
        "model": resolved_model,
        "messages": messages,
        "temperature": resolved_temperature,
    }
    if resolved_max_tokens is not None:
        request["max_tokens"] = resolved_max_tokens
    if norm_tools:
        request["tools"] = norm_tools

    last_err: Optional[Exception] = None
    response = None
    for attempt in range(DEFAULT_API_MAX_RETRIES):
        try:
            response = client.chat.completions.create(**request)
            break
        except Exception as exc:  # noqa: BLE001 - retry on any transient error
            last_err = exc
            if attempt == DEFAULT_API_MAX_RETRIES - 1:
                raise
            time.sleep(DEFAULT_API_BACKOFF_S)

    msg = response.choices[0].message
    content = msg.content or None

    tool_calls: list[ToolCall] = []
    for tc in (msg.tool_calls or []):
        raw_args = tc.function.arguments or ""
        try:
            args = json.loads(raw_args) if raw_args else {}
        except json.JSONDecodeError:
            args = {"_raw_arguments": raw_args}
        tool_calls.append(
            ToolCall(
                id=tc.id or uuid.uuid4().hex[:12],
                name=tc.function.name or "",
                arguments=args if isinstance(args, dict) else {},
            )
        )

    return LLMResponse(
        content=content,
        tool_calls=tool_calls,
        stop_reason=response.choices[0].finish_reason,
        raw=response,
    )
