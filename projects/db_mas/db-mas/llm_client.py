"""Thin wrapper around the OpenAI Chat Completions API pointed at a vLLM endpoint.

Uses Chat Completions (not the Responses API) since it's the standard,
broadly-supported interface for vLLM-hosted tool calling.
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from openai import APITimeoutError, OpenAI

import config

_client: Optional[OpenAI] = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        # max_retries=0: the SDK's own built-in retry (default 2) silently
        # retries on httpx.TimeoutException too, stacking with call_llm's
        # own retry loop below -- up to 3 physical attempts per logical
        # call, each waiting the full read timeout, invisible in any log.
        # All retry behavior is handled in exactly one place (call_llm).
        _client = OpenAI(base_url=config.BASE_URL, api_key=config.API_KEY, max_retries=0)
    return _client


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class LLMResult:
    content: Optional[str]
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: Optional[str] = None
    usage: dict = field(default_factory=dict)
    raw: Any = None


def _extract_tool_calls(message: Any) -> list[ToolCall]:
    tool_calls = []
    for tc in getattr(message, "tool_calls", None) or []:
        raw_args = tc.function.arguments or ""
        try:
            args = json.loads(raw_args) if raw_args else {}
        except json.JSONDecodeError:
            args = {"_raw_arguments": raw_args}
        tool_calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=args))
    return tool_calls


def call_llm(
    messages: list[dict[str, Any]],
    tools: Optional[list[dict[str, Any]]] = None,
    tool_choice: Any = None,
    model: Optional[str] = None,
    temperature: float = 0.2,
    max_retries: int = 15,
    base_wait_s: float = 1.0,
    max_wait_s: float = 30.0,
) -> LLMResult:
    """One Chat Completions round-trip with capped-exponential-backoff retry.

    Timeouts are NOT retried: a request that already burned the full
    client-side read timeout (600s, see get_client()) without a response
    gets no benefit from an identical retry -- the model host is already
    slow/contended, and firing a duplicate request just adds more load on
    top of whatever is still in flight server-side (the client-side
    timeout does not cancel it). A timeout is logged and raised
    immediately, letting the caller's own fallback logic (e.g. the
    coordinator's forced-fallback path) take over instead of silently
    multiplying wall time with no visibility.

    Retries ARE used for other transient errors (connection resets, 429s,
    5xx) -- these fail fast (no 600s wait involved), so more attempts cost
    little. max_retries=15 restores the effective retry budget the old
    code had by accident: get_client() disables the SDK's own internal
    retry (max_retries=0) specifically to stop it from silently retrying
    ON TIMEOUT (which used to stack with this loop, up to 3x per attempt
    here, causing multi-hour compounding) -- but that same SDK setting used
    to ALSO give 3x "free" attempts against fast-failing errors like 500s,
    which never caused the compounding problem. Restoring that lost
    redundancy at this level (5 attempts x 3 -> 15) fixes the 500-error
    regression measured after the timeout fix, without reintroducing any
    timeout-retry compounding (timeouts still get exactly one attempt).
    Backoff is capped at max_wait_s so a higher max_retries can't blow up
    total wait time (uncapped 2**attempt at 15 retries would reach ~4.5h).
    """
    client = get_client()
    resolved_model = model or config.MODEL

    kwargs: dict[str, Any] = {
        "model": resolved_model,
        "messages": messages,
        "temperature": temperature,
    }
    if tools:
        kwargs["tools"] = tools
    if tool_choice is not None:
        kwargs["tool_choice"] = tool_choice

    last_err: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(**kwargs)
            break
        except APITimeoutError:
            # stderr, not stdout: SubprocessEvaluator parses the LAST line of
            # stdout as the case's JSON envelope and only persists stderr as
            # the debug artifact (case_<id>.stderr) -- a stdout print here
            # would be silently discarded, invisible in every post-hoc log.
            print(
                f"[llm_client] timeout on attempt {attempt + 1}/{max_retries} "
                f"(model={resolved_model}) -- not retrying, raising immediately",
                file=sys.stderr,
                flush=True,
            )
            raise
        except Exception as exc:  # noqa: BLE001 - retry on other transient errors
            last_err = exc
            print(
                f"[llm_client] retryable error on attempt {attempt + 1}/{max_retries} "
                f"(model={resolved_model}): {exc!r}",
                file=sys.stderr,
                flush=True,
            )
            if attempt == max_retries - 1:
                raise
            time.sleep(min(base_wait_s * (2**attempt), max_wait_s))
    else:
        raise last_err  # pragma: no cover - unreachable, loop always breaks or raises

    choice = response.choices[0]
    message = choice.message
    usage = {}
    if response.usage:
        usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }

    return LLMResult(
        content=message.content,
        tool_calls=_extract_tool_calls(message),
        finish_reason=choice.finish_reason,
        usage=usage,
        raw=response,
    )
