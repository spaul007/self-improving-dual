"""Thin synchronous wrapper over the LLM API.

The one place that talks to the model. Swap providers/endpoints here (and in
config.py) and the whole MAS follows.

Unlike math_mas's llm_client.py (async, AsyncOpenAI), this is fully
synchronous — plain `OpenAI` client, plain `time.sleep` backoff, no
asyncio/semaphore machinery. Batch throughput across questions, if wanted,
is a plain ThreadPoolExecutor in mas_workflow.run_many.
"""

import functools
import json
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI

import config


def retry(tries: int, delay: float, max_delay: float):
    def deco(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            d = delay
            for attempt in range(1, tries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception:
                    if attempt == tries:
                        raise
                    time.sleep(min(max_delay, d))
                    d *= 2
        return wrapper
    return deco


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class LLMResult:
    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str | None = None


def _extract_tool_calls(message: Any) -> list[ToolCall]:
    out = []
    for tc in getattr(message, "tool_calls", None) or []:
        raw_args = tc.function.arguments or ""
        try:
            args = json.loads(raw_args) if raw_args else {}
        except json.JSONDecodeError:
            args = {"_raw_arguments": raw_args}
        out.append(ToolCall(id=tc.id, name=tc.function.name, arguments=args))
    return out


class LLMClient:
    """A synchronous OpenAI client bundled with the model name it should call."""

    def __init__(self, cfg: "config.LLMConfig | None" = None):
        cfg = cfg or config.MAIN_LLM
        self.model = cfg.model
        self.raw = OpenAI(api_key=cfg.api_key, base_url=cfg.base_url)

    @retry(config.MAX_RETRIES, config.RETRY_DELAY, config.RETRY_MAX_DELAY)
    def call(
        self,
        prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Single-turn, no tools — used by Decomposer/Extractor."""
        resp = self.raw.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=config.TEMPERATURE if temperature is None else temperature,
            max_tokens=config.MAX_TOKENS if max_tokens is None else max_tokens,
            extra_body={
                "top_k": config.TOP_K,
                "chat_template_kwargs": {"enable_thinking": config.ENABLE_THINKING},
            },
        )
        return resp.choices[0].message.content or ""

    @retry(config.MAX_RETRIES, config.RETRY_DELAY, config.RETRY_MAX_DELAY)
    def call_messages(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        tool_choice: Any = "auto",
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResult:
        """Multi-turn, optionally with tools — used by Retriever/Concluder."""
        kwargs = dict(
            model=self.model,
            messages=messages,
            temperature=config.TEMPERATURE if temperature is None else temperature,
            max_tokens=config.MAX_TOKENS if max_tokens is None else max_tokens,
            extra_body={
                "top_k": config.TOP_K,
                "chat_template_kwargs": {"enable_thinking": config.ENABLE_THINKING},
            },
        )
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice
        resp = self.raw.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        return LLMResult(
            content=choice.message.content,
            tool_calls=_extract_tool_calls(choice.message),
            finish_reason=choice.finish_reason,
        )


_default_client: LLMClient | None = None
_client_lock = threading.Lock()


def get_client() -> LLMClient:
    """Process-wide default client, created on first use.

    Guarded by a lock: mas_workflow.run_many drives this from a
    ThreadPoolExecutor, so the first few worker threads can race on the
    check-then-create below without it -- unlike math_mas's asyncio version,
    which is single-threaded and never had this race.
    """
    global _default_client
    if _default_client is None:
        with _client_lock:
            if _default_client is None:
                _default_client = LLMClient()
    return _default_client
