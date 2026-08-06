"""Thin async wrapper over the LLM API.

The one place that talks to the model. Swap providers/endpoints here (and in
config.py) and the whole MAS follows.
"""

import asyncio
import functools

from openai import AsyncOpenAI

import config

_semaphore: asyncio.Semaphore | None = None


def _get_semaphore() -> asyncio.Semaphore:
    """Lazily create the concurrency limiter bound to the running loop."""
    global _semaphore
    if _semaphore is None:
        _semaphore = asyncio.Semaphore(config.LLM_CONCURRENCY)
    return _semaphore


def async_retry(tries: int, delay: float, max_delay: float):
    def deco(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            d = delay
            for attempt in range(1, tries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception:
                    if attempt == tries:
                        raise
                    await asyncio.sleep(min(max_delay, d))
                    d *= 2
        return wrapper
    return deco


class LLMClient:
    """An AsyncOpenAI client bundled with the model name it should call."""

    def __init__(self, cfg: config.LLMConfig | None = None):
        cfg = cfg or config.MAIN_LLM
        self.model = cfg.model
        self.raw = AsyncOpenAI(api_key=cfg.api_key, base_url=cfg.base_url)

    @async_retry(config.MAX_RETRIES, config.RETRY_DELAY, config.RETRY_MAX_DELAY)
    async def acall(
        self,
        prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        async with _get_semaphore():
            resp = await self.raw.chat.completions.create(
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


_default_client: LLMClient | None = None


def get_client() -> LLMClient:
    """Process-wide default client, created on first use."""
    global _default_client
    if _default_client is None:
        _default_client = LLMClient()
    return _default_client
