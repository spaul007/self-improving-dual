"""LLM access from the host.

Pier has no first-party LLM client for agents -- every built-in adapter shells out to a
CLI installed in the container. We are host-orchestrated, so we call litellm directly.
litellm is a hard dependency of pier (1.98.0 in this venv), so nothing new is installed.

MUST be async: trials run as concurrent asyncio tasks in ONE event loop bounded by a
semaphore, so a synchronous litellm.completion() would stall EVERY other concurrent trial.
"""

from __future__ import annotations

import asyncio

from . import settings


class LLMError(Exception):
    """Raised inside a role; the role loop contains it. Never escapes run()."""


class LLM:
    """One instance per trial. No module-level state (concurrent trials share a loop)."""

    def __init__(self, model: str, api_base: str | None, logger,
                 api_key: str = "dummy", deadline=None) -> None:
        self.model = model
        self.api_base = api_base
        self.api_key = api_key
        self.logger = logger
        self.deadline = deadline
        self.n_calls = 0
        self.n_retries = 0
        self.n_context_exceeded = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.cached_tokens = 0
        self.cost_usd = 0.0

    def _kwargs(self) -> dict:
        return {
            "model": self.model,
            "api_base": self.api_base,
            "api_key": self.api_key,
            "max_tokens": settings.MAX_TOKENS,
            "timeout": settings.LLM_TIMEOUT_SEC,
            "drop_params": True,
            **settings.SAMPLING,
            "extra_body": {
                "chat_template_kwargs": {"reasoning_effort": settings.REASONING_EFFORT}
            },
        }

    async def chat(self, messages: list[dict], tools: list[dict] | None = None,
                   tool_choice=None, thinking: bool = True) -> dict:
        """Returns {content, reasoning, tool_calls, usage}. Raises LLMError on failure.

        thinking=False turns Qwen's <think> block OFF for this call (`enable_thinking: false`
        via chat_template_kwargs). Used by the summariser: EXP-027 yaegi's summariser
        deliberated for 8-25K chars and then emitted NOTHING outside the think block, four
        times in a row (`finish=stop`, not `length`); a summary needs no deliberation."""
        import litellm

        kw = self._kwargs()
        if not thinking:
            kw["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
        if tools:
            kw["tools"] = tools
            # "auto" lets the model answer in prose instead of calling the tool. The report
            # call needs the finish schema FORCED -- with "auto", 3 of 3 wall-terminated
            # roles in v8 smoke run 5 just kept working and fell to _synthesize.
            kw["tool_choice"] = tool_choice or "auto"

        last = None
        for attempt in range(settings.LLM_MAX_RETRIES):
            if self.deadline is not None and self.deadline.expired():
                raise LLMError("soft deadline reached before LLM call")
            try:
                r = await litellm.acompletion(messages=messages, **kw)
                self.n_calls += 1
                return self._unpack(r)
            except asyncio.CancelledError:
                raise
            except BaseException as e:  # noqa: BLE001
                last = e
                name = type(e).__name__
                if "ContextWindow" in name or "context_length" in str(e).lower():
                    self.n_context_exceeded += 1
                    raise LLMError(f"context window exceeded: {e}") from e
                self.n_retries += 1
                self.logger.warning("LLM call failed (%s), attempt %d/%d: %s",
                                    name, attempt + 1, settings.LLM_MAX_RETRIES, e)
                await asyncio.sleep(min(2 ** attempt, 15))
        raise LLMError(f"LLM failed after {settings.LLM_MAX_RETRIES} attempts: {last}")

    def _unpack(self, r) -> dict:
        choice = r.choices[0]
        msg = choice.message
        usage = getattr(r, "usage", None)
        if usage:
            self.prompt_tokens += getattr(usage, "prompt_tokens", 0) or 0
            self.completion_tokens += getattr(usage, "completion_tokens", 0) or 0
            # vLLM reports prefix-cache hits under prompt_tokens_details.cached_tokens.
            # This is the metric that prices COMPACTION honestly: a compaction rewrites the
            # head of the conversation, so the shared prefix is destroyed and the next call
            # re-prefills from the summary. A run where cached_tokens collapses after each
            # compaction is paying far more than the summary call alone suggests.
            det = getattr(usage, "prompt_tokens_details", None)
            if det is not None:
                self.cached_tokens += (getattr(det, "cached_tokens", 0) or 0)
        # None, not 0, when the server does not report it: server C returns
        # `prompt_tokens_details: null` on every response while its /metrics shows an 18%
        # prefix-cache hit rate -- a 0 here was the falsy-zero trap for every earlier run.
        _cached_this_call = None
        if usage is not None:
            _det = getattr(usage, "prompt_tokens_details", None)
            if _det is not None:
                _cached_this_call = getattr(_det, "cached_tokens", 0) or 0
        calls = []
        for tc in (getattr(msg, "tool_calls", None) or []):
            import json
            args = getattr(tc.function, "arguments", "") or "{}"
            try:
                args = json.loads(args) if isinstance(args, str) else args
            except Exception:  # noqa: BLE001 -- a malformed args blob must not kill the role
                args = {"_raw": str(args)}
            calls.append({"id": getattr(tc, "id", ""), "name": tc.function.name,
                          "arguments": args})
        return {
            # WHY finish_reason IS LOAD-BEARING: a reasoning model can spend the entire
            # max_tokens budget inside its thinking block and return content="" with no tool
            # call and finish_reason="length". That is a TRUNCATED generation, not a decision
            # to stop -- but it is indistinguishable from a real stop if you only look at
            # `tool_calls`. Measured on the v8 smoke: two turns with 30,526 and 33,410 chars
            # of reasoning (~8.3K tokens against MAX_TOKENS=8192) were read as `end_turn`,
            # and PATCH ended with 0 edits on 3 of 3 tasks.
            "finish_reason": getattr(choice, "finish_reason", None),
            "content": getattr(msg, "content", None) or "",
            # vLLM 0.20.2 returns `reasoning`, NOT `reasoning_content`.
            "reasoning": getattr(msg, "reasoning", None)
                         or getattr(msg, "reasoning_content", None),
            "tool_calls": calls,
            "usage": {"prompt_tokens": getattr(usage, "prompt_tokens", 0) if usage else 0,
                      "completion_tokens": getattr(usage, "completion_tokens", 0) if usage else 0,
                      "cached_tokens": _cached_this_call},
        }

    def snapshot(self) -> dict:
        return {"n_calls": self.n_calls, "n_retries": self.n_retries,
                "n_context_exceeded": self.n_context_exceeded,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "cached_tokens": self.cached_tokens,
                "cost_usd": self.cost_usd}
