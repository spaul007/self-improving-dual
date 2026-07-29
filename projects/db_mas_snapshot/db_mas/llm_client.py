"""Thin async wrapper over the LLM API, including the tool-calling loop.

The one place that talks to the model. Swap providers/endpoints here (and in
config.py) and the whole MAS follows.

Two entry points:
- ``acall``            — plain single-turn completion (lead DBA, compress tool).
- ``acall_with_tools`` — multi-round ReAct loop over native OpenAI
                         tool-calling (the investigators' query_db loop).
"""

import asyncio
import functools
import json
from typing import Any, Callable

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


# ---------------------------------------------------------------------------
# Context-length management for the tool loop. Snapshot table dumps can be tens
# of KB; feeding them all back untrimmed would overflow the context window. We
# (a) cap each single tool result and (b) keep the whole assembled input under
# MAX_INPUT_TOKENS by truncating tool-result contents (oldest first, evenly).
# ---------------------------------------------------------------------------
# Conservative chars-per-token used to convert token budgets to char budgets.
# Under-estimating chars/token makes truncation MORE aggressive => safely under
# the real limit.
_CHARS_PER_TOKEN = 3.0


def _tokens_to_chars(tokens: int) -> int:
    return int(tokens * _CHARS_PER_TOKEN)


def _truncate_text(text: str, max_chars: int) -> str:
    """Truncate `text` to <= max_chars, appending a marker noting the cut."""
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    marker = f"\n...[truncated {len(text) - max_chars} chars]"
    keep = max(0, max_chars - len(marker))
    return text[:keep] + marker


def _message_chars(m: dict[str, Any]) -> int:
    n = len(m.get("content") or "")
    for tc in (m.get("tool_calls") or []):
        n += len((tc.get("function") or {}).get("arguments") or "")
    return n


def _enforce_context_budget(messages: list[dict[str, Any]], max_input_tokens: int) -> None:
    """Shrink tool-result message contents in place so the total stays in budget.

    Only role:"tool" message contents are truncated; the user prompt and
    assistant messages — including tool_calls, which must stay paired with
    their tool responses — are left intact.
    """
    budget = _tokens_to_chars(max_input_tokens)
    total = sum(_message_chars(m) for m in messages)
    if total <= budget:
        return
    tool_msgs = [m for m in messages if m.get("role") == "tool" and m.get("content")]
    if not tool_msgs:
        return
    fixed = total - sum(len(m["content"]) for m in tool_msgs)
    tool_budget = max(0, budget - fixed)
    per_msg = max(200, tool_budget // len(tool_msgs))
    for m in tool_msgs:
        if len(m["content"]) > per_msg:
            m["content"] = _truncate_text(m["content"], per_msg)


def _convert_to_str(result: Any) -> str:
    """Stringify a tool result for feeding back to the model (MARBLE parity)."""
    if isinstance(result, bool):
        return str(result)
    if isinstance(result, (dict, list)):
        return json.dumps(result)
    return str(result)


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

    @async_retry(config.MAX_RETRIES, config.RETRY_DELAY, config.RETRY_MAX_DELAY)
    async def _achat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        tool_choice: str | None,
        temperature: float,
        max_tokens: int,
    ):
        """One chat-completion call, returning the raw message object (so
        tool_calls are preserved)."""
        request_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "extra_body": {
                "top_k": config.TOP_K,
                "chat_template_kwargs": {"enable_thinking": config.ENABLE_THINKING},
            },
        }
        if tools is not None:
            request_kwargs["tools"] = tools
            request_kwargs["tool_choice"] = tool_choice
        async with _get_semaphore():
            resp = await self.raw.chat.completions.create(**request_kwargs)
        return resp.choices[0].message

    async def acall_with_tools(
        self,
        prompt: str,
        tools: list[dict[str, Any]],
        apply_tool_fn: Callable[[str, dict[str, Any]], Any],
        temperature: float | None = None,
        max_tokens: int | None = None,
        max_rounds: int | None = None,
    ) -> tuple[str, list[dict[str, Any]]]:
        """Multi-round ReAct loop over native OpenAI tool-calling.

        The model may call any of `tools`; each call is dispatched via
        `apply_tool_fn` and its result fed back as a role:"tool" message, until
        the model emits a plain text answer or `max_rounds` is reached (then one
        final answer is forced with tools disabled). Returns (final_text,
        tool_log); each log entry is {"name", "args", "replay", "status",
        "result", "result_len"} with `result` truncated for trajectory size —
        the full dump is reproducible from the task's snapshot file.

        Tool handlers here are in-memory snapshot lookups, so they are invoked
        inline (no thread pool needed).
        """
        temperature = config.TEMPERATURE if temperature is None else temperature
        max_tokens = config.MAX_TOKENS if max_tokens is None else max_tokens
        max_rounds = config.TOOL_MAX_ROUNDS if max_rounds is None else max_rounds

        messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
        tool_log: list[dict[str, Any]] = []
        per_result_chars = _tokens_to_chars(config.MAX_TOOL_RESULT_TOKENS)

        for _ in range(max_rounds):
            _enforce_context_budget(messages, config.MAX_INPUT_TOKENS)
            msg = await self._achat(messages, tools, "auto", temperature, max_tokens)
            tool_calls = getattr(msg, "tool_calls", None)
            if not tool_calls:
                return (msg.content or ""), tool_log

            # Record the assistant turn (with its tool_calls) so the follow-up
            # role:"tool" messages have a matching assistant message.
            messages.append({
                "role": "assistant",
                "content": msg.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in tool_calls
                ],
            })

            for tc in tool_calls:
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                except (json.JSONDecodeError, TypeError):
                    args = {}
                result = apply_tool_fn(name, args)
                # `replay` is replay bookkeeping — keep it out of the payload the
                # model sees so the result stays MARBLE-shaped.
                replay = result.pop("replay", None) if isinstance(result, dict) else None
                result_str = _convert_to_str(result)
                tool_log.append({
                    "name": name,
                    "args": args,
                    "replay": replay,
                    "status": result.get("status") if isinstance(result, dict) else None,
                    "result": _truncate_text(result_str, config.TRAJECTORY_TOOL_RESULT_CHARS),
                    "result_len": len(result_str),
                })
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": _truncate_text(result_str, per_result_chars),
                })

        # Rounds exhausted: force one final answer with tools disabled.
        _enforce_context_budget(messages, config.MAX_INPUT_TOKENS)
        msg = await self._achat(messages, None, None, temperature, max_tokens)
        return (msg.content or ""), tool_log

    async def acheck_tool_support(self) -> tuple[bool, str]:
        """Preflight: verify the (vLLM) endpoint accepts OpenAI tool-calling.

        Issues one tiny completion with a dummy tool + tool_choice="auto".
        """
        dummy_tool = [{
            "type": "function",
            "function": {
                "name": "_noop",
                "description": "Return ok.",
                "parameters": {
                    "type": "object",
                    "properties": {"x": {"type": "string"}},
                    "required": [],
                    "additionalProperties": False,
                },
            },
        }]
        try:
            await self.raw.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "Say hi."}],
                max_tokens=8,
                temperature=0.0,
                tools=dummy_tool,
                tool_choice="auto",
            )
            return True, "Tool-calling supported by endpoint."
        except Exception as e:  # noqa: BLE001
            return False, (
                f"Endpoint rejected tool-calling ({type(e).__name__}: {e}).\n"
                "If this is a vLLM server, relaunch it with:\n"
                "  --enable-auto-tool-choice --tool-call-parser hermes\n"
                "(Qwen models use the 'hermes' tool-call parser.)"
            )


_default_client: LLMClient | None = None


def get_client() -> LLMClient:
    """Process-wide default client, created on first use."""
    global _default_client
    if _default_client is None:
        _default_client = LLMClient()
    return _default_client
