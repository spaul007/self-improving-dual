"""LLM API wrapper: one place to swap providers.

Wraps an OpenAI-compatible endpoint with retry/backoff, JSON extraction and
a native tool-dispatch loop. The `openai` package is used purely as an HTTP
client — no agent framework. Each agent call is a stateless chat completion
(plus its own tool rounds).

Every message produced inside a call can be mirrored into a `trace` list in
the benchmark's messages.json format, and every completion increments a
shared per-case call counter so the workflow can enforce a hard LLM budget.
"""

import json
import re
import threading
import time

from openai import APIError, APIConnectionError, APITimeoutError, OpenAI, RateLimitError

from config import MASConfig

THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)

RETRYABLE = (APIConnectionError, APITimeoutError, RateLimitError, APIError)


class JSONParseError(Exception):
    pass


class LLMBudgetExceeded(Exception):
    """Raised when a case has spent its cfg.max_llm_calls completions."""


class CallCounter:
    """Per-case LLM-call budget shared by all agents of one task.
    Thread-safe: line-item scouts of one case may run concurrently."""

    def __init__(self, limit: int):
        self.limit = limit
        self.count = 0
        self._lock = threading.Lock()

    def tick(self):
        with self._lock:
            self.count += 1
            over = self.count > self.limit
        if over:
            raise LLMBudgetExceeded(f"exceeded {self.limit} LLM calls for this case")


class LLMClient:
    def __init__(self, cfg: MASConfig):
        self.cfg = cfg
        self.client = OpenAI(base_url=cfg.server.url, api_key=cfg.server.key, timeout=600)

    def chat_raw(self, messages, tools=None, counter: CallCounter | None = None,
                 thinking: bool = True, max_tokens: int | None = None):
        """One completion with retry/backoff.

        Returns (message, finish_reason). `thinking=False` disables the
        model's reasoning pass (vLLM chat-template kwarg) — used for
        mechanical turns where reasoning only costs latency."""
        if counter is not None:
            counter.tick()
        delay = self.cfg.backoff_seconds
        for attempt in range(self.cfg.request_retries + 1):
            try:
                kwargs = {"tools": tools} if tools else {}
                if not thinking:
                    kwargs["extra_body"] = {
                        "chat_template_kwargs": {"enable_thinking": False}}
                resp = self.client.chat.completions.create(
                    model=self.cfg.server.served_model_name,
                    messages=messages,
                    temperature=self.cfg.temperature,
                    max_tokens=max_tokens or self.cfg.max_tokens,
                    **kwargs,
                )
                return resp.choices[0].message, resp.choices[0].finish_reason
            except RETRYABLE:
                if attempt == self.cfg.request_retries:
                    raise
                time.sleep(delay)
                delay *= 2

    def chat_json(self, system: str, user: str, tools=None, tool_handlers=None,
                  trace: list | None = None, counter: CallCounter | None = None,
                  thinking: bool = True) -> dict:
        """Call the model and parse a single JSON object from its final reply.

        With `tools`/`tool_handlers`, runs the tool-dispatch loop: while the
        model emits tool_calls, each is executed via its handler (a callable
        taking the raw JSON-arguments string and returning a result string)
        and appended as a tool message, up to cfg.max_tool_rounds rounds
        (after which tools are withheld so the model must answer).

        Reasoning traces (<think>...</think>) are stripped before parsing.
        On parse failure the model is re-asked with the error appended, up
        to cfg.json_retries times. All messages are mirrored into `trace`.
        """
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        if trace is not None:
            trace.extend(messages)

        def emit(msg):
            messages.append(msg)
            if trace is not None:
                trace.append(msg)

        tool_rounds = 0
        json_attempts = 0
        forced_answer = False   # set after a truncated generation: answer, don't think
        while True:
            offer_tools = tools if (tools and tool_rounds < self.cfg.max_tool_rounds) else None
            if forced_answer:
                offer_tools = None
            msg, finish_reason = self.chat_raw(
                messages, tools=offer_tools, counter=counter,
                thinking=thinking and not forced_answer)

            if msg.tool_calls:
                tool_rounds += 1
                emit({
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": [
                        {"id": tc.id, "type": "function",
                         "function": {"name": tc.function.name,
                                      "arguments": tc.function.arguments}}
                        for tc in msg.tool_calls
                    ],
                })
                for tc in msg.tool_calls:
                    handler = (tool_handlers or {}).get(tc.function.name)
                    if handler is None:
                        result = json.dumps({"error": f"unknown tool {tc.function.name}"},
                                            ensure_ascii=False)
                    else:
                        try:
                            result = handler(tc.function.arguments or "{}")
                        except Exception as e:  # tool bugs must not kill the agent turn
                            result = json.dumps({"error": str(e)}, ensure_ascii=False)
                    emit({"role": "tool", "tool_call_id": tc.id, "content": result})
                continue

            raw = msg.content or ""
            emit({"role": "assistant", "content": raw})
            try:
                return extract_json(raw)
            except JSONParseError as e:
                # Salvage: the model often composes the JSON inside its
                # reasoning channel and then truncates before echoing it
                # into `content`. Reading it is plumbing, not deciding.
                for attr in ("reasoning", "reasoning_content"):
                    hidden = getattr(msg, attr, None)
                    if hidden:
                        try:
                            return extract_json(hidden)
                        except JSONParseError:
                            pass
                json_attempts += 1
                if json_attempts > self.cfg.json_retries:
                    raise JSONParseError(f"no valid JSON after retries: {e}")
                if finish_reason == "length":
                    # Ran out of budget mid-generation: re-ask without tools
                    # and without the reasoning pass so the answer fits.
                    forced_answer = True
                    emit({
                        "role": "user",
                        "content": (
                            "You ran out of output budget before finishing. Answer NOW "
                            "with only the JSON object required by your schema — no "
                            "analysis, no tool calls, no commentary."
                        ),
                    })
                else:
                    emit({
                        "role": "user",
                        "content": (
                            f"Your reply could not be parsed as JSON ({e}). "
                            "Respond again with a single valid JSON object and nothing else."
                        ),
                    })


def extract_json(text: str) -> dict:
    text = THINK_RE.sub("", text).strip()

    candidates = [m.strip() for m in FENCE_RE.findall(text)]
    candidates.append(text)
    start = text.find("{")
    if start != -1:
        candidates.append(_balanced_braces(text, start))

    for cand in candidates:
        if not cand:
            continue
        try:
            obj = json.loads(cand)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    raise JSONParseError("no JSON object found in model output")


def _balanced_braces(text: str, start: int) -> str:
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if escape:
                escape = False
            elif c == "\\":
                escape = True
            elif c == '"':
                in_str = False
        elif c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return ""
