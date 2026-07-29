"""Shared agent mechanics: prompt assembly, LLM calls, tolerant JSON parsing.

`BaseAgent` is single-turn (Decomposer, Extractor): build one prompt, call the
LLM once, parse JSON out of the response. `ToolAgent` extends it with
`run_with_tools` for multi-turn LLM<->tool loops (Retriever, Concluder). The
run loops live here so a fix lands once for every agent that uses them.

wikihop_mas uses structured JSON hand-offs between agents (not math_mas's
<answer> tag convention) because the controller branches on typed, multi-field
output at several points -- see mas_prompt_cfg.yaml for the schema-by-example
each agent's task prompt embeds.
"""

import json
from dataclasses import dataclass, field
from typing import Callable

import config
import llm_client


# --------------------------------------------------------------------------
# Tolerant JSON extraction
# --------------------------------------------------------------------------
def parse_json_output(raw: str) -> dict:
    """Extract the first balanced {...} block from `raw` and json.loads it.

    Strips ``` code fences first. Never raises -- on failure returns
    {"_parse_error": raw} so one bad sample doesn't kill the whole task.
    """
    if not raw:
        return {"_parse_error": raw}

    text = raw.strip()
    if text.startswith("```"):
        parts = text.split("```")
        if len(parts) >= 2:
            text = parts[1]
            if text.lstrip().lower().startswith("json"):
                text = text.lstrip()[4:]

    start = text.find("{")
    if start == -1:
        return {"_parse_error": raw}

    depth = 0
    in_string = False
    escape = False
    end = None
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break

    if end is None:
        return {"_parse_error": raw}
    try:
        return json.loads(text[start:end])
    except json.JSONDecodeError:
        return {"_parse_error": raw}


# --------------------------------------------------------------------------
# Single-turn agents (Decomposer, Extractor)
# --------------------------------------------------------------------------
@dataclass
class AgentOutput:
    agent: str
    prompt: str
    raw: str
    parsed: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"agent": self.agent, "prompt": self.prompt, "raw": self.raw, "parsed": self.parsed}


@dataclass
class BaseAgent:
    """An LLM agent defined by a frozen `role` and an editable `task` template.

    Subclasses set `name`; role/task are loaded from mas_prompt_cfg.yaml.
    """

    name: str = ""
    role: str = field(default="", init=False)
    task: str = field(default="", init=False)

    def __post_init__(self):
        if not self.name:
            raise ValueError("BaseAgent requires a `name` matching mas_prompt_cfg.yaml")
        self.role, self.task = config.agent_prompt(self.name)

    def format_task(self, **kwargs) -> str:
        """Fill the editable task template. Shared by build_prompt (single-turn
        agents) and ToolAgent.run_with_tools (tool-calling agents' user message)."""
        try:
            return self.task.format(**kwargs)
        except (KeyError, IndexError):
            # A hand-edited task template may contain stray braces; degrade to
            # literal substitution rather than failing the whole task.
            body = self.task
            for k, v in kwargs.items():
                body = body.replace("{" + k + "}", str(v))
            return body

    def build_prompt(self, **kwargs) -> str:
        """Frozen role instruction, then the editable task instruction."""
        return f"{self.role}\n\n{self.format_task(**kwargs)}"

    def run(self, **kwargs) -> AgentOutput:
        prompt = self.build_prompt(**kwargs)
        raw = llm_client.get_client().call(prompt)
        return AgentOutput(agent=self.name, prompt=prompt, raw=raw, parsed=parse_json_output(raw))


# --------------------------------------------------------------------------
# Multi-turn tool-calling agents (Retriever, Concluder)
# --------------------------------------------------------------------------
@dataclass
class ToolAgentOutput:
    agent: str
    raw: str
    parsed: dict = field(default_factory=dict)
    trace: list[dict] = field(default_factory=list)
    rounds_used: int = 0
    retrieved_paragraphs: list = field(default_factory=list)   # Retriever-only; empty for Concluder

    def to_dict(self) -> dict:
        return {
            "agent": self.agent, "raw": self.raw, "parsed": self.parsed,
            "trace": self.trace, "rounds_used": self.rounds_used,
        }


@dataclass
class ToolAgent(BaseAgent):
    """Base for agents that do multi-turn LLM<->tool-calling."""

    def run_with_tools(
        self,
        tools: list[dict],
        tool_handlers: dict[str, Callable[[dict], str]],
        max_rounds: int,
        **kwargs,
    ) -> ToolAgentOutput:
        messages = [
            {"role": "system", "content": self.role},
            {"role": "user", "content": self.format_task(**kwargs)},
        ]
        result, trace = run_tool_loop(messages, tools, tool_handlers, max_rounds)
        raw = result.content or ""
        return ToolAgentOutput(
            agent=self.name, raw=raw, parsed=parse_json_output(raw),
            trace=trace, rounds_used=_rounds_used(trace),
        )


def run_tool_loop(
    messages: list[dict],
    tools: list[dict],
    tool_handlers: dict[str, Callable[[dict], str]],
    max_rounds: int,
) -> tuple["llm_client.LLMResult", list[dict]]:
    """Bounded LLM<->tool loop: model proposes tool_calls, we execute them and
    feed results back, until the model stops calling tools or `max_rounds` is
    hit (in which case a final tool-free turn is forced)."""
    trace: list[dict] = []
    for round_idx in range(max_rounds):
        result = llm_client.get_client().call_messages(messages, tools=tools, tool_choice="auto")
        if not result.tool_calls:
            return result, trace
        messages.append(_assistant_tool_call_message(result))
        for tc in result.tool_calls:
            handler = tool_handlers.get(tc.name)
            output = handler(tc.arguments) if handler else f"Unknown tool: {tc.name}"
            trace.append({"round": round_idx, "tool": tc.name, "arguments": tc.arguments, "result": output})
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": output})
    # Round cap hit while the model still wanted to call tools: force a text-only final turn.
    final = llm_client.get_client().call_messages(messages, tools=None, tool_choice="none")
    return final, trace


def _assistant_tool_call_message(result: "llm_client.LLMResult") -> dict:
    return {
        "role": "assistant",
        "content": result.content,
        "tool_calls": [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
            }
            for tc in result.tool_calls
        ],
    }


def _rounds_used(trace: list[dict]) -> int:
    return len({t["round"] for t in trace}) if trace else 0
