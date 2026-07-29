"""Shared LLM-call-loop / tool-dispatch / transcript-logging machinery used by
both the specialist agents and the coordinator agent."""
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from llm_client import LLMResult, call_llm


@dataclass
class SpecialistFindings:
    agent_id: str
    label: str
    supports_label: bool
    evidence: str
    confidence: float
    forced_fallback: bool = False


@dataclass
class CoordinatorVerdict:
    predicted_root_causes: List[str]
    reasoning: str
    forced_fallback: bool = False
    validation_error: Optional[str] = None


def assistant_tool_call_message(result: LLMResult) -> dict:
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


def tool_result_message(tool_call_id: str, content: str) -> dict:
    return {"role": "tool", "tool_call_id": tool_call_id, "content": content}


def new_usage_totals() -> Dict[str, int]:
    return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def accumulate_usage(totals: Dict[str, int], usage: Dict[str, int]) -> None:
    for k in totals:
        totals[k] += usage.get(k, 0)


def run_tool_loop(
    messages: List[dict],
    tools: List[dict],
    terminal_tool_names: set,
    tool_handlers: Dict[str, Callable[[dict], str]],
    max_turns: int,
    usage_totals: Dict[str, int],
) -> Tuple[Optional[str], dict, bool]:
    """Generic loop: call the LLM with `tools`, dispatch non-terminal tool calls via
    `tool_handlers`, and return as soon as one of `terminal_tool_names` is called.

    Returns (terminal_tool_name, terminal_arguments, forced_fallback).
    """
    for _ in range(max_turns):
        result = call_llm(messages, tools=tools, tool_choice="auto")
        accumulate_usage(usage_totals, result.usage)

        if not result.tool_calls:
            messages.append({"role": "assistant", "content": result.content or ""})
            messages.append(
                {
                    "role": "user",
                    "content": "Please continue by calling one of the available tools.",
                }
            )
            continue

        messages.append(assistant_tool_call_message(result))

        # If a non-terminal tool (e.g. query_db) was called in the same turn as a
        # terminal one (e.g. report_findings), the terminal call's arguments were
        # written *before* that tool's result was seen -- don't accept it as final;
        # let the non-terminal result land and give the model another turn to decide.
        used_handler_this_turn = any(tc.name in tool_handlers for tc in result.tool_calls)

        terminal_call = None
        for tc in result.tool_calls:
            if tc.name in terminal_tool_names:
                if used_handler_this_turn:
                    messages.append(
                        tool_result_message(
                            tc.id,
                            "Deferred: a tool result was just returned above. Please "
                            "consider it, then call this again if you're still done.",
                        )
                    )
                else:
                    terminal_call = tc
                    messages.append(tool_result_message(tc.id, "Recorded."))
            elif tc.name in tool_handlers:
                output = tool_handlers[tc.name](tc.arguments)
                messages.append(tool_result_message(tc.id, output))
            else:
                messages.append(tool_result_message(tc.id, f"Unknown tool: {tc.name}"))
        if terminal_call:
            return terminal_call.name, terminal_call.arguments, False

    # Turn budget exhausted without a terminal tool call: force one final attempt,
    # pinning tool_choice to a terminal tool so the model can't dodge it.
    forced_tool = next(
        t for t in tools if t["function"]["name"] in terminal_tool_names
    )
    forced_name = forced_tool["function"]["name"]
    result = call_llm(
        messages,
        tools=[forced_tool],
        tool_choice={"type": "function", "function": {"name": forced_name}},
    )
    accumulate_usage(usage_totals, result.usage)
    if result.tool_calls:
        tc = result.tool_calls[0]
        messages.append(assistant_tool_call_message(result))
        messages.append(tool_result_message(tc.id, "Recorded (forced)."))
        return tc.name, tc.arguments, True

    return None, {}, True
