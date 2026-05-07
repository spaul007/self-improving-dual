"""Naive single-shot agent — the round-0 seed. Intentionally brittle: one LLM
call, no real loop, fragile final-answer parsing. Easy for the optimizer to
beat.

Interface contract (preserved across rounds):
    def run_task(task: Task) -> AgentOutput
"""
from __future__ import annotations

from platform_core.llm_wrapper import call_llm
from platform_core.runner import AgentOutput, Task

from tool_wrapper import ToolWrapper


def run_task(task: Task) -> AgentOutput:
    wrapper = ToolWrapper()
    schema = wrapper.get_schema()

    system = (
        "You are a helpful assistant. Answer the user's task. "
        "When you have the final answer, write it on a line that starts with "
        "'Final answer:' and nothing else after."
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": task.description},
    ]

    response = call_llm(messages=messages, tools=schema)

    if response.tool_calls:
        call = response.tool_calls[0]
        result = wrapper.execute(call.name, call.arguments)
        return AgentOutput(result=f"Tool {call.name} returned: {result}")

    text = response.content or ""
    for line in text.splitlines():
        line = line.strip()
        if line.lower().startswith("final answer:"):
            return AgentOutput(result=line.split(":", 1)[1].strip())
    return AgentOutput(result=text.strip())
