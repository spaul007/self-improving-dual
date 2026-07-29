"""Shared agent mechanics: prompt assembly, the LLM call, the tool loop.

Every sub-agent subclasses `BaseAgent` and only declares *which* prompt section
it uses and whether it may call the query_db tool. The run loop itself lives
here so a fix lands once for all agents.
"""

from dataclasses import dataclass, field
from typing import Any

import config
import llm_client
from tools.immutable.query_db import TOOL_DESCRIPTIONS, apply_tool


@dataclass
class AgentOutput:
    """One agent's contribution to a task."""

    agent: str
    prompt: str
    raw: str
    answer: str
    short: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "agent": self.agent,
            "prompt": self.prompt,
            "raw": self.raw,
            "answer": self.answer,
            "short": self.short,
            "tool_calls": self.tool_calls,
        }


@dataclass
class BaseAgent:
    """An LLM agent defined by a frozen `role` and an editable `task` template.

    Subclasses set `name` (matching mas_prompt_cfg.yaml) and `uses_tools`.
    Tool-using agents run the query_db ReAct loop against the task's bound
    snapshot; the others make one plain completion.
    """

    name: str = ""
    uses_tools: bool = False
    role: str = field(default="", init=False)
    task: str = field(default="", init=False)

    def __post_init__(self):
        if not self.name:
            raise ValueError("BaseAgent requires a `name` matching mas_prompt_cfg.yaml")
        self.role, self.task = config.agent_prompt(self.name)

    def build_prompt(self, question: str, context: str = "") -> str:
        """Frozen role instruction, then the editable task instruction."""
        try:
            body = self.task.format(question=question, context=context)
        except (KeyError, IndexError):
            # A hand-edited task template may contain stray braces; degrade to
            # literal substitution rather than failing the whole task.
            body = self.task.replace("{question}", str(question)).replace(
                "{context}", str(context)
            )
        return f"{self.role}\n\n{body}"

    async def arun(self, question: str, context: str = "") -> AgentOutput:
        prompt = self.build_prompt(question, context)
        tool_calls: list[dict[str, Any]] = []
        if self.uses_tools and config.TOOLS_ENABLED:
            raw, tool_calls = await llm_client.get_client().acall_with_tools(
                prompt, TOOL_DESCRIPTIONS, apply_tool
            )
        else:
            raw = await llm_client.get_client().acall(prompt)
        # The "answer" of a database agent is its report/diagnosis verbatim:
        # evidence reports have no extractable span, and the lead's labels are
        # read out by tools/immutable/label_extraction.py at scoring time.
        return AgentOutput(
            agent=self.name,
            prompt=prompt,
            raw=raw,
            answer=(raw or "").strip(),
            tool_calls=tool_calls,
        )
