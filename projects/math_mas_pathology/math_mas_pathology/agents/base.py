"""Shared agent mechanics: prompt assembly, the LLM call, answer extraction.

Every sub-agent subclasses `BaseAgent` and only declares *which* prompt section
it uses. The run loop itself lives here so a fix lands once for all agents.
"""

from dataclasses import dataclass, field

import config
import llm_client
from tools.immutable.answer_extraction import extract_answer


@dataclass
class AgentOutput:
    """One agent's contribution to a task."""

    agent: str
    prompt: str
    raw: str
    answer: str
    short: str = ""
    # Additive, backward-compatible: free-form per-agent diagnostics that
    # don't belong in the benchmark-facing fields above (e.g. the
    # reflector's selective-deafness bookkeeping, see
    # agents/reflector/workflow.py). None unless an agent explicitly sets it.
    meta: dict | None = None

    def to_dict(self) -> dict:
        return {
            "agent": self.agent,
            "prompt": self.prompt,
            "raw": self.raw,
            "answer": self.answer,
            "short": self.short,
            "meta": self.meta,
        }


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

    async def arun(
        self, question: str, context: str = "", temperature: float | None = None
    ) -> AgentOutput:
        prompt = self.build_prompt(question, context)
        raw = await llm_client.get_client().acall(prompt, temperature=temperature)
        return AgentOutput(
            agent=self.name,
            prompt=prompt,
            raw=raw,
            answer=extract_answer(raw),
        )
