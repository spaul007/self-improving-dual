"""Reflector agent — stage 2 of the sequential math MAS."""

from dataclasses import dataclass

from agents.base import AgentOutput, BaseAgent
from agents.reflector import prompt


@dataclass
class ReflectorAgent(BaseAgent):
    """Reviews the predictor's solution, criticizes it, and emits the final answer.

    `context` is the predictor's solution — either its compressed briefing or the
    full text, per `config.USE_COMPRESSED_CONTEXT`. This agent's answer is the
    MAS's answer.
    """

    name: str = prompt.NAME

    async def arun(self, question: str, context: str = "") -> AgentOutput:
        return await super().arun(question, context)
