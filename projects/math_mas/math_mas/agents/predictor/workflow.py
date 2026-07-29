"""Predictor agent — stage 1 of the sequential math MAS."""

from dataclasses import dataclass

from agents.base import AgentOutput, BaseAgent
from agents.predictor import prompt


@dataclass
class PredictorAgent(BaseAgent):
    """Solves the problem from scratch, step by step.

    Receives no context (it is the first stage) and emits a full solution with
    the final answer in <answer> tags.
    """

    name: str = prompt.NAME

    async def arun(self, question: str, context: str = "") -> AgentOutput:
        return await super().arun(question, context)
