"""Lead DBA — stage 2 (terminal) of the database RCA MAS."""

from dataclasses import dataclass

from agents.base import AgentOutput, BaseAgent
from agents.lead_dba import prompt


@dataclass
class LeadDBAAgent(BaseAgent):
    """Reconciles the five investigators' evidence into the final diagnosis.

    `context` is the five briefings (compressed by default, full reports when
    `MAS_USE_COMPRESSED_CONTEXT=0`). Has NO tools — it only weighs evidence,
    matching MARBLE/crewai where the lead does not query the database. Its
    answer is the MAS's answer and must end with the `FINAL: <LABEL>[, ...]`
    line that tools/immutable/label_extraction.py reads.
    """

    name: str = prompt.NAME
    uses_tools: bool = False

    async def arun(self, question: str, context: str = "") -> AgentOutput:
        return await super().arun(question, context)
