"""Index investigator — stage 1 of the database RCA MAS."""

from dataclasses import dataclass

from agents.base import AgentOutput, BaseAgent
from agents.index_investigator import prompt


@dataclass
class IndexInvestigatorAgent(BaseAgent):
    """Examines REDUNDANT_INDEX as the candidate root cause.

    Runs in parallel with the four other investigators, queries the snapshot
    via query_db (pg_stat_user_indexes, pg_indexes), and reports evidence plus
    a high/medium/low likelihood verdict. Never makes the final decision.
    """

    name: str = prompt.NAME
    uses_tools: bool = True

    async def arun(self, question: str, context: str = "") -> AgentOutput:
        return await super().arun(question, context)
