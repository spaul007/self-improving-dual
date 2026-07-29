"""Lock investigator — stage 1 of the database RCA MAS."""

from dataclasses import dataclass

from agents.base import AgentOutput, BaseAgent
from agents.lock_investigator import prompt


@dataclass
class LockInvestigatorAgent(BaseAgent):
    """Examines LOCK_CONTENTION as the candidate root cause.

    Runs in parallel with the four other investigators, queries the snapshot
    via query_db (pg_locks, pg_stat_activity), and reports evidence plus a
    high/medium/low likelihood verdict. Never makes the final decision.
    """

    name: str = prompt.NAME
    uses_tools: bool = True

    async def arun(self, question: str, context: str = "") -> AgentOutput:
        return await super().arun(question, context)
