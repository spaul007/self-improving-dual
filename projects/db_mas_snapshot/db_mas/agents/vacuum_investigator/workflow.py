"""Vacuum investigator — stage 1 of the database RCA MAS."""

from dataclasses import dataclass

from agents.base import AgentOutput, BaseAgent
from agents.vacuum_investigator import prompt


@dataclass
class VacuumInvestigatorAgent(BaseAgent):
    """Examines VACUUM (inappropriate/aggressive vacuuming) as the candidate
    root cause.

    Runs in parallel with the four other investigators, queries the snapshot
    via query_db (pg_stat_all_tables / pg_stat_user_tables, and VACUUM
    statements in pg_stat_statements), and reports evidence plus a
    high/medium/low likelihood verdict. Never makes the final decision.
    """

    name: str = prompt.NAME
    uses_tools: bool = True

    async def arun(self, question: str, context: str = "") -> AgentOutput:
        return await super().arun(question, context)
