"""Verifier agent — stage 2 of the pathological 3-stage math MAS.

Implements pathology 1, repetition-then-ignore (see README.md "Communication
Pathologies"): `arun_repeated` calls the inherited single-shot `arun` `n`
times with byte-identical `(question, context)` on every call -- no turn
index, no prior answer, no running history is ever fed back in, so only
sampling temperature can make one turn differ from another. All `n` turns
are computed and kept (for instrumentation), but only the last is ever read
by the rest of the pipeline (`mas_workflow.py`).
"""

from dataclasses import dataclass, field

import config
from agents.base import AgentOutput, BaseAgent
from agents.verifier import prompt


@dataclass
class VerifierAgent(BaseAgent):
    """Independently re-checks the predictor's solution, possibly many times.

    `context` is always the predictor's first-draft solution -- never the
    verifier's own prior turn -- so repeated calls are genuinely "the same
    question asked again", not a refinement loop.
    """

    name: str = prompt.NAME

    async def arun(self, question: str, context: str = "") -> AgentOutput:
        return await super().arun(question, context, temperature=config.VERIFIER_TEMPERATURE)

    async def arun_repeated(self, question: str, context: str, n: int) -> "VerifierResult":
        n = max(1, n)
        turns = [await self.arun(question, context) for _ in range(n)]
        return VerifierResult(
            turns=turns,
            final=turns[-1],
            rounds_run=n,
            repetition_pathology_active=n > 1,
        )


@dataclass
class VerifierResult:
    """All of a verifier's repeated turns, plus which one the pipeline used."""

    turns: list[AgentOutput] = field(default_factory=list)
    final: AgentOutput | None = None
    rounds_run: int = 0
    repetition_pathology_active: bool = False

    def to_dict(self) -> dict:
        return {
            "turns": [t.to_dict() for t in self.turns],
            "final": self.final.to_dict() if self.final else None,
            "rounds_run": self.rounds_run,
            "repetition_pathology_active": self.repetition_pathology_active,
        }
