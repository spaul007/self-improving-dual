"""Reflector agent — stage 3 (terminal) of the pathological math MAS.

Implements pathology 3, selective deafness (see README.md "Communication
Pathologies"): before assembling its prompt, the reflector deterministically
truncates whatever `context` it receives down to only its last sentence
(`tools/mutable/deafen.py`) -- dropping every earlier sentence, including
any caveats, hedges, or corrections they contained. This happens regardless
of *which* context object was handed to it (see mas_workflow.py's pathology
2, stale context injection, for which context that actually is).
"""

from dataclasses import dataclass

import config
from agents.base import AgentOutput, BaseAgent
from agents.reflector import prompt
from tools.mutable.deafen import deafen


@dataclass
class ReflectorAgent(BaseAgent):
    """Reviews a proposed solution, criticizes it, and emits the final answer.

    `context` is whatever mas_workflow.py hands it -- either the predictor's
    stale first draft or the verifier's fresher conclusion, per pathology 2.
    This agent's answer is the MAS's answer.
    """

    name: str = prompt.NAME

    def build_prompt(self, question: str, context: str = "") -> str:
        if config.ENABLE_SELECTIVE_DEAFNESS:
            context = deafen(context).last_sentence
        return super().build_prompt(question, context)

    async def arun(self, question: str, context: str = "") -> AgentOutput:
        deafen_result = deafen(context)
        out = await super().arun(question, context)
        out.meta = {
            "context_received": context,
            "context_used": (
                deafen_result.last_sentence if config.ENABLE_SELECTIVE_DEAFNESS else context
            ),
            "sentences_dropped": deafen_result.n_dropped_sentences if config.ENABLE_SELECTIVE_DEAFNESS else 0,
            "chars_dropped": deafen_result.n_dropped_chars if config.ENABLE_SELECTIVE_DEAFNESS else 0,
            "deafness_active": config.ENABLE_SELECTIVE_DEAFNESS,
        }
        return out
