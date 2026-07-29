"""Extractor agent -- single LLM call: pulls a grounded (answer, quote,
source) triple for one hop out of the Retriever's paragraphs.
"""

from dataclasses import dataclass

from agents.base import AgentOutput, BaseAgent
from agents.extractor import prompt
from mas_state import Paragraph


def _format_context(paragraphs: list[Paragraph]) -> str:
    if not paragraphs:
        return "(no sentences retrieved)"
    return "\n".join(f'[title="{p.title}", sent_id={p.sent_id}] "{p.text}"' for p in paragraphs)


@dataclass
class ExtractorAgent(BaseAgent):
    name: str = prompt.NAME

    def run(self, sub_question: str, paragraphs: list[Paragraph]) -> AgentOutput:
        return super().run(sub_question=sub_question, context=_format_context(paragraphs))
