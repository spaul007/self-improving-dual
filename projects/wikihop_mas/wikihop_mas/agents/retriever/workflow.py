"""Retriever agent -- multi-turn tool-calling: search_context over this
question's own paragraphs, closed-book. Bounded to config.RETRIEVER_MAX_ROUNDS
LLM<->tool round-trips (agents/base.py::run_tool_loop).
"""

from dataclasses import dataclass

import config
from agents.base import ToolAgent, ToolAgentOutput
from agents.retriever import prompt
from mas_state import Paragraph
from tools.immutable.search_context import BM25Index, SEARCH_CONTEXT_TOOL, make_search_context_handler


@dataclass
class RetrieverAgent(ToolAgent):
    name: str = prompt.NAME

    def run(self, sub_question: str, index: BM25Index, retry_hint: str = "") -> ToolAgentOutput:
        collected: list[Paragraph] = []
        handler = make_search_context_handler(index, collected)
        out = self.run_with_tools(
            tools=[SEARCH_CONTEXT_TOOL],
            tool_handlers={"search_context": handler},
            max_rounds=config.RETRIEVER_MAX_ROUNDS,
            sub_question=sub_question,
            retry_hint=retry_hint,
        )
        out.retrieved_paragraphs = collected
        return out
