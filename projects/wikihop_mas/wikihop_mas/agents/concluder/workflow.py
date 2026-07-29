"""Concluder/Reflector agent -- multi-turn tool-calling: aggregates all hop
results, judges per-hop grounding, and (for comparisons between two hop
answers) calls compare_values instead of doing date/number arithmetic itself.

Called at most twice by the controller (mas_workflow.py): once per question,
plus a mandatory second and final call if a hop was retried for grounding.
"""

from dataclasses import dataclass

import config
from agents.base import ToolAgent, ToolAgentOutput
from agents.concluder import prompt
from mas_state import HopResult
from tools.immutable.compare_values import COMPARE_VALUES_TOOL, make_compare_values_handler


def _format_hops_summary(hops: dict[int, HopResult]) -> str:
    lines = []
    for hop_id in sorted(hops):
        h = hops[hop_id]
        lines.append(
            f"Hop {hop_id} -- sub-question: {h.sub_question}\n"
            f"  answer: {h.extractor_answer}\n"
            f'  quote: "{h.extractor_quote}" (source: {h.extractor_source})\n'
            f"  quote verified verbatim in context: {h.quote_verified}"
        )
    return "\n\n".join(lines) if lines else "(no hops completed)"


@dataclass
class ConcluderAgent(ToolAgent):
    name: str = prompt.NAME

    def run(self, question: str, hops: dict[int, HopResult]) -> ToolAgentOutput:
        return self.run_with_tools(
            tools=[COMPARE_VALUES_TOOL],
            tool_handlers={"compare_values": make_compare_values_handler()},
            max_rounds=config.CONCLUDER_MAX_ROUNDS,
            question=question,
            hops_summary=_format_hops_summary(hops),
        )
