"""Decomposer agent -- stage 1: classifies the question's reasoning type and
emits a hop-plan (independent-vs-dependent sub-questions) that the controller
in mas_workflow.py dispatches on.

Classifies the question itself; must NOT read the dataset's gold `type` field
at inference time (that would be an oracle shortcut). config.ORACLE_TYPE is
the explicit, opt-in debug/ablation toggle for bypassing this.
"""

from dataclasses import dataclass

import config
from agents.base import AgentOutput, BaseAgent
from agents.decomposer import prompt
from mas_state import HopPlan, SubQuestion


@dataclass
class DecomposerAgent(BaseAgent):
    name: str = prompt.NAME

    def run(self, question: str) -> AgentOutput:
        return super().run(question=question)


def parse_hop_plan(out: AgentOutput, question: str) -> HopPlan:
    """Turn the Decomposer's (possibly malformed) JSON into a HopPlan.

    Never raises: on a parse failure, degrades to a single independent hop
    over the whole question so the pipeline can still produce *some* answer
    rather than losing the task -- same "no sample lost to one bad parse"
    discipline as parse_json_output itself.
    """
    parsed = out.parsed
    if "_parse_error" in parsed:
        return HopPlan(
            predicted_type="unknown", dependency="independent",
            sub_questions=[SubQuestion(hop_id=1, text=question)], raw=out.raw,
        )

    predicted_type = str(parsed.get("type", "unknown"))
    dependency = parsed.get("dependency")
    if dependency not in ("independent", "dependent"):
        dependency = "dependent" if predicted_type in config.DEPENDENT_TYPES else "independent"

    sub_questions = []
    for i, sq in enumerate(parsed.get("sub_questions", []) or [], start=1):
        if not isinstance(sq, dict):
            continue
        sub_questions.append(SubQuestion(
            hop_id=int(sq.get("hop_id", i)),
            text=str(sq.get("question", "")),
            depends_on=sq.get("depends_on"),
        ))
    if not sub_questions:
        sub_questions = [SubQuestion(hop_id=1, text=question)]
        dependency = "independent"

    return HopPlan(predicted_type=predicted_type, dependency=dependency,
                    sub_questions=sub_questions, raw=out.raw)
