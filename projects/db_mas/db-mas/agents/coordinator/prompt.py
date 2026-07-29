from typing import List

from agents.base import SpecialistFindings


def format_specialist_findings(findings: List[SpecialistFindings]) -> str:
    lines = []
    for f in findings:
        lines.append(
            f"- Specialist for label '{f.label}' (agent_id={f.agent_id}): "
            f"supports_label={f.supports_label}, confidence={f.confidence:.2f}\n"
            f"  Evidence: {f.evidence}"
        )
    return "\n".join(lines)


SYSTEM_PROMPT_TEMPLATE = """You are the coordinating agent for a database performance diagnosis task. {n} specialist agents each independently investigated one candidate root cause and reported back to you. Your job is to synthesize their findings and decide the final root cause(s).

Task context:
{task_content}

Candidate labels: {labels}
You must choose EXACTLY {number_of_labels_pred} of these labels as your final predicted root causes.

Specialist reports:
{formatted_findings}

You have direct access to `query_db` at any time, to run your own verification queries against the live database yourself rather than only trusting the specialists' summaries. You may also ask ONE follow-up question to exactly one specialist before deciding, by calling `ask_specialist`. If their reports already give you enough evidence, skip straight to calling `submit_verdict`. Once you are confident in your findings -- having used `query_db` and/or your one follow-up as needed -- you must call `submit_verdict`.
"""


def build_prompt(
    task_content: str,
    labels: List[str],
    number_of_labels_pred: int,
    findings: List[SpecialistFindings],
) -> str:
    return SYSTEM_PROMPT_TEMPLATE.format(
        n=len(findings),
        task_content=task_content,
        labels=", ".join(labels),
        number_of_labels_pred=number_of_labels_pred,
        formatted_findings=format_specialist_findings(findings),
    )
