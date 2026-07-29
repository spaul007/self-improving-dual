from typing import Dict, List, Optional

import config
from agents.base import (
    CoordinatorVerdict,
    SpecialistFindings,
    accumulate_usage,
    assistant_tool_call_message,
    new_usage_totals,
    tool_result_message,
)
from agents.coordinator.prompt import build_prompt
from agents.coordinator.tools import ASK_SPECIALIST_TOOL, SUBMIT_VERDICT_TOOL
from common_tools.immutable.query_db import QUERY_DB_TOOL, query_db
from llm_client import call_llm


class CoordinatorAgent:
    def __init__(
        self,
        task_content: str,
        labels: List[str],
        number_of_labels_pred: int,
        specialist_findings: List[SpecialistFindings],
        specialists_by_id: Dict[str, object],
        max_followups: Optional[int] = None,
    ):
        self.task_content = task_content
        self.labels = labels
        self.number_of_labels_pred = number_of_labels_pred
        self.specialist_findings = specialist_findings
        self.specialists_by_id = specialists_by_id
        self.max_followups = (
            max_followups if max_followups is not None else config.MAX_COORDINATOR_FOLLOWUPS
        )
        self.usage = new_usage_totals()
        self.messages = [
            {
                "role": "system",
                "content": build_prompt(task_content, labels, number_of_labels_pred, specialist_findings),
            },
            {
                "role": "user",
                "content": "Please review the specialist reports. You may call query_db to verify anything directly, ask_specialist for one follow-up, or submit_verdict when ready.",
            },
        ]
        self.verdict: Optional[CoordinatorVerdict] = None

    def _call(self, tools, tool_choice="auto"):
        result = call_llm(self.messages, tools=tools, tool_choice=tool_choice)
        accumulate_usage(self.usage, result.usage)
        return result

    def _validate(self, predicted: List[str]) -> Optional[str]:
        if len(predicted) != self.number_of_labels_pred:
            return f"expected exactly {self.number_of_labels_pred} labels, got {len(predicted)}"
        invalid = [l for l in predicted if l not in self.labels]
        if invalid:
            return f"labels not in candidate set {self.labels}: {invalid}"
        return None

    def run(self) -> CoordinatorVerdict:
        followups_used = 0
        tools = [QUERY_DB_TOOL, ASK_SPECIALIST_TOOL, SUBMIT_VERDICT_TOOL]
        max_turns = self.max_followups + config.MAX_COORDINATOR_QUERY_CALLS + 2
        verdict_args = None
        forced = False

        for _ in range(max_turns):
            result = self._call(tools, tool_choice="auto")
            if not result.tool_calls:
                self.messages.append({"role": "assistant", "content": result.content or ""})
                self.messages.append(
                    {
                        "role": "user",
                        "content": "Please call query_db, ask_specialist, or submit_verdict.",
                    }
                )
                continue

            self.messages.append(assistant_tool_call_message(result))

            # If the model bundled query_db or ask_specialist together with
            # submit_verdict in this same turn (parallel tool calls), the verdict was
            # decided *before* that new information existed -- it must not be accepted
            # as final. Detect that case first so the submit_verdict branch below can
            # defer instead of return.
            used_info_gathering_this_turn = any(
                tc.name == "query_db"
                or (tc.name == "ask_specialist" and followups_used < self.max_followups)
                for tc in result.tool_calls
            )

            submit_call = None
            for tc in result.tool_calls:
                if tc.name == "submit_verdict":
                    if used_info_gathering_this_turn:
                        self.messages.append(
                            tool_result_message(
                                tc.id,
                                "Deferred: new information (a query result or follow-up "
                                "answer) was just returned above. Please review it, then "
                                "call submit_verdict again.",
                            )
                        )
                    else:
                        submit_call = tc
                        self.messages.append(tool_result_message(tc.id, "Recorded."))
                elif tc.name == "query_db":
                    output = query_db(tc.arguments.get("sql", ""))
                    self.messages.append(tool_result_message(tc.id, output))
                elif tc.name == "ask_specialist" and followups_used < self.max_followups:
                    followups_used += 1
                    agent_id = tc.arguments.get("agent_id")
                    question = tc.arguments.get("question", "")
                    specialist = self.specialists_by_id.get(agent_id)
                    if specialist is None:
                        answer = (
                            f"No specialist with agent_id '{agent_id}' found. "
                            f"Valid agent_ids: {list(self.specialists_by_id)}"
                        )
                    else:
                        answer = specialist.answer_followup(question)
                    self.messages.append(tool_result_message(tc.id, answer))
                    # The one allowed follow-up is used up; query_db remains available.
                    tools = [QUERY_DB_TOOL, SUBMIT_VERDICT_TOOL]
                else:
                    self.messages.append(
                        tool_result_message(tc.id, "This tool is no longer available.")
                    )
            if submit_call:
                verdict_args = submit_call.arguments
                break

        if verdict_args is None:
            result = self._call(
                [SUBMIT_VERDICT_TOOL],
                tool_choice={"type": "function", "function": {"name": "submit_verdict"}},
            )
            if result.tool_calls:
                tc = result.tool_calls[0]
                self.messages.append(assistant_tool_call_message(result))
                self.messages.append(tool_result_message(tc.id, "Recorded (forced)."))
                verdict_args = tc.arguments
            else:
                verdict_args = {
                    "predicted_root_causes": [],
                    "reasoning": "Model failed to produce a verdict.",
                }
            forced = True

        predicted = verdict_args.get("predicted_root_causes", [])
        reasoning = verdict_args.get("reasoning", "")
        validation_error = self._validate(predicted)

        if validation_error:
            self.messages.append(
                {
                    "role": "user",
                    "content": (
                        f"Invalid verdict: {validation_error}. You must choose exactly "
                        f"{self.number_of_labels_pred} labels from {self.labels}. Please call "
                        "submit_verdict again with a corrected list."
                    ),
                }
            )
            result = self._call(
                [SUBMIT_VERDICT_TOOL],
                tool_choice={"type": "function", "function": {"name": "submit_verdict"}},
            )
            if result.tool_calls:
                tc = result.tool_calls[0]
                self.messages.append(assistant_tool_call_message(result))
                self.messages.append(tool_result_message(tc.id, "Recorded."))
                predicted = tc.arguments.get("predicted_root_causes", predicted)
                reasoning = tc.arguments.get("reasoning", reasoning)
                validation_error = self._validate(predicted)

        self.verdict = CoordinatorVerdict(
            predicted_root_causes=predicted,
            reasoning=reasoning,
            forced_fallback=forced,
            validation_error=validation_error,
        )
        return self.verdict
