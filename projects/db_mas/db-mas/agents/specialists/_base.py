"""Shared specialist mechanics. Each of the 5 specialist folders subclasses this
with only its own prompt/label -- the LLM loop and tools are identical."""
from typing import Optional

import config
from agents.base import (
    SpecialistFindings,
    accumulate_usage,
    assistant_tool_call_message,
    new_usage_totals,
    run_tool_loop,
    tool_result_message,
)
from common_tools.immutable.query_db import QUERY_DB_TOOL, query_db
from common_tools.mutable.report_findings import REPORT_FINDINGS_TOOL
from llm_client import call_llm


class SpecialistAgent:
    LABEL: str = ""
    SYSTEM_PROMPT_TEMPLATE: str = ""

    def __init__(self, agent_id: str, task_content: str, max_tool_calls: Optional[int] = None):
        self.agent_id = agent_id
        self.task_content = task_content
        self.max_tool_calls = max_tool_calls or config.MAX_SPECIALIST_TOOL_CALLS
        self.usage = new_usage_totals()
        self.messages = [
            {
                "role": "system",
                "content": self.SYSTEM_PROMPT_TEMPLATE.format(task_content=task_content),
            },
            {
                "role": "user",
                "content": "Please begin your investigation using query_db, then call report_findings when you have enough evidence.",
            },
        ]
        self.findings: Optional[SpecialistFindings] = None

    def run(self) -> SpecialistFindings:
        tools = [QUERY_DB_TOOL, REPORT_FINDINGS_TOOL]
        terminal_name, args, forced = run_tool_loop(
            messages=self.messages,
            tools=tools,
            terminal_tool_names={"report_findings"},
            tool_handlers={"query_db": lambda a: query_db(a.get("sql", ""))},
            max_turns=self.max_tool_calls,
            usage_totals=self.usage,
        )
        if terminal_name == "report_findings":
            self.findings = SpecialistFindings(
                agent_id=self.agent_id,
                # Always the specialist's own fixed assignment, never the model's
                # self-reported echo of it -- which label this agent investigates is a
                # system fact, not something the model's output should be trusted to
                # restate correctly (it could typo it or use the wrong one entirely).
                label=self.LABEL,
                supports_label=bool(args.get("supports_label", False)),
                evidence=args.get("evidence", ""),
                confidence=float(args.get("confidence", 0.0)),
                forced_fallback=forced,
            )
        else:
            self.findings = SpecialistFindings(
                agent_id=self.agent_id,
                label=self.LABEL,
                supports_label=False,
                evidence="The model failed to produce structured findings.",
                confidence=0.0,
                forced_fallback=True,
            )
        return self.findings

    def answer_followup(self, question: str, max_extra_turns: int = 3) -> str:
        """Handle the Coordinator's one allowed follow-up question. May issue a
        few more query_db calls before answering in plain text."""
        self.messages.append(
            {"role": "user", "content": f"[Coordinator follow-up question]: {question}"}
        )
        tools = [QUERY_DB_TOOL]
        for _ in range(max_extra_turns):
            result = call_llm(self.messages, tools=tools, tool_choice="auto")
            accumulate_usage(self.usage, result.usage)
            if not result.tool_calls:
                content = result.content or ""
                if _looks_like_unparsed_tool_call(content):
                    # The server's tool-call parser failed to structure this turn's
                    # output (observed live: a Qwen3.5 turn came back with a raw
                    # "<tool_call><function=query_db>..." template sitting in
                    # `content` instead of populating `tool_calls`). Accepting that
                    # text as a real answer would hand the Coordinator garbage, so
                    # nudge for a plain-text retry instead of returning it as-is.
                    # Record the assistant's own (malformed) turn first -- matching
                    # the same pattern used in run_tool_loop's no-tool-call branch --
                    # so the retry has full context instead of silently vanishing it.
                    self.messages.append({"role": "assistant", "content": content})
                    self.messages.append(
                        {
                            "role": "user",
                            "content": (
                                "That did not come through as a valid answer or tool "
                                "call. Please answer the follow-up question in plain "
                                "text, or call query_db properly if you need one more "
                                "query."
                            ),
                        }
                    )
                    continue
                self.messages.append({"role": "assistant", "content": content})
                return content
            self.messages.append(assistant_tool_call_message(result))
            for tc in result.tool_calls:
                if tc.name == "query_db":
                    output = query_db(tc.arguments.get("sql", ""))
                    self.messages.append(tool_result_message(tc.id, output))

        # Extra-turn budget exhausted: force a plain-text-only reply.
        result = call_llm(self.messages, tools=None)
        accumulate_usage(self.usage, result.usage)
        answer = result.content or ""
        self.messages.append({"role": "assistant", "content": answer})
        return answer


def _looks_like_unparsed_tool_call(content: str) -> bool:
    """Heuristic for a tool-call template that leaked into the text channel
    instead of being parsed into a structured tool_calls entry."""
    markers = ("<tool_call", "<function=", "<|tool_call|>")
    return any(marker in content for marker in markers)
