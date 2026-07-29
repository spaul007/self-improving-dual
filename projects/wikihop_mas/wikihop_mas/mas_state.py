"""Shared "blackboard" state for the wikihop MAS controller.

math_mas's hand-off is a simple chain of two `AgentOutput`s; wikihop_mas needs
more because the controller branches (independent vs. chained hops) and loops
(bounded grounding retry), so every stage's output is accumulated on one
`MASState` object instead of being passed hand-to-hand.
"""

from dataclasses import dataclass, field
from typing import Literal, Optional

Dependency = Literal["independent", "dependent"]


@dataclass
class SubQuestion:
    hop_id: int
    text: str                          # may contain "{hop1_answer}" for a dependent hop2
    depends_on: Optional[int] = None   # hop_id it depends on, else None


@dataclass
class HopPlan:
    predicted_type: str                # Decomposer's own classification, NOT the gold label
    dependency: Dependency
    sub_questions: list[SubQuestion]
    raw: str = ""                      # Decomposer's raw LLM output, for the trajectory log

    def to_dict(self) -> dict:
        return {
            "predicted_type": self.predicted_type,
            "dependency": self.dependency,
            "sub_questions": [
                {"hop_id": sq.hop_id, "text": sq.text, "depends_on": sq.depends_on}
                for sq in self.sub_questions
            ],
            "raw": self.raw,
        }


@dataclass
class Paragraph:
    title: str
    sent_id: int
    text: str


@dataclass
class HopResult:
    hop_id: int
    sub_question: str
    retriever_trace: list[dict] = field(default_factory=list)   # [{round, tool, arguments, result}]
    retriever_raw: str = ""
    retrieved_paragraphs: list[Paragraph] = field(default_factory=list)
    retriever_rounds_used: int = 0
    extractor_raw: str = ""
    extractor_answer: str = ""
    extractor_quote: str = ""
    extractor_source: Optional[dict] = None        # {"title":..., "sent_id":...}
    quote_verified: Optional[bool] = None           # grounding_check.verify_quote (deterministic)
    llm_grounded: Optional[bool] = None             # Concluder's semantic judgment (drives retry)
    retry_count: int = 0                            # capped at config.MAX_HOP_RETRIES

    def to_dict(self) -> dict:
        return {
            "hop_id": self.hop_id,
            "sub_question": self.sub_question,
            "retriever_trace": self.retriever_trace,
            "retriever_raw": self.retriever_raw,
            "retrieved_paragraphs": [
                {"title": p.title, "sent_id": p.sent_id, "text": p.text} for p in self.retrieved_paragraphs
            ],
            "retriever_rounds_used": self.retriever_rounds_used,
            "extractor_raw": self.extractor_raw,
            "extractor_answer": self.extractor_answer,
            "extractor_quote": self.extractor_quote,
            "extractor_source": self.extractor_source,
            "quote_verified": self.quote_verified,
            "llm_grounded": self.llm_grounded,
            "retry_count": self.retry_count,
        }


@dataclass
class ConcluderCall:
    phase: int                          # 1 or 2
    raw: str
    tool_trace: list[dict] = field(default_factory=list)
    hop_grounding: list[dict] = field(default_factory=list)
    final_answer: str = ""
    reasoning: str = ""

    def to_dict(self) -> dict:
        return {
            "phase": self.phase,
            "raw": self.raw,
            "tool_trace": self.tool_trace,
            "hop_grounding": self.hop_grounding,
            "final_answer": self.final_answer,
            "reasoning": self.reasoning,
        }


@dataclass
class MASState:
    unique_id: str
    question: str
    gold_answer: str
    gold_type: str                      # from the dataset, eval-only — never fed to any agent
    context_paragraphs: list[Paragraph]
    gold_supporting_facts: list[tuple[str, int]]
    evidences: list[dict] = field(default_factory=list)   # analysis-only, never fed to agents

    hop_plan: Optional[HopPlan] = None
    hops: dict[int, HopResult] = field(default_factory=dict)
    concluder_calls: list[ConcluderCall] = field(default_factory=list)

    final_answer: str = ""
    final_answer_pre_retry: str = ""    # for the retry-delta ablation metric
    concluder_rounds: int = 0           # 1, or 2 if a hop retry fired
    elapsed_s: float = 0.0
    error: Optional[str] = None

    def predicted_supporting_facts(self) -> list[list]:
        seen: set[tuple[str, int]] = set()
        out: list[list] = []
        for hop in self.hops.values():
            src = hop.extractor_source
            if not src:
                continue
            key = (src.get("title"), src.get("sent_id"))
            if key not in seen and key[0] is not None:
                seen.add(key)
                out.append([src.get("title"), src.get("sent_id")])
        return out

    def to_record(self) -> dict:
        return {
            "unique_id": self.unique_id,
            "question": self.question,
            "gold_answer": self.gold_answer,
            "gold_type": self.gold_type,
            "predicted_type": self.hop_plan.predicted_type if self.hop_plan else "",
            "prediction": self.final_answer,
            "final_answer_pre_retry": self.final_answer_pre_retry,
            "gold_supporting_facts": [list(sf) for sf in self.gold_supporting_facts],
            "predicted_supporting_facts": self.predicted_supporting_facts(),
            "concluder_rounds": self.concluder_rounds,
            "trajectory": {
                "hop_plan": self.hop_plan.to_dict() if self.hop_plan else None,
                "hops": {str(k): v.to_dict() for k, v in self.hops.items()},
                "concluder_calls": [c.to_dict() for c in self.concluder_calls],
            },
            "elapsed_s": self.elapsed_s,
            "error": self.error,
        }
