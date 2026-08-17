"""Shared LLM-call loop and state dataclasses for all agents.

Every agent call is stateless: [assembled system prompt, user message]
-> one JSON object (with optional native tool rounds in between).
Task state is carried explicitly in `State` by the workflow.
"""

from dataclasses import dataclass, field

from llm_client import JSONParseError, LLMClient
from mas_prompt_cfg import build_system_prompt


def call_agent(llm: LLMClient, prompt_module, level: int, user_msg: str,
               tools=None, tool_handlers=None, trace=None, counter=None,
               task_override=None, schema_override=None, thinking=True) -> dict:
    """The single shared LLM-call / tool-dispatch loop: assemble system
    prompt, call the LLM (executing native tool calls via tool_handlers),
    parse JSON. Returns {"_error": ...} instead of raising so one bad agent
    turn degrades gracefully inside the pipeline."""
    system = build_system_prompt(prompt_module, level, task_override, schema_override)
    try:
        return llm.chat_json(system, user_msg, tools=tools,
                             tool_handlers=tool_handlers, trace=trace, counter=counter,
                             thinking=thinking)
    except JSONParseError as e:
        return {"_error": str(e)}


@dataclass
class LineItem:
    """One requested item, as parsed from the query. `constraints` is the
    dict of stated constraints the agents check the products against."""
    item_id: int
    constraints: dict
    quantity: int = 1

    @property
    def description(self) -> str:
        return str(self.constraints.get("description", ""))


@dataclass
class Candidate:
    """A verified product candidate for one line item."""
    product: dict
    verified: bool = True

    @property
    def product_id(self):
        return self.product.get("product_id")

    @property
    def price(self):
        return float(self.product.get("price", 0.0))


@dataclass
class State:
    """Per-case working state assembled by the workflow."""
    query: str = ""
    user_info: dict = field(default_factory=dict)
    line_items: list = field(default_factory=list)          # [LineItem]
    budget: dict | None = None                              # {"min":..,"max":..} | None
    candidates: dict = field(default_factory=dict)          # item_id -> [product dict]
    scout_notes: dict = field(default_factory=dict)         # item_id -> str
    plan: dict | None = None                                # optimizer's chosen cart
    issues: list = field(default_factory=list)              # accumulated problems

    @property
    def destination(self) -> str:
        return self.user_info.get("shipping_addresses", {}).get("province", "")

    @property
    def owned_coupons(self) -> dict:
        return self.user_info.get("coupons", {}) or {}

    @property
    def is_vip(self) -> bool:
        return bool(self.user_info.get("is_vip", False))
