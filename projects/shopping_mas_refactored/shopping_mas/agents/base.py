"""Shared LLM-call loop and state dataclasses for all agents.

Every agent call is stateless: [assembled system prompt, user message]
-> one JSON object (with optional native tool rounds in between).
Task state is carried explicitly between stages via the frozen
AgentMessage/inbox contract in agents/immutable/message.py (not a
shared mutable object) -- see that module's docstring for the full
per-sender content shape. `destination`/`owned_coupons`/`is_vip` below
operate directly on a `user_info: dict` (the one piece of per-case input
every stage either receives directly or can derive from its own inbox),
replacing the three same-named properties a prior revision kept on a
shared `State` dataclass.
"""

from dataclasses import dataclass

from agents.llm_backbone import get_backbone_config
from llm_client import JSONParseError, LLMClient
from mas_prompt_cfg import build_system_prompt


def call_agent(llm: LLMClient, prompt_module, level: int, user_msg: str, *,
               agent_name: str, tools=None, tool_handlers=None, trace=None,
               counter=None, task_override=None, schema_override=None,
               thinking=True) -> dict:
    """The single shared LLM-call / tool-dispatch loop: assemble system
    prompt, call the LLM (executing native tool calls via tool_handlers),
    parse JSON. Returns {"_error": ...} instead of raising so one bad agent
    turn degrades gracefully inside the pipeline.

    `agent_name` selects this stage's backbone LLM settings from
    mas_llm_backbone.yaml (see agents/llm_backbone.py) -- any field left
    null there falls back to this call's own `thinking` argument / the
    process-wide MASConfig.server/temperature/max_tokens. Keyword-only
    (and required) so a missed call site fails immediately rather than
    silently always resolving the "default" backbone section."""
    backbone = get_backbone_config(agent_name)
    effective_thinking = thinking if backbone["enable_thinking"] is None else backbone["enable_thinking"]
    system = build_system_prompt(prompt_module, level, task_override, schema_override)
    try:
        return llm.chat_json(system, user_msg, tools=tools,
                             tool_handlers=tool_handlers, trace=trace, counter=counter,
                             thinking=effective_thinking,
                             model=backbone["model"], base_url=backbone["base_url"],
                             temperature=backbone["temperature"],
                             max_tokens=backbone["max_tokens"])
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


def destination(user_info: dict) -> str:
    return user_info.get("shipping_addresses", {}).get("province", "")


def owned_coupons(user_info: dict) -> dict:
    return user_info.get("coupons", {}) or {}


def is_vip(user_info: dict) -> bool:
    return bool(user_info.get("is_vip", False))
