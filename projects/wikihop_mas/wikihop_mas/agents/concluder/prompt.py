"""Concluder prompt accessors.

The text lives in mas_prompt_cfg.yaml under `agents.concluder`:
  - role : FROZEN   -- the agent's identity
  - task : EDITABLE -- fair game for prompt optimization
"""

import config

NAME = "concluder"


def role() -> str:
    """Frozen role instruction."""
    return config.agent_prompt(NAME)[0]


def task() -> str:
    """Editable task instruction (placeholders: {question}, {hops_summary})."""
    return config.agent_prompt(NAME)[1]
