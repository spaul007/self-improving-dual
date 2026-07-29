"""Reflector prompt accessors.

The text lives in mas_prompt_cfg.yaml under `agents.reflector`:
  - role : FROZEN   -- the agent's identity
  - task : EDITABLE -- fair game for prompt optimization
"""

import config

NAME = "reflector"


def role() -> str:
    """Frozen role instruction."""
    return config.agent_prompt(NAME)[0]


def task() -> str:
    """Editable task instruction (placeholders: {question}, {context})."""
    return config.agent_prompt(NAME)[1]
