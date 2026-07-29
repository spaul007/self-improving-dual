"""Decomposer prompt accessors.

The text lives in mas_prompt_cfg.yaml under `agents.decomposer`:
  - role : FROZEN   -- the agent's identity
  - task : EDITABLE -- fair game for prompt optimization
"""

import config

NAME = "decomposer"


def role() -> str:
    """Frozen role instruction."""
    return config.agent_prompt(NAME)[0]


def task() -> str:
    """Editable task instruction (placeholders: {question})."""
    return config.agent_prompt(NAME)[1]
