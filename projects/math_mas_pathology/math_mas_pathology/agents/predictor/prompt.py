"""Predictor prompt accessors.

The text lives in mas_prompt_cfg.yaml under `agents.predictor`:
  - role : FROZEN   -- the agent's identity
  - task : EDITABLE -- fair game for prompt optimization
"""

import config

NAME = "predictor"


def role() -> str:
    """Frozen role instruction."""
    return config.agent_prompt(NAME)[0]


def task() -> str:
    """Editable task instruction (placeholders: {question}, {context})."""
    return config.agent_prompt(NAME)[1]
