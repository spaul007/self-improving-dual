"""Insert investigator prompt accessors.

The text lives in mas_prompt_cfg.yaml under `agents.insert_investigator`:
  - role : FROZEN   -- the agent's identity
  - task : EDITABLE -- fair game for prompt optimization
"""

import config

NAME = "insert_investigator"
# The candidate root cause this investigator is assigned to examine.
CANDIDATE = "INSERT_LARGE_DATA"


def role() -> str:
    """Frozen role instruction."""
    return config.agent_prompt(NAME)[0]


def task() -> str:
    """Editable task instruction (placeholders: {question})."""
    return config.agent_prompt(NAME)[1]
