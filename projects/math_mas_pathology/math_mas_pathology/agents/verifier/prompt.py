"""Verifier prompt accessors.

The text lives in mas_prompt_cfg.yaml under `agents.verifier`:
  - role : FROZEN   -- the agent's identity
  - task : EDITABLE -- fair game for prompt optimization

The task template is identical on every call the verifier receives (see
`workflow.py::VerifierAgent.arun_repeated`) -- no turn index or prior-answer
placeholder exists here on purpose.
"""

import config

NAME = "verifier"


def role() -> str:
    """Frozen role instruction."""
    return config.agent_prompt(NAME)[0]


def task() -> str:
    """Editable task instruction (placeholders: {question}, {context})."""
    return config.agent_prompt(NAME)[1]
