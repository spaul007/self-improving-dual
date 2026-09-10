"""Per-agent backbone LLM config, loaded from ``mas_llm_backbone.yaml``
(sibling to ``workflow.py``, i.e. the task_agent root -- one directory up
from this file). Mirrors the ``mas_prompt_cfg.yaml`` pattern already used
by ``math_mas``/``db_mas``/``wikihop_mas`` (``Path(__file__).resolve()``
-relative default + env-var override + a process-lifetime cache).

Every field in the YAML may be omitted or ``null`` -- ``get_backbone_config``
then returns ``None`` for that field, which ``call_llm``
(``platform_core/llm_wrapper.py``) already treats as "no override, use the
LLM_* env-var default", so a config with all-null fields (the seed's own
shipped default) is byte-identical to today's env-var-only behavior. Real
values only start taking effect once something (a human, or the HGM's
``llm_backbone_selection`` block hook) writes them in.
"""
from __future__ import annotations

import functools
import os
from pathlib import Path
from typing import Any, Optional

import yaml

_CONFIG_ENV_VAR = "MAS_LLM_BACKBONE_CFG"
_DEFAULT_PATH = Path(__file__).resolve().parent.parent / "mas_llm_backbone.yaml"

_FIELDS = ("model", "base_url", "temperature", "max_output_tokens", "reasoning_effort")


@functools.lru_cache(maxsize=1)
def _load() -> dict[str, Any]:
    path = Path(os.environ.get(_CONFIG_ENV_VAR, str(_DEFAULT_PATH)))
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_backbone_config(agent_name: str) -> dict[str, Optional[Any]]:
    """Merge the YAML's ``default`` section with ``agents.<agent_name>``
    (per-agent keys win). Always returns exactly the 5 keys in ``_FIELDS``;
    any key absent from both sections comes back as ``None``.

    Note on credentials: every call still authenticates via the single
    process-wide ``OPENAI_API_KEY`` (see ``platform_core/llm_wrapper.py``).
    A run whose pipeline default is local vLLM but whose
    ``mas_llm_backbone.yaml`` overrides one role to OpenRouter needs
    ``OPENAI_API_KEY`` set to a real OpenRouter key BEFORE launch -- this is
    safe even for the local-default calls, since local OpenAI-compatible
    servers ignore auth entirely and accept any non-empty string."""
    data = _load()
    default = data.get("default") or {}
    per_agent = (data.get("agents") or {}).get(agent_name) or {}
    merged = {**default, **per_agent}
    return {field: merged.get(field) for field in _FIELDS}
