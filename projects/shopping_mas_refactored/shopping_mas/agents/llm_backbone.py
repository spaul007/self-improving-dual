"""Per-agent backbone LLM config, loaded from ``mas_llm_backbone.yaml``
(sibling to ``workflow.py``, i.e. the task_agent root -- one directory up
from this file). Mirrors travel_mas_refactored's own
agents/llm_backbone.py: same env-var-override + process-lifetime-cache
pattern, fields adapted to this project's own LLMClient knobs (no
``reasoning_effort`` concept here -- that's an OpenAI-Responses-API-only
param this project's raw-Chat-Completions ``llm_client.py`` never had;
``max_tokens`` not ``max_output_tokens``, matching this project's own
naming).

Every field in the YAML may be omitted or ``null`` -- ``get_backbone_config``
then returns ``None`` for that field, which ``call_agent``/``LLMClient``
already treat as "no override, use this run's single global
MASConfig.server/temperature/max_tokens/force_enable_thinking", so a
config with all-null fields (this project's own shipped default) is
byte-identical to today's single-global-backbone behavior. Real values
only start taking effect once something (a human, or the HGM's
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

_FIELDS = ("model", "base_url", "temperature", "max_tokens", "enable_thinking")


@functools.lru_cache(maxsize=1)
def _load() -> dict[str, Any]:
    path = Path(os.environ.get(_CONFIG_ENV_VAR, str(_DEFAULT_PATH)))
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_backbone_config(agent_name: str) -> dict[str, Optional[Any]]:
    """Merge the YAML's ``default`` section with ``agents.<agent_name>``
    (per-agent keys win). Always returns exactly the keys in ``_FIELDS``;
    any key absent from both sections comes back as ``None``.

    Note on credentials: every call still authenticates via this
    project's single process-wide ``MAS_API_KEY`` (see
    ``LLMClient._client_for``, which only varies ``base_url``). A run
    whose pipeline default is one vLLM server but whose
    ``mas_llm_backbone.yaml`` overrides one role to a server needing a
    different key would need that handled separately -- not needed for
    this knob's actual use cases (same-cluster vLLM endpoints, which
    ignore auth entirely and accept any non-empty string)."""
    data = _load()
    default = data.get("default") or {}
    per_agent = (data.get("agents") or {}).get(agent_name) or {}
    merged = {**default, **per_agent}
    return {field: merged.get(field) for field in _FIELDS}
