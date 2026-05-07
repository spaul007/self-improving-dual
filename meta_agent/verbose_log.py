"""Helpers for the verbose-logging mode.

When ``META_AGENT_VERBOSE`` is set in the environment (gated by the YAML
``verbose: true`` flag at startup; see ``runtime_env.apply_verbose``),
each major component writes its full prompts and raw LLM responses into
``round_NNN/verbose/`` so a human can inspect them after the fact.

When the env var is unset, every writer in this module is a cheap
no-op — components don't need to thread a flag through their call
graphs.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


_ENV_VAR = "META_AGENT_VERBOSE"


def is_enabled() -> bool:
    return os.environ.get(_ENV_VAR) == "1"


def verbose_dir(round_dir: Path) -> Path:
    """Return ``<round_dir>/verbose`` and create it if missing. Caller is
    responsible for guarding with :func:`is_enabled` first if they want
    to avoid the mkdir on disabled runs."""
    path = Path(round_dir) / "verbose"
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_text(round_dir: Path, name: str, body: str) -> None:
    if not is_enabled():
        return
    target = verbose_dir(round_dir) / name
    target.write_text(body, encoding="utf-8")


def write_json(round_dir: Path, name: str, obj: Any) -> None:
    if not is_enabled():
        return
    target = verbose_dir(round_dir) / name
    target.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
