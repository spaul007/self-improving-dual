"""Shared "reset a round's task_agent workspace from its base" helper.

Extracted from ``AgentEditor._copy_workspace`` so any component that needs
to start a fresh round's ``out_dir/task_agent`` as an exact copy of
``base_dir/task_agent`` -- today: ``AgentEditor`` itself, and
``LLMBackboneSelector`` (see ``meta_agent/backbone_selector.py``) -- shares
one implementation instead of duplicating the rmtree+copytree.
"""
from __future__ import annotations

import shutil
from pathlib import Path


def reset_workspace(base_dir: Path, out_dir: Path) -> None:
    src = base_dir / "task_agent"
    dst = out_dir / "task_agent"
    if dst.exists():
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst)
