"""Single source of truth for locating the vendored wikihop_mas/ core.

`scorer_impl.py` calls `ensure_on_path()` before importing wikihop_mas's own
`eval.metrics` module.

Resolved relative to *this file* -- `adapter/` lives at the project level (a
sibling of `benchmark/`, exactly like `wikihop_mas/` itself) and is never
copied per-round by the editor (only the seed dir's contents are, into
`round_NNN/task_agent/`), so this stays correct even once HGM starts
creating those copies. `wikihop_mas/workflow.py`'s own path resolution
doesn't need this same treatment -- it's fully self-contained and imports
only `mas_workflow`/`run_inference`, which sit right next to it in every
copy.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_DEFAULT_WIKIHOPMAS_ROOT = str(Path(__file__).resolve().parent.parent / "wikihop_mas")
WIKIHOPMAS_ROOT = os.environ.get("WIKIHOPMAS_ROOT", _DEFAULT_WIKIHOPMAS_ROOT)


def ensure_on_path() -> None:
    if WIKIHOPMAS_ROOT not in sys.path:
        sys.path.insert(0, WIKIHOPMAS_ROOT)
