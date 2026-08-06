"""Single source of truth for locating the vendored math_mas_pathology/ core.

`scorer_impl.py` calls `ensure_on_path()` before importing this project's own
`eval.metrics` module.

Resolved relative to *this file* -- `adapter/` lives at the project level (a
sibling of `benchmark/`, exactly like `math_mas_pathology/` itself) and is
never copied per-round by the editor (only the seed dir's contents are, into
`round_NNN/task_agent/`), so this stays correct even once HGM starts creating
those copies. `math_mas_pathology/workflow.py`'s own path resolution doesn't
need this same treatment -- it's fully self-contained and imports only
`mas_workflow`, which sits right next to it in every copy.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_DEFAULT_ROOT = str(Path(__file__).resolve().parent.parent / "math_mas_pathology")
MATH_MAS_PATHOLOGY_ROOT = os.environ.get("MATHMASPATHOLOGY_ROOT", _DEFAULT_ROOT)


def ensure_on_path() -> None:
    if MATH_MAS_PATHOLOGY_ROOT not in sys.path:
        sys.path.insert(0, MATH_MAS_PATHOLOGY_ROOT)
