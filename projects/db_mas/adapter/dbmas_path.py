"""Single source of truth for locating the vendored db-mas/ core.

Every adapter module that needs to import db-mas's own code
(`task_translation.py`, `scorer_impl.py`) calls `ensure_on_path()` first.

Resolved relative to *this file* -- `adapter/` lives at the project level
(a sibling of `seed/` and `benchmark/`, exactly like `db-mas/` itself) and is
never copied per-round by the editor (only `seed/`'s contents are, into
`round_NNN/task_agent/`), so this stays correct even once HGM starts
creating those copies. `seed/workflow.py`'s old self-relative resolution
did NOT have this property (found while doing this refactor -- see
`seed/workflow.py`'s docstring for the fix on that side).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_DEFAULT_DB_MAS_ROOT = str(Path(__file__).resolve().parent.parent / "db-mas")
DB_MAS_ROOT = os.environ.get("DBMAS_ROOT", _DEFAULT_DB_MAS_ROOT)


def ensure_on_path() -> None:
    if DB_MAS_ROOT not in sys.path:
        sys.path.insert(0, DB_MAS_ROOT)
