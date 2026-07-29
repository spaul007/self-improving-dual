"""Single source of truth for locating the vendored db_mas/ core.

`scorer_impl.py` calls `ensure_on_path()` before importing this project's own
`eval.metrics` module.

Resolved relative to *this file* -- `adapter/` lives at the project level (a
sibling of `benchmark/`, exactly like `db_mas/` itself) and is never copied
per-round by the editor (only the seed dir's contents are, into
`round_NNN/task_agent/`), so this stays correct even once HGM starts creating
those copies. `db_mas/workflow.py`'s own path resolution doesn't need this
same treatment -- it's fully self-contained and imports only `mas_workflow`/
`config`, which sit right next to it in every copy.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_DEFAULT_DB_MAS_SNAPSHOT_ROOT = str(Path(__file__).resolve().parent.parent / "db_mas")
DB_MAS_SNAPSHOT_ROOT = os.environ.get("DBMASSNAPSHOT_ROOT", _DEFAULT_DB_MAS_SNAPSHOT_ROOT)


def ensure_on_path() -> None:
    if DB_MAS_SNAPSHOT_ROOT not in sys.path:
        sys.path.insert(0, DB_MAS_SNAPSHOT_ROOT)
