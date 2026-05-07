"""Shared CSV loader for travel tools.

The reference repo uses pandas. We use the stdlib ``csv`` module to keep the
platform free of heavy dependencies — pandas can still be added back if a tool
needs it. Each row is returned as a ``dict[str, str]`` (everything as strings,
matching the reference's ``dtype=str`` behaviour to preserve precision on
lat/lon and price fields).
"""
from __future__ import annotations

import csv
import os
import threading
from pathlib import Path
from typing import Optional

_CACHE: dict[str, list[dict[str, str]]] = {}
_LOCK = threading.Lock()


DB_ROOT_ENV = "TRAVEL_DATABASE_ROOT"
SAMPLE_ID_ENV = "TRAVEL_SAMPLE_ID"


def database_root() -> Optional[Path]:
    """Return the per-sample CSV root.

    Order of precedence:
      1. ``$TRAVEL_DATABASE_ROOT`` if set (user override via YAML
         ``env:`` block, or pre-set in the shell).
      2. The project-relative default ``projects/travel/data/database_en``
         when that directory exists. This means a typical run needs no
         explicit env wiring — drop the data under
         ``projects/travel/data/database_en/`` and the tools find it.
      3. ``None`` if neither is available; tools surface a "database
         not loaded" sentinel string.
    """
    val = os.environ.get(DB_ROOT_ENV)
    if val:
        return Path(val)
    # Project-relative default. ``_csv.py`` lives at
    # projects/travel/tools/_csv.py; repo root is four parents up.
    candidate = Path(__file__).resolve().parents[3] / "projects" / "travel" / "data" / "database_en"
    return candidate if candidate.exists() else None


def sample_id() -> Optional[str]:
    return os.environ.get(SAMPLE_ID_ENV)


def csv_path(relative_filename: str) -> Optional[Path]:
    """Return ``<root>/id_<sample>/<relative_filename>`` or None if either env
    var is missing."""
    root = database_root()
    sid = sample_id()
    if root is None or sid is None:
        return None
    return root / f"id_{sid}" / relative_filename


def load_csv(path: Path) -> list[dict[str, str]]:
    key = str(path)
    with _LOCK:
        if key in _CACHE:
            return _CACHE[key]
    if not path.exists():
        return []
    rows: list[dict[str, str]] = []
    with open(path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append({k: ("" if v is None else str(v)) for k, v in row.items()})
    with _LOCK:
        _CACHE[key] = rows
    return rows


def load_for_tool(relative_filename: str) -> tuple[list[dict[str, str]], Optional[Path]]:
    """Load the per-sample CSV for a tool. Returns (rows, path). If the
    database is unconfigured or the file is missing, returns ([], path-or-None)."""
    path = csv_path(relative_filename)
    if path is None or not path.exists():
        return [], path
    return load_csv(path), path


def db_not_loaded_message(thing: str) -> str:
    """Uniform sentinel string when database access is unavailable."""
    root = database_root()
    sid = sample_id()
    if root is None:
        return f"{thing} database not loaded: TRAVEL_DATABASE_ROOT env var is not set"
    if sid is None:
        return f"{thing} database not loaded: TRAVEL_SAMPLE_ID env var is not set"
    return f"{thing} database not loaded for sample {sid} under {root}"
