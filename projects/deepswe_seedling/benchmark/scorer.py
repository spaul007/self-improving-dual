"""Framework-mandated scorer module (auto-imported for @register side effects).
Real logic lives in ``adapter/`` (same convention as travel_mas_refactored/db_mas).
Importing ``adapter.validators`` registers the project's validators."""
from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from adapter.scorer_impl import DeepSWESeedlingScorer, score  # noqa: E402,F401
from adapter import validators  # noqa: E402,F401
