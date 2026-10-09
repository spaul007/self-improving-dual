"""Framework-mandated scorer module (auto-imported for @register side effects).
Real logic lives in ``adapter/`` (same convention as travel_mas_refactored/db_mas).
Importing ``adapter.validators`` registers the project's validators."""
from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Imported by FULL package path: several projects have a top-level package named
# `adapter`, and in one process whichever is imported first owns that name -- the
# travel project's `adapter` silently shadowed this one (config._load_project_components
# swallows import errors), leaving `deepswe_seedling_default` unregistered.
from projects.deepswe_seedling.adapter.scorer_impl import DeepSWESeedlingScorer, score  # noqa: E402,F401
from projects.deepswe_seedling.adapter import validators  # noqa: E402,F401
from projects.deepswe_seedling.adapter import gatherer_impl  # noqa: E402,F401
