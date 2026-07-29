"""Framework-mandated entry point: `meta_agent.evaluator._load_scorer` /
`meta_agent.config._load_project_components` require this exact module name
and location (the only per-project file the framework auto-imports for
`@register` side effects, and where `SubprocessEvaluator` looks up the
module-level `score()` fallback). All real logic lives in `adapter/` -- see
`adapter/scorer_impl.py` (+ `adapter/gatherer_impl.py` /
`adapter/summarizer_impl.py`, imported here as a side effect so their
`MathMASFeedbackGatherer`/`MathMASBehaviorSummarizer` registrations run too).

This file is never copied per-round (only the seed dir's contents are), so
its `sys.path` bootstrap can safely be resolved relative to its own
`__file__`.
"""
from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from adapter.scorer_impl import MathMASScorer, score  # noqa: E402,F401
from adapter import gatherer_impl, summarizer_impl  # noqa: E402,F401
