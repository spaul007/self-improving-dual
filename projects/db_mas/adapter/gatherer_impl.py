"""db_mas's named feedback-gatherer extension point.

The framework's own history (see meta_agent/feedback_gatherer.py's module
docstring) shows per-project gatherer subclasses used to exist and were
deliberately merged into two hooks on the *scorer* instead: `aggregate()`
for round-level project_metrics, and an optional `error_categorizer` for
failure-report grouping (skipped for db_mas for now). Every other project
in this repo (travel/shopping/math) just uses `type: "default"` in YAML.

db_mas registers its own named class anyway, per explicit request, so it has
a stable registry slot to add real overrides to later without a rename or a
YAML change. Behavior is DefaultFeedbackGatherer's throughout (tool_usage,
tool_error_rate, log_excerpt, failure_report, and dispatch to
DBMASScorer.aggregate() for project_metrics); the one override is
`pass_threshold`, read from `gatherer_config.json` in this same directory
(see `_load_pass_threshold`) rather than only from a YAML `gatherer.config`
block -- this benchmark's `score` is F1, which structurally can never reach
DefaultFeedbackGatherer's built-in default of 1.0 (see db-mas's
design_doc.md on why exact_match/precision are capped below 1.0), so the
unmodified default would mark every case as "failing" in the failure report
regardless of whether it actually passed.
"""
from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any

from meta_agent.feedback_gatherer import DefaultFeedbackGatherer
from meta_agent.registry import register

_CONFIG_PATH = Path(__file__).resolve().parent / "gatherer_config.json"


def _load_pass_threshold(default: float = 1.0) -> float:
    """Read `pass_threshold` from gatherer_config.json in this directory.
    Missing file, unreadable JSON, or a missing key all fall back to
    `default` silently -- this is a soft override, not a required file."""
    if not _CONFIG_PATH.exists():
        return default
    try:
        data = json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default
    return float(data.get("pass_threshold", default))


@register("gatherer", "db_mas_default")
class DBMASFeedbackGatherer(DefaultFeedbackGatherer):
    def __init__(self, **kwargs: Any) -> None:
        # An explicit `pass_threshold` in YAML's gatherer.config still wins
        # (setdefault only fills it in when absent) -- this file is a
        # project-level default, not a hard override.
        kwargs.setdefault("pass_threshold", _load_pass_threshold())
        super().__init__(**kwargs)


# meta_agent.config._build_with_injection decides which kwargs (notably
# `scorer`) to inject into a gatherer by checking
# `inspect.signature(cls).parameters` -- a bare `__init__(self, **kwargs)`
# hides DefaultFeedbackGatherer's real parameter names from that check, so
# `scorer` was silently never being injected here: self.scorer stayed None,
# and `_project_metrics` (feedback_gatherer.py) returns `{}` whenever
# `self.scorer` is falsy, with no warning. Every db_mas round's
# project_metrics has been empty because of this. Restoring the parent's
# real signature (bypassing the **kwargs-only view) fixes the introspection
# without changing __init__'s actual behavior.
DBMASFeedbackGatherer.__init__.__signature__ = inspect.signature(DefaultFeedbackGatherer.__init__)
