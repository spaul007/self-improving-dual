"""travel_mas's named behavior-summarizer extension point.

Pure passthrough -- ``BehaviorSummarizer``'s base
``_extract_failure_hint`` already recognizes the generic scorer
conventions ``TravelCompositeScorer.score()`` actually uses (a flat
``failed_checks`` list, a nested ``{name: {"passed": bool}}``-shaped map
via ``hard_constraints``, a bare ``error`` string), so no override is
needed to get a useful hint out of the box.

Registered under its own name anyway, matching db_mas/math_mas's
convention, so travel_mas has a stable registry slot to add real
overrides to later without a rename or a YAML change. Not currently
referenced by any travel_mas config (none set a ``summarizer:`` block) --
available for future opt-in.
"""
from __future__ import annotations

from meta_agent.behavior_summarizer import BehaviorSummarizer
from meta_agent.registry import register


@register("summarizer", "travel_mas_default")
class TravelMASBehaviorSummarizer(BehaviorSummarizer):
    pass
