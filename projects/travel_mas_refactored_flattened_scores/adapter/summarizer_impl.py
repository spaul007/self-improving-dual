"""travel_mas_refactored_flattened_scores's named behavior-summarizer
extension point -- continuous-scoring sibling of
``projects/travel_mas_refactored/adapter/summarizer_impl.py`` (never
modified; see this project's own scorer_impl.py for why this clone
exists).

Pure passthrough -- ``BehaviorSummarizer``'s base
``_extract_failure_hint`` already recognizes the generic scorer
conventions ``TravelFlattenedScoresScorer.score()`` actually uses (a flat
``failed_checks`` list, a nested ``{name: {"passed": bool}}``-shaped map
via ``hard_constraints``, a bare ``error`` string), so no override is
needed to get a useful hint out of the box.

Registered under its own name anyway, matching db_mas/math_mas's
convention, so travel_mas_refactored_flattened_scores has a stable
registry slot to add real overrides to later without a rename or a YAML
change. Not currently referenced by any
travel_mas_refactored_flattened_scores config (none set a ``summarizer:``
block) -- available for future opt-in.

Registered as ``"travel_mas_refactored_flattened_scores_default"`` (not
``"travel_mas_refactored_default"``, the project this was cloned from) so
both projects can be imported in the same process without a registry
collision -- the same convention travel_mas_refactored itself already
uses against its own sibling travel_mas.
"""
from __future__ import annotations

from meta_agent.behavior_summarizer import BehaviorSummarizer
from meta_agent.registry import register


@register("summarizer", "travel_mas_refactored_flattened_scores_default")
class TravelFlattenedScoresBehaviorSummarizer(BehaviorSummarizer):
    pass
