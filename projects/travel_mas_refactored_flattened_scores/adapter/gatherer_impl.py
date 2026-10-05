"""travel_mas_refactored_flattened_scores's named feedback-gatherer
extension point -- continuous-scoring sibling of
``projects/travel_mas_refactored/adapter/gatherer_impl.py`` (never
modified; see this project's own scorer_impl.py for why this clone
exists).

Pure passthrough -- ``TravelFlattenedScoresScorer``'s composite score can
reach 1.0 (it's a real weighted average, not a structurally-capped metric
like db_mas's F1), so ``DefaultFeedbackGatherer``'s built-in default
(``pass_threshold=1.0``) is already correct out of the box.

Registered under its own name anyway, matching db_mas/math_mas's
convention (a named class instead of bare ``type: "default"``) so
travel_mas_refactored_flattened_scores has a stable registry slot to add
real overrides to later without a rename or a YAML change. Not currently
referenced by any travel_mas_refactored_flattened_scores config (they use
``gatherer: {type: "default"}`` directly) -- available for future opt-in.

Registered as ``"travel_mas_refactored_flattened_scores_default"`` (not
``"travel_mas_refactored_default"``, the project this was cloned from) so
both projects can be imported in the same process without a registry
collision -- the same convention travel_mas_refactored itself already
uses against its own sibling travel_mas.
"""
from __future__ import annotations

from meta_agent.feedback_gatherer import DefaultFeedbackGatherer
from meta_agent.registry import register


@register("gatherer", "travel_mas_refactored_flattened_scores_default")
class TravelFlattenedScoresFeedbackGatherer(DefaultFeedbackGatherer):
    pass
