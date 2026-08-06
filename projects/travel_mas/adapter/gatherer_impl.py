"""travel_mas's named feedback-gatherer extension point.

Pure passthrough -- ``TravelCompositeScorer``'s composite score can reach
1.0 (it's a real weighted average, not a structurally-capped metric like
db_mas's F1), so ``DefaultFeedbackGatherer``'s built-in default
(``pass_threshold=1.0``) is already correct out of the box.

Registered under its own name anyway, matching db_mas/math_mas's
convention (a named class instead of bare ``type: "default"``) so
travel_mas has a stable registry slot to add real overrides to later
without a rename or a YAML change. Not currently referenced by any
travel_mas config (they use ``gatherer: {type: "default"}`` directly) --
available for future opt-in.
"""
from __future__ import annotations

from meta_agent.feedback_gatherer import DefaultFeedbackGatherer
from meta_agent.registry import register


@register("gatherer", "travel_mas_default")
class TravelMASFeedbackGatherer(DefaultFeedbackGatherer):
    pass
