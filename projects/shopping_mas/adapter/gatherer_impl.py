"""shopping_mas's named feedback-gatherer extension point.

Pure passthrough -- like the sibling projects/shopping project, shopping_mas's
`score` is strictly binary (case_score 0.0/1.0), so `DefaultFeedbackGatherer`'s
built-in default (`pass_threshold=1.0`) is already correct out of the box.

Registered under its own name anyway, matching every other project's
convention, so shopping_mas has a stable registry slot to add real overrides
to later without a rename or a YAML change.
"""
from __future__ import annotations

from meta_agent.feedback_gatherer import DefaultFeedbackGatherer
from meta_agent.registry import register


@register("gatherer", "shopping_mas_default")
class ShoppingMasFeedbackGatherer(DefaultFeedbackGatherer):
    pass
