"""math_mas's named feedback-gatherer extension point.

Pure passthrough -- unlike db_mas's `DBMASFeedbackGatherer`, no
`pass_threshold` override is needed here: math_mas's `score` is strictly
binary (0.0/1.0, exact-match after normalization), so
`DefaultFeedbackGatherer`'s built-in default (`pass_threshold=1.0`) is
already correct out of the box. (db_mas needed an override specifically
because its F1 score can structurally never reach 1.0.)

Registered under its own name anyway, matching every other project
(travel/shopping/math use `type: "default"`; db_mas and this project use a
named class instead) so math_mas has a stable registry slot to add real
overrides to later without a rename or a YAML change.
"""
from __future__ import annotations

from meta_agent.feedback_gatherer import DefaultFeedbackGatherer
from meta_agent.registry import register


@register("gatherer", "math_mas_default")
class MathMASFeedbackGatherer(DefaultFeedbackGatherer):
    pass
