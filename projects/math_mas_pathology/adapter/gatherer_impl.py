"""math_mas_pathology's named feedback-gatherer extension point.

Pure passthrough -- like math_mas, this project's `score` is strictly binary
(0.0/1.0, exact-match after normalization), so `DefaultFeedbackGatherer`'s
built-in default (`pass_threshold=1.0`) is already correct out of the box.

Registered under its own name anyway, matching math_mas's convention, so
this project has a stable registry slot to add real overrides to later
without a rename or a YAML change.
"""
from __future__ import annotations

from meta_agent.feedback_gatherer import DefaultFeedbackGatherer
from meta_agent.registry import register


@register("gatherer", "math_mas_pathology_default")
class MathMASPathologyFeedbackGatherer(DefaultFeedbackGatherer):
    pass
