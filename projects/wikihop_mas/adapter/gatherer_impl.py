"""wikihop_mas's named feedback-gatherer extension point.

Pure passthrough. wikihop_mas's `score` is answer_f1 (continuous, see
adapter/scorer_impl.py's module docstring for why F1 over EM) -- but for
well-formed short entity/date answers, an exact-match prediction always
normalizes to F1=1.0 (identical token multisets), so
DefaultFeedbackGatherer's built-in `pass_threshold=1.0` default still
correctly buckets "genuinely imperfect" cases as failing in the failure
report, same as math_mas's strictly-binary case. Unlike db_mas's F1 (which
structurally can never reach 1.0 for its benchmark), that concern doesn't
apply here, so no `pass_threshold` override is needed.

Registered under its own name anyway, matching math_mas/db_mas's
convention, so wikihop_mas has a stable registry slot to add real
overrides to later without a rename or a YAML change.
"""
from __future__ import annotations

from meta_agent.feedback_gatherer import DefaultFeedbackGatherer
from meta_agent.registry import register


@register("gatherer", "wikihop_mas_default")
class WikihopMASFeedbackGatherer(DefaultFeedbackGatherer):
    pass
