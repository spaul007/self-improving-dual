"""db_mas_snapshot's named feedback-gatherer extension point.

Pure passthrough -- no `__init__` override, no `pass_threshold` override.
Unlike the other, already-integrated `db_mas` project (whose F1 score
structurally can never reach 1.0, requiring a `pass_threshold` override),
this project's headline score is `recall`, which genuinely CAN reach 1.0
(confirmed: `number_of_labels_pred == len(root_causes)` exactly for every
task), so `DefaultFeedbackGatherer`'s built-in `pass_threshold=1.0` default
is already correct out of the box. This also means the signature-
introspection pitfall documented in the other db_mas project's
`gatherer_impl.py` (a custom `__init__(self, **kwargs)` hiding the parent's
real parameter names from `meta_agent.config._build_with_injection`'s
`inspect.signature(cls)`-based scorer injection) never arises here, since no
`__init__` is defined at all. If a `pass_threshold` override is ever added
later, see that other project's fix (restoring `__init__.__signature__`
explicitly) before adding a bare `__init__(self, **kwargs)`.

Registered under its own name anyway, matching every other project in this
family (db_mas/math_mas/wikihop_mas all use a named class), so this project
has a stable registry slot to add real overrides to later without a rename
or a YAML change.
"""
from __future__ import annotations

from meta_agent.feedback_gatherer import DefaultFeedbackGatherer
from meta_agent.registry import register


@register("gatherer", "db_mas_snapshot_default")
class DBMasSnapshotFeedbackGatherer(DefaultFeedbackGatherer):
    pass
