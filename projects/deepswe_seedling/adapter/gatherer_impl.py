"""Project gatherer ``deepswe_seedling``: the default gatherer + a ranked per-node dossier index.

After every feedback compile (each EVALUATE batch and finalize top-up) it (re)writes
``<round>/logs/DOSSIERS.md`` from the per-run ``dossier.json`` files pier_case wrote, so the
block suggester and the editor (``agentic_log_access``) have one entry point into the evidence.
No ``__init__`` override: the parent's signature (and hence ``scorer`` injection) is inherited
unchanged -- see projects/db_mas/adapter/gatherer_impl.py for the **kwargs trap.
"""
from __future__ import annotations

from pathlib import Path

from meta_agent.feedback_gatherer import DefaultFeedbackGatherer
from meta_agent.registry import register

from .dossier import write_index


@register("gatherer", "deepswe_seedling")
class DeepSWESeedlingGatherer(DefaultFeedbackGatherer):
    def compile(self, round_number, base_round, strategy, eval_result, round_dir):
        fb = super().compile(round_number, base_round, strategy, eval_result, round_dir)
        try:
            write_index(Path(round_dir))
        except Exception as exc:  # noqa: BLE001 -- the index is a convenience
            print(f"[deepswe_seedling gatherer] DOSSIERS.md not written: {exc!r}", flush=True)
        return fb
