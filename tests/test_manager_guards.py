"""Managers that don't support the agentic editor's assignment, the edit
memory or paired evaluation say so clearly, and hill climbing accepts the
optional components main_loop may pass.

    PYTHONPATH=. python3 -m unittest tests.test_manager_guards
"""
from __future__ import annotations

import unittest
from pathlib import Path

_EVOLVE = dict(evaluator=None, gatherer=None, seed_dir=Path("."), benchmark_dir=Path("."),
               experiment_dir=Path("."), max_rounds=10, score_target=None)
_CATEGORIZER = "projects.travel.travel_error_categorizer:categorize_errors"


class _AssignmentEditor:
    steering = "assignment"


class DualGuardTests(unittest.TestCase):
    def test_expand_eval_size_rejected(self) -> None:
        from meta_agent.managers.hgm_dual import HGMDualManager

        with self.assertRaises(ValueError):
            HGMDualManager(error_categorizer=_CATEGORIZER, expand_eval_size=4)

    def test_edit_memory_rejected(self) -> None:
        from meta_agent.managers.hgm_dual import HGMDualManager

        with self.assertRaisesRegex(ValueError, "edit_memory"):
            HGMDualManager(error_categorizer=_CATEGORIZER).evolve(
                editor=object(), edit_memory=object(), **_EVOLVE)

    def test_assignment_editor_rejected(self) -> None:
        from meta_agent.managers.hgm_dual import HGMDualManager

        with self.assertRaisesRegex(ValueError, "assignment"):
            HGMDualManager(error_categorizer=_CATEGORIZER).evolve(
                editor=_AssignmentEditor(), **_EVOLVE)


class HillClimbingTests(unittest.TestCase):
    def test_optional_components_accepted_edit_memory_rejected(self) -> None:
        import inspect

        from meta_agent.managers.hill_climbing import HillClimbingManager

        params = inspect.signature(HillClimbingManager.evolve).parameters
        for name in ("summarizer", "failure_summarizer", "block_suggester", "unit_selector",
                     "edit_memory"):
            self.assertIn(name, params)
        with self.assertRaisesRegex(ValueError, "edit_memory"):
            HillClimbingManager().evolve(editor=object(), edit_memory=object(), **_EVOLVE)


class HGMGuardTests(unittest.TestCase):
    def test_edit_memory_needs_paired_evaluation(self) -> None:
        from meta_agent.managers.hgm import HGMManager

        with self.assertRaisesRegex(ValueError, "expand_eval_size"):
            HGMManager().evolve(editor=object(), edit_memory=object(), **_EVOLVE)


if __name__ == "__main__":
    unittest.main()
