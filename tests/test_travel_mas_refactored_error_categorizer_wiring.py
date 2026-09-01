"""Every travel_mas_refactored config must wire up an error_categorizer,
and keep the supplementary hardest-cases top-off trimmed to 1.

Real gap this guards against: gatherer.config was `{}` for travel_mas_refactored
across every config (confirmed live 2026-09-01) -- without an error_categorizer,
DefaultFeedbackGatherer's failure report degrades to "3 arbitrary hardest
cases" instead of grouping failures by category (e.g. reasonable_transfer_time,
traceable_accommodation) with representative examples, starving the editor
and block_suggester of the causal detail needed to diagnose root causes
instead of symptoms. projects/travel/travel_error_categorizer.py is pure
schema-logic over dimension_details/hard_constraints/failed_checks, which
travel_mas_refactored's own scorer emits identically (confirmed live against
real reval case data) -- same categorizer already reused verbatim for the
sibling travel_mas project's hgm_dual config. n_hard_cases (already a plain
DefaultFeedbackGatherer kwarg, no new code needed) is a separate top-off of
lowest-scoring cases shown ALONGSIDE the categorized examples, not just a
no-categorizer fallback -- trimmed from its default of 3 to 1 the same day.

    PYTHONPATH=. python3 -m unittest tests.test_travel_mas_refactored_error_categorizer_wiring
"""
from __future__ import annotations

import glob
import unittest

import yaml

_EXPECTED_CATEGORIZER = "projects.travel.travel_error_categorizer:categorize_errors"
_EXPECTED_N_HARD_CASES = 1


class TravelMasRefactoredErrorCategorizerWiringTests(unittest.TestCase):
    def test_every_travel_mas_refactored_config_wires_the_categorizer(self) -> None:
        checked = 0
        missing: list[str] = []
        for path in sorted(glob.glob("configs/*.yaml")):
            with open(path, encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            if cfg.get("project") != "travel_mas_refactored":
                continue
            checked += 1
            gc = (cfg.get("gatherer") or {}).get("config", {})
            if gc.get("error_categorizer") != _EXPECTED_CATEGORIZER:
                missing.append(
                    f"{path}: error_categorizer={gc.get('error_categorizer')!r}"
                )
            if gc.get("n_hard_cases") != _EXPECTED_N_HARD_CASES:
                missing.append(f"{path}: n_hard_cases={gc.get('n_hard_cases')!r}")

        # Sanity: this test is only meaningful if it actually found configs
        # to check -- a silent 0-checked pass would hide a real regression
        # (e.g. every config renamed away from project: "travel_mas_refactored").
        self.assertGreater(checked, 0, "no travel_mas_refactored configs found")
        self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()
