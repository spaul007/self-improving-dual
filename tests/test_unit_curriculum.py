"""Tests for meta_agent/unit_curriculum.py -- pure logic (load_units_map,
rollup_unit_counts, unit_failure_rate_for, choose_unit, UnitCurriculum), no
LLM, no HGMTree/AgentFeedback. Mirrors tests/test_curriculum.py's offline
style.

    PYTHONPATH=. python3 -m unittest tests.test_unit_curriculum
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from meta_agent.unit_curriculum import (
    UnitCurriculum,
    choose_unit,
    describe_unit_from_counts,
    load_units_map,
    rollup_unit_counts,
    unit_failure_rate_for,
)

UNITS = {
    "Business Hours": ("commonsense:Business Hours:",),
    "Activity Diversity": ("commonsense:Activity Diversity:",),
    "hard constraints: transport": ("hard:train_", "hard:flight_"),
}


class LoadUnitsMapTests(unittest.TestCase):
    def _write(self, tmpdir: str, content: str) -> str:
        p = Path(tmpdir) / "units.json"
        p.write_text(content)
        return str(p)

    def test_valid_file_loads_correctly(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = self._write(d, json.dumps({
                "_comment": "ignored",
                "Business Hours": ["commonsense:Business Hours:"],
                "hard constraints: transport": ["hard:train_", "hard:flight_"],
            }))
            units = load_units_map(path)
            self.assertEqual(units["Business Hours"], ("commonsense:Business Hours:",))
            self.assertEqual(units["hard constraints: transport"], ("hard:train_", "hard:flight_"))
            self.assertNotIn("_comment", units)

    def test_missing_file_raises(self) -> None:
        with self.assertRaises(ValueError):
            load_units_map("/nonexistent/path/units.json")

    def test_malformed_json_raises(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = self._write(d, "{not valid json")
            with self.assertRaises(ValueError):
                load_units_map(path)

    def test_non_dict_body_raises(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = self._write(d, json.dumps(["a", "b"]))
            with self.assertRaises(ValueError):
                load_units_map(path)

    def test_non_list_value_raises(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = self._write(d, json.dumps({"Unit A": "not_a_list"}))
            with self.assertRaises(ValueError):
                load_units_map(path)

    def test_empty_prefix_list_raises(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = self._write(d, json.dumps({"Unit A": []}))
            with self.assertRaises(ValueError):
                load_units_map(path)

    def test_empty_object_raises(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = self._write(d, json.dumps({}))
            with self.assertRaises(ValueError):
                load_units_map(path)


class RollupUnitCountsTests(unittest.TestCase):
    def test_correct_summation(self) -> None:
        counts = {
            "commonsense:Business Hours:dining_within_service_hours": 27,
            "commonsense:Business Hours:attraction_visit_within_opening_hours": 21,
            "hard:train_departure_window": 3,
            "hard:flight_departure_window": 2,
        }
        rolled = rollup_unit_counts(counts, UNITS)
        self.assertEqual(rolled["Business Hours"], 48)
        self.assertEqual(rolled["hard constraints: transport"], 5)
        self.assertEqual(rolled["Activity Diversity"], 0)

    def test_unmatched_checks_dropped(self) -> None:
        counts = {"commonsense:Unrelated Dimension:some_check": 99}
        rolled = rollup_unit_counts(counts, UNITS)
        self.assertEqual(sum(rolled.values()), 0)

    def test_every_unit_always_present(self) -> None:
        rolled = rollup_unit_counts({}, UNITS)
        self.assertEqual(set(rolled), set(UNITS))


class UnitFailureRateForTests(unittest.TestCase):
    def test_matches_hand_computed_sum(self) -> None:
        metrics = {
            "top_failed_checks": [
                ["commonsense:Business Hours:dining_within_service_hours", 27],
                ["commonsense:Business Hours:attraction_visit_within_opening_hours", 21],
            ],
        }
        rate = unit_failure_rate_for("Business Hours", metrics, 60, UNITS)
        self.assertAlmostEqual(rate, 48 / 60)

    def test_none_when_no_evals(self) -> None:
        self.assertIsNone(unit_failure_rate_for("Business Hours", {}, 0, UNITS))

    def test_none_unit_returns_none(self) -> None:
        self.assertIsNone(unit_failure_rate_for(None, {}, 10, UNITS))

    def test_no_plan_rate_correction_applied(self) -> None:
        # documents the intentional over-count-when-shared-case behavior is
        # bounded by the SAME no-plan correction Curriculum's own
        # failure_rate_for uses -- not a separate, undocumented mechanism.
        metrics = {
            "top_failed_checks": [["commonsense:Business Hours:dining_within_service_hours", 10]],
            "no_plan_rate": 0.5,
        }
        # scored_evals = 60 * (1 - 0.5) = 30
        rate = unit_failure_rate_for("Business Hours", metrics, 60, UNITS)
        self.assertAlmostEqual(rate, 10 / 30)


class ChooseUnitTests(unittest.TestCase):
    def test_mechanical_fallback_when_no_chooser(self) -> None:
        counts = {"A": 10, "B": 30}
        chosen = choose_unit(counts, {}, 2, units={}, check_descriptions={})
        self.assertEqual(chosen, "B")

    def test_single_candidate_skips_chooser(self) -> None:
        called = []
        def chooser(counts, descs):
            called.append(1)
            return "A"
        counts = {"A": 10}
        chosen = choose_unit(counts, {}, 2, units={}, check_descriptions={}, chooser=chooser)
        self.assertEqual(chosen, "A")
        self.assertEqual(called, [])

    def test_chooser_result_used_when_valid(self) -> None:
        counts = {"A": 10, "B": 30}
        chosen = choose_unit(
            counts, {}, 2, units={"A": ("x",), "B": ("y",)}, check_descriptions={},
            chooser=lambda c, d: "A",
        )
        self.assertEqual(chosen, "A")

    def test_chooser_invalid_answer_falls_back(self) -> None:
        counts = {"A": 10, "B": 30}
        chosen = choose_unit(
            counts, {}, 2, units={"A": ("x",), "B": ("y",)}, check_descriptions={},
            chooser=lambda c, d: "NOT_A_CANDIDATE",
        )
        self.assertEqual(chosen, "B")

    def test_chooser_exception_falls_back(self) -> None:
        def boom(c, d):
            raise RuntimeError("llm call failed")
        counts = {"A": 10, "B": 30}
        chosen = choose_unit(
            counts, {}, 2, units={"A": ("x",), "B": ("y",)}, check_descriptions={}, chooser=boom,
        )
        self.assertEqual(chosen, "B")

    def test_description_building_failure_also_falls_back(self) -> None:
        # units={} (a candidate absent from the units map) makes
        # describe_unit_from_counts itself raise -- must still fall back
        # cleanly, same as the chooser raising directly.
        counts = {"A": 10, "B": 30}
        chosen = choose_unit(
            counts, {}, 2, units={}, check_descriptions={}, chooser=lambda c, d: "A",
        )
        self.assertEqual(chosen, "B")

    def test_none_when_all_attempts_exhausted(self) -> None:
        counts = {"A": 10}
        chosen = choose_unit(counts, {"A": 2}, 2, units={}, check_descriptions={})
        self.assertIsNone(chosen)

    def test_none_when_all_zero_counts(self) -> None:
        counts = {"A": 0, "B": 0}
        chosen = choose_unit(counts, {}, 2, units={}, check_descriptions={})
        self.assertIsNone(chosen)


class DescribeUnitFromCountsTests(unittest.TestCase):
    def test_lists_member_checks_with_counts_and_descriptions(self) -> None:
        counts = {
            "commonsense:Business Hours:dining_within_service_hours": 27,
            "commonsense:Business Hours:attraction_visit_within_opening_hours": 21,
        }
        descs = {"commonsense:Business Hours:dining_within_service_hours": "Meals outside hours"}
        text = describe_unit_from_counts("Business Hours", counts, UNITS, descs)
        self.assertIn("dining_within_service_hours: 27", text)
        self.assertIn("Meals outside hours", text)
        self.assertIn("attraction_visit_within_opening_hours: 21", text)


class UnitCurriculumTests(unittest.TestCase):
    def test_initial_pick_from_seed_counts(self) -> None:
        seed_counts = {
            "commonsense:Business Hours:dining_within_service_hours": 27,
            "commonsense:Activity Diversity:diverse_attraction_options": 5,
        }
        uc = UnitCurriculum(seed_counts, units=UNITS, max_attempts=2)
        self.assertEqual(uc.current_goal, "Business Hours")
        self.assertFalse(uc.done)

    def test_advance_repicks_using_fresh_check_counts_not_stale_seed(self) -> None:
        seed_counts = {"commonsense:Business Hours:dining_within_service_hours": 27}
        uc = UnitCurriculum(seed_counts, units=UNITS, max_attempts=2, resolution_threshold=0.1)
        self.assertEqual(uc.current_goal, "Business Hours")
        # Resolve Business Hours; new check_counts now favors Activity Diversity.
        new_counts = {"commonsense:Activity Diversity:diverse_attraction_options": 40}
        uc.advance_if_ready(failure_rate=0.0, check_counts=new_counts)
        self.assertEqual(uc.current_goal, "Activity Diversity")

    def test_attempts_accumulate_across_repicks_of_the_same_unit(self) -> None:
        seed_counts = {"commonsense:Business Hours:dining_within_service_hours": 27}
        uc = UnitCurriculum(seed_counts, units=UNITS, max_attempts=2, patience=1)
        uc.record_expand()
        # patience exhausted -> re-picks; same unit still has the highest (only) count, chosen again.
        uc.advance_if_ready(failure_rate=0.9, check_counts=seed_counts)
        self.assertEqual(uc._attempts["Business Hours"], 1)
        self.assertEqual(uc.current_goal, "Business Hours")
        uc.record_expand()
        # second exhaustion -> attempts hits max_attempts -> no candidates left -> done.
        uc.advance_if_ready(failure_rate=0.9, check_counts=seed_counts)
        self.assertEqual(uc._attempts["Business Hours"], 2)
        self.assertTrue(uc.done)

    def test_done_when_choose_unit_returns_none(self) -> None:
        uc = UnitCurriculum({}, units=UNITS, max_attempts=2)
        self.assertTrue(uc.done)
        self.assertIsNone(uc.current_goal)
        self.assertIsNone(uc.directive())

    def test_directive_contains_escape_hatch_and_unit_name(self) -> None:
        seed_counts = {"commonsense:Business Hours:dining_within_service_hours": 27}
        uc = UnitCurriculum(seed_counts, units=UNITS, max_attempts=2)
        d = uc.directive()
        self.assertIn("ESCAPE HATCH", d)
        self.assertIn("Business Hours", d)

    def test_unit_selector_used_when_configured(self) -> None:
        class FakeSelector:
            def choose(self, unit_counts, descriptions, attempts, max_attempts):
                return "Activity Diversity"

        seed_counts = {
            "commonsense:Business Hours:dining_within_service_hours": 27,
            "commonsense:Activity Diversity:diverse_attraction_options": 5,
        }
        uc = UnitCurriculum(
            seed_counts, units=UNITS, max_attempts=2, unit_selector=FakeSelector(),
        )
        # FakeSelector always returns Activity Diversity, overriding the mechanical max(cands) (Business Hours).
        self.assertEqual(uc.current_goal, "Activity Diversity")

    def test_snapshot_roundtrips(self) -> None:
        seed_counts = {"commonsense:Business Hours:dining_within_service_hours": 27}
        uc = UnitCurriculum(seed_counts, units=UNITS, max_attempts=2)
        snap = uc.snapshot(current_failure_rate=0.45)
        self.assertEqual(snap.current_goal, "Business Hours")
        self.assertFalse(snap.done)


if __name__ == "__main__":
    unittest.main()
