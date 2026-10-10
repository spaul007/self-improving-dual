"""Pass/fail contrast pairing (meta_agent/case_reflections.py::find_contrast) and its section in
the per-case files.

    PYTHONPATH=. python3 -m pytest tests/test_case_contrast.py -q
"""
from __future__ import annotations

import json
import unittest

from meta_agent.case_reflections import find_contrast
from meta_agent.models import CaseResult
from tests.test_case_reflections import SECRET, _CaseFixture, _rec


def _run(node, parent, k, passed, score, limited=None):
    return {"node": node, "parent": parent, "k": k, "passed": passed, "score": score, "limited": limited}


class FindContrastTests(unittest.TestCase):
    def test_same_node_beats_parent_child_beats_any(self) -> None:
        runs = [_run(0, None, 1, False, 0.0), _run(1, 0, 1, True, 1.0),
                _run(3, 2, 1, True, 1.0), _run(3, 2, 2, False, 0.5)]
        c = find_contrast(runs)
        self.assertEqual((c["high"]["node"], c["high"]["k"], c["low"]["node"], c["low"]["k"]), (3, 1, 3, 2))
        self.assertEqual((c["tier"], c["relation"]), (0, "same node"))
        c = find_contrast(runs[:3])            # no repeat: parent -> child, despite node 3's equal gap
        self.assertEqual((c["tier"], c["high"]["node"], c["low"]["node"]), (1, 1, 0))
        c = find_contrast([_run(1, 0, 1, True, 1.0), _run(2, 0, 1, False, 0.0)])   # siblings
        self.assertEqual(c["tier"], 2)
        self.assertIn("different nodes", c["relation"])

    def test_largest_gap_within_a_tier(self) -> None:
        c = find_contrast([_run(1, 0, 1, True, 1.0), _run(0, None, 1, False, 0.7),
                           _run(2, 1, 1, False, 0.1)])
        self.assertEqual((c["high"]["node"], c["low"]["node"]), (1, 2))
        self.assertAlmostEqual(c["gap"], 0.9)

    def test_gap_threshold_without_a_pass(self) -> None:
        runs = [_run(0, None, 1, False, 0.30), _run(1, 0, 1, False, 0.45)]
        self.assertIsNone(find_contrast(runs))                       # gap 0.15 < 0.2
        self.assertIsNotNone(find_contrast(runs, min_gap=0.1))
        self.assertIsNone(find_contrast([_run(0, None, 1, True, 1.0), _run(1, 0, 1, True, 1.0)]))
        self.assertIsNone(find_contrast([_run(0, None, 1, True, 1.0)]))

    def test_budget_limited_run_is_never_the_low_side(self) -> None:
        runs = [_run(0, None, 1, False, 0.0, limited="timeout after 600s"), _run(1, 0, 1, True, 1.0)]
        self.assertIsNone(find_contrast(runs))
        runs.append(_run(2, 1, 1, False, 0.2))
        c = find_contrast(runs)
        self.assertEqual((c["high"]["node"], c["low"]["node"]), (1, 2))
        # ...but a limited run may be the HIGH side (it did well despite the cap)
        c = find_contrast([_run(0, None, 1, True, 1.0, limited="cap"), _run(1, 0, 1, False, 0.0)])
        self.assertEqual(c["high"]["node"], 0)


class ContrastSectionTests(_CaseFixture, unittest.TestCase):
    """tests/test_case_reflections.py's fixture -- case A: root 0.40 FAIL, node 1 1.00 PASS,
    node 2 twice: 0.60 FAIL then 1.00 PASS."""

    def _a(self, **kw):
        summary = self._build(**kw)
        return summary, (self.run / "case_reflections" / "A.md").read_text()

    def test_contrast_section_prefers_the_same_node_repeat(self) -> None:
        summary, a = self._a()
        self.assertIn("## Pass/fail contrast", a)
        self.assertIn("high: node 2 (parent 1) · eval 2 · score 1.00 PASSED", a)
        self.assertIn("low:  node 2 (parent 1) · eval 1 · score 0.60 FAILED", a)
        self.assertIn("relation: same node · gap 0.40", a)
        self.assertIn("| PATCH | keep: keep the interface read |", a)
        self.assertLess(a.index("## Pass/fail contrast"), a.index("### node 0"))   # in the header
        self.assertEqual(summary["A"]["contrast"]["high"], (2, 2))
        self.assertEqual(summary["A"]["contrast"]["low"], (2, 1))
        self.assertTrue(summary["A"]["unstable"])
        self.assertFalse(summary["B/x y"]["unstable"])
        self.assertAlmostEqual(summary["A"]["spread"], 0.6)

    def test_where_why_shown_in_contrast_but_redacted(self) -> None:
        n2 = self.nodes[2]
        r = _rec(2, 1, "A", "PATCH", 1, False, 0.6, lesson="read the interface", conf=60, ts=3,
                 terms=[SECRET])
        r["parsed"].update(where=f"step 4 ran {SECRET}", why="skipped the interface", catch="a type check")
        (n2.round_dir / "reflections" / "A.PATCH.e1.json").write_text(json.dumps(r))
        _, a = self._a()
        self.assertIn("where: step 4 ran [redacted] / why: skipped the interface / catch: a type check", a)
        self.assertNotIn(SECRET, a)

    def test_budget_limited_lows_excluded_and_named(self) -> None:
        limited = lambda c: "timeout after 600s" if c.case_id == "A" and c.score < 1.0 else None  # noqa: E731
        summary, a = self._a(limited_fn=limited)
        self.assertNotIn("## Pass/fail contrast", a)
        self.assertFalse(summary["A"]["unstable"])
        self.assertIn("cut short by a budget", a)
        # a hook that raises is ignored, never fatal
        summary, _ = self._a(limited_fn=lambda c: 1 / 0)
        self.assertTrue(summary["A"]["unstable"])

    def test_gap_threshold_is_configurable(self) -> None:
        # node 2's repeat differs by 0.40 AND pass/fail, so it qualifies at any gap; with only
        # failing runs left, a high threshold removes the contrast.
        for n in self.nodes:
            n.case_results[:] = [CaseResult(case_id="A", passed=False, score=s)
                                 for s in ([0.3] if n.node_id == 0 else [0.45] if n.node_id == 1 else [])]
        self.assertTrue(self._build(contrast_min_gap=0.1)["A"]["unstable"])
        self.assertFalse(self._build(contrast_min_gap=0.2)["A"]["unstable"])

    def test_index_has_unstable_and_spread_columns(self) -> None:
        self._build()
        rows = [ln for ln in (self.run / "case_reflections" / "INDEX.md").read_text().splitlines()
                if ln.startswith("| A |")]
        cells = [c.strip() for c in rows[0].strip("|").split("|")]
        self.assertEqual(len(cells), 9)
        self.assertEqual(cells[7], "yes")
        self.assertEqual(cells[8], "0.60")                          # max - min score

    def test_contrast_off_under_exposure_off(self) -> None:
        _, a = self._a(exposure="off")
        self.assertNotIn("## Pass/fail contrast", a)



class TravelLimitedByBudgetTests(unittest.TestCase):
    """The travel scorers' ``limited_by_budget`` hook: a run cut short by the case timeout, a
    stage's iteration cap or a grader conversion failure is not a behavioural low."""

    def test_both_travel_scorers(self) -> None:
        import importlib

        for proj in ("travel_mas_refactored", "travel_mas_refactored_gemma"):
            try:
                mod = importlib.import_module(f"projects.{proj}.adapter.scorer_impl")
            except Exception as exc:  # noqa: BLE001
                self.skipTest(f"travel scorer not importable here: {exc!r}")
            s = mod.TravelCompositeScorer()
            C = lambda **kw: CaseResult(case_id="t", passed=False, score=0.1, **kw)  # noqa: E731
            self.assertIn("timeout after 900s", s.limited_by_budget(C(error="timeout after 900s")))
            self.assertIn("sightseeing hit its iteration cap", s.limited_by_budget(C(details={
                "agent_metadata": {"sightseeing_budget_exhausted": True, "hotel_budget_exhausted": False}})))
            self.assertIn("conversion failed", s.limited_by_budget(C(details={"conversion_error_type": "json"})))
            self.assertIsNone(s.limited_by_budget(C(details={"agent_metadata": {"stage_iterations": {"a": 3}}})))
            self.assertIsNone(s.limited_by_budget(C()))


if __name__ == "__main__":
    unittest.main()
