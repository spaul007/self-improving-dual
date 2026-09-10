"""Unit tests for belief prediction scoring + the calibration report.

Pure module — no LLM, no I/O.

    PYTHONPATH=. python3 -m unittest tests.test_belief_scoring
"""
from __future__ import annotations

import unittest

from meta_agent.belief_contract import parse_document
from meta_agent.belief_scoring import (
    UNCOVERED_P,
    Scored,
    per_belief_stats,
    per_version_brier,
    render_calibration_report,
    track_lines,
)
from meta_agent.belief_scoring import resolve as _resolve
from tests.test_belief_contract import GOOD


def resolve(*args, **kw):
    """These tests exercise the DELTA label rule (the pre-v7 default, kept
    for ablation); the module default is now the judge, so pass it
    explicitly. Judge-mode behaviour is covered in tests/test_judge_signal.py."""
    kw.setdefault("label_source", "delta")
    return _resolve(*args, **kw)


def _rec(delta, n_shared, impl, reason="gate fired on case a", tags=None):
    body = ("## Edit 1\n- **name**: `route-check`\n"
            "- **category level 1 (strategy)**: `add-verifier`\n"
            "- **category level 2 (area)**: `routing`\n"
            "- **what**: Adds a route verifier\n- **why**: routes were wrong")
    text = body + "\n\n## Outcome\n- **performance**: child 0.5000 over 10 " \
                  "evaluated cases (vs parent on 8 shared: child 0.5000, " \
                  "parent 0.4500, Δ +0.0500)\n"
    return {"delta": delta, "n_shared": n_shared, "impl_sound": impl,
            "impl_reason": reason, "text": text, "body": body,
            "tags": tags if tags is not None else
            [{"edit": 1, "strategy": "add-verifier", "area": "routing"}]}


def _pred(strategy=("s", 0.7), implementation=("i", 0.4), bv=1, iv=0,
          coverable=True):
    return {"belief_version": bv, "instruction_version": iv,
            "coverable": coverable,
            "strategy": {"slug": strategy[0], "p": strategy[1]} if strategy else None,
            "implementation": ({"slug": implementation[0], "p": implementation[1]}
                               if implementation else None)}


class TestResolve(unittest.TestCase):
    def test_sound_node_scores_both_kinds(self):
        new, pending, skipped = resolve({1: _pred()}, {1: _rec(0.05, 8, True)},
                                        threshold=0.02, min_shared=8,
                                        already=set(), n_updates=3)
        self.assertEqual((pending, skipped), ([], []))
        by_kind = {s.kind: s for s in new}
        self.assertAlmostEqual(by_kind["implementation"].brier, (0.4 - 1) ** 2)
        self.assertEqual(by_kind["implementation"].y, 1)
        self.assertAlmostEqual(by_kind["strategy"].brier, (0.7 - 1) ** 2)
        self.assertEqual((by_kind["strategy"].slug, by_kind["strategy"].node,
                          by_kind["strategy"].resolved_at_update), ("s", 1, 3))

    def test_unsound_scores_implementation_only(self):
        new, pending, _ = resolve({1: _pred()}, {1: _rec(0.05, 8, False, "never fired")},
                                  threshold=0.02, min_shared=8, already=set(),
                                  n_updates=1)
        self.assertEqual([s.kind for s in new], ["implementation"])
        self.assertEqual(new[0].y, 0)
        self.assertAlmostEqual(new[0].brier, 0.16)
        self.assertEqual(len(pending), 1)
        self.assertIn("unsound", pending[0]["reason"])

    def test_no_verdict_is_pending(self):
        new, pending, _ = resolve({1: _pred()}, {1: _rec(0.05, 8, None)},
                                  threshold=0.02, min_shared=8, already=set(),
                                  n_updates=1)
        self.assertEqual(new, [])
        self.assertEqual(pending[0]["reason"], "awaiting implementation verdict")

    def test_uncovered_scores_at_half(self):
        new, _, skipped = resolve({1: _pred(strategy=None, implementation=None)},
                                  {1: _rec(-0.05, 8, True)}, threshold=0.02,
                                  min_shared=8, already=set(), n_updates=1)
        self.assertEqual(skipped, [])
        self.assertEqual({s.p for s in new}, {UNCOVERED_P})
        self.assertEqual({s.brier for s in new}, {0.25})
        self.assertEqual({s.slug for s in new}, {None})
        self.assertEqual({s.kind: s.y for s in new},
                         {"strategy": 0, "implementation": 1})

    def test_uncoverable_uncovered_is_skipped_not_charged(self):
        # First node of a brand-new strategy: no belief could have covered
        # it, so silence is not charged — the prediction is retired.
        new, pending, skipped = resolve(
            {1: _pred(strategy=None, implementation=None, coverable=False)},
            {1: _rec(-0.05, 8, True)}, threshold=0.02, min_shared=8,
            already=set(), n_updates=1)
        self.assertEqual((new, pending), ([], []))
        self.assertEqual({(k["node"], k["kind"]) for k in skipped},
                         {(1, "strategy"), (1, "implementation")})
        self.assertIn("first node of its strategy", skipped[0]["reason"])
        # ...but a COVERED prediction on an uncoverable node still scores.
        new2, _, skipped2 = resolve(
            {1: _pred(strategy=("s", 0.7), implementation=None, coverable=False)},
            {1: _rec(0.05, 8, True)}, threshold=0.02, min_shared=8,
            already=set(), n_updates=1)
        self.assertEqual([s.kind for s in new2], ["strategy"])
        self.assertEqual([k["kind"] for k in skipped2], ["implementation"])
        # Missing key (older prediction files) means coverable.
        p = _pred(strategy=None, implementation=None)
        p.pop("coverable")
        new3, _, skipped3 = resolve({1: p}, {1: _rec(0.05, 8, True)},
                                    threshold=0.02, min_shared=8,
                                    already=set(), n_updates=1)
        self.assertEqual((len(new3), skipped3), (2, []))

    def test_matched_sub_edit_verdict_overrides_node_verdict(self):
        # Node judged unsound overall, but the matched sub-edit (1) is sound:
        # the strategy belief IS scored; the implementation belief matched
        # sub-edit 2, which is unsound.
        rec = _rec(0.05, 8, False, "node-level unsound")
        rec["impl_by_edit"] = {1: True, 2: False}
        rec["impl_reason_by_edit"] = {1: "edit 1 fine", 2: "edit 2 dead"}
        pred = _pred(strategy=("s", 0.7), implementation=("i", 0.4))
        pred["strategy"]["matched_edit"] = 1
        pred["implementation"]["matched_edit"] = 2
        new, pending, skipped = resolve({1: pred}, {1: rec}, threshold=0.02,
                                        min_shared=8, already=set(), n_updates=1)
        by = {s.kind: s for s in new}
        self.assertEqual((pending, skipped), ([], []))
        self.assertEqual(by["strategy"].y, 1)
        self.assertEqual(by["strategy"].impl_reason, "edit 1 fine")
        self.assertEqual(by["implementation"].y, 0)
        self.assertEqual(by["implementation"].impl_reason, "edit 2 dead")
        # Without per-edit verdicts the node verdict still governs.
        rec2 = _rec(0.05, 8, False, "node-level unsound")
        new2, pending2, _ = resolve({1: pred}, {1: rec2}, threshold=0.02,
                                    min_shared=8, already=set(), n_updates=1)
        self.assertEqual([s.kind for s in new2], ["implementation"])
        self.assertEqual(len(pending2), 1)

    def test_helped_uses_threshold(self):
        new, _, _ = resolve({1: _pred()}, {1: _rec(0.019, 8, True)}, threshold=0.02,
                            min_shared=8, already=set(), n_updates=1)
        self.assertEqual({s.kind: s.y for s in new}["strategy"], 0)

    def test_below_min_shared_or_unmeasured_is_open(self):
        new, pending, skipped = resolve(
            {1: _pred(), 2: _pred()},
            {1: _rec(0.05, 3, True), 2: _rec(None, 0, True)},
            threshold=0.02, min_shared=8, already=set(), n_updates=1)
        self.assertEqual((new, pending, skipped), ([], [], []))

    def test_already_prevents_rescoring(self):
        new, pending, skipped = resolve(
            {1: _pred()}, {1: _rec(0.05, 8, True)}, threshold=0.02, min_shared=8,
            already={(1, "strategy"), (1, "implementation")}, n_updates=1)
        self.assertEqual((new, pending, skipped), ([], [], []))

    def test_missing_record_is_skipped(self):
        new, pending, skipped = resolve({7: _pred()}, {}, threshold=0.02,
                                        min_shared=8, already=set(), n_updates=1)
        self.assertEqual((new, pending, skipped), ([], [], []))


def _scored(node, kind, slug, p, y, iv=0):
    return Scored(node=node, kind=kind, slug=slug, p=p, y=y, brier=(p - y) ** 2,
                  belief_version=1, instruction_version=iv,
                  resolved_at_update=2, n_shared=8, delta=0.05)


class TestAggregates(unittest.TestCase):
    def test_per_version_and_per_belief(self):
        rows = [_scored(1, "strategy", "s", 0.7, 1, iv=0),
                _scored(2, "strategy", "s", 0.7, 0, iv=1),
                _scored(3, "implementation", None, 0.5, 1, iv=1)]
        pv = per_version_brier(rows)
        self.assertEqual(pv[0][0], 1)
        self.assertAlmostEqual(pv[0][1], 0.09)
        self.assertEqual(pv[1][0], 2)
        pb = per_belief_stats(rows)
        self.assertEqual(pb[("strategy", "s")]["n"], 2)
        self.assertIn(("implementation", "(uncovered)"), pb)

    def test_track_lines(self):
        beliefs = parse_document(GOOD).beliefs
        rows = [_scored(1, "strategy", "add-verifier-helps", 0.65, 1),
                _scored(2, "strategy", "add-verifier-helps", 0.65, 0)]
        lines = track_lines(beliefs, rows, {"add-verifier-impl": 1})
        self.assertIn("n=2 · Brier", lines["add-verifier-helps"])
        self.assertIn("1 yes, 2 no", lines["add-verifier-helps"])
        self.assertIn("no scored predictions yet", lines["add-verifier-impl"])
        self.assertIn("1 misquoted citation", lines["add-verifier-impl"])

    def test_roundtrip_dict(self):
        s = _scored(1, "strategy", "s", 0.7, 1)
        self.assertEqual(Scored.from_dict(s.as_dict()), s)


class TestReport(unittest.TestCase):
    def test_sections_and_content(self):
        parsed = parse_document(GOOD)
        records = {1: _rec(0.05, 8, True),
                   2: _rec(-0.05, 8, True, tags=[{"edit": 1, "strategy": "add-verifier",
                                                  "area": "speed"}]),
                   3: _rec(0.05, 8, None), 4: _rec(None, 0, None)}
        rows = [_scored(1, "strategy", "add-verifier-helps", 0.65, 1),
                _scored(2, "strategy", None, 0.5, 0)]
        report = render_calibration_report(
            parsed=parsed, scored=rows,
            pending=[{"node": 3, "reason": "awaiting implementation verdict"}],
            records=records,
            predictions={1: _pred(), 2: _pred(strategy=None), 3: _pred(),
                         4: _pred(strategy=("add-verifier-helps", 0.65))},
            soft_violations=[], joins=[{"node": 1, "belief_id": "add-verifier-helps"}],
            version_history=[{"version": 0}, {"version": 1}], current_version=1,
            min_shared=8, n_skipped=3, label_source="delta")
        for header in ("## Calibration report", "### Per belief", "### Worst misses",
                       "### Uncovered measured nodes", "### Not scored",
                       "### Open predictions", "### Citation checks"):
            self.assertIn(header, report)
        self.assertIn("2 scored prediction(s)", report)
        self.assertIn("3 prediction(s) skipped, not scored: the first node of a "
                      "new strategy", report)
        self.assertIn("belief:add-verifier-helps (strategy strategy=add-verifier "
                      "area=routing, p=0.65): n=1", report)
        self.assertIn("cited by 1 proposal(s)", report)
        self.assertIn("node 2 · tags add-verifier/speed · strategy: not helped", report)
        self.assertIn("belief:add-verifier-helps covers `add-verifier` only in area "
                      "`routing`; a strategy-wide scope would have covered it", report)
        self.assertIn("node 3 · awaiting implementation verdict", report)
        self.assertIn("node 4 · belief:add-verifier-helps p=0.65 (strategy)", report)
        self.assertIn("v0: n=2, Brier 0.186", report)
        self.assertIn("v1: no scored predictions (current)", report)
        self.assertIn("inline citation(s) match the records", report)
        self.assertIn("labels: strategy y = Δ vs parent", report)
        self.assertNotIn("Judge vs score", report)          # delta mode: no diagnostic

    def test_judge_mode_report_has_the_diagnostic_sections(self):
        report = render_calibration_report(
            parsed=parse_document(GOOD), scored=[], pending=[], records={},
            predictions={}, soft_violations=[], joins=[], version_history=[],
            current_version=0, min_shared=8)          # default label source: judge
        self.assertIn("labels: strategy y = the judge's effect verdict", report)
        self.assertIn("### Judge vs score — a diagnostic, not a yardstick", report)
        self.assertIn("(no strategy row has a well-measured score yet)", report)
        self.assertIn("### Verdict changes since scoring", report)


if __name__ == "__main__":
    unittest.main()
