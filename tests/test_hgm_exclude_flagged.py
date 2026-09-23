"""HGMManager ``exclude_flagged_cases``: scorer-flagged infrastructure
failures (``details["excluded"]``) are ATTEMPTED but carry no utility.

    PYTHONPATH=. python3 -m pytest -q tests/test_hgm_exclude_flagged.py
"""
from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from meta_agent.managers.hgm import HGMManager
from meta_agent.managers.hgm_tree import HGMNode
from meta_agent.models import CaseResult, EvaluationResult


def _case(cid: str, score: float, excluded: bool = False) -> CaseResult:
    return CaseResult(case_id=cid, passed=score >= 1.0, score=score,
                      details={"excluded": True} if excluded else {})


def _batch(*cases: CaseResult) -> EvaluationResult:
    return EvaluationResult(score=0.0, per_case=list(cases))


class ExcludeFlaggedTests(unittest.TestCase):
    def test_default_off_records_flagged_case_as_a_zero(self) -> None:
        with TemporaryDirectory() as d:
            node = HGMNode(0, None, Path(d))
            m = HGMManager()
            self.assertFalse(m.exclude_flagged_cases)
            m._record_batch(node, _batch(_case("a", 1.0), _case("b", 0.0, excluded=True)))
        self.assertEqual(node.n_evals, 2)
        self.assertEqual(node.n_excluded, 0)
        self.assertEqual(node.mean_utility, 0.5)

    def test_flagged_case_is_attempted_but_carries_no_utility(self) -> None:
        with TemporaryDirectory() as d:
            node = HGMNode(0, None, Path(d))
            m = HGMManager(exclude_flagged_cases=True)
            m._record_batch(node, _batch(_case("a", 1.0), _case("b", 0.0, excluded=True)))
        self.assertEqual(node.n_evals, 1)
        self.assertEqual(node.n_excluded, 1)
        self.assertEqual(node.n_attempted, 2)
        self.assertEqual(node.mean_utility, 1.0)
        self.assertEqual((node.n_success, node.n_failure), (1.0, 0.0))
        self.assertEqual(sorted(node.evaluated_case_ids), ["a", "b"])
        # Feedback still sees the infra case.
        self.assertEqual([c.case_id for c in node.case_results], ["a", "b"])

    def test_flagged_case_is_not_resampled(self) -> None:
        with TemporaryDirectory() as d:
            node = HGMNode(0, None, Path(d))
            m = HGMManager(exclude_flagged_cases=True)
            m._train_case_ids = ["a", "b"]
            m._tree.add(node)
            m._record_batch(node, _batch(_case("a", 1.0), _case("b", 0.0, excluded=True)))
            self.assertEqual(m._evaluable(), [])

    def test_finalist_with_an_exclusion_is_still_lcb_eligible(self) -> None:
        with TemporaryDirectory() as d:
            m = HGMManager(exclude_flagged_cases=True)
            m._train_case_ids = ["a", "b", "c"]
            full = HGMNode(0, None, Path(d))
            partial = HGMNode(1, 0, Path(d))
            m._tree.add(full)
            m._tree.add(partial)
            m._record_batch(full, _batch(_case("a", 1.0), _case("b", 1.0), _case("c", 0.0, excluded=True)))
            m._record_batch(partial, _batch(_case("a", 1.0)))
            self.assertEqual(m._fully_evaluated_ids(), {0})

    def test_all_excluded_node_is_not_a_finalist(self) -> None:
        with TemporaryDirectory() as d:
            m = HGMManager(exclude_flagged_cases=True)
            m._train_case_ids = ["a"]
            n = HGMNode(0, None, Path(d))
            m._tree.add(n)
            m._record_batch(n, _batch(_case("a", 0.0, excluded=True)))
            self.assertEqual(m._fully_evaluated_ids(), set())


class ExcludeCrashedTests(unittest.TestCase):
    def _crash(self, cid):
        return CaseResult(case_id=cid, passed=False, score=0.0, error="timeout after 60s")

    def test_default_off_records_crash_as_zero(self) -> None:
        with TemporaryDirectory() as d:
            node = HGMNode(0, None, Path(d))
            HGMManager(exclude_flagged_cases=True)._record_batch(node, _batch(_case("a", 1.0), self._crash("b")))
        self.assertEqual((node.n_evals, node.n_excluded), (2, 0))

    def test_on_crash_is_attempted_without_utility(self) -> None:
        with TemporaryDirectory() as d:
            node = HGMNode(0, None, Path(d))
            HGMManager(exclude_crashed_cases=True)._record_batch(node, _batch(_case("a", 1.0), self._crash("b")))
        self.assertEqual((node.n_evals, node.n_excluded, node.n_attempted), (1, 1, 2))
        self.assertEqual(node.mean_utility, 1.0)


if __name__ == "__main__":
    unittest.main()
