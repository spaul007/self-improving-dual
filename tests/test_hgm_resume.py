"""Pause / finalize-now / resume of the HGM manager, the resume-state fixes
(atomic state files, durable resume count, exact RNG continuation, curriculum
restore, corrupt-round quarantine) and the resume config guard.

    PYTHONPATH=. python -m pytest tests/test_hgm_resume.py -q
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

from tests.test_hgm_smoke import _StubEditor, _StubEvaluator

N_TRAIN = 20
TRAIN = [f"c{i}" for i in range(N_TRAIN)]


class _Killed(BaseException):
    """Simulates SIGKILL mid-batch (not an Exception, so nothing swallows it)."""


class _HookEvaluator(_StubEvaluator):
    """Runs ``hook(call_no, exp_dir)`` before each evaluator.run."""

    def __init__(self, exp: Path, hook=None) -> None:
        super().__init__()
        self.exp, self.hook = exp, hook

    def run(self, round_dir, benchmark_dir, *, case_ids=None):
        if self.hook is not None:
            self.hook(self.run_calls + 1, self.exp)
        return super().run(round_dir, benchmark_dir, case_ids=case_ids)


class ResumeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="hgm_resume_"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.seed = self.tmp / "seed"
        self.seed.mkdir()
        (self.seed / "workflow.py").write_text("def run_task(t):\n    return None\n")
        self.exp = self.tmp / "exp"
        self.exp.mkdir()

    def _evolve(self, hook=None, resume=False, eval_budget=40):
        from meta_agent.feedback_gatherer import DefaultFeedbackGatherer
        from meta_agent.managers.hgm import HGMManager

        m = HGMManager(eval_budget=eval_budget, init_expansions=2, eval_batch_size=4,
                       alpha=0.6, seed=7, snapshot_tree=True)
        out = m.evolve(
            editor=_StubEditor(), evaluator=_HookEvaluator(self.exp, hook),
            gatherer=DefaultFeedbackGatherer(), seed_dir=self.seed,
            benchmark_dir=self.tmp / "bench", experiment_dir=self.exp, max_rounds=30,
            score_target=None, train_case_ids=TRAIN, eval_case_ids=None,
            **({"resume": True} if resume else {}),
        )
        return m, out

    def _assert_consistent(self, m) -> None:
        for nid, n in m._tree.nodes.items():
            side = json.loads((n.round_dir / "hgm_node.json").read_text())
            self.assertEqual(side["n_evals"], n.n_evals, nid)
            self.assertTrue(all(v == 1 for v in side["case_eval_counts"].values()), nid)
            self.assertLessEqual(n.n_attempted, N_TRAIN, nid)

    def test_pause_then_resume_spends_exact_budget(self) -> None:
        from meta_agent.managers.hgm import RunPaused

        def hook(call, exp):
            if call == 4:
                (exp / "PAUSE").touch()

        with self.assertRaises(RunPaused):
            self._evolve(hook)
        state = json.loads((self.exp / "loop_state.json").read_text())
        paused_at = state["budget_spent"]
        self.assertLess(paused_at, 40)
        self.assertIn("PAUSED", state["current_action"])
        n_nodes_paused = state["n_nodes"]

        m, out = self._evolve(resume=True)
        self.assertEqual(m._budget_spent, 40)
        self.assertFalse((self.exp / "PAUSE").exists())
        self.assertGreaterEqual(len(m._tree.nodes), n_nodes_paused)
        self.assertIn("RESUME #1", (self.exp / "RESUMES.log").read_text())
        self.assertIn(out.best_round, m._tree.nodes)
        self._assert_consistent(m)
        # root was NOT re-evaluated on resume
        self.assertEqual(m._tree[0].n_evals, N_TRAIN)

    def test_kill_mid_batch_reruns_it_uncharged(self) -> None:
        def hook(call, exp):
            if call == 5:
                raise _Killed()

        with self.assertRaises(_Killed):
            self._evolve(hook)
        m, _ = self._evolve(resume=True)
        self.assertEqual(m._budget_spent, 40)
        spent = sum(n.n_attempted for nid, n in m._tree.nodes.items() if nid != 0)
        # finalize top-ups are uncharged; everything else matches the budget
        self.assertGreaterEqual(spent, 40)
        self._assert_consistent(m)

    def test_half_expanded_round_is_marked_aborted(self) -> None:
        from meta_agent.managers.hgm import RunPaused

        def hook(call, exp):
            if call == 3:
                (exp / "PAUSE").touch()

        with self.assertRaises(RunPaused):
            self._evolve(hook)
        state = json.loads((self.exp / "loop_state.json").read_text())
        ghost = self.exp / f"round_{state['next_id']:03d}"
        (ghost / "logs").mkdir(parents=True)          # editor was killed here
        m, _ = self._evolve(resume=True)
        self.assertTrue(list(self.exp.glob(f"{ghost.name}.aborted-*")))
        self.assertEqual(m._budget_spent, 40)
        self._assert_consistent(m)

    def test_finalize_now_stops_spending_and_finalizes(self) -> None:
        def hook(call, exp):
            if call == 4:
                (exp / "FINALIZE_NOW").touch()

        m, out = self._evolve(hook)
        self.assertLess(m._budget_spent, 40)
        self.assertIn(out.best_round, m._tree.nodes)
        self.assertIn("FINISHED", json.loads((self.exp / "loop_state.json").read_text())["current_action"])

    def test_resume_can_extend_budget(self) -> None:
        self._evolve(eval_budget=24)
        m, _ = self._evolve(resume=True, eval_budget=40)
        self.assertEqual(m._budget_spent, 40)


class ResumeStateFixTests(ResumeTests):
    """Resume-state bugs found in EXP-042: a truncated loop_state.json reset the
    resume counter (RESUMES.log "#1, #1, #2") and the uncharged-eval count; RNG
    state was not persisted; a corrupt feedback.json crashed the resume."""

    def _tree_signature(self, m):
        return {nid: (n.parent_id, tuple(sorted(n.evaluated_case_ids)))
                for nid, n in m._tree.nodes.items()}

    def _pause_at(self, call_no):
        def hook(call, exp):
            if call == call_no:
                (exp / "PAUSE").touch()
        return hook

    def test_truncated_loop_state_keeps_resume_count(self) -> None:
        from meta_agent.managers.hgm import RunPaused

        with self.assertRaises(RunPaused):
            self._evolve(self._pause_at(4))
        (self.exp / "loop_state.json").write_text("")          # quota-truncated
        with self.assertRaises(RunPaused):
            self._evolve(self._pause_at(2), resume=True)
        (self.exp / "loop_state.json").write_text("{")         # half-written
        m, _ = self._evolve(resume=True)
        log = (self.exp / "RESUMES.log").read_text()
        self.assertIn("RESUME #1", log)
        self.assertIn("RESUME #2", log)
        self.assertEqual(m._resume_count, 2)
        self.assertEqual(m._budget_spent, 40)
        self._assert_consistent(m)

    def test_resume_continues_the_exact_rng_sequence(self) -> None:
        from meta_agent.managers.hgm import RunPaused

        straight, _ = self._evolve()
        expected = self._tree_signature(straight)
        shutil.rmtree(self.exp)
        self.exp.mkdir()
        with self.assertRaises(RunPaused):
            self._evolve(self._pause_at(5))
        resumed, _ = self._evolve(resume=True)
        self.assertIn("RNG state restored exactly", (self.exp / "RESUMES.log").read_text())
        self.assertEqual(self._tree_signature(resumed), expected)

    def test_state_files_are_written_atomically(self) -> None:
        self._evolve()
        leftovers = [p for p in self.exp.rglob(".*.tmp")]
        self.assertEqual(leftovers, [])
        json.loads((self.exp / "loop_state.json").read_text())
        self.assertIn("rng_state", json.loads((self.exp / "loop_state.json").read_text()))

    def test_corrupt_feedback_is_quarantined_not_fatal(self) -> None:
        from meta_agent.managers.hgm import RunPaused

        with self.assertRaises(RunPaused):
            self._evolve(self._pause_at(5))
        last = sorted(p for p in self.exp.glob("round_*") if p.name[6:].isdigit())[-1]
        (last / "feedback.json").write_text('{"round_number": 3, "trunc')
        m, _ = self._evolve(resume=True)
        self.assertTrue(list(self.exp.glob(f"{last.name}.corrupt-*")))
        self.assertEqual(m._budget_spent, 40)

    def test_dual_manager_refuses_resume_clearly(self) -> None:
        from meta_agent.managers.hgm_dual import HGMDualManager

        m = HGMDualManager(eval_budget=8, error_categorizer="projects.travel.travel_error_categorizer:categorize_errors")
        with self.assertRaises(NotImplementedError):
            m.evolve(editor=None, evaluator=None, gatherer=None, seed_dir=self.seed,
                     benchmark_dir=self.tmp, experiment_dir=self.exp, max_rounds=30,
                     score_target=None, train_case_ids=TRAIN, eval_case_ids=None, resume=True)


class CurriculumStateTests(unittest.TestCase):
    def test_check_curriculum_round_trip(self) -> None:
        from meta_agent.curriculum import Curriculum

        c = Curriculum([("a", 5), ("b", 3), ("c", 1)], patience=2)
        c.record_expand(); c.record_expand()
        c.advance_if_ready(failure_rate=0.9)
        state = c.state_dict()
        d = Curriculum([("a", 5), ("b", 3), ("c", 1)], patience=2)
        d.load_state(state)
        self.assertEqual((d.current_goal, d.state_dict()), (c.current_goal, state))

    def test_unit_curriculum_restore_does_not_rechoose(self) -> None:
        from meta_agent.unit_curriculum import UnitCurriculum

        class _Selector:
            calls = 0

            def choose(self, *a, **k):
                _Selector.calls += 1
                return "u2"

        units = {"u1": ("a",), "u2": ("b",)}
        state = {"kind": "unit", "attempts": {"u1": 1}, "resolved": ["u1"],
                 "rounds_on_current": 3, "current": "u2", "done": False}
        cur = UnitCurriculum.restore(state, units=units, max_attempts=3, unit_selector=_Selector())
        self.assertEqual(_Selector.calls, 0)
        self.assertEqual(cur.current_goal, "u2")
        self.assertEqual(cur.state_dict(), state)


class ResumeConfigGuardTests(unittest.TestCase):
    def test_allowed_and_forbidden(self) -> None:
        from main_loop import resume_config_diff

        a = "manager:\n  config:\n    eval_budget: 100\n    alpha: 0.5\nloop:\n  max_rounds: 20\n"
        ok = "manager:\n  config:\n    eval_budget: 200\n    alpha: 0.5\nloop:\n  max_rounds: 30\n"
        bad = "manager:\n  config:\n    eval_budget: 100\n    alpha: 0.6\nloop:\n  max_rounds: 20\n"
        allowed, forbidden = resume_config_diff(a, ok)
        self.assertEqual(forbidden, [])
        self.assertEqual({k for k, *_ in allowed}, {"manager.config.eval_budget", "loop.max_rounds"})
        _, forbidden = resume_config_diff(a, bad)
        self.assertEqual([k for k, *_ in forbidden], ["manager.config.alpha"])


if __name__ == "__main__":
    unittest.main()
