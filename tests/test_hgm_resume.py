"""``HGMManager.evolve(resume=True)`` and ``main_loop.py --resume``.

A run is killed (KeyboardInterrupt from the editor or the evaluator) and
continued from its round dirs: the tree, feedback and spend come back as
they were, an interrupted editor session's dir is moved aside and its id
reused, a child killed before its paired evaluation gets it on resume, and
the continuation is deterministic.

    PYTHONPATH=. python3 -m unittest tests.test_hgm_resume
"""
from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from tests.test_hgm_paired_eval import _Editor


class _Kill(KeyboardInterrupt):
    pass


class _KillingEditor(_Editor):
    def __init__(self, kill_at: int) -> None:
        super().__init__()
        self.kill_at = kill_at

    def apply(self, feedback, base_dir, out_dir, *, context=None, has_suggestion=False):
        if self.calls + 1 == self.kill_at:
            self.calls += 1
            raise _Kill(f"killed during editor call {self.calls}")
        return super().apply(feedback, base_dir, out_dir, context=context,
                             has_suggestion=has_suggestion)


class _KillingEvaluator:
    def __init__(self, kill_at: int = 0) -> None:
        from tests.test_hgm_smoke import _StubEvaluator

        self.inner = _StubEvaluator()
        self.kill_at = kill_at
        self.calls = 0

    def run(self, round_dir, benchmark_dir, *, case_ids=None):
        self.calls += 1
        if self.calls == self.kill_at:
            raise _Kill(f"killed during evaluation {self.calls}")
        return self.inner.run(round_dir, benchmark_dir, case_ids=case_ids)


def _shape(manager) -> list[tuple]:
    return [(nid, n.parent_id, n.edit_failed, n.n_evals, round(n.mean_utility, 9))
            for nid, n in sorted(manager._tree.nodes.items())]


class _Run(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="hgm_resume_"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.seed = self.tmp / "seed"
        self.seed.mkdir()
        (self.seed / "workflow.py").write_text("def run_task(task):\n    return None\n")
        self.exp = self.tmp / "exp"
        self.exp.mkdir()

    def manager(self, **kw):
        from meta_agent.managers.hgm import HGMManager

        opts = dict(eval_budget=40, init_expansions=2, eval_batch_size=4, alpha=0.6, seed=7,
                    snapshot_tree=True, finalize_top_k=0, eval_repeats=2)
        opts.update(kw)
        return HGMManager(**opts)

    def evolve(self, manager, editor, evaluator=None, *, resume=False, exp=None):
        from meta_agent.feedback_gatherer import DefaultFeedbackGatherer

        kwargs = {"resume": True} if resume else {}
        return manager.evolve(
            editor=editor, evaluator=evaluator or _KillingEvaluator(),
            gatherer=DefaultFeedbackGatherer(), seed_dir=self.seed,
            benchmark_dir=self.tmp / "b", experiment_dir=exp or self.exp, max_rounds=30,
            score_target=None, train_case_ids=[f"c{i}" for i in range(12)],
            eval_case_ids=None, **kwargs)

    def killed(self, fn) -> None:
        with self.assertRaises(_Kill):
            fn()


class ResumeTests(_Run):
    def test_kill_during_an_edit_then_resume(self) -> None:
        a = self.manager()
        self.killed(lambda: self.evolve(a, _KillingEditor(kill_at=5)))
        before = _shape(a)
        spent_before = a._budget_spent
        self.assertTrue((self.exp / "round_005").is_dir())         # the killed session's dir

        b = self.manager()
        self.killed(lambda: self.evolve(b, _KillingEditor(kill_at=1), resume=True))
        self.assertEqual(_shape(b), before)                        # restored exactly
        self.assertEqual(b._budget_spent, spent_before)
        self.assertTrue((self.exp / "interrupted" / "round_005.1").is_dir())

        c = self.manager()
        outcome = self.evolve(c, _Editor(), resume=True)
        self.assertTrue((self.exp / "interrupted" / "round_005.2").is_dir())
        self.assertIn(5, c._tree.nodes)                            # the id was reused
        self.assertLessEqual(c._budget_spent, 40)
        self.assertIn(outcome.best_round, c._tree.nodes)
        log = json.loads((self.exp / "resume_log.json").read_text())["resumes"]
        self.assertEqual(len(log), 2)
        self.assertEqual(log[0]["moved"], ["round_005 -> interrupted/round_005.1"])
        events = [json.loads(l)["event"] for l in
                  (self.exp / "snapshots" / "tree_snapshots.jsonl").read_text().splitlines()]
        self.assertEqual(events.count("resume"), 2)

    def test_continuation_is_deterministic(self) -> None:
        a = self.manager()
        self.killed(lambda: self.evolve(a, _KillingEditor(kill_at=4)))
        twin = self.tmp / "twin"
        shutil.copytree(self.exp, twin)
        b, c = self.manager(), self.manager()
        self.evolve(b, _Editor(), resume=True)
        self.evolve(c, _Editor(), resume=True, exp=twin)
        self.assertEqual(_shape(b), _shape(c))

    def test_child_killed_before_its_paired_evaluation_gets_it_on_resume(self) -> None:
        # Seed pre-eval is evaluation 1; the first paired eval is 2.
        a = self.manager(expand_eval_size=4)
        self.killed(lambda: self.evolve(a, _Editor(), _KillingEvaluator(kill_at=2)))
        self.assertEqual(a._tree[1].n_evals, 0)
        b = self.manager(expand_eval_size=4)
        self.evolve(b, _Editor(), resume=True)
        log = json.loads((self.exp / "resume_log.json").read_text())["resumes"][0]
        self.assertEqual(log["paired_on_resume"], [1])
        self.assertGreaterEqual(b._tree[1].n_evals, 4)

    def test_recovered_batch_and_missing_sidecar(self) -> None:
        a = self.manager()
        self.killed(lambda: self.evolve(a, _KillingEditor(kill_at=5)))
        n1 = a._tree[1].n_evals
        side = self.exp / "round_001" / "hgm_node.json"
        data = json.loads(side.read_text())
        data["n_evals"] = n1 - 4           # the sidecar missed the last batch
        side.write_text(json.dumps(data))
        (self.exp / "round_002" / "hgm_node.json").unlink()
        b = self.manager()
        self.killed(lambda: self.evolve(b, _KillingEditor(kill_at=1), resume=True))
        self.assertEqual(b._tree[1].n_evals, n1)
        self.assertEqual(json.loads(side.read_text())["n_evals"], n1)
        self.assertTrue((self.exp / "round_002" / "hgm_node.json").is_file())
        log = json.loads((self.exp / "resume_log.json").read_text())["resumes"][0]
        self.assertIn(1, log["recovered"])

    def test_streak_and_edit_failed_nodes(self) -> None:
        a = self.manager()
        editor = _KillingEditor(kill_at=5)
        editor.fail = {3, 4}
        self.killed(lambda: self.evolve(a, editor))
        b = self.manager(max_consecutive_edit_failures=3)
        self.killed(lambda: self.evolve(b, _KillingEditor(kill_at=1), resume=True))
        self.assertEqual(b._consecutive_edit_failures, 2)
        self.assertTrue(b._tree[3].edit_failed and b._tree[4].edit_failed)
        self.assertEqual(b._tree[3].parent_id, a._tree[3].parent_id)

    def test_refuses_without_the_seed_evaluation(self) -> None:
        (self.exp / "round_000").mkdir()
        with self.assertRaisesRegex(ValueError, "seed's pre-evaluation"):
            self.evolve(self.manager(), _Editor(), resume=True)

    def test_finalize_in_progress_is_not_repeated(self) -> None:
        a = self.manager(finalize_top_k=2)
        self.evolve(a, _Editor())
        b = self.manager(finalize_top_k=2)
        calls = []
        b._finalize_top_k = lambda *args: calls.append(1)
        self.evolve(b, _Editor(), resume=True)
        self.assertTrue(b._skip_finalize_top_k)
        self.assertEqual(calls, [])


class CurriculumRestoreTests(unittest.TestCase):
    def test_check_curriculum(self) -> None:
        import dataclasses

        from meta_agent.curriculum import Curriculum

        c = Curriculum([("a", 5), ("b", 3), ("c", 1)], patience=2)
        c.record_expand(); c.record_expand()
        c.advance_if_ready(failure_rate=None)
        c.record_expand()
        snap = dataclasses.asdict(c.snapshot(current_failure_rate=None))
        r = Curriculum([("a", 5), ("b", 3), ("c", 1)], patience=2)
        r.restore(snap)
        self.assertEqual(dataclasses.asdict(r.snapshot(current_failure_rate=None)), snap)

    def test_unit_curriculum_without_the_selector(self) -> None:
        import dataclasses

        from meta_agent.unit_curriculum import UnitCurriculum

        class _NoCall:
            def choose(self, *a, **k):
                raise AssertionError("the unit selector must not run on restore")

        units = {"u1": ("a", "b"), "u2": ("c",)}
        c = UnitCurriculum({"a": 4, "c": 1}, units=units, max_attempts=2, patience=1)
        c.record_expand()
        c.advance_if_ready(failure_rate=None, check_counts={"a": 4, "c": 1})
        c.record_expand()
        snap = dataclasses.asdict(c.snapshot(current_failure_rate=None))
        r = UnitCurriculum.restored(snap, units=units, max_attempts=2, patience=1,
                                    unit_selector=_NoCall())
        self.assertEqual(dataclasses.asdict(r.snapshot(current_failure_rate=None)), snap)


class MemoryOrphanTests(unittest.TestCase):
    def test_evaluated_child_no_window_saw_is_adopted(self) -> None:
        import random

        from meta_agent.edit_memory.layer import EditMemoryLayer
        from meta_agent.managers.hgm_tree import HGMNode, HGMTree
        from meta_agent.models import CaseResult

        with tempfile.TemporaryDirectory() as d:
            lay = EditMemoryLayer(lambda **kw: None, window_size=4)
            lay.setup(Path(d))
            tree = HGMTree(rng=random.Random(0))
            tree.add(HGMNode(node_id=0, parent_id=None, round_dir=Path(d)))
            for nid, arm in ((1, "with"), (2, "without")):
                n = HGMNode(node_id=nid, parent_id=0, round_dir=Path(d), memory_arm=arm,
                            memory_version=1)
                n.record(CaseResult(case_id="c", passed=True, score=1.0))
                tree.add(n)
            lay.window = [1]                  # node 2's expand_eval was never delivered
            lay.on_event("resume", tree)
            self.assertEqual(lay.window, [1, 2])
            self.assertEqual(lay.node_arms, {1: ("with", 1), 2: ("without", 1)})
            self.assertEqual(lay.pulls, {"with": 1, "without": 1})


class MainLoopResumeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="main_resume_"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.run_dir = self.tmp / "run"
        (self.run_dir / "round_000").mkdir(parents=True)
        (self.run_dir / "round_000" / "hgm_node.json").write_text("{}")
        (self.run_dir / "config.snapshot.yaml").write_text("manager: {type: hgm}\n")

    def test_records_the_config_it_continues_with(self) -> None:
        from main_loop import _prepare_resume

        path = _prepare_resume(self.run_dir, None)
        self.assertEqual(path, self.run_dir / "config.snapshot.yaml")
        self.assertTrue((self.run_dir / "config.resume_001.yaml").is_file())
        _prepare_resume(self.run_dir, None)
        self.assertTrue((self.run_dir / "config.resume_002.yaml").is_file())

    def test_relative_run_dir_reaches_evolve_as_an_absolute_path(self) -> None:
        """`--resume runs/<run>` (relative to the shell's cwd) must reach the
        manager as an absolute path: the evaluator derives each case's trace
        and scratch paths from the round dirs and runs the case from inside
        round_NNN/task_agent/, where a relative path no longer resolves
        (2026-10-02: every re-evaluation of a restored node crashed with
        FileNotFoundError on logs/trace.jsonl and scored 0)."""
        import os
        from types import SimpleNamespace
        from unittest import mock

        import main_loop

        class _Stop(Exception):
            pass

        seen: dict = {}

        def evolve(**kw):
            seen.update(kw)
            raise _Stop

        fw = SimpleNamespace(
            manager=SimpleNamespace(evolve=evolve), editor=None, evaluator=None, gatherer=None,
            seed_dir=None, benchmark_dir=None, train_case_ids=None, eval_case_ids=None,
            summarizer=None, failure_summarizer=None, block_suggester=None, unit_selector=None,
            edit_memory=None)
        cfg = SimpleNamespace(loop=SimpleNamespace(max_rounds=1, score_target=None))
        cwd = os.getcwd()
        os.chdir(self.tmp)
        self.addCleanup(os.chdir, cwd)
        with mock.patch.object(main_loop.cfg_mod, "load", return_value=cfg), \
                mock.patch.object(main_loop.cfg_mod, "build_components", return_value=fw), \
                mock.patch.object(main_loop.runtime_env, "apply_all"), \
                mock.patch.object(main_loop, "_preflight_keys"):
            with self.assertRaises(_Stop):
                main_loop.run(None, resume=Path("run"))
        self.assertTrue(seen["experiment_dir"].is_absolute())
        self.assertEqual(seen["experiment_dir"], self.run_dir.resolve())
        self.assertTrue(seen["resume"])
        self.assertTrue((self.run_dir / "config.resume_001.yaml").is_file())

    def test_refusals(self) -> None:
        from main_loop import _prepare_resume

        other = self.tmp / "other.yaml"
        other.write_text("manager: {type: hgm}\nedit_memory: {type: agentic}\n")
        with self.assertRaisesRegex(ValueError, "edit_memory"):
            _prepare_resume(self.run_dir, other)
        (self.run_dir / "run_summary.md").write_text("done")
        with self.assertRaisesRegex(ValueError, "already finished"):
            _prepare_resume(self.run_dir, None)
        (self.run_dir / "run_summary.md").unlink()
        (self.run_dir / "round_000" / "hgm_node.json").unlink()
        with self.assertRaisesRegex(ValueError, "pre-evaluation"):
            _prepare_resume(self.run_dir, None)

    def test_managers_without_resume_say_so(self) -> None:
        from meta_agent.managers.hgm_dual import HGMDualManager

        with self.assertRaisesRegex(ValueError, "resume"):
            HGMDualManager(error_categorizer="projects.travel.travel_error_categorizer:categorize_errors"
                           ).evolve(editor=object(), evaluator=None, gatherer=None,
                                    seed_dir=Path("."), benchmark_dir=Path("."),
                                    experiment_dir=Path("."), max_rounds=5, score_target=None,
                                    resume=True)


if __name__ == "__main__":
    unittest.main()
