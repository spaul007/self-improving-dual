"""Integration (evolve_next only): every opt-in from the PR series switched on at once --
first_batch_on_expand, status_report, exclude_flagged_cases, and the reflector -- through a
pause -> truncated loop_state -> resume cycle, with stub editor/evaluator and a fake LLM.

    PYTHONPATH=. python -m pytest tests/test_integration_all_flags.py -q
"""
from __future__ import annotations

import json
import re
import shutil
import tempfile
import unittest
from pathlib import Path

from tests.test_hgm_resume import TRAIN, _HookEvaluator
from tests.test_hgm_smoke import _StubEditor
from tests.test_reflector import SECRET_TEST, _FakeScorer, _fake_chat


class AllFlagsIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="all_flags_"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.seed = self.tmp / "seed"
        self.seed.mkdir()
        (self.seed / "workflow.py").write_text("def run_task(t):\n    return None\n")

    def _evolve(self, exp: Path, hook=None, resume=False, contexts=None):
        from meta_agent.feedback_gatherer import DefaultFeedbackGatherer
        from meta_agent.managers.hgm import HGMManager
        from meta_agent.reflector import Reflector

        ctx = contexts if contexts is not None else []

        class Editor(_StubEditor):
            def apply(self, feedback, base_dir, out_dir, *, context=None, has_suggestion=False, **kw):
                ctx.append(context or "")
                return super().apply(feedback, base_dir, out_dir, context=context,
                                     has_suggestion=has_suggestion)

        m = HGMManager(eval_budget=40, init_expansions=2, eval_batch_size=4, alpha=0.6, seed=7,
                       snapshot_tree=True, first_batch_on_expand=True, status_report=True,
                       exclude_flagged_cases=True)
        refl = Reflector(scorer=_FakeScorer(), chat_caller=_fake_chat([]), phases=["root", "expand"])
        out = m.evolve(editor=Editor(), evaluator=_HookEvaluator(exp, hook),
                       gatherer=DefaultFeedbackGatherer(), seed_dir=self.seed,
                       benchmark_dir=self.tmp / "bench", experiment_dir=exp, max_rounds=30,
                       score_target=None, train_case_ids=TRAIN, eval_case_ids=None,
                       reflector=refl, **({"resume": True} if resume else {}))
        return m, out

    @staticmethod
    def _sig(m):
        return {nid: (n.parent_id, tuple(sorted(n.evaluated_case_ids))) for nid, n in m._tree.nodes.items()}

    @staticmethod
    def _index_rows(exp: Path) -> dict:
        rows = {}
        for line in (exp / "case_reflections" / "INDEX.md").read_text().splitlines():
            cells = [c.strip() for c in line.strip("|").split("|")]
            if len(cells) == 7 and cells[2].isdigit():
                rows[cells[0]] = (int(cells[2]), int(cells[3]))   # evals, passed
        return rows

    def _pause_at(self, n):
        def hook(call, e):
            if call == n:
                (e / "PAUSE").touch()
        return hook

    def test_pause_resume_with_everything_on_is_exact(self) -> None:
        from meta_agent.managers.hgm import RunPaused

        straight_dir = self.tmp / "straight"
        straight_dir.mkdir()
        straight, _ = self._evolve(straight_dir)
        for pause_call in (3, 6):
            exp = self.tmp / f"exp{pause_call}"
            exp.mkdir()
            with self.assertRaises(RunPaused):
                self._evolve(exp, self._pause_at(pause_call))
            contexts: list[str] = []
            m, _ = self._evolve(exp, resume=True, contexts=contexts)
            self.assertEqual(self._sig(m), self._sig(straight), pause_call)   # exact continuation
            self.assertEqual(m._budget_spent, 40)
            unevaluated = [nid for nid, n in m._tree.nodes.items() if not n.edit_failed and n.n_evals == 0]
            self.assertEqual(unevaluated, [])                            # first_batch_on_expand
            self.assertTrue((exp / "STATUS.md").is_file())               # status_report
            self.assertTrue(list((m._tree[0].round_dir / "reflections").glob("*.json")))
            self.assertTrue(any("Task-agent reflections" in c for c in contexts))
            self.assertFalse(any(SECRET_TEST in c for c in contexts))    # lessons_only redaction
            self.assertIn("rng_state", json.loads((exp / "loop_state.json").read_text()))
            # per-test-case files: rebuilt across the resume, same pass rates as the straight run,
            # entries from the root AND children, and no hidden-test name anywhere
            self.assertEqual(self._index_rows(exp), self._index_rows(straight_dir), pause_call)
            texts = [f.read_text() for f in (exp / "case_reflections").glob("*.md")]
            self.assertTrue(any("### node 0 " in t for t in texts))
            self.assertTrue(any(re.search(r"### node [1-9]", t) for t in texts))
            self.assertFalse(any(SECRET_TEST in t for t in texts))

    def test_corrupt_loop_state_still_spends_exact_budget(self) -> None:
        """RNG state lives in loop_state.json, so a corrupt one cannot continue the
        exact sequence (logged re-seed fallback) -- but budget, resume count and
        new-node evaluation must still be right."""
        from meta_agent.managers.hgm import RunPaused

        exp = self.tmp / "exp"
        exp.mkdir()
        with self.assertRaises(RunPaused):
            self._evolve(exp, self._pause_at(3))
        (exp / "loop_state.json").write_text("{")
        m, _ = self._evolve(exp, resume=True)
        self.assertEqual(m._budget_spent, 40)
        self.assertEqual(m._resume_count, 1)
        self.assertEqual([nid for nid, n in m._tree.nodes.items()
                          if not n.edit_failed and n.n_evals == 0], [])


if __name__ == "__main__":
    unittest.main()
