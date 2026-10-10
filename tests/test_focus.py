"""Reliability focus axis (meta_agent/focus.py, meta_agent/focus_bandit.py, HGMManager._select_focus):
target selection, steering text, the bandit's tallies, first-batch targeting, and exact resume.

    PYTHONPATH=. python3 -m pytest tests/test_focus.py -q
"""
from __future__ import annotations

import json
import random
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from meta_agent.focus import focus_body, reliability_targets
from meta_agent.focus_bandit import FocusBandit
from tests.test_hgm_resume import TRAIN, _HookEvaluator
from tests.test_hgm_smoke import _StubEditor
from tests.test_reflector import SECRET_TEST, _FakeScorer, _fake_chat

CASES = TRAIN[:8]


def _s(evals, passes, spread, unstable=True, file=None):
    return {"evals": evals, "passes": passes, "spread": spread, "unstable": unstable, "mean": 0.5,
            "file": file or "x.md", "contrast": None}


class TargetsAndBodyTests(unittest.TestCase):
    def test_targets_unstable_min_evals_parent_first_then_spread(self) -> None:
        summary = {"a": _s(3, 1, 0.4), "b": _s(5, 2, 0.9), "c": _s(2, 1, 1.0),       # c: too few evals
                   "d": _s(4, 4, 0.0, unstable=False), "e": _s(3, 2, 0.5)}
        self.assertEqual([t["case_id"] for t in reliability_targets(summary)], ["b", "e", "a"])
        self.assertEqual([t["case_id"] for t in reliability_targets(summary, prefer=["a"])], ["a", "b", "e"])
        self.assertEqual([t["case_id"] for t in reliability_targets(summary, k=1)], ["b"])
        self.assertEqual([t["case_id"] for t in reliability_targets(summary, min_evals=2)][0], "c")
        self.assertEqual(reliability_targets({}), [])

    def test_body_only_for_reliability_with_targets(self) -> None:
        t = reliability_targets({"a": _s(3, 1, 0.4, file="a.md")})
        self.assertEqual(focus_body(None, t), "")
        self.assertEqual(focus_body("default", t), "")
        self.assertEqual(focus_body("reliability", []), "")
        b = focus_body("reliability", t)
        self.assertIn("## Reliability focus for this EXPAND", b)
        self.assertIn("`a` -> a.md: 1/3 passed, score spread 0.40", b)
        self.assertIn("Pass/fail contrast", b)


class FocusBanditTests(unittest.TestCase):
    @staticmethod
    def _tree(spec):
        """spec: {node_id: (parent, mean, focus)}"""
        nodes, fb = {}, {}
        for nid, (par, mean, focus) in spec.items():
            nodes[nid] = SimpleNamespace(edit_failed=False, n_evals=4, parent_id=par, mean_utility=mean,
                                         n_success=mean * 4, n_failure=(1 - mean) * 4)
            fb[nid] = SimpleNamespace(strategy=SimpleNamespace(focus=focus))
        return SimpleNamespace(nodes=nodes), fb

    def test_boolean_increase_credits_the_stamped_focus(self) -> None:
        tree, fb = self._tree({0: (None, 0.5, None), 1: (0, 0.7, "reliability"), 2: (0, 0.3, "default"),
                               3: (0, 0.6, "reliability")})
        sel = FocusBandit(rng=random.Random(0)).select(tree, fb)
        p = sel.posteriors
        self.assertEqual((p["reliability"].n_success, p["reliability"].n_failure, p["reliability"].n_evals),
                         (2.0, 0.0, 2))
        self.assertEqual((p["default"].n_success, p["default"].n_failure), (0.0, 1.0))
        self.assertIn(sel.focus, ("default", "reliability"))
        frac = FocusBandit(rng=random.Random(0), reward_metric="fractional_score").select(tree, fb)
        self.assertAlmostEqual(frac.posteriors["reliability"].n_success, 5.2)
        with self.assertRaises(ValueError):
            FocusBandit(rng=random.Random(0), reward_metric="bogus")

    def test_allowed_restricts_the_choice(self) -> None:
        tree, fb = self._tree({0: (None, 0.5, None)})
        for seed in range(10):
            self.assertEqual(FocusBandit(rng=random.Random(seed)).select(tree, fb, allowed=["default"]).focus,
                             "default")


class _VaryingEvaluator(_HookEvaluator):
    """Scores depend on (case, node) so cases flip between nodes -> pass/fail contrasts exist."""

    @staticmethod
    def _score_for(cid: str, node: int) -> float:
        return 1.0 if (int(cid[1:]) + node) % 3 else 0.2

    def run(self, round_dir, benchmark_dir, *, case_ids=None):
        from meta_agent.models import CaseResult, EvaluationResult

        if self.hook is not None:
            self.hook(self.run_calls + 1, self.exp)
        self.run_calls += 1
        node = int(Path(round_dir).name[6:])
        per = [CaseResult(case_id=c, passed=self._score_for(c, node) >= 0.5, score=self._score_for(c, node))
               for c in (case_ids or [])]
        passed = sum(c.passed for c in per)
        return EvaluationResult(score=sum(c.score for c in per) / max(len(per), 1), passed=passed,
                                failed=len(per) - passed, per_case=per)


class FocusManagerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="focus_"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.seed = self.tmp / "seed"
        self.seed.mkdir()
        (self.seed / "workflow.py").write_text("def run_task(t):\n    return None\n")

    def _evolve(self, exp: Path, focus, hook=None, resume=False, contexts=None, reflector=True, **kw):
        from meta_agent.feedback_gatherer import DefaultFeedbackGatherer
        from meta_agent.managers.hgm import HGMManager
        from meta_agent.reflector import Reflector

        exp.mkdir(exist_ok=True)
        ctx = contexts if contexts is not None else []

        class Editor(_StubEditor):
            def apply(self, feedback, base_dir, out_dir, *, context=None, has_suggestion=False, **_):
                ctx.append(context or "")
                return super().apply(feedback, base_dir, out_dir, context=context,
                                     has_suggestion=has_suggestion)

        m = HGMManager(eval_budget=40, init_expansions=2, eval_batch_size=4, alpha=0.6, seed=7,
                       first_batch_on_expand=True, status_report=True, focus_selection_strategy=focus,
                       focus_min_evals=2, **kw)
        refl = Reflector(scorer=_FakeScorer(), chat_caller=_fake_chat([]), phases=["root", "expand"]) \
            if reflector else None
        m.evolve(editor=Editor(), evaluator=_VaryingEvaluator(exp, hook), gatherer=DefaultFeedbackGatherer(),
                 seed_dir=self.seed, benchmark_dir=self.tmp / "bench", experiment_dir=exp, max_rounds=30,
                 score_target=None, train_case_ids=CASES, eval_case_ids=None, reflector=refl,
                 **({"resume": True} if resume else {}))
        return m

    @staticmethod
    def _sig(m):
        return {nid: (n.parent_id, tuple(sorted(n.evaluated_case_ids)),
                      getattr(m._feedback[nid].strategy, "focus", None),
                      tuple(getattr(m._feedback[nid].strategy, "focus_targets", None) or ()))
                for nid, n in m._tree.nodes.items()}

    def test_config_validation(self) -> None:
        from meta_agent.managers.hgm import HGMManager

        with self.assertRaises(ValueError):
            HGMManager(eval_budget=10, focus_selection_strategy="sometimes")
        with self.assertRaises(ValueError):
            HGMManager(eval_budget=10, focus_selection_strategy="adaptive", focus_targets_k=0)
        with self.assertRaises(ValueError):                     # needs a reflector writing case files
            self._evolve(self.tmp / "noref", "adaptive", reflector=False)

    def test_forced_reliability_targets_first_batch_context_and_delta(self) -> None:
        contexts: list[str] = []
        m = self._evolve(self.tmp / "rel", "reliability", contexts=contexts)
        rel = [nid for nid in m._tree.nodes if getattr(m._feedback[nid].strategy, "focus", None) == "reliability"]
        self.assertTrue(rel, "no reliability EXPAND happened")
        self.assertTrue(any("## Reliability focus for this EXPAND" in c for c in contexts))
        self.assertFalse(any(SECRET_TEST in c for c in contexts))
        for nid in rel:
            node, st = m._tree[nid], m._feedback[nid].strategy
            self.assertTrue(st.focus_targets)
            first = list(node.evaluated_case_ids)[:4]      # insertion order = first batch first
            self.assertTrue(set(st.focus_targets) <= set(first), (nid, st.focus_targets, first))
            rec = json.loads((node.round_dir / "focus.json").read_text())
            self.assertEqual([t["case_id"] for t in rec["targets"]], st.focus_targets)
            self.assertIsNotNone(rec["targets_delta"])
            self.assertGreater(rec["targets_delta"]["n"], 0)
            self.assertTrue(json.loads((node.round_dir / "strategy.json").read_text()).get("focus"))
        status = (self.tmp / "rel" / "STATUS.md").read_text()
        self.assertIn("· reliability (targets Δ", status)
        # nodes expanded before any case had a contrast fall back to "default"
        self.assertTrue(all(getattr(m._feedback[n].strategy, "focus", None) in ("default", "reliability")
                            for n in m._tree.nodes if n != 0))

    def test_default_focus_is_byte_identical_to_off(self) -> None:
        off_ctx, def_ctx = [], []
        off = self._evolve(self.tmp / "off", None, contexts=off_ctx)
        dflt = self._evolve(self.tmp / "dflt", "default", contexts=def_ctx)
        self.assertEqual(off_ctx, def_ctx)
        self.assertEqual({k: v[:2] for k, v in self._sig(off).items()},
                         {k: v[:2] for k, v in self._sig(dflt).items()})
        self.assertTrue(all(getattr(off._feedback[n].strategy, "focus", None) is None for n in off._tree.nodes))
        self.assertFalse(list((self.tmp / "off").glob("round_*/focus.json")))

    def test_adaptive_pause_resume_is_exact(self) -> None:
        from meta_agent.managers.hgm import RunPaused

        straight = self._evolve(self.tmp / "straight", "adaptive")
        self.assertIn("_focus_rng", json.loads((self.tmp / "straight" / "loop_state.json").read_text())["rng_state"])
        for pause_call in (4, 7):
            exp = self.tmp / f"exp{pause_call}"

            def hook(call, e, n=pause_call):
                if call == n:
                    (e / "PAUSE").touch()
            with self.assertRaises(RunPaused):
                self._evolve(exp, "adaptive", hook=hook)
            m = self._evolve(exp, "adaptive", resume=True)
            self.assertEqual(self._sig(m), self._sig(straight), pause_call)
            self.assertEqual(m._budget_spent, 40)


if __name__ == "__main__":
    unittest.main()
