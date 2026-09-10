"""Unit tests for belief-mode steering (meta_agent/steering.py) and the
manager branch that selects it.

No LLM anywhere.

    PYTHONPATH=. python3 -m unittest tests.test_steering
"""
from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from meta_agent.managers.hgm import HGMManager
from meta_agent.managers.hgm_tree import HGMNode, HGMTree
from meta_agent.models import AgentFeedback, CaseResult, EvaluationResult, EvolutionStrategy
from meta_agent.steering import compact_sibling_lines, render_belief_steering
from tests.test_edit_archive import write_record

LEGACY_MARKERS = ("aim to beat it", "make a DIFFERENT change", "BUILD ON",
                  "How to read the belief document", "Edit memory — the run's",
                  "Make targeted improvement", "What has been tried, by strategy",
                  "### Every edit, oldest first", "propose a promising exploratory",
                  "ABSOLUTE benchmark score", "most affect the score")

DOC = ("### belief:add-verifier-helps — t\n- kind: strategy\n"
       "- scope: strategy=add-verifier\n- predict: p=0.65\n"
       "- evidence: [node 2: Δ+0.0500/8]\n- next: extend it\n"
       "- track: n=1 · Brier 0.12\n")


def _body(what, extra=""):
    return ("## Edit 1\n- **name**: `x`\n"
            "- **category level 1 (strategy)**: `add-verifier`\n"
            "- **category level 2 (area)**: `routing`\n"
            f"- **what**: {what}\n- **why**: y" + extra)


class TestRender(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        # node 2: helped, with a dead component + suspect verifier + unsound
        write_record(self.tmp, 2, 1, _body(
            "Adds a route gate",
            "\n- **new tools**: `gate` **0 calls** · SUSPECT VERIFIER"
            "\n\n## Analysis\n- **implementation**: unsound — never fired"),
            delta=0.05, n_shared=8)
        # node 3: hurt, clean
        write_record(self.tmp, 3, 1, _body("Reworks the budget"), delta=-0.05,
                     n_shared=9)

    def _render(self, **kw):
        base = dict(
            experiment_dir=self.tmp, parent_id=1,
            lineage=[(0, "Seed agent (HGM tree root)."), (1, "Pre-compute routes")],
            parent_score=(0.5, 10),
            run_context={"seed_mean": 0.45, "seed_n": 60, "best_mean": 0.55,
                         "best_n": 32, "best_node": 3},
            siblings=[(2, "Adds a route gate\nsecond line", False),
                      (3, "Reworks the budget", False), (4, "Broken edit", True)],
            belief_doc=DOC, calibration_line="2 scored prediction(s) so far.",
            threshold=0.02, min_shared=8)
        base.update(kw)
        return render_belief_steering(**base)

    def test_sections_and_order(self):
        out = self._render()
        idx = [out.index(h) for h in (
            "## Objective",
            "## What the judge found on this parent (node 1)",
            "## Scope of this edit",
            "## Edits already applied along this lineage (root → parent)",
            "## Edits already tried directly off this parent (node 1)",
            "## Belief document")]
        self.assertEqual(idx, sorted(idx))
        self.assertTrue(out.startswith("## Objective\nFix what the judge found."))
        # The score is one labelled context line, never the objective.
        self.assertIn("Score context (a noisy reference, not the objective): "
                      "seed 0.4500/60 · best so far 0.5500/32 (node 3) · this "
                      "parent (node 1) 0.5000/10.", out)
        self.assertNotIn("ABSOLUTE", out)
        self.assertIn("(no record for this parent)", out)   # node 1 has no record
        self.assertIn("ONE targeted, coherent change", out)
        self.assertIn("trace.log", out)
        self.assertIn("  [depth 1] Pre-compute routes", out)
        self.assertNotIn("[depth 0]", out)
        # Sibling lines: judge first (even when not yet judged), score as context.
        node2 = ('- node 2: judge not yet judged · score helped Δ+0.0500/8 shared · '
                 'child 0.5000/10 · "Adds a route gate" · flags: dead component, '
                 'suspect verifier, implementation unsound')
        self.assertIn(node2, out)
        self.assertLess(node2.index("judge"), node2.index("score "))
        self.assertIn('- node 3: judge not yet judged · score hurt Δ-0.0500/9 shared · '
                      'child 0.4000/11 · "Reworks the budget"', out)
        self.assertIn("Per sibling: the judge's effect verdict", out)
        self.assertIn('- node 4: edit failed · "Broken edit"', out)
        self.assertIn("retrievable by node id", out)
        self.assertIn("2 scored prediction(s) so far.", out)
        self.assertIn("- track: n=1 · Brier 0.12", out)
        self.assertTrue(out.rstrip().endswith("- track: n=1 · Brier 0.12"))
        for marker in LEGACY_MARKERS:
            self.assertNotIn(marker, out)

    def test_belief_document_is_never_cut(self):
        big = DOC + "".join(
            f"### belief:b{i} — t\n- kind: strategy\n- scope: strategy=s{i}\n"
            f"- predict: p=0.5\n- evidence: {'e' * 450}\n- next: n\n"
            for i in range(450))
        self.assertGreater(len(big), 200_000)
        out = self._render(belief_doc=big)
        self.assertIn(big.strip(), out)
        self.assertNotIn("chars elided", out)

    def test_unevaluated_parent_no_siblings_no_doc(self):
        out = self._render(parent_score=None, siblings=[], belief_doc="",
                           calibration_line="", lineage=[(0, "seed")],
                           run_context={})
        self.assertIn("this parent (node 1) not yet evaluated.", out)
        self.assertNotIn("seed 0.", out)
        self.assertNotIn("## Edits already applied along this lineage", out)
        self.assertIn("(none yet)", out)
        self.assertIn("(no belief document yet", out)

    def test_judged_parent_and_judged_sibling(self):
        """The judge's findings lead: the parent gets its own section and a
        judged sibling shows verdict, targets and regressions before the
        score; the seed parent gets an explicit no-edit note."""
        write_record(self.tmp, 1, 0, _body(
            "Parent edit",
            "\n\n## Analysis\n- **implementation**: sound — gate fired on 8 cases"
            "\n- **implementation (edit 1)**: sound — gate fired on 8 cases"
            "\n- **effect (edit 1)**: improved (strong; targets: opening_hours) "
            "— opening_hours fixed on 6 of 8 cases where the gate fired"
            "\n- **regressions**: budget 1->3 fails (-2)"),
            delta=0.03, n_shared=16)
        write_record(self.tmp, 5, 1, _body(
            "Adds a meal validator",
            "\n\n## Analysis\n- **implementation**: sound — ran"
            "\n- **effect (edit 1)**: no_effect (moderate; targets: essential_meal_coverage) "
            "— the check fired but the final plans did not change"),
            delta=-0.01, n_shared=3)
        out = self._render(siblings=[(5, "Adds a meal validator", False)])
        self.assertIn("## What the judge found on this parent (node 1)\n"
                      "- judge improved (strong; targets: opening_hours) · "
                      "regressions: budget 1->3 fails (-2) · implementation sound "
                      '— "opening_hours fixed on 6 of 8 cases where the gate fired"',
                      out)
        self.assertIn("- node 5: judge no_effect (moderate; targets: "
                      "essential_meal_coverage) · score Δ", out)
        self.assertLess(out.index("judge no_effect"), out.index("score Δ"))
        seed = self._render(parent_id=0, lineage=[(0, "seed")], siblings=[])
        self.assertIn("## What the judge found on this parent (node 0)\n"
                      "(node 0 is the seed — no edit to judge)", seed)

    def test_compact_lines_without_records(self):
        lines = compact_sibling_lines({}, [(9, "goal", False), (10, "g2", True)],
                                      threshold=0.02, min_shared=8)
        self.assertEqual(lines, ['- node 9: unmeasured · "goal"',
                                 '- node 10: edit failed · "g2"'])


class _EM:
    """The attributes the manager reads off an edit-memory component."""

    def __init__(self, mode, doc=DOC):
        self.steering = True
        self.steering_mode = mode
        self.steering_token_budget = 48000
        self.verdict_threshold = 0.02
        self.min_shared = 8
        self.doc = doc

    def render_belief_block(self):
        return self.doc

    def belief_calibration_line(self):
        return "1 scored prediction(s) so far."


def _fb(n, goal):
    return AgentFeedback(round_number=n, base_round=max(n - 1, 0),
                         strategy=EvolutionStrategy(target_files=["workflow.py"],
                                                    optimization_goal=goal,
                                                    proposed_changes="p"),
                         eval_result=EvaluationResult(score=0.5))


class TestManagerBranch(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        write_record(self.tmp, 2, 1, _body("Adds a route gate"), delta=0.05,
                     n_shared=8)
        self.mgr = HGMManager()
        self.mgr._experiment_dir = self.tmp
        self.mgr._tree = HGMTree()
        for nid, parent in ((0, None), (1, 0), (2, 1)):
            node = HGMNode(node_id=nid, parent_id=parent,
                           round_dir=self.tmp / f"round_{nid:03d}")
            for i in range(4):
                node.record(CaseResult(case_id=f"c{i}", passed=True,
                                       score=0.5 + 0.05 * nid))
            self.mgr._tree.add(node)
        self.mgr._feedback = {0: _fb(0, "Seed agent (HGM tree root)."),
                              1: _fb(1, "Pre-compute routes"),
                              2: _fb(2, "Adds a route gate")}
        self.parent = self.mgr._tree[1]

    def test_belief_mode_uses_the_steering_module(self):
        self.mgr._edit_memory = _EM("belief")
        ctx = self.mgr._render_expand_context(self.parent)
        self.assertTrue(ctx.startswith("## Objective"))
        self.assertIn("this parent (node 1) 0.5500/4", ctx)
        self.assertIn("best so far 0.6000/4 (node 2)", ctx)
        self.assertIn("  [depth 1] Pre-compute routes", ctx)
        self.assertIn('- node 2: judge not yet judged · score helped Δ+0.0500/8', ctx)
        self.assertIn("## What the judge found on this parent (node 1)", ctx)
        self.assertIn("## Belief document", ctx)
        self.assertIn("### belief:add-verifier-helps", ctx)
        self.assertIn("1 scored prediction(s) so far.", ctx)
        for marker in LEGACY_MARKERS:
            self.assertNotIn(marker, ctx)

    def test_full_mode_keeps_the_legacy_context(self):
        self.mgr._edit_memory = _EM("full")
        ctx = self.mgr._render_expand_context(self.parent)
        self.assertIn("aim to beat it", ctx)
        self.assertIn("Make targeted improvement", ctx)
        self.assertNotIn("## Objective", ctx)

    def test_no_edit_memory_keeps_the_legacy_context(self):
        self.mgr._edit_memory = None
        ctx = self.mgr._render_expand_context(self.parent)
        self.assertIn("aim to beat it", ctx)
        self.assertIn("Make targeted improvement", ctx)
        self.assertNotIn("## Objective", ctx)


if __name__ == "__main__":
    unittest.main()
