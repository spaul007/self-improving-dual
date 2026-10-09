"""`blocks:` on/off section + the skills-block fixes (harness_heavy exclusion,
implementation strategy dropped for skill-library edits, empty-pool and ranking checks)."""
from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from meta_agent.config import (
    BlockToggle,
    ResolvedSkills,
    apply_blocks_to_manager_config,
    apply_skills_to_manager_config,
)
from meta_agent.block_suggester import default_blocks


def _skills():
    return ResolvedSkills("skills/", "roles", ["a"], "# idx\n", "guide")


class BlocksConfigTests(unittest.TestCase):
    def test_empty_section_is_a_no_op(self) -> None:
        cfg = {"block_selection_strategy": "adaptive"}
        self.assertEqual(apply_blocks_to_manager_config(dict(cfg), {}, skills_enabled=False), cfg)

    def test_disabling_removes_from_default_set(self) -> None:
        out = apply_blocks_to_manager_config(
            {}, {"foundation_capability": BlockToggle(enabled=False),
                 "llm_backbone_selection": BlockToggle(enabled=False)}, skills_enabled=False)
        self.assertEqual(out["active_blocks"],
                         [b for b in default_blocks()
                          if b not in ("foundation_capability", "llm_backbone_selection")])

    def test_errors(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown block"):
            apply_blocks_to_manager_config({}, {"nope": BlockToggle()}, skills_enabled=False)
        with self.assertRaisesRegex(ValueError, "disabled in `blocks:` but listed"):
            apply_blocks_to_manager_config({"active_blocks": ["verifiers", "mixed"]},
                                           {"mixed": BlockToggle(enabled=False)}, skills_enabled=False)
        with self.assertRaisesRegex(ValueError, "disagrees with skills.enabled"):
            apply_blocks_to_manager_config({}, {"skills": BlockToggle(enabled=True)}, skills_enabled=False)

    def test_composes_with_skills(self) -> None:
        cfg = apply_blocks_to_manager_config(
            {"block_selection_strategy": "adaptive"},
            {"foundation_capability": BlockToggle(enabled=False), "skills": BlockToggle(enabled=True)},
            skills_enabled=True)
        cfg = apply_skills_to_manager_config(cfg, _skills())
        self.assertIn("skills", cfg["active_blocks"])
        self.assertNotIn("foundation_capability", cfg["active_blocks"])
        self.assertEqual(cfg["block_edit_scopes"]["skills"], ["skills/"])

    def test_ranking_must_list_skills(self) -> None:
        with self.assertRaisesRegex(ValueError, "block_initial_ranking must also list"):
            apply_skills_to_manager_config(
                {"block_selection_strategy": "adaptive",
                 "block_initial_ranking": ["verifiers", "mixed"]}, _skills())


class SkillsBlockFixTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="blocks_"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.seed = self.tmp / "seed"
        self.seed.mkdir()
        (self.seed / "workflow.py").write_text("def run_task(t):\n    return None\n")

    def test_empty_pool_fails_fast(self) -> None:
        from meta_agent.managers.hgm import HGMManager

        with self.assertRaisesRegex(ValueError, "leaves no block for a harness_heavy"):
            HGMManager(block_selection_strategy="non_adaptive",
                       active_blocks=["skills", "llm_backbone_selection"],
                       implementation_strategy_selection_strategy="non_adaptive")

    def test_harness_heavy_never_targets_skills(self) -> None:
        from meta_agent.managers.hgm import _HARNESS_HEAVY_BLOCK_EXCLUDE, HGMManager

        self.assertIn("skills", _HARNESS_HEAVY_BLOCK_EXCLUDE)
        m = HGMManager(block_selection_strategy="non_adaptive", active_blocks=["skills", "verifiers"])
        picks = {m._select_block(None, exclude=set(_HARNESS_HEAVY_BLOCK_EXCLUDE)) for _ in range(50)}
        self.assertEqual(picks, {"verifiers"})

    def test_skills_edit_records_no_implementation_strategy(self) -> None:
        from meta_agent.feedback_gatherer import DefaultFeedbackGatherer
        from meta_agent.managers.hgm import HGMManager
        from tests.test_hgm_smoke import _StubEditor, _StubEvaluator

        exp = self.tmp / "exp"
        exp.mkdir()
        m = HGMManager(eval_budget=8, init_expansions=2, eval_batch_size=4, alpha=0.6, seed=3,
                       block_selection_strategy="non_adaptive", active_blocks=["skills"],
                       implementation_strategy_selection_strategy="llm_heavy")
        m.evolve(editor=_StubEditor(), evaluator=_StubEvaluator(), gatherer=DefaultFeedbackGatherer(),
                 seed_dir=self.seed, benchmark_dir=self.tmp / "b", experiment_dir=exp, max_rounds=30,
                 score_target=None, train_case_ids=[f"c{i}" for i in range(8)], eval_case_ids=None)
        strategies = [json.loads(p.read_text()) for p in exp.glob("round_*/strategy.json")
                      if p.parent.name != "round_000"]
        self.assertTrue(strategies)
        for st in strategies:
            self.assertEqual(st.get("block"), "skills")
            self.assertIsNone(st.get("implementation_strategy"))


if __name__ == "__main__":
    unittest.main()
