"""Config-gated skill evolution (`skills: {enabled: ...}`).

Disabled (default): no skills block, no guide text, no library seeding -- prompts unchanged.
Enabled: the opt-in `skills` block joins active_blocks with an edit scope of the library,
the editor and block suggester get the project's skills guide, and the root starts with an
EMPTY library (header-only INDEX.md; seed skill files removed).
"""
from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

from meta_agent import config as cfg_mod
from meta_agent.managers.hgm import HGMManager

CFG = "configs/hgm_travel_full_scale_block_tagged_X100Y180.yaml"
SEED = Path("projects/travel_mas_refactored/seed")


def _load(enabled: bool, active_blocks=None):
    cfg = cfg_mod.load(CFG)
    cfg.skills = cfg_mod.SkillsSpec(enabled=enabled)
    if active_blocks is not None:
        cfg.manager.config["active_blocks"] = active_blocks
    return cfg


class GatingTests(unittest.TestCase):
    def test_disabled_changes_nothing(self):
        fw = cfg_mod.build_components(_load(False))
        self.assertIsNone(fw.manager.skills_library)
        self.assertIsNone(getattr(fw.editor, "skills_guide", None))
        self.assertEqual(fw.editor._skills_section(), "")
        if fw.block_suggester is not None:
            self.assertIsNone(fw.block_suggester.skills_guide)
            self.assertEqual(fw.block_suggester._render_skills(), "")
        blocks = getattr(fw.manager._block_bandit, "blocks", ()) if fw.manager._block_bandit else ()
        self.assertNotIn("skills", blocks)
        self.assertNotIn("skills", fw.manager.block_edit_scopes)

    def test_skills_block_rejected_when_disabled(self):
        with self.assertRaises(ValueError):
            cfg_mod.build_components(_load(False, ["verifiers", "skills"]))

    def test_enabled_wires_block_scope_guide_and_library(self):
        fw = cfg_mod.build_components(_load(True, ["verifiers", "mixed"]))
        m = fw.manager
        self.assertIn("skills", m.active_blocks)
        self.assertEqual(m.block_edit_scopes["skills"], ["skills/"])
        self.assertEqual(m.skills_library["dir"], "skills/")
        self.assertIn("Valid stages: flight, train, sightseeing, accounting", m.skills_library["index_header"])
        self.assertIn("SKILL LIBRARY", fw.editor.skills_guide)
        self.assertIn("with_inline_skills", fw.editor.skills_guide)   # project guide body
        self.assertIn("SKILL LIBRARY", fw.editor._skills_section())
        if fw.block_suggester is not None:
            self.assertIn("## Skill library", fw.block_suggester._render_skills())

    def test_enabled_with_default_blocks_adds_skills(self):
        cfg = _load(True)
        cfg.manager.config.pop("active_blocks", None)
        fw = cfg_mod.build_components(cfg)
        self.assertIn("skills", fw.manager.active_blocks)

    def test_missing_guide_is_an_error(self):
        cfg = _load(True)
        cfg.skills.guide = "projects/travel_mas_refactored/does_not_exist.md"
        with self.assertRaises(FileNotFoundError):
            cfg_mod.build_components(cfg)


class EmptyLibraryTests(unittest.TestCase):
    def test_root_library_starts_empty(self):
        with tempfile.TemporaryDirectory() as d:
            agent = Path(d)
            (agent / "skills").mkdir()
            (agent / "skills" / "old.md").write_text("seed skill")
            (agent / "skills" / "INDEX.md").write_text("- old | stages: flight | x\n")
            m = HGMManager()
            m.skills_library = {"dir": "skills/", "index_header": "# Skill index\n"}
            m._seed_empty_skill_library(agent)
            self.assertEqual(sorted(p.name for p in (agent / "skills").iterdir()), ["INDEX.md"])
            self.assertEqual((agent / "skills" / "INDEX.md").read_text(), "# Skill index\n")

    def test_no_library_when_disabled(self):
        with tempfile.TemporaryDirectory() as d:
            HGMManager()._seed_empty_skill_library(Path(d))
            self.assertFalse((Path(d) / "skills").exists())


class TravelLoaderTests(unittest.TestCase):
    def _skills_module(self, root: Path):
        spec = importlib.util.spec_from_file_location("tskills", SEED / "agents" / "skills.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.SKILLS_DIR = root / "skills"
        return mod

    def test_inert_without_library_or_with_empty_index(self):
        with tempfile.TemporaryDirectory() as d:
            mod = self._skills_module(Path(d))
            self.assertEqual(mod.with_inline_skills("PROMPT", "flight"), "PROMPT")
            (Path(d) / "skills").mkdir()
            (Path(d) / "skills" / "INDEX.md").write_text("# Skill index\n\nOne line per skill: `- <name> | ...`\n")
            self.assertEqual(mod.with_inline_skills("PROMPT", "flight"), "PROMPT")

    def test_inlines_only_for_tagged_stage(self):
        with tempfile.TemporaryDirectory() as d:
            lib = Path(d) / "skills"
            lib.mkdir()
            (lib / "INDEX.md").write_text("# h\n- gap-check | stages: sightseeing | when scheduling\n")
            (lib / "gap-check.md").write_text("# gap-check\nSTEPS")
            mod = self._skills_module(Path(d))
            self.assertIn("STEPS", mod.with_inline_skills("P", "sightseeing"))
            self.assertEqual(mod.with_inline_skills("P", "flight"), "P")


if __name__ == "__main__":
    unittest.main()
