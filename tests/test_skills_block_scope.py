"""The opt-in "skills" block and the enforced per-block edit scope."""
import tempfile
import unittest
from pathlib import Path

from meta_agent.agent_editor import AgentEditor
from meta_agent.block_suggester import OPT_IN_BLOCKS, _BLOCK_BODIES, default_blocks
from meta_agent.managers.hgm import HGMManager


class SkillsBlockTests(unittest.TestCase):
    def test_skills_block_exists_and_is_opt_in(self):
        self.assertIn("skills", _BLOCK_BODIES)
        self.assertIn("skills", OPT_IN_BLOCKS)
        self.assertNotIn("skills", default_blocks())

    def test_active_blocks_can_name_skills(self):
        m = HGMManager(active_blocks=["skills", "verifiers"], block_selection_strategy="adaptive")
        self.assertIn("skills", m._block_bandit.blocks)

    def test_scope_validation(self):
        with self.assertRaises(ValueError):
            HGMManager(block_edit_scopes={"nope": ["x/"]})
        with self.assertRaises(ValueError):
            HGMManager(active_blocks=["verifiers"], block_edit_scopes={"skills": ["skills/"]})
        with self.assertRaises(ValueError):
            HGMManager(active_blocks=["skills"], block_edit_scopes={"skills": []})
        m = HGMManager(active_blocks=["skills", "verifiers"], block_edit_scopes={"skills": ["skills/"]})
        self.assertEqual(m._edit_scope_kwargs("skills"), {"edit_scope": ["skills/"]})
        self.assertEqual(m._edit_scope_kwargs("verifiers"), {})
        self.assertEqual(m._edit_scope_kwargs(None), {})


class EditScopeTests(unittest.TestCase):
    def _editor(self, scope):
        ed = AgentEditor.__new__(AgentEditor)
        ed.mutable_exclude = ["workflow.py"]
        ed._edit_scope = scope
        return ed

    def test_in_scope_prefix_and_exact(self):
        ed = self._editor(["skills/", "INDEX.md"])
        self.assertTrue(ed._in_scope("skills/a.md"))
        self.assertTrue(ed._in_scope("INDEX.md"))
        self.assertFalse(ed._in_scope("skillsX/a.md"))
        self.assertFalse(ed._in_scope("agents/flight.py"))
        self.assertTrue(self._editor(None)._in_scope("agents/flight.py"))

    def test_scope_violations_diff(self):
        with tempfile.TemporaryDirectory() as d:
            base, out = Path(d) / "base", Path(d) / "out"
            for root in (base, out):
                (root / "task_agent" / "skills").mkdir(parents=True)
                (root / "task_agent" / "agents").mkdir()
                (root / "task_agent" / "agents" / "f.py").write_text("x=1\n")
                (root / "task_agent" / "skills" / "a.md").write_text("a\n")
            ed = self._editor(["skills/"])
            (out / "task_agent" / "skills" / "b.md").write_text("new\n")
            (out / "task_agent" / "skills" / "a.md").write_text("changed\n")
            self.assertEqual(ed._scope_violations(out, base), [])
            (out / "task_agent" / "agents" / "f.py").write_text("x=2\n")
            (out / "task_agent" / "agents" / "g.py").write_text("y=1\n")
            v = ed._scope_violations(out, base)
            self.assertEqual(len(v), 2)
            self.assertTrue(all("edit scope violation" in e for e in v))
            self.assertEqual(self._editor(None)._scope_violations(out, base), [])

    def test_prompt_section_only_when_scoped(self):
        self.assertEqual(self._editor(None)._format_edit_scope(), [])
        txt = "".join(self._editor(["skills/"])._format_edit_scope())
        self.assertIn("Edit scope for this EXPAND (enforced)", txt)
        self.assertIn("skills/", txt)


if __name__ == "__main__":
    unittest.main()
