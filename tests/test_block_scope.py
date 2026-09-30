"""``block_suggester.block_scope``: the editor-facing part of each block body.

    PYTHONPATH=. python3 -m unittest tests.test_block_scope
"""
from __future__ import annotations

import unittest

from meta_agent.block_suggester import _BLOCK_BODIES, block_scope


class BlockScopeTests(unittest.TestCase):
    def test_every_block_has_a_scope_without_suggester_instructions(self) -> None:
        for block in _BLOCK_BODIES:
            with self.subTest(block=block):
                text = block_scope(block)
                self.assertIn("Scope:", text)
                self.assertNotIn("## Block:", text)
                self.assertNotIn("Output a short markdown suggestion", text)
                self.assertNotIn("{{", text)
                if "{{" not in _BLOCK_BODIES[block]:
                    self.assertIn(text, _BLOCK_BODIES[block])   # verbatim excerpt

    def test_plain_blocks_are_just_the_scope_paragraph(self) -> None:
        text = block_scope("verifiers")
        self.assertTrue(text.startswith("Scope: "))
        self.assertEqual(text.count("\n\n"), 0)

    def test_backbone_block_keeps_its_hard_constraint_and_renders_the_catalog(self) -> None:
        text = block_scope("llm_backbone_selection")
        self.assertTrue(text.startswith("HARD CONSTRAINT"))
        self.assertIn("\n\nScope: ", text)
        custom = block_scope("llm_backbone_selection",
                             backbone_catalog=[{"slug": "my/model-x", "note": "test entry"}])
        self.assertIn("my/model-x", custom)

    def test_unknown_block_raises(self) -> None:
        with self.assertRaises(KeyError):
            block_scope("nope")


if __name__ == "__main__":
    unittest.main()
