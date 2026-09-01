"""Tests for meta_agent/block_suggester.py's strategies.md support:
_parse_strategies_md, BlockSuggester._render_strategies, and end-to-end
injection into the system prompt passed to the LLM call.

    PYTHONPATH=. python3 -m unittest tests.test_block_suggester_strategies
"""
from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from meta_agent.block_suggester import BlockSuggester, _parse_strategies_md


class ParseStrategiesMdTests(unittest.TestCase):
    def test_general_and_block_sections_split_correctly(self) -> None:
        text = (
            "# Strategies\n"
            "some preamble text, ignored (no recognized header yet)\n\n"
            "## General\n"
            "- general strategy one\n"
            "- general strategy two\n\n"
            "## Block: verifiers\n"
            "- verifiers strategy\n\n"
            "## Block: individual_subagent\n"
            "- subagent strategy\n"
        )
        sections = _parse_strategies_md(text)
        self.assertEqual(set(sections), {"general", "verifiers", "individual_subagent"})
        self.assertIn("general strategy one", sections["general"])
        self.assertIn("general strategy two", sections["general"])
        self.assertEqual(sections["verifiers"], "- verifiers strategy")
        self.assertEqual(sections["individual_subagent"], "- subagent strategy")

    def test_unrecognized_header_ignored(self) -> None:
        text = "## Not A Real Section\nsome text\n\n## General\nreal content\n"
        sections = _parse_strategies_md(text)
        self.assertEqual(sections, {"general": "real content"})

    def test_empty_section_body_omitted(self) -> None:
        text = "## General\n\n## Block: verifiers\n- has content\n"
        sections = _parse_strategies_md(text)
        self.assertNotIn("general", sections)
        self.assertEqual(sections["verifiers"], "- has content")

    def test_empty_input_yields_empty_dict(self) -> None:
        self.assertEqual(_parse_strategies_md(""), {})

    def test_block_header_whitespace_tolerant(self) -> None:
        text = "##   Block:   collaboration_workflow   \ncontent\n"
        sections = _parse_strategies_md(text)
        self.assertEqual(sections, {"collaboration_workflow": "content"})

    def test_no_content_before_first_header_is_dropped(self) -> None:
        text = "this line has no header yet\n## General\nreal\n"
        sections = _parse_strategies_md(text)
        self.assertEqual(sections, {"general": "real"})


class RenderStrategiesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="strategies_test_"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.path = self.tmp / "strategies.md"

    def _bs(self, **kwargs) -> BlockSuggester:
        return BlockSuggester(llm_caller=lambda **kw: None, **kwargs)

    def test_disabled_by_default(self) -> None:
        bs = self._bs()
        self.assertEqual(bs.strategies_path, None)
        self.assertEqual(bs._render_strategies("verifiers"), "")

    def test_missing_file_is_a_silent_noop(self) -> None:
        bs = self._bs(strategies_path=str(self.tmp / "does_not_exist.md"))
        self.assertEqual(bs._render_strategies("verifiers"), "")

    def test_general_and_block_both_included(self) -> None:
        self.path.write_text(
            "## General\n- gen rule\n\n## Block: verifiers\n- verifiers rule\n",
            encoding="utf-8",
        )
        bs = self._bs(strategies_path=str(self.path))
        rendered = bs._render_strategies("verifiers")
        self.assertIn("gen rule", rendered)
        self.assertIn("verifiers rule", rendered)
        self.assertIn("## Strategies to consider", rendered)

    def test_block_without_a_section_gets_general_only(self) -> None:
        self.path.write_text("## General\n- gen rule\n", encoding="utf-8")
        bs = self._bs(strategies_path=str(self.path))
        rendered = bs._render_strategies("foundation_capability")
        self.assertIn("gen rule", rendered)

    def test_no_matching_sections_at_all_yields_empty_string(self) -> None:
        self.path.write_text(
            "## Block: verifiers\n- only verifiers has content\n", encoding="utf-8"
        )
        bs = self._bs(strategies_path=str(self.path))
        self.assertEqual(bs._render_strategies("individual_subagent"), "")

    def test_relative_path_resolved_against_cwd(self) -> None:
        self.path.write_text("## General\n- rel rule\n", encoding="utf-8")
        import os

        old_cwd = os.getcwd()
        os.chdir(self.tmp)
        try:
            bs = self._bs(strategies_path="strategies.md")
            rendered = bs._render_strategies("verifiers")
        finally:
            os.chdir(old_cwd)
        self.assertIn("rel rule", rendered)


class SuggestIntegrationTests(unittest.TestCase):
    """Confirms the rendered strategies text actually reaches the system
    prompt handed to the LLM call, end-to-end through suggest()."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="suggest_strategies_test_"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.agent_dir = self.tmp / "agent"
        self.agent_dir.mkdir()
        self.out_dir = self.tmp / "out"
        self.out_dir.mkdir()
        self.strategies_path = self.tmp / "strategies.md"
        self.strategies_path.write_text(
            "## General\n- always ground values in real data\n\n"
            "## Block: verifiers\n- make sure checks are acted on downstream\n",
            encoding="utf-8",
        )

    def test_system_prompt_includes_strategies_section(self) -> None:
        captured: dict[str, str] = {}

        def fake_llm(**kwargs):
            captured["system"] = kwargs["messages"][0]["content"]
            return SimpleNamespace(content="a suggestion")

        bs = BlockSuggester(
            llm_caller=fake_llm, strategies_path=str(self.strategies_path)
        )
        result = bs.suggest(
            block="verifiers",
            agent_dir=self.agent_dir,
            out_dir=self.out_dir,
            node_id=0,
        )
        self.assertEqual(result, "a suggestion")
        self.assertIn("always ground values in real data", captured["system"])
        self.assertIn("make sure checks are acted on downstream", captured["system"])

    def test_disabled_by_default_system_prompt_has_no_strategies_section(self) -> None:
        captured: dict[str, str] = {}

        def fake_llm(**kwargs):
            captured["system"] = kwargs["messages"][0]["content"]
            return SimpleNamespace(content="a suggestion")

        bs = BlockSuggester(llm_caller=fake_llm)  # strategies_path unset
        bs.suggest(
            block="verifiers", agent_dir=self.agent_dir, out_dir=self.out_dir, node_id=0,
        )
        self.assertNotIn("Strategies to consider", captured["system"])


if __name__ == "__main__":
    unittest.main()
