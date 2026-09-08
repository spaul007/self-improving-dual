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

    def test_mixed_gets_every_block_section_not_just_one(self) -> None:
        self.path.write_text(
            "## General\n- gen rule\n\n"
            "## Block: verifiers\n- verifiers rule\n\n"
            "## Block: individual_subagent\n- subagent rule\n\n"
            "## Block: collaboration_workflow\n- collab rule\n\n"
            "## Block: foundation_capability\n- foundation rule\n",
            encoding="utf-8",
        )
        bs = self._bs(strategies_path=str(self.path))
        rendered = bs._render_strategies("mixed")
        for expected in (
            "gen rule", "verifiers rule", "subagent rule",
            "collab rule", "foundation rule",
        ):
            self.assertIn(expected, rendered)

    def test_mixed_with_no_sections_at_all_yields_empty_string(self) -> None:
        self.path.write_text("no headers here\n", encoding="utf-8")
        bs = self._bs(strategies_path=str(self.path))
        self.assertEqual(bs._render_strategies("mixed"), "")

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

    def test_mixed_block_is_accepted_and_gets_every_strategies_section(self) -> None:
        captured: dict[str, str] = {}

        def fake_llm(**kwargs):
            captured["system"] = kwargs["messages"][0]["content"]
            return SimpleNamespace(content="a mixed suggestion")

        bs = BlockSuggester(
            llm_caller=fake_llm, strategies_path=str(self.strategies_path)
        )
        result = bs.suggest(
            block="mixed", agent_dir=self.agent_dir, out_dir=self.out_dir, node_id=0,
        )
        self.assertEqual(result, "a mixed suggestion")
        self.assertIn("## Block: mixed", captured["system"])
        self.assertIn("All strategies in strategies.md are applicable", captured["system"])
        # Both sections from self.strategies_path reach a mixed-block
        # prompt, not just one filtered slice.
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

    def test_curriculum_directive_reaches_the_system_prompt(self) -> None:
        captured: dict[str, str] = {}

        def fake_llm(**kwargs):
            captured["system"] = kwargs["messages"][0]["content"]
            return SimpleNamespace(content="a suggestion")

        bs = BlockSuggester(llm_caller=fake_llm)  # strategies_path unset
        bs.suggest(
            block="verifiers", agent_dir=self.agent_dir, out_dir=self.out_dir, node_id=0,
            curriculum_directive="Focus on `reasonable_transfer_time`.",
        )
        self.assertIn("## Current curriculum focus", captured["system"])
        self.assertIn("reasonable_transfer_time", captured["system"])

    def test_curriculum_directive_none_reproduces_byte_identical_system_prompt(
        self,
    ) -> None:
        captured: dict[str, str] = {}

        def fake_llm(**kwargs):
            captured["system"] = kwargs["messages"][0]["content"]
            return SimpleNamespace(content="a suggestion")

        bs = BlockSuggester(llm_caller=fake_llm)

        bs.suggest(
            block="verifiers", agent_dir=self.agent_dir, out_dir=self.out_dir, node_id=0,
        )
        system_omitted = captured["system"]

        bs.suggest(
            block="verifiers", agent_dir=self.agent_dir, out_dir=self.out_dir, node_id=0,
            curriculum_directive=None,
        )
        system_explicit_none = captured["system"]

        self.assertEqual(system_omitted, system_explicit_none)
        self.assertNotIn("Current curriculum focus", system_omitted)

    def test_no_plan_rate_reaches_the_feedback_digest(self) -> None:
        # no_plan_rate is a plain float inside project_metrics -- now
        # rendered generically via render_metrics (colon format), not the
        # old special-cased "no_plan_rate=X" line.
        from meta_agent.models import AgentFeedback, EvaluationResult, EvolutionStrategy

        captured: dict[str, str] = {}

        def fake_llm(**kwargs):
            captured["user"] = kwargs["messages"][1]["content"]
            return SimpleNamespace(content="a suggestion")

        bs = BlockSuggester(llm_caller=fake_llm)
        feedback = AgentFeedback(
            round_number=0, base_round=0,
            strategy=EvolutionStrategy(
                target_files=[], optimization_goal="g", proposed_changes="x",
            ),
            eval_result=EvaluationResult(score=0.3, passed=0, failed=10),
            project_metrics={"no_plan_rate": 0.42},
        )
        bs.suggest(
            block="verifiers", agent_dir=self.agent_dir, out_dir=self.out_dir,
            node_id=0, feedback=feedback,
        )
        self.assertIn("project metrics:", captured["user"])
        self.assertIn("no_plan_rate: 0.420", captured["user"])

    def test_no_plan_rate_absent_omits_the_project_metrics_section(self) -> None:
        from meta_agent.models import AgentFeedback, EvaluationResult, EvolutionStrategy

        captured: dict[str, str] = {}

        def fake_llm(**kwargs):
            captured["user"] = kwargs["messages"][1]["content"]
            return SimpleNamespace(content="a suggestion")

        bs = BlockSuggester(llm_caller=fake_llm)
        feedback = AgentFeedback(
            round_number=0, base_round=0,
            strategy=EvolutionStrategy(
                target_files=[], optimization_goal="g", proposed_changes="x",
            ),
            eval_result=EvaluationResult(score=0.3, passed=0, failed=10),
        )
        bs.suggest(
            block="verifiers", agent_dir=self.agent_dir, out_dir=self.out_dir,
            node_id=0, feedback=feedback,
        )
        self.assertNotIn("no_plan_rate", captured["user"])
        self.assertNotIn("project metrics:", captured["user"])

    def test_project_metrics_top_failed_checks_reaches_the_feedback_digest(self) -> None:
        # The core regression this change fixes: top_failed_checks (and
        # anything else in project_metrics) used to be entirely invisible
        # to the block suggester -- only no_plan_rate was special-cased.
        from meta_agent.models import AgentFeedback, EvaluationResult, EvolutionStrategy

        captured: dict[str, str] = {}

        def fake_llm(**kwargs):
            captured["user"] = kwargs["messages"][1]["content"]
            return SimpleNamespace(content="a suggestion")

        bs = BlockSuggester(llm_caller=fake_llm)
        feedback = AgentFeedback(
            round_number=0, base_round=0,
            strategy=EvolutionStrategy(
                target_files=[], optimization_goal="g", proposed_changes="x",
            ),
            eval_result=EvaluationResult(score=0.3, passed=0, failed=10),
            project_metrics={
                "top_failed_checks": [["check_a", 5], ["check_b", 2]]
            },
        )
        bs.suggest(
            block="verifiers", agent_dir=self.agent_dir, out_dir=self.out_dir,
            node_id=0, feedback=feedback,
        )
        self.assertIn("check_a", captured["user"])
        self.assertIn("check_b", captured["user"])

    def test_project_metrics_cap_is_ten_not_five_or_fifteen(self) -> None:
        from meta_agent.models import AgentFeedback, EvaluationResult, EvolutionStrategy

        captured: dict[str, str] = {}

        def fake_llm(**kwargs):
            captured["user"] = kwargs["messages"][1]["content"]
            return SimpleNamespace(content="a suggestion")

        bs = BlockSuggester(llm_caller=fake_llm)
        feedback = AgentFeedback(
            round_number=0, base_round=0,
            strategy=EvolutionStrategy(
                target_files=[], optimization_goal="g", proposed_changes="x",
            ),
            eval_result=EvaluationResult(score=0.3, passed=0, failed=10),
            project_metrics={
                "top_failed_checks": [[f"check_{i}", 15 - i] for i in range(15)]
            },
        )
        bs.suggest(
            block="verifiers", agent_dir=self.agent_dir, out_dir=self.out_dir,
            node_id=0, feedback=feedback,
        )
        shown = sum(
            1 for i in range(15) if f"check_{i}:" in captured["user"]
        )
        self.assertEqual(shown, 10)


if __name__ == "__main__":
    unittest.main()
